import re
import csv
import fitz
import string
from docx import Document
from transformers import BartTokenizer, BartForConditionalGeneration


def load_model_and_tokenizer():
    """
    Load the BART model and tokenizer from the Hugging Face Transformers library.

    Returns:
        BartTokenizer, BartForConditionalGeneration: The tokenizer and model objects.
    """
    model_name_or_path = "facebook/bart-large-cnn"
    tokenizer = BartTokenizer.from_pretrained(model_name_or_path)
    model = BartForConditionalGeneration.from_pretrained(model_name_or_path)
    return model, tokenizer

def extract_text_from_pdf(pdf_path):
    """
    Extracts text from a PDF file using the PyMuPDF library (fitz).

    Args:
        pdf_path (str): Path to the PDF file.

    Returns:
        str: Cleaned text extracted from the PDF.
    """
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return clean_text(text)

def extract_text_from_word(docx_path):
    """
    Extracts text from a Word document using the python-docx library.

    Args:
        docx_path (str): Path to the Word document.

    Returns:
        str: Cleaned text extracted from the Word document.
    """
    doc = Document(docx_path)
    return "\n".join([clean_text(paragraph.text) for paragraph in doc.paragraphs])

def clean_text(text):
    """
    Cleans the text by replacing specific characters with their desired replacements.
    
    Args:
        text (str): The input text to clean.
    
    Returns:
        str: The cleaned text.
    """
    replacements = {
        "’": "'",
        "–": "-"
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text

def load_criteria(file_path):
    """
    Load criteria and context filters from a text file.

    Args:
        file_path (str): Path to the text file containing criteria.

    Returns:
        List, Dict: List of criteria words and a dictionary of primary words and their associated context filters.
    """
    criteria = []
    context_filters = {}

    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        if ":" in line:
            word, context = line.split(":", 1)
            word = word.strip()
            context_terms = [term.strip().strip('"') for term in context.split(",")]
            context_filters[word] = context_terms
            criteria.append(word)
        else:
            word = line.strip()
            criteria.append(word)

    return criteria, context_filters

def find_matching_criteria_with_window(text, criteria, context_filters, window_size=3):
    """
    Find matches for criteria in the text, considering context filters and wildcards.

    Args:
        text (str): The input text to search.
        criteria (list): List of primary words to match.
        context_filters (dict): Dictionary of primary words and their associated context filters.
        window_size (int): The size of the window for context filtering.

    Returns:
        list: List of matches, including primary words and word-context combinations.
    """
    matches = []

    for criterion in criteria:
        # If the criterion has a wildcard, build a regex for it
        if "*" in criterion:
            regex = r'\b' + re.escape(criterion).replace(r'\*', r'\w*') + r'\b'
        else:
            regex = r'\b' + re.escape(criterion) + r'\b'

        # Find matches for the primary word
        for match in re.finditer(regex, text, flags=re.IGNORECASE):
            word = match.group(0).lower().strip(string.punctuation)  # Remove trailing punctuation

            # If there are no context filters, add the word to matches
            if (criterion not in context_filters) and (word not in matches):
                matches.append(word)
            elif criterion in context_filters:
                # Handle context filters with wildcards
                context_terms = context_filters[criterion]
                context_regexes = [
                    (r'\b' + re.escape(term).replace(r'\*', r'\w*') + r'\b') if "*" in term
                    else (r'\b' + re.escape(term) + r'\b')
                    for term in context_terms
                ]

                # Get the surrounding words for the context window
                start_idx = match.start()
                words_before = text[:start_idx].split()[-window_size+1:]
                words_after = text[start_idx + len(match.group(0)):].split()[:window_size]
                context_window = words_before + [word] + words_after

                # Look for context terms in the context window
                for context_regex in context_regexes:
                    for w in context_window:
                        w_cleaned = w.strip(string.punctuation)  # Remove trailing punctuation from context word
                        if re.search(context_regex, w_cleaned, flags=re.IGNORECASE):
                            combination = f"{word} {w_cleaned}"  # Use the cleaned words from the text
                            if combination not in matches:
                                matches.append(combination)
    return matches

def format_prompt(matches):
    """
    Create a prompt based on the found criteria matches.
    """
    if not matches:
        return "No relevant criteria found in the text."
    
    # Format the prompt with the matched terms
    prompt = "Summarize this text focusing only on details related to: " + ", ".join(matches) + "."
    return prompt

def split_text_by_paragraphs(text):
    """
    Splits text into paragraphs based on likely paragraph boundaries.
    - Ensures list items stay together within the same paragraph.
    - Separates sections based on common section header keywords (e.g., "RADIOGRAPHIC EVALUATION").
    """
    # Normalize line breaks
    normalized_text = text.replace('\r\n', '\n').replace('\r', '\n')

    # Define common section headers that should act as paragraph boundaries
    section_headers = ["CLINICAL EXAMINATION", "RADIOGRAPHIC EVALUATION", "WATERS VIEW", "IMPRESSION"]
    header_pattern = r'(' + '|'.join(section_headers) + r')\.?'

    # Split based on double newlines, numbered lists, or section headers
    paragraphs = re.split(r'\n\s*\n|\n(?=\d+\.\s)|\n(?=\-)|\n(?=\*)|' + header_pattern, normalized_text)

    merged_paragraphs = []
    current_paragraph = ""

    for para in paragraphs:
        if para is None:
            continue

        para = para.strip()  # Remove leading and trailing whitespace

        if para in section_headers:
            # Treat section header as a separate paragraph
            if current_paragraph:
                merged_paragraphs.append(current_paragraph.strip())
            current_paragraph = para  # Start new paragraph with the section header
        elif re.match(r'^\d+\.\s|^[\-*]\s', para) or (current_paragraph and len(current_paragraph) < 150):
            # Add list items to the current paragraph
            current_paragraph += "\n" + para
        else:
            # Append current paragraph if it's not empty and reset for the new paragraph
            if current_paragraph:
                merged_paragraphs.append(current_paragraph.strip())
            current_paragraph = para  # Start a new paragraph

    # Add any remaining text as the last paragraph
    if current_paragraph:
        merged_paragraphs.append(current_paragraph.strip())

    return merged_paragraphs

def create_chunks_from_paragraphs(text, max_chunk_size=1500):
    """
    Splits the text into chunks based on paragraph boundaries.

    Args:
        text (str): Text to split into chunks.
        max_chunk_size (int, optional): Maximum number of characters to create the chunk. Defaults to 1500.

    Returns:
        List: List of text chunks.
    """
    paragraphs = split_text_by_paragraphs(text)
    chunks = []
    current_chunk = ""

    for para in paragraphs:
        # Remove double spaces within paragraphs
        para = re.sub(r'\s{2,}', ' ', para)

        if len(current_chunk) + len(para) + 1 <= max_chunk_size:
            current_chunk += para + "\n\n"  # Add paragraph with a newline separator
        else:
            chunks.append(current_chunk.strip())
            current_chunk = para + "\n\n"  # Start the new chunk

    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks

def extract_key_value_pairs(text):
    """
    Extracts key-value pairs from text where the format is 'Key: Value'.

    Args:
        text (str): Text to extract key-value pairs from.

    Returns:
        dict: A dictionary containing extracted key-value pairs.
    """
    key_value_dict = {}
    pattern = r"([A-Za-z ]+):\s*(\d+)"
    matches = re.findall(pattern, text)
    
    for key, value in matches:
        key = key.strip()
        # TODO: Change the key check to the ones who needs integer 
        int_keys = ["Age", "Weight", "Height"]
        if key in int_keys:
            value = int(value)
        key_value_dict[key] = value
    
    return key_value_dict

def save_dict_to_csv(data_dict, output_file_path):
    """
    Save a dictionary to a CSV file.

    Args:
        data_dict (dict): The dictionary to save, where keys are column headers and values are their corresponding values.
        output_file_path (str): Path to the CSV file to save the data.
    """
    with open(output_file_path, mode='w', newline='', encoding='utf-8') as csv_file:
        writer = csv.writer(csv_file)
        
        # Write headers
        writer.writerow(["Key", "Value"])
        
        # Write key-value pairs
        for key, value in data_dict.items():
            writer.writerow([key, value])
    print(f"CSV file saved to {output_file_path}")
    
def initialize_key_value_summary():
    """
    Initialize a dictionary with default values based on the expected types.

    Returns:
        dict: Dictionary with default values assigned based on the types.
    """
    KEYS_AND_TYPES = {
    "patient_id": str,
    "headache_intensity": str,
    "headache_frequency": str,
    "headache_location": str,
    "average_daily_pain_intensity": str,
    "diet_score": str,
    "tmj_pain_rating": str,
    "tmj_disability_rating": str,
    "jaw_function_score": str,
    "jaw_clicking": bool,
    "jaw_locking": bool,
    "muscle_pain_score": str,
    "muscle_spasm_present": bool,
    "muscle_tenderness_present": bool,
    "muscle_soreness_present": bool,
    "joint_pain_areas": str,
    "joint_pain_level": str,
    "joint_arthritis_present": bool,
    "neck_pain_present": bool,
    "back_pain_present": bool,
    "earache_present": bool,
    "tinnitus_present": bool,
    "vertigo_present": bool,
    "hearing_loss_present": bool,
    "hearing_sensitivity_present": bool,
    "sleep_apnea_diagnosed": bool,
    "sleep_disorder_type": str,
    "airway_obstruction_present": bool,
    "anxiety_present": bool,
    "depression_present": bool,
    "stress_present": bool,
    "autoimune_condition": str,
    "autoimmune_condition_error": str,
    "fibromyalgia_present": bool,
    "chronic_fatigue_present": bool,
    "current_medications": str,
    "previous_medications": str,
    "adverse_reactions": str,
    "appliance_history": str,
    "current_appliance": str,
    "cpap_used": bool,
    "apap_used": bool,
    "bipap_used": bool,
    "physical_therapy_status": str,
    "pain_onset_date": str,
    "pain_duration": str,
    "pain_frequency": str,
    "pain_triggers": str,
    "pain_relieving_factors": str,
    "pain_aggravating_factors": str
    }
    
    defaults = {
        str: "none",
        bool: False,
    }
    return {key: defaults[expected_type] for key, expected_type in KEYS_AND_TYPES.items()}
