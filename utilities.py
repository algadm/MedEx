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
    section_headers = ["CLINICAL EXAMINATION", "RADIOGRAPHIC EVALUATION"]
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

def create_chunks_from_paragraphs(text, max_chunk_size=1800):
    section_headers = ["CLINICAL EXAMINATION", "CLINICAL EVALUATION", "RADIOGRAPHIC EXAMINATION", "RADIOGRAPHIC EVALUATION"]

    def split_to_sentences(paragraph, max_size):
        """
        Splits a paragraph into sentences that fit within the max size.
        """
        sentences = re.split(r'(?<=[.!?])\s+', paragraph)  # Split on sentence boundaries
        chunk = ""
        chunks = []

        for sentence in sentences:
            if len(chunk) + len(sentence) + 1 <= max_size:
                chunk += sentence + " "
            else:
                if chunk:
                    chunks.append(chunk.strip())
                chunk = sentence + " "

        if chunk:
            chunks.append(chunk.strip())

        return chunks

    paragraphs = split_text_by_paragraphs(text)
    chunks = []
    current_chunk = ""

    i = 0
    while i < len(paragraphs):
        para = re.sub(r'\s{2,}', ' ', paragraphs[i])
        # Check if this paragraph starts with a section header
        if any(para.lower().startswith(header.lower()) for header in section_headers):
            # Accumulate the entire section: header + following non-header paragraphs.
            section_paragraphs = [para]
            j = i + 1
            while j < len(paragraphs):
                next_para = re.sub(r'\s{2,}', ' ', paragraphs[j])
                if any(next_para.lower().startswith(header.lower()) for header in section_headers):
                    break
                section_paragraphs.append(next_para)
                j += 1
            section_text = "\n\n".join(section_paragraphs)
            
            # If current chunk plus the whole section fits, add it.
            if len(current_chunk) + len(section_text) + 1 <= max_chunk_size:
                current_chunk += section_text + "\n\n"
            else:
                # Flush current_chunk if not empty.
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = ""
                # If the section itself is small enough, start a new chunk with it.
                if len(section_text) <= max_chunk_size:
                    current_chunk = section_text + "\n\n"
                else:
                    # Otherwise, process the section piece by piece.
                    for sec_para in section_paragraphs:
                        sec_para = re.sub(r'\s{2,}', ' ', sec_para)
                        if len(sec_para) <= max_chunk_size:
                            if len(current_chunk) + len(sec_para) + 1 <= max_chunk_size:
                                current_chunk += sec_para + "\n\n"
                            else:
                                if current_chunk:
                                    chunks.append(current_chunk.strip())
                                current_chunk = sec_para + "\n\n"
                        else:
                            # Split long paragraphs by sentences.
                            para_sentences = split_to_sentences(sec_para, max_chunk_size)
                            for sentence_chunk in para_sentences:
                                if len(current_chunk) + len(sentence_chunk) + 1 <= max_chunk_size:
                                    current_chunk += sentence_chunk + " "
                                else:
                                    if current_chunk:
                                        chunks.append(current_chunk.strip())
                                    current_chunk = sentence_chunk + " "
            i = j  # Move past the entire section.
        else:
            # Regular paragraph (non-header)
            if len(current_chunk) + len(para) + 1 <= max_chunk_size:
                current_chunk += para + "\n\n"
            else:
                # If the paragraph itself is too long, split it.
                para_sentences = split_to_sentences(para, max_chunk_size)
                for sentence_chunk in para_sentences:
                    if len(current_chunk) + len(sentence_chunk) + 1 <= max_chunk_size:
                        current_chunk += sentence_chunk + " "
                    else:
                        if current_chunk:
                            chunks.append(current_chunk.strip())
                        current_chunk = sentence_chunk + " "
            i += 1


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
    "patient_age": str,
    "headache_intensity": str,
    "headache_frequency": str,
    "headache_location": str,
    "migraine_history": str,
    "migraine_frequency": str,
    "average_daily_pain_intensity": str,
    "diet_score": str,
    "tmj_pain_rating": str,
    "disability_rating": str,
    "jaw_function_score": str,
    "jaw_clicking": str,
    "jaw_crepitus": str,
    "jaw_locking": str,
    "maximum_opening": str,
    "maximum_opening_without_pain": str,
    "disc_displacement": str,
    "muscle_pain_score": str,
    "muscle_pain_location": str,
    "muscle_spasm_present": str,
    "muscle_tenderness_present": str,
    "muscle_stiffness_present": str,
    "muscle_soreness_present": str,
    "joint_pain_areas": str,
    "joint_arthritis_location": str,
    "neck_pain_present": str,
    "back_pain_present": str,
    "earache_present": str,
    "tinnitus_present": str,
    "vertigo_present": str,
    "hearing_loss_present": str,
    "hearing_sensitivity_present": str,
    "sleep_apnea_diagnosed": str,
    "sleep_disorder_type": str,
    "airway_obstruction_present": str,
    "anxiety_present": str,
    "depression_present": str,
    "stress_present": str,
    "autoimmune_condition": str,
    "fibromyalgia_present": str,
    "current_medications": str,
    "previous_medications": str,
    "adverse_reactions": str,
    "appliance_history": str,
    "current_appliance": str,
    "cpap_used": str,
    "apap_used": str,
    "bipap_used": str,
    "physical_therapy_status": str,
    "pain_onset_date": str,
    "pain_duration": str,
    "pain_frequency": str,
    "onset_triggers": str,
    "pain_relieving_factors": str,
    "pain_aggravating_factors": str
    }
    
    defaults = {
        str: "",
    }
    return {key: defaults[expected_type] for key, expected_type in KEYS_AND_TYPES.items()}
