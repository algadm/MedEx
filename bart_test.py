import argparse
import os
import re
import json
import fitz
import torch
import string
from docx import Document
from transformers import BartTokenizer, BartForConditionalGeneration
from utilities import initialize_key_value_summary

def load_model_and_tokenizer():
    model_name_or_path = "/home/lucia/Documents/Alban/MedSummarizer/fine_tuned_model_table_2"
    tokenizer = BartTokenizer.from_pretrained(model_name_or_path)
    model = BartForConditionalGeneration.from_pretrained(model_name_or_path)
    return model, tokenizer

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return clean_text(text)

def extract_text_from_word(docx_path):
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
    criteria = []
    context_filters = {}

    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    for line in lines:
        line = line.strip()
        if not line:  # skip empty lines
            continue
        
        if ":" in line:  # if line has a colon, it's a word with context filters
            word, context = line.split(":", 1)
            word = word.strip()
            context_terms = [term.strip().strip('"') for term in context.split(",")]
            context_filters[word] = context_terms
            criteria.append(word)
        else:
            # If no colon, just add the word to the criteria list
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

def generate_combined_summary(model, tokenizer, text, criteria_path, patient_id, max_chunk_size=2300):
    chunks = create_chunks_from_paragraphs(text, max_chunk_size=max_chunk_size)
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    summaries = []
    for chunk in chunks:
        dict = initialize_key_value_summary()
        keys = list(dict.keys())
        prompt = f"Using this list: {keys}\nSummarize this text: {chunk}\n"
        
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        summary_ids = model.generate(
            inputs["input_ids"], 
            max_length=300,
            min_length=3,
            length_penalty=1.0,
            num_beams=8,
            do_sample=True,
            temperature=0.9,
            top_k=40,
            top_p=0.9
        )
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        summaries.append(summary)
        

    final_summary = "\n\n".join(summaries)
    return final_summary

def process_notes(notes_folder, output_folder, criteria_path, model, tokenizer):
    patient_files = {}
    
    for file_name in os.listdir(notes_folder):
        if not (file_name.endswith(".pdf") or file_name.endswith(".docx")):
            continue
        patient_id = file_name.split("_")[0]
        if patient_id not in patient_files:
            patient_files[patient_id] = []
        patient_files[patient_id].append(file_name)

    for patient_id, files in patient_files.items():
        print(f"Processing patient {patient_id}...")
        combined_text = ""
        for file_name in files:
            file_path = os.path.join(notes_folder, file_name)
            if file_name.endswith(".pdf"):
                combined_text += extract_text_from_pdf(file_path)
            elif file_name.endswith(".docx"):
                combined_text += extract_text_from_word(file_path)

        summary = generate_combined_summary(model, tokenizer, combined_text, criteria_path, patient_id)
        output_file_path = os.path.join(output_folder, f"{patient_id}_summary.txt")
        
        with open(output_file_path, 'w', encoding='utf-8') as output_file:
            print(f"Saved summary to {output_file_path}")
            output_file.write(f"{summary}\n")

def main(notes_folder, output_folder, criteria_file):
    model, tokenizer = load_model_and_tokenizer()

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    process_notes(notes_folder, output_folder, criteria_file, model, tokenizer)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Summarize clinical notes using BART based on specific criteria")
    parser.add_argument('--notes_folder', type=str, required=True, help="Path to the folder containing clinical notes")
    parser.add_argument('--output_folder', type=str, required=True, help="Path to the folder to save the summaries")
    parser.add_argument('--criteria_file', type=str, required=True, help="Path to the text file containing criteria")
    
    args = parser.parse_args()
    main(args.notes_folder, args.output_folder, args.criteria_file)
