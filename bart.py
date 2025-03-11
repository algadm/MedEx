import argparse
import os
import re
import json
import fitz
import torch
import string
from docx import Document
from transformers import BartTokenizer, BartForConditionalGeneration
from utilities import create_chunks_from_paragraphs

def load_model_and_tokenizer():
    model_name_or_path = "facebook/bart-large-cnn"
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


def generate_combined_summary(model, tokenizer, text, patient_id, max_chunk_size=3500):
    # if patient_id == "B001":
    #         print(text)
    chunks = create_chunks_from_paragraphs(text, max_chunk_size=max_chunk_size)
    
    path = "/home/lucia/Documents/Alban/data/CLINICAL_NOTES/text_tab_word_with_3500_context"
    os.makedirs(path, exist_ok=True)
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    summaries = []
    for chunk in chunks:
        with open(os.path.join(path, f"{patient_id}_Word_text.txt"), 'a', encoding='utf-8') as output_file:
            output_file.write(f"```\n{chunk}\n```\n\n")
        
        # prompt = f"Prompt: {crit}\nText: {chunk}\n"
        
        # inputs = tokenizer(prompt, return_tensors="pt").to(device)
        # summary_ids = model.generate(
        #     inputs["input_ids"], 
        #     max_length=300,
        #     min_length=3,
        #     length_penalty=1.0,
        #     num_beams=8,
        #     do_sample=True,
        #     temperature=0.9,
        #     top_k=40,
        #     top_p=0.9
        # )
        # summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        # summaries.append(summary)
        
        # print("Input:", prompt)
        # print("Summary:", summary)
        summaries = ""

    final_summary = "\n\n".join(summaries)
    return final_summary

def process_notes(notes_folder, output_folder, model, tokenizer):
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

        summary = generate_combined_summary(model, tokenizer, combined_text, patient_id)
        output_file_path = os.path.join(output_folder, f"{patient_id}_summary.txt")
        
        with open(output_file_path, 'w', encoding='utf-8') as output_file:
            print(f"Saved summary to {output_file_path}")
            output_file.write(f"{summary}\n")

def main(notes_folder, output_folder):
    model, tokenizer = load_model_and_tokenizer()

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    process_notes(notes_folder, output_folder, model, tokenizer)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Summarize clinical notes using BART based on specific criteria")
    parser.add_argument('--notes_folder', type=str, required=True, help="Path to the folder containing clinical notes")
    parser.add_argument('--output_folder', type=str, required=True, help="Path to the folder to save the summaries")
    
    args = parser.parse_args()
    main(args.notes_folder, args.output_folder)
