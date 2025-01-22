import argparse
import os
import re
import fitz
import torch
from docx import Document
from transformers import pipeline
from langchain.text_splitter import RecursiveCharacterTextSplitter

def load_summarization_pipeline():
    device = 0 if torch.cuda.is_available() else -1  # -1 indicates CPU, 0 indicates the first GPU
    # Load the summarization pipeline
    summarization_pipeline = pipeline("summarization", model="/home/lucia/Documents/Alban/MedSummarizer/fine_tuned_model4", device=device)
    return summarization_pipeline

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def extract_text_from_word(docx_path):
    doc = Document(docx_path)
    return "\n".join([paragraph.text for paragraph in doc.paragraphs])

def correct_broken_words(text):
    # Replace patterns like "left -to-right" to "left-to-right"
    text = re.sub(r'\b(\w+)\s*-\s*(\w+)\b', r'\1-\2', text)
    # Replace patterns like "better tha n" to "better than"
    text = re.sub(r'\b(\w+)\s*(\w{1})\s+(\w+)\b', r'\1\2\3', text)
    # Correct cases where a word was split, e.g., "fro ntal" to "frontal"
    text = re.sub(r'\b(\w{1,2})\s+(\w+)', r'\1\2', text)
    
    return text

def split_text_with_langchain(text, max_chunk_size=1500, min_chunk_overlap=200):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_chunk_size,
        chunk_overlap=min_chunk_overlap,
        length_function=len  # Uses character count as the length metric
    )
    chunks = text_splitter.split_text(text)
    return chunks

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
    matches = []

    # Iterate over each criterion
    for criterion in criteria:
        # Use re.finditer to get both the word and its position in the text
        for match in re.finditer(r'\b' + re.escape(criterion) + r'\w*\b', text, flags=re.IGNORECASE):
            word = match.group(0).lower()

            # If there is no context filter, just add the word to the matches
            if (criterion not in context_filters) and (word not in matches):
                matches.append(word)  # Add the word to matches
            elif criterion in context_filters:
                # There are context filters, so check both the word and its context
                context_terms = context_filters[criterion]
                
                start_idx = match.start()

                # Get the surrounding words for the context window
                words_before = text[:start_idx].split()[-window_size+1:]
                words_after = text[start_idx + len(word):].split()[:window_size]
                context_window = words_before + [word] + words_after

                # Look for context terms in the context window
                for context_term in context_terms:
                    if any(re.search(r'\b' + re.escape(context_term) + r'\b', w, flags=re.IGNORECASE) for w in context_window):
                        combination = f"{word} {context_term}"
                        if combination not in matches:
                            matches.append(combination)  # If context matches, add the word

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

def generate_combined_summary(summarization_pipeline, text, criteria_path, patient_id, max_chunk_size=2300):
    text = text.replace('½', '1/2')
    # Split text into smaller chunks if it exceeds max_words
    chunks = create_chunks_from_paragraphs(text, max_chunk_size=max_chunk_size)

    # Load criteria and context filters from the file
    criteria, context_filters = load_criteria(criteria_path)
    
    summaries = []

    # Process each chunk separately
    for chunk in chunks:
        # Find all occurrences of criteria words or partial words in the chunk
        matches = find_matching_criteria_with_window(chunk, criteria, context_filters)
        
        # Format a prompt based on the found criteria matches
        crit = format_prompt(matches)
        prompt = f"Prompt: {crit}\nText: {chunk}\n"
        
        # Generate a summary using the pipeline
        summary = summarization_pipeline(
            prompt, 
            max_length=300, 
            min_length=3, 
            do_sample=True,
            temperature=0.9,
            top_k=40,
            top_p=0.9)
        summaries.append(summary[0]["summary_text"])

    # Combine all chunk summaries into a single summary
    final_summary = "\n\n".join(summaries)
    return final_summary

def process_notes(notes_folder, output_folder, criteria_path, summarization_pipeline):
    
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

        summary = generate_combined_summary(summarization_pipeline, combined_text, criteria_path, patient_id)

        output_file_path = os.path.join(output_folder, f"{patient_id}_summary.txt")
        with open(output_file_path, 'w', encoding='utf-8') as output_file:
            print(f"Saved summary to {output_file_path}")
            output_file.write(f"{summary}\n")

def main(notes_folder, output_folder, criteria_file):
    summarization_pipeline = load_summarization_pipeline()

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    process_notes(notes_folder, output_folder, criteria_file, summarization_pipeline)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Summarize clinical notes using BART based on specific criteria")
    
    parser.add_argument('--notes_folder', type=str, required=True, help="Path to the folder containing clinical notes")
    parser.add_argument('--output_folder', type=str, required=True, help="Path to the folder to save the summaries")
    parser.add_argument('--criteria_file', type=str, required=True, help="Path to the text file containing criteria")

    args = parser.parse_args()

    main(args.notes_folder, args.output_folder, args.criteria_file)