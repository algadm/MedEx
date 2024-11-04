import argparse
import os
import re
import fitz
import torch
from docx import Document
from transformers import BartTokenizer, BartForConditionalGeneration
from langchain.text_splitter import RecursiveCharacterTextSplitter

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
    return text

def extract_text_from_word(docx_path):
    doc = Document(docx_path)
    return "\n".join([paragraph.text for paragraph in doc.paragraphs])

def load_criteria(criteria_file):
    with open(criteria_file, 'r') as file:
        criteria = [line.strip() for line in file.readlines()]
    return criteria

def correct_broken_words(text):
    # Replace patterns like "left -to-right" to "left-to-right"
    text = re.sub(r'\b(\w+)\s*-\s*(\w+)\b', r'\1-\2', text)
    # Replace patterns like "better tha n" to "better than"
    text = re.sub(r'\b(\w+)\s*(\w{1})\s+(\w+)\b', r'\1\2\3', text)
    # Correct cases where a word was split, e.g., "fro ntal" to "frontal"
    text = re.sub(r'\b(\w{1,2})\s+(\w+)', r'\1\2', text)
    
    return text

def split_text_with_langchain(text, max_chunk_size=1500, min_chunk_overlap=200):
    """
    Splits text into chunks using LangChain's RecursiveCharacterTextSplitter.
    
    Parameters:
    - text (str): The input text to split.
    - max_chunk_size (int): The maximum number of characters in each chunk.
    - min_chunk_overlap (int): Minimum overlap in characters between chunks.
    
    Returns:
    - List[str]: A list of text chunks.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_chunk_size,
        chunk_overlap=min_chunk_overlap,
        length_function=len  # Uses character count as the length metric
    )
    chunks = text_splitter.split_text(text)
    return chunks

def remove_irrelevant_text(text):
    # Patterns to identify unwanted content
    patterns_to_remove = [
        r"suicide prevention lifeline", 
        r"samaritans",
        r"1-800-\d{3}-\d{4}",  # US phone numbers
        r"08457 \d{2} \d{2} \d{2}",  # UK numbers
        r"www\.\w+\.\w+",  # URLs
    ]
    
    # Combine all patterns into one regex to match any of them within a sentence
    combined_pattern = r"(?:\s*\b" + r"|".join(patterns_to_remove) + r"\b.*?[\.\!\?])"
    
    # Use re.sub to remove entire sentences containing the patterns
    cleaned_text = re.sub(combined_pattern, "", text, flags=re.IGNORECASE)
    
    # Return the cleaned text
    return cleaned_text.strip()

def generate_combined_summary(model, tokenizer, text, relevant_criteria, patient_id):
    # Split text into smaller chunks if it exceeds max_words
    chunks = split_text_with_langchain(text, max_chunk_size=1500, min_chunk_overlap=200)

    # Move the model to the GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    summaries = []

    # Process each chunk separately
    for i, chunk in enumerate(chunks):
        prompt = f"""
            Please provide a comprehensive and detailed summary of the following clinical note of patient '{patient_id}':

            ```{chunk}```
        """
        
        # Tokenize and generate a summary
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        summary_ids = model.generate(
            inputs["input_ids"], 
            max_length=300,
            min_length=200,
            length_penalty=1.0,
            num_beams=4,
            do_sample=True,
            temperature=0.9,
            top_k=40,
            top_p=0.9
        )

        # Decode and get the generated summary
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        summaries.append(remove_irrelevant_text(summary))

    # Combine all chunk summaries into a single summary
    final_summary = " ".join(summaries)
    return final_summary

def process_notes(notes_folder, output_folder, criteria_file, model, tokenizer):
    # Load the criteria from the provided file
    criteria = load_criteria(criteria_file)
    
    # Group files by patientId
    patient_files = {}
    for file_name in os.listdir(notes_folder):
        if not (file_name.endswith(".pdf") or file_name.endswith(".docx")):
            continue
        patient_id = file_name.split("_")[0]  # Extract patientId from the file name
        if patient_id not in patient_files:
            patient_files[patient_id] = []
        patient_files[patient_id].append(file_name)

    # Process each patient's files together
    for patient_id, files in patient_files.items():
        print(f"Processing patient {patient_id}...")
        combined_text = ""
        for file_name in files:
            file_path = os.path.join(notes_folder, file_name)
            if file_name.endswith(".pdf"):
                combined_text += extract_text_from_pdf(file_path)
            elif file_name.endswith(".docx"):
                combined_text += extract_text_from_word(file_path)

        summary = generate_combined_summary(model, tokenizer, combined_text, criteria, patient_id)

        # Create an output file for the patient's summary
        output_file_path = os.path.join(output_folder, f"{patient_id}_summary.txt")
        with open(output_file_path, 'w', encoding='utf-8') as output_file:
            print(f"Saved summary to {output_file_path}")
            output_file.write(f"{summary}\n")

def main(notes_folder, output_folder, criteria_file):
    # Load the tokenizer and model for BART
    model, tokenizer = load_model_and_tokenizer()

    # Process the clinical notes in the specified folder
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    process_notes(notes_folder, output_folder, criteria_file, model, tokenizer)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Summarize clinical notes using BART based on specific criteria")
    
    # Add arguments for folder paths and other parameters
    parser.add_argument('--notes_folder', type=str, required=True, help="Path to the folder containing clinical notes")
    parser.add_argument('--output_folder', type=str, required=True, help="Path to the folder to save the summaries")
    parser.add_argument('--criteria_file', type=str, required=True, help="Path to the text file containing criteria")

    args = parser.parse_args()

    main(args.notes_folder, args.output_folder, args.criteria_file)