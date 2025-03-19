import argparse
import os
import re
import fitz
import torch
import string
from docx import Document
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from utilities import initialize_key_value_summary

def load_model_and_tokenizer():
    model_name_or_path = "PanaceaAI/BiomedGPT-Base-Pretrained"
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path, 
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16, 
        device_map="auto"
    )
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
    replacements = {
        "’": "'",
        "–": "-"
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text

def create_chunks_from_paragraphs(text, max_chunk_size=1500):
    paragraphs = text.split("\n\n")
    chunks = []
    current_chunk = ""

    for para in paragraphs:
        para = re.sub(r'\s{2,}', ' ', para)

        if len(current_chunk) + len(para) + 1 <= max_chunk_size:
            current_chunk += para + "\n\n"
        else:
            chunks.append(current_chunk.strip())
            current_chunk = para + "\n\n"

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks

def generate_combined_summary(model, tokenizer, text, max_chunk_size=3500, model_max_tokens=512):
    chunks = create_chunks_from_paragraphs(text, max_chunk_size=max_chunk_size)
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    summaries = []
    
    generation_config = GenerationConfig(
        max_length=800,
        min_length=100,
        length_penalty=1.0,
        num_beams=8,
        do_sample=True,
        temperature=0.9,
        top_k=40,
        top_p=0.9,
    )
    
    for chunk in chunks:
        input_prompt = f"""Summarize the following clinical text into structured key-value format:\n\n{chunk}\n\nSummary:"""
        inputs = tokenizer(input_prompt, return_tensors="pt", truncation=True, max_length=model_max_tokens, padding="max_length").to(device)
        
        summary_ids = model.generate(
            inputs["input_ids"], 
            generation_config=generation_config
        )
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        summaries.append(summary)
        
    final_summary = "\n----------------------------------------------------------------------------------------------------\n".join(summaries)
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

        summary = generate_combined_summary(model, tokenizer, combined_text)
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
    parser = argparse.ArgumentParser(description="Summarize clinical notes using BiomedGPT-Base-Pretrained")
    parser.add_argument('--notes_folder', type=str, required=True, help="Path to the folder containing clinical notes")
    parser.add_argument('--output_folder', type=str, required=True, help="Path to the folder to save the summaries")
    
    args = parser.parse_args()
    main(args.notes_folder, args.output_folder)