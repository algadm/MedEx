import argparse
import os
import re
import json
import fitz
import torch
import string
from docx import Document
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from utilities import initialize_key_value_summary, create_chunks_from_paragraphs

def load_model_and_tokenizer():
    model_name_or_path = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
    
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    
    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path, 
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="auto",
        quantization_config=quantization_config
    )
    
    return model, tokenizer, torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

def generate_text(model, tokenizer, device, prompt, max_length=300):
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=2048,
        return_attention_mask=True
    ).to(device)
    
    with torch.no_grad():
        output = model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=512,
            pad_token_id=tokenizer.pad_token_id,
            temperature=0.7,
            top_p=0.9,
            do_sample=True
        )

    return tokenizer.decode(output[0], skip_special_tokens=True)

def generate_combined_summary(model, tokenizer, device, text, max_chunk_size=12000, max_length=300):
    """
    Generates a summary using DeepSeek.
    """
    chunks = create_chunks_from_paragraphs(text, max_chunk_size=max_chunk_size)
    dict = initialize_key_value_summary()
    keys = list(dict.keys())
    
    summaries = []
    for chunk in chunks:
        prompt = f"Each patient has the following dicionnary: {dict}. For each of the keys, summarize the following text: {chunk}"
        summary = generate_text(model, tokenizer, device, prompt, max_length=max_length)
        summaries.append(summary)
        

    final_summary = "\n\n".join(summaries)
    return final_summary

def process_notes(notes_folder, output_folder, device, model, tokenizer):
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

        summary = generate_combined_summary(model, tokenizer, device, combined_text)
        output_file_path = os.path.join(output_folder, f"{patient_id}_summary.txt")
        
        with open(output_file_path, 'w', encoding='utf-8') as output_file:
            print(f"Saved summary to {output_file_path}")
            output_file.write(f"{summary}\n")

def main(notes_folder, output_folder):
    model, tokenizer, device = load_model_and_tokenizer()

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    process_notes(notes_folder, output_folder, device, model, tokenizer)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Summarize clinical notes using DeepSeek")
    parser.add_argument('--notes_folder', type=str, required=True, help="Path to the folder containing clinical notes")
    parser.add_argument('--output_folder', type=str, required=True, help="Path to the folder to save the summaries")
    
    args = parser.parse_args()
    main(args.notes_folder, args.output_folder)
