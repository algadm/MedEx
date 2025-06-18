import argparse
import os
import fitz
import torch
from docx import Document
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from utilities import initialize_key_value_summary, create_chunks_from_paragraphs

def load_model_and_tokenizer(lora_adapter_path):
    base_model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)

    # Set padding token to avoid warnings
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        trust_remote_code=True,
        device_map="auto",
        quantization_config=quantization_config,
        torch_dtype=torch.float16,
        attn_implementation="flash_attention_2"
    )

    # Now load LoRA adapter
    model = PeftModel.from_pretrained(base_model, lora_adapter_path)
    # model = model.merge_and_unload()
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return model, tokenizer, device

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = "".join(page.get_text() for page in doc)
    return clean_text(text)

def extract_text_from_word(docx_path):
    doc = Document(docx_path)
    return "\n".join(clean_text(paragraph.text) for paragraph in doc.paragraphs)

def clean_text(text):
    replacements = {"’": "'", "–": "-"}
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text

def generate_text(model, tokenizer, device, chunk, max_new_tokens=300):
    # Build exact prompt used during fine-tuning
    prompt = f"Summarize: {chunk}\nSummary: "
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        output = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=200,
            min_new_tokens=30,
            pad_token_id=tokenizer.pad_token_id,
            repetition_penalty=2.0,
            # Add these new parameters for better structure
            num_beams=4,
            no_repeat_ngram_size=3,
            early_stopping=True,
            do_sample=True,
            top_p=0.95,
            temperature=0.6,
        )

    # Only return newly generated tokens (not repeating input prompt)
    generated_tokens = output[:, inputs["input_ids"].shape[1]:]
    decoded_output = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
    return decoded_output.strip()

def generate_combined_summary(model, tokenizer, device, text, max_chunk_size=3500, max_new_tokens=300):
    chunks = create_chunks_from_paragraphs(text, max_chunk_size=max_chunk_size)
    summaries = []

    for chunk in chunks:
        summary = generate_text(model, tokenizer, device, chunk, max_new_tokens=max_new_tokens)
        summaries.append(summary)

    final_summary = "\n----------------------------------------------------------------------------------------------------\n".join(summaries)
    return final_summary

def process_notes(notes_folder, output_folder, device, model, tokenizer):
    patient_files = {}

    for file_name in os.listdir(notes_folder):
        if not (file_name.endswith(".pdf") or file_name.endswith(".docx") or file_name.endswith(".txt")):
            continue
        patient_id = file_name.split("_")[0]
        patient_files.setdefault(patient_id, []).append(file_name)

    for patient_id, files in patient_files.items():
        print(f"Processing patient {patient_id}...")
        combined_text = ""
        for file_name in files:
            file_path = os.path.join(notes_folder, file_name)
            if file_name.endswith(".pdf"):
                combined_text += extract_text_from_pdf(file_path)
            elif file_name.endswith(".docx"):
                combined_text += extract_text_from_word(file_path)
            elif file_name.endswith(".txt"):
                with open(file_path, 'r', encoding='utf-8') as file:
                    combined_text += clean_text(file.read())

        summary = generate_combined_summary(model, tokenizer, device, combined_text)
        output_file_path = os.path.join(output_folder, f"{patient_id}_summary.txt")

        with open(output_file_path, 'w', encoding='utf-8') as output_file:
            print(f"Saved summary to {output_file_path}")
            output_file.write(f"{summary}\n")

def main(notes_folder, output_folder, lora_adapter_path):
    model, tokenizer, device = load_model_and_tokenizer(lora_adapter_path)

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    process_notes(notes_folder, output_folder, device, model, tokenizer)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Summarize clinical notes using fine-tuned DeepSeek")
    parser.add_argument('--notes_folder', type=str, required=True)
    parser.add_argument('--output_folder', type=str, required=True)
    parser.add_argument('--lora_adapter_path', type=str, required=True)
    
    args = parser.parse_args()
    main(args.notes_folder, args.output_folder, args.lora_adapter_path)
