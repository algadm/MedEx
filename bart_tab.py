import argparse
import os
import torch
from utilities import *

def generate_combined_summary(model, tokenizer, text, patient_id, output_folder, max_chunk_size=2300):
    """
    Generate a combined summary for a given text based on specific criteria

    Args:
        model (BartForConditionalGeneration): Model for summarization
        tokenizer (BartTokenizer): Tokenizer for the model
        text (str): Text to summarize
        patient_id (str): Patient ID
        output_folder (str): Path to the folder to save the summary
        max_chunk_size (int, optional): Maximum number of characters to create the chunk. Defaults to 2300.
    """
    chunks = create_chunks_from_paragraphs(text, max_chunk_size=max_chunk_size)
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    key_value_summary = {}
    for chunk in chunks:  
        dict = initialize_key_value_summary()
        keys = list(dict.keys())
        prompt = f"Here is the list you need to refer to: {keys}\nText: {chunk}\n"
        print(prompt)
        
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
        
        extracted_data = extract_key_value_pairs(summary)
        key_value_summary.update(extracted_data)
    
    if output_folder:
        csv_file_path = os.path.join(output_folder, f"{patient_id}_summary.csv")
        save_dict_to_csv(key_value_summary, csv_file_path)

def process_notes(notes_folder, output_folder, model, tokenizer):
    """
    Process clinical notes to generate summaries based on specific criteria

    Args:
        notes_folder (str): Path to the folder containing clinical notes
        output_folder (str): Path to the folder to save the summaries
        model (BartForConditionalGeneration): Model for summarization
        tokenizer (BartTokenizer): Tokenizer for the model
    """
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

        generate_combined_summary(model, tokenizer, combined_text, patient_id, output_folder)

def main(notes_folder, output_folder):
    """
    Main function to summarize clinical notes using BART based on specific criteria

    Args:
        notes_folder (str): Path to the folder containing clinical notes
        output_folder (str): Path to the folder to save the summaries
    """
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
