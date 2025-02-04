import os
import re
import csv
import fitz
import argparse
import evaluate
import numpy as np
import pandas as pd
from docx import Document
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from transformers import BartForConditionalGeneration, BartTokenizer, Trainer, TrainingArguments, DataCollatorForSeq2Seq, EarlyStoppingCallback

from utilities import extract_text_from_pdf, extract_text_from_word, create_chunks_from_paragraphs

def clean_summary(summary):
    """
    Cleans a single summary string by removing unnecessary newlines and ensuring consistent formatting.
    """
    return summary.replace("\n", "\\").strip()

def load_text(file_path):
    if file_path.endswith(".pdf"):
        return clean_text(extract_text_from_pdf(file_path))
    elif file_path.endswith(".docx"):
        return clean_text(extract_text_from_word(file_path))
    elif file_path.endswith(".txt"):
        with open(file_path, 'r', encoding='utf-8') as file:
            return clean_text(file.read())
    else:
        raise ValueError("Unsupported file format. Supported formats: .txt, .pdf, .docx")
    
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

def save_notes_and_summaries_to_csv(notes_folder, summaries_folder, output_csv, max_chunk_size=2300):
    texts, summaries = [], []
    
    for note_filename in os.listdir(notes_folder):
        if note_filename.endswith((".txt", ".pdf", ".docx")):
            patient_id = note_filename.split("_")[0]
            summary_filename = f"{patient_id}_summary.txt"
            note_path, summary_path = os.path.join(notes_folder, note_filename), os.path.join(summaries_folder, summary_filename)
            
            if os.path.exists(note_path) and os.path.exists(summary_path):
                # Extract the full text from the note
                text = load_text(note_path)
                summary = load_text(summary_path)
                
                # Split the text into chunks
                chunks = create_chunks_from_paragraphs(text, max_chunk_size=max_chunk_size)
                
                # Split summary into corresponding chunks
                summary_chunks = split_summary_text(summary)
                # summary_chunks = [clean_summary(chunk) for chunk in summary_chunks]
                
                # Ensure chunks and summaries are paired correctly
                if len(chunks) != len(summary_chunks):
                    print(f"Warning: Mismatch between text chunks and summary chunks for {patient_id}")
                
                # Add each chunk with its corresponding summary
                for chunk, summary_chunk in zip(chunks, summary_chunks):
                    texts.append(chunk)
                    summaries.append(summary_chunk)

    data = pd.DataFrame({"text": texts, "summary": summaries})
    data.to_csv(output_csv, index=False, quoting=csv.QUOTE_ALL)
    print(f"Data saved to {output_csv}")
    return data

def split_summary_text(summary_text):
    """
    Splits the summary text by '-----' delimiter, indicating different chunk summaries.
    """
    return [chunk.strip() for chunk in summary_text.split("-"*100) if chunk.strip()]

def extract_chunks(file_content):
    # Use regex to find each prompt and its respective block
    pattern = r"Prompt:.*?(?=Prompt:|$)"  # Matches each chunk starting with 'Prompt:' until the next 'Prompt:' or end of text
    chunks = re.findall(pattern, file_content, flags=re.DOTALL)
    # Strip trailing/leading whitespace for cleaner results
    return [chunk.strip() for chunk in chunks]

def split_data(input_csv, output_dir):
    data = pd.read_csv(input_csv)
    train_data, temp_data = train_test_split(data, test_size=0.2, random_state=42)
    val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)
    
    train_data.to_csv(os.path.join(output_dir, "train.csv"), index=False, quoting=csv.QUOTE_ALL)
    val_data.to_csv(os.path.join(output_dir, "validation.csv"), index=False, quoting=csv.QUOTE_ALL)
    test_data.to_csv(os.path.join(output_dir, "test.csv"), index=False, quoting=csv.QUOTE_ALL)
    print("Data split and saved.")

    return train_data, val_data, test_data

def fine_tune(training_path, validation_path, output_dir):
    dataset = load_dataset("csv", data_files={"train": training_path, "validation": validation_path})

    model_name = "facebook/bart-large-cnn"
    model = BartForConditionalGeneration.from_pretrained(model_name)
    tokenizer = BartTokenizer.from_pretrained(model_name)

    def preprocess_function(examples):
        model_inputs = tokenizer(examples["text"], max_length=1024, truncation=True, padding="max_length")
        labels = tokenizer(examples["summary"], max_length=150, truncation=True, padding="max_length").input_ids
        model_inputs["labels"] = labels
        return model_inputs

    tokenized_datasets = dataset.map(preprocess_function, batched=True)

    # Use DataCollatorForSeq2Seq for sequence-to-sequence tasks
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=1e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=20,
        fp16=True,
        load_best_model_at_end=True,
        metric_for_best_model="eval_rouge1",
        greater_is_better=True,
    )

    def compute_metrics(eval_preds):
        metric = evaluate.load("rouge")
        logits, labels = eval_preds
        
        # Ensure logits is a single array in case it's a tuple
        if isinstance(logits, tuple):
            logits = logits[0]
            
        # Get predictions by taking the argmax (most probable token) along the last axis
        predictions = np.argmax(logits, axis=-1)
        
        # Decode the predictions and labels
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        
        # Replace label tokens set to -100 (ignore index) with the padding token for decoding
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Compute ROUGE scores
        result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
        
        # Since result now contains float values directly, no `.mid` is needed.
        result = {key: value * 100 for key, value in result.items()}
        return result

    early_stopping_callback = EarlyStoppingCallback(early_stopping_patience=3)  # Early stopping after no improvement for 3 epochs
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[early_stopping_callback],
    )

    trainer.train()
    results = trainer.evaluate()
    print(results)

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Model saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare data or fine-tune model for summarization.")
    parser.add_argument("--prepare_data", action="store_true", help="If set, prepares the data by saving clinical notes to CSV.")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory with clinical notes with prompts in .txt format.")
    parser.add_argument("--summaries_dir", type=str, required=True, help="Directory with summaries in .txt format.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the output model or CSV.")
    parser.add_argument("--train_csv", type=str, help="Path to the training CSV for fine-tuning.")
    parser.add_argument("--val_csv", type=str, help="Path to the validation CSV for fine-tuning.")
    
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.prepare_data:
        output_csv = os.path.join(args.output_dir, "clinical_data.csv")
        data = save_notes_and_summaries_to_csv(args.input_dir, args.summaries_dir, output_csv)
        split_data(output_csv, args.output_dir)
    else:
        if args.train_csv and args.val_csv:
            fine_tune(args.train_csv, args.val_csv, args.output_dir)
        else:
            print("Error: Provide both --train_csv and --val_csv for fine-tuning.")