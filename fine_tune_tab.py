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
from utilities import initialize_key_value_summary
from transformers import BartForConditionalGeneration, BartTokenizer, Trainer, TrainingArguments, DataCollatorForSeq2Seq, EarlyStoppingCallback

from utilities import extract_text_from_pdf, extract_text_from_word, create_chunks_from_paragraphs

def clean_summary(summary):
    """Cleans a single summary string by removing unnecessary newlines and ensuring consistent formatting.

    Args:
        summary (str): Text to clean.

    Returns:
        str: Text cleaned.
    """
    return summary.replace("\n", "\\").strip()

def load_text(file_path):
    """Reads the text from a file and returns it as a string.

    Args:
        file_path (str): Path to the file to read.

    Raises:
        ValueError: Raised if the file format is not supported.

    Returns:
        str: Text read from the file.
    """
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

def save_notes_and_summaries_to_csv(notes_folder, summaries_folder, output_csv, max_chunk_size=3500):
    """Save clinical notes and summaries to a CSV file.

    Args:
        notes_folder (str): Path to the folder containing clinical notes.
        summaries_folder (str): Path to the folder containing the summaries.
        output_csv (str): Path to save the output CSV file.
        max_chunk_size (int, optional): Maximum number of characters used for each chunk. Defaults to 3500.
    """
    texts, summaries = [], []
    
    for note_filename in os.listdir(notes_folder):
        if note_filename.endswith((".txt", ".pdf", ".docx")):
            patient_id = note_filename.split("_")[0]
            summary_filename = f"{patient_id}_summary.txt"
            note_path, summary_path = os.path.join(notes_folder, note_filename), os.path.join(summaries_folder, summary_filename)
            
            if os.path.exists(note_path) and os.path.exists(summary_path):
                dict = initialize_key_value_summary()
                keys = list(dict.keys())
                
                text = load_text(note_path)
                summary = load_text(summary_path)
                
                chunks = create_chunks_from_paragraphs(text, max_chunk_size=max_chunk_size)
                summary_chunks = split_summary_text(summary)
                
                if len(chunks) != len(summary_chunks):
                    print(f"Warning: Mismatch between text chunks and summary chunks for {patient_id}")
                
                for chunk, summary_chunk in zip(chunks, summary_chunks):
                    texts.append(chunk)
                    summaries.append(summary_chunk)

    data = pd.DataFrame({"text": texts, "summary": summaries})
    data.to_csv(output_csv, index=False, quoting=csv.QUOTE_ALL)
    print(f"Data saved to {output_csv}")

def split_summary_text(summary_text):
    """Splits the summary text by a long '-' delimiter, indicating different chunk summaries.

    Args:
        summary_text (str): The summary text to split.

    Returns:
        list[str]: A list containing the split summary chunks.
    """
    return [chunk.strip() for chunk in summary_text.split("-"*100) if chunk.strip()]

def assign_patient_ids(data):
    """Fills missing patient_id values based on the last seen patient_id.

    Args:
        data (pd.DataFrame): The input data.

    Returns:
        pd.DataFrame: The data with patient_id values filled.
    """
    current_patient_id = None
    patient_ids = []
    
    for index, row in data.iterrows():
        # Extract patient_id from the 'summary' column using regex
        extracted_id = pd.Series(row['summary']).str.extract(r'patient_id: (\w+)')[0].values[0]
        
        if pd.notna(extracted_id):
            current_patient_id = extracted_id
        
        patient_ids.append(current_patient_id)
    
    data["patient_id"] = patient_ids
    return data

def split_data_by_patient(input_csv, output_dir):
    """Splits the data into training, validation, and test sets based on unique patient IDs.

    Args:
        input_csv (str): Path to the input CSV file.
        output_dir (str): Path to save the output CSV files.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: A tuple containing the training, validation, and test data.
    """
    data = pd.read_csv(input_csv)
    data = assign_patient_ids(data)
    data = data.dropna(subset=["patient_id"])
    unique_patients = data["patient_id"].unique()

    train_patients, temp_patients = train_test_split(unique_patients, train_size=0.8, random_state=42)
    val_patients, test_patients = train_test_split(temp_patients, test_size=0.5, random_state=42)

    train_data = data[data["patient_id"].isin(train_patients)]
    val_data = data[data["patient_id"].isin(val_patients)]
    test_data = data[data["patient_id"].isin(test_patients)]
    
    train_data = train_data.drop(columns=["patient_id"])
    val_data = val_data.drop(columns=["patient_id"])
    test_data = test_data.drop(columns=["patient_id"])

    train_data.to_csv(os.path.join(output_dir, "train.csv"), index=False, quoting=csv.QUOTE_ALL)
    val_data.to_csv(os.path.join(output_dir, "validation.csv"), index=False, quoting=csv.QUOTE_ALL)
    test_data.to_csv(os.path.join(output_dir, "test.csv"), index=False, quoting=csv.QUOTE_ALL)

    print("Data split by patient and saved.")
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

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

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
            
        predictions = np.argmax(logits, axis=-1)
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
        result = {key: value * 100 for key, value in result.items()}
        return result

    early_stopping_callback = EarlyStoppingCallback(early_stopping_patience=3)
    
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

    model.config.forced_bos_token_id = 0 
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
        save_notes_and_summaries_to_csv(args.input_dir, args.summaries_dir, output_csv)
        split_data_by_patient(output_csv, args.output_dir)
    else:
        if args.train_csv and args.val_csv:
            fine_tune(args.train_csv, args.val_csv, args.output_dir)
        else:
            print("Error: Provide both --train_csv and --val_csv for fine-tuning.")