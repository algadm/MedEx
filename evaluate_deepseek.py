import os
import time
import torch
import evaluate
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, DataCollatorForSeq2Seq
from peft import PeftModel
from datasets import load_dataset
from torch.utils.data import DataLoader
import numpy as np

def load_val_dataset(val_csv, tokenizer, max_length=1024):
    dataset = load_dataset("csv", data_files={"validation": val_csv})["validation"]

    def preprocess(example):
        input = f"Summarize: {example['text']}\nSummary: {example['summary']}"
        tokenized = tokenizer(
            input,
            truncation=True,
            max_length=max_length,
            padding="longest",
        )
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    return dataset.map(preprocess, batched=False)

def evaluate_checkpoint(model_dir, val_csv, max_length=1024, batch_size=8):
    model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Fix pad_token to avoid warnings
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        quantization_config=bnb_config,
        torch_dtype=torch.float16,
        attn_implementation="flash_attention_2",
    )

    model = PeftModel.from_pretrained(base_model, model_dir)
    model.eval()

    val_dataset = load_val_dataset(val_csv, tokenizer, max_length)
    val_dataset = val_dataset.remove_columns(['text', 'summary'])

    # Use HF data collator to handle dynamic padding per batch
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size,
        shuffle=False,
        collate_fn=data_collator
    )

    metric = evaluate.load("rouge")
    decoded_preds, decoded_labels = [], []

    total_samples = 0
    total_time = 0
    total_loss = 0  # <== Add total loss counter

    for batch_idx, batch in enumerate(val_loader):
        start_time = time.time()

        input_ids = batch["input_ids"].to("cuda")
        attention_mask = batch["attention_mask"].to("cuda")
        labels = batch["labels"].to("cuda")

        with torch.no_grad():
            # Compute loss (teacher forcing mode)
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            batch_loss = outputs.loss.item()
            total_loss += batch_loss * input_ids.shape[0]  # sum loss across batch
            # Generate predictions
            generated_outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=300,
                pad_token_id=tokenizer.pad_token_id,
            )

        for output, label in zip(generated_outputs, labels):
            pred = tokenizer.decode(output, skip_special_tokens=True)
            label_ids = [tok if tok != -100 else tokenizer.pad_token_id for tok in label]
            label_text = tokenizer.decode(label_ids, skip_special_tokens=True)

            decoded_preds.append(pred)
            decoded_labels.append(label_text)

        batch_time = time.time() - start_time
        total_samples += len(input_ids)
        total_time += batch_time

        print(f"Batch {batch_idx+1} - Batch time: {batch_time:.3f} sec - Batch loss: {batch_loss:.4f}")

    avg_time_per_sample = total_time / total_samples
    avg_loss = total_loss / total_samples

    print(f"\nAverage inference time per sample: {avg_time_per_sample:.3f} sec")
    print(f"Average validation loss: {avg_loss:.4f}")

    results = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    results = {k: round(v * 100, 2) for k, v in results.items()}
    results["eval_loss"] = round(avg_loss, 4)  # include loss in output dict
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", required=True, help="Path to checkpoint dir (e.g., output_dir/checkpoint-1000)")
    parser.add_argument("--val_csv", required=True, help="Validation CSV path")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for evaluation")
    args = parser.parse_args()

    scores = evaluate_checkpoint(args.checkpoint_dir, args.val_csv, batch_size=args.batch_size)
    print(f"Scores for {args.checkpoint_dir}: {scores}")
