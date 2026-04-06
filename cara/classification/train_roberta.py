import os

import numpy as np
from datasets import load_dataset
from evaluate import load
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)

# --- 1. CONFIGURATION ---
MODEL_NAME = "roberta-base" # Change to "roberta-large" if you have >16GB VRAM
DATA_FILE = os.environ.get("DATA_FILE", "predictions_rationale_vllm_8B captioned.jsonl")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "./metameme_roberta_model")

# --- 2. DATA FLATTENING FUNCTION ---
def flatten_meme_data(example):
    """
    Converts the structured JSON into a single paragraph for RoBERTa.
    """
    ocr = example.get("ocr_text", "")
    reasoning = example.get("reasoning", "")
    
    # Safely parse the hateful arguments
    hateful_args = example.get("hateful", [])
    if hateful_args and isinstance(hateful_args, list):
        hateful_str = " ".join([f"'{item.get('metaphor', '')}' implies {item.get('meaning', '')}" for item in hateful_args])
    else:
        hateful_str = "None."

    # Safely parse the benign arguments
    benign_args = example.get("benign", [])
    if benign_args and isinstance(benign_args, list):
        benign_str = " ".join([f"'{item.get('metaphor', '')}' implies {item.get('meaning', '')}" for item in benign_args])
    else:
        benign_str = "None."

    # Construct the final text block
    full_text = (
        f"Meme Text: {ocr}. "
        f"Contextual Reasoning: {reasoning} "
        f"Hateful Interpretation: {hateful_str} "
        f"Benign Interpretation: {benign_str}"
    )
    
    return {"text": full_text, "labels": example["label"]}

# --- 3. METRICS ---
clf_metrics = load("evaluate-metric/roc_auc") # We use AUROC for binary classification
acc_metric = load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    # Convert logits to probabilities using softmax
    probs = np.exp(logits) / np.sum(np.exp(logits), axis=-1, keepdims=True)
    predictions = np.argmax(logits, axis=-1)
    
    # probability of the positive class (Hateful = 1)
    prob_pos = probs[:, 1]
    
    auc = clf_metrics.compute(prediction_scores=prob_pos, references=labels)["roc_auc"]
    acc = acc_metric.compute(predictions=predictions, references=labels)["accuracy"]
    return {"accuracy": acc, "roc_auc": auc}

# --- 4. MAIN TRAINING PIPELINE ---
def main():
    print(f"Loading dataset from {DATA_FILE}...")
    # Load dataset
    dataset = load_dataset("json", data_files=DATA_FILE, split="train")
    
    # Split into 90% Train, 10% Validation
    dataset = dataset.train_test_split(test_size=0.1, seed=42)
    
    print("Flattening JSON structure into narrative text...")
    dataset = dataset.map(flatten_meme_data, remove_columns=dataset["train"].column_names)

    print(f"Loading {MODEL_NAME} tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    def tokenize_function(examples):
        # max_length=512 is the maximum sequence length for RoBERTa
        return tokenizer(examples["text"], padding=False, truncation=True, max_length=512)

    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Initialize model for binary classification
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

    # Setup Training Arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        learning_rate=2e-5,           # Standard LR for full fine-tuning
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=4,           # 4 epochs is usually sufficient for 4,000 samples
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="roc_auc", # Optimize for AUROC
        greater_is_better=True,
        report_to="none"              # Set to "wandb" if you use Weights & Biases
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    print("Starting Training...")
    trainer.train()
    
    print(f"Training Complete! Saving best model to {OUTPUT_DIR}")
    trainer.save_model(OUTPUT_DIR)

if __name__ == "__main__":
    main()