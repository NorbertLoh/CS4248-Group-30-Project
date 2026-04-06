import numpy as np
import os
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    DataCollatorWithPadding
)

# --- 1. CONFIGURATION ---
MODEL_DIR = os.environ.get("MODEL_DIR", "./metameme_roberta_model") # Your saved best model
DATA_FILE = os.environ.get(
    "DATA_FILE",
    "datapreparation/output/predictions_rationale_vllm_8B captioned.jsonl",
) # Your Qwen-generated data
MODEL_NAME = os.environ.get("MODEL_NAME", "roberta-base") # Must match what you trained

# --- 2. DATA FLATTENING FUNCTION ---
def flatten_meme_data(example):
    """Recreates the exact same formatting used during training."""
    ocr = example.get("ocr_text", "")
    reasoning = example.get("reasoning", "")
    
    hateful_args = example.get("hateful", [])
    if hateful_args and isinstance(hateful_args, list):
        hateful_str = " ".join([f"'{item.get('metaphor', '')}' implies {item.get('meaning', '')}" for item in hateful_args])
    else:
        hateful_str = "None."

    benign_args = example.get("benign", [])
    if benign_args and isinstance(benign_args, list):
        benign_str = " ".join([f"'{item.get('metaphor', '')}' implies {item.get('meaning', '')}" for item in benign_args])
    else:
        benign_str = "None."

    full_text = (
        f"Meme Text: {ocr}. "
        f"Contextual Reasoning: {reasoning} "
        f"Hateful Interpretation: {hateful_str} "
        f"Benign Interpretation: {benign_str}"
    )
    
    return {"text": full_text, "labels": example["label"]}

# --- 3. MAIN SCRIPT ---
def main():
    print(f"Loading best model from {MODEL_DIR}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)

    print("Loading and preparing validation dataset...")
    dataset = load_dataset("json", data_files=DATA_FILE, split="train")
    # MUST match the seed from training to get the exact same test set
    dataset = dataset.train_test_split(test_size=0.1, seed=42) 
    
    eval_dataset = dataset["test"]
    eval_dataset = eval_dataset.map(flatten_meme_data, remove_columns=dataset["test"].column_names)

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding=False, truncation=True, max_length=512)

    tokenized_eval = eval_dataset.map(tokenize_function, batched=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Use Trainer just for inference
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    print("Generating predictions...")
    predictions = trainer.predict(tokenized_eval)
    logits = predictions.predictions
    labels = predictions.label_ids

    # Convert raw logits to probabilities
    probs = np.exp(logits) / np.sum(np.exp(logits), axis=-1, keepdims=True)
    prob_pos = probs[:, 1] # Probability that the meme is Hateful (Class 1)

    print("\n--- THRESHOLD SWEEP RESULTS ---")
    print(f"{'Threshold':<10} | {'Accuracy':<10} | {'F1-Score':<10}")
    print("-" * 35)

    best_acc = 0
    best_acc_threshold = 0.5
    best_f1 = 0
    best_f1_threshold = 0.5

    # Sweep from 0.10 to 0.90 in increments of 0.05
    thresholds = np.arange(0.1, 0.95, 0.05)
    
    for t in thresholds:
        # If probability is >= threshold, predict 1 (Hateful), else 0 (Benign)
        preds = (prob_pos >= t).astype(int)
        
        acc = accuracy_score(labels, preds)
        f1 = f1_score(labels, preds)
        
        print(f"{t:<10.2f} | {acc:<10.4f} | {f1:<10.4f}")

        if acc > best_acc:
            best_acc = acc
            best_acc_threshold = t
            
        if f1 > best_f1:
            best_f1 = f1
            best_f1_threshold = t

    print("\n--- FINAL OPTIMAL SETTINGS ---")
    print(f"To maximize ACCURACY, use threshold: {best_acc_threshold:.2f} (Yields {best_acc:.4f})")
    print(f"To maximize F1-SCORE, use threshold: {best_f1_threshold:.2f} (Yields {best_f1:.4f})")
    
    # Show confusion matrix for the accuracy winner
    best_preds = (prob_pos >= best_acc_threshold).astype(int)
    cm = confusion_matrix(labels, best_preds)
    print("\nConfusion Matrix at Best Accuracy Threshold:")
    print(f"True Negatives (Correctly Benign): {cm[0][0]}")
    print(f"False Positives (Hallucinated Hate): {cm[0][1]}")
    print(f"False Negatives (Missed Hate): {cm[1][0]}")
    print(f"True Positives (Correctly Hateful): {cm[1][1]}")

if __name__ == "__main__":
    main()