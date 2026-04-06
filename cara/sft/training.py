import json
import os
from PIL import Image # <-- FIX 2: Added PIL import
from datasets import Dataset
from unsloth import FastVisionModel, is_bf16_supported, UnslothVisionDataCollator # <-- FIX 1: Added Collator import
from trl import SFTTrainer, SFTConfig
import torch

# --- 1. CONFIGURATION ---
DATA_PATH = "datapreparation/output/predictions_rationale_vllm.jsonl"
MODEL_ID = "unsloth/Qwen3-VL-8B-Instruct-bnb-4bit"
MAX_SEQ_LENGTH = 2048

# --- 2. DATA FORMATTING FUNCTION ---
def format_evidence(evidence_list):
    if not evidence_list or not isinstance(evidence_list, list):
        return "None provided."
    
    text_blocks = []
    for item in evidence_list:
        if isinstance(item, dict):
            meta = item.get("metaphor", "").strip()
            mean = item.get("meaning", "").strip()
            if meta and mean:
                text_blocks.append(f"- Visual Evidence: {meta}\n  Interpretation: {mean}")
    return "\n".join(text_blocks)


def resolve_image_path(img_field):
    img_value = (img_field or "").strip()
    if not img_value:
        return ""

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    candidates = [
        img_value,
        os.path.join(repo_root, img_value),
        os.path.join(repo_root, "datapreparation", "output", img_value),
        os.path.join(repo_root, "facebook-data", img_value),
        os.path.join(repo_root, "facebook-data", "img", os.path.basename(img_value)),
        os.path.join(repo_root, "datapreparation", "output", "facebook-images", os.path.basename(img_value)),
    ]

    for candidate in candidates:
        abs_candidate = os.path.abspath(candidate)
        if os.path.exists(abs_candidate):
            return abs_candidate

    return ""

def load_and_format_dataset(jsonl_path):
    formatted_data = []
    stats = {
        "total": 0,
        "blank": 0,
        "parse_error": 0,
        "parsing_error": 0,
        "missing_img": 0,
        "loaded": 0,
    }

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                stats["blank"] += 1
                continue

            stats["total"] += 1
            try:
                sample = json.loads(line)
            except json.JSONDecodeError:
                stats["parse_error"] += 1
                continue

            hateful = sample.get("hateful") or []
            benign = sample.get("benign") or []

            if hateful and hateful[0].get("meaning") == "PARSING_ERROR":
                stats["parsing_error"] += 1
                continue
            if benign and benign[0].get("meaning") == "PARSING_ERROR":
                stats["parsing_error"] += 1
                continue

            img_path = resolve_image_path(sample.get("img", ""))
            if not img_path:
                stats["missing_img"] += 1
                continue

            target_label = "Hateful" if sample.get("label") == 1 else "Benign"

            prompt_text = (
                f"You are a dialectical judge evaluating a meme. Review the text, the image, and the two conflicting arguments below. Determine if the meme is ultimately Hateful or Benign.\n\n"
                f"### MEME TEXT:\n{sample.get('ocr_text', 'No text')}\n\n"
                f"### ARGUMENT A (Why it might be Hateful):\n{format_evidence(hateful)}\n\n"
                f"### ARGUMENT B (Why it might be Benign):\n{format_evidence(benign)}\n\n"
                f"### FINAL VERDICT:"
            )

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": Image.open(img_path).convert("RGB")},
                        {"type": "text", "text": prompt_text},
                    ],
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": target_label}],
                },
            ]

            formatted_data.append({"messages": messages})
            stats["loaded"] += 1

    print(
        "Dataset load stats: "
        f"total={stats['total']}, blank={stats['blank']}, parse_error={stats['parse_error']}, "
        f"parsing_error={stats['parsing_error']}, missing_img={stats['missing_img']}, loaded={stats['loaded']}"
    )

    if not formatted_data:
        raise ValueError(
            "No valid training samples were loaded. Check DATA_PATH and image path resolution."
        )

    return Dataset.from_list(formatted_data)

# --- 3. MODEL SETUP ---
def main():
    print("Loading dataset...")
    dataset = load_and_format_dataset(DATA_PATH)
    print(f"Loaded {len(dataset)} valid training samples.")

    print("Initializing Unsloth Model...")
    model, tokenizer = FastVisionModel.from_pretrained(
        model_name = MODEL_ID,
        load_in_4bit = True, 
        use_gradient_checkpointing = "unsloth",
    )

    print("Applying LoRA Adapters...")
    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers     = False, 
        finetune_language_layers   = True,  
        finetune_attention_modules = True,
        finetune_mlp_modules       = True,
        r = 16,            
        lora_alpha = 16, 
        lora_dropout = 0,
        bias = "none",
        random_state = 3407,
    )

    # --- 4. TRAINING LOOP ---
    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        data_collator = UnslothVisionDataCollator(model, tokenizer), # <-- FIX 1: Add the Data Collator here
        train_dataset = dataset,
        dataset_text_field = "", 
        max_seq_length = MAX_SEQ_LENGTH,
        dataset_num_proc = 2,
        args = SFTConfig(
            per_device_train_batch_size = 2,
            gradient_accumulation_steps = 4, 
            warmup_steps = 10,
            # max_steps = 30, 
            num_train_epochs = 1,
            learning_rate = 2e-4,
            fp16 = not is_bf16_supported(), # <-- FIX 3: Re-added Precision flags
            bf16 = is_bf16_supported(),     # <-- FIX 3: Re-added Precision flags
            logging_steps = 1,
            optim = "adamw_8bit",
            weight_decay = 0.01,
            lr_scheduler_type = "linear",
            seed = 3407,
            output_dir = "outputs/explainhm-judge-lora",
        ),
    )

    print("Starting Training...")
    trainer.train()

    print("Training Complete. Saving LoRA adapter...")
    model.save_pretrained("judge-qwen3-lora")
    tokenizer.save_pretrained("judge-qwen3-lora")
    
    print("Save to GGUF 4bit")
    model.save_pretrained_gguf("judge-qwen3-lora-gguf", tokenizer, quantization_method = "q4_k_m")

if __name__ == "__main__":
    main()