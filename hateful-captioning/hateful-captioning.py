import os
import json
import torch
from PIL import Image
from tqdm import tqdm
from typing import List
from pydantic import BaseModel

# Unsloth & Qwen utilities
from unsloth import FastVisionModel
from qwen_vl_utils import process_vision_info

# --- 1. CONFIGURATION ---
# Using the pre-quantized 4-bit Unsloth model
MODEL_ID = "unsloth/Qwen3-VL-8B-Instruct-bnb-4bit"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FACEBOOK_DATA_DIR = os.path.join(BASE_DIR, "..", "facebook-data")
DATASET_JSONL = os.path.join(FACEBOOK_DATA_DIR, "train.jsonl")
RUN_SUFFIX = 2
SAMPLES_PER_LABEL = 200
TARGET_LABELS = (0, 1)
BATCH_SIZE = 8 
OUTPUT_FILE = os.path.join(BASE_DIR, f"captions_output{RUN_SUFFIX}.jsonl")

# --- 2. STRUCTURED OUTPUT SCHEMA ---
class Metaphor(BaseModel):
    metaphor: str
    meaning: str

class MemeAnalysis(BaseModel):
    img_captions: List[str]
    meme_captions: List[str]
    title: str
    metaphors: List[Metaphor]

# --- 3. DATA SELECTION ---
def select_balanced_samples_chunked(dataset_jsonl, samples_per_label, target_labels, run_suffix):
    counts = dict.fromkeys(target_labels, 0)
    skipped = dict.fromkeys(target_labels, 0)
    selected = []
    skip_target = (run_suffix - 1) * samples_per_label

    with open(dataset_jsonl, 'r') as f_in:
        for line in f_in:
            data = json.loads(line)
            try:
                label = int(data.get('label'))
            except (ValueError, TypeError): continue

            if label in counts:
                if skipped[label] < skip_target:
                    skipped[label] += 1
                    continue
                if counts[label] < samples_per_label:
                    selected.append(data)
                    counts[label] += 1
                if all(counts[l] >= samples_per_label for l in target_labels):
                    break
                    
    print(f"Run {run_suffix} | Skipped first {skip_target} per label | Collected: {counts}")
    return selected

# --- 4. PROCESSING LOGIC ---
def process_memes():
    print(f"Loading 4-bit Unsloth model: {MODEL_ID}")
    
    # Load Unsloth's optimized 4-bit model and processor
    model, processor = FastVisionModel.from_pretrained(
        MODEL_ID,
        load_in_4bit=True, # Forces standard bnb 4-bit loading
    )
    
    # Crucial: Enable Unsloth's 2x faster inference mode
    FastVisionModel.for_inference(model)

    selected_data = select_balanced_samples_chunked(
        DATASET_JSONL, SAMPLES_PER_LABEL, TARGET_LABELS, RUN_SUFFIX
    )

    prompt_text = (
        "Analyze this meme and extract its components. "
        "1. Provide all literal 'img_captions' describing exactly what is visually seen. "
        "2. Provide 1 'meme_captions' explaining the underlying joke or message. "
        "3. Give the meme a short 'title'. "
        "4. Identify 'metaphors' where a visual element represents a broader meaning. "
        "You MUST output exactly in valid JSON format matching this schema:\n"
        f"{MemeAnalysis.model_json_schema()}"
    )

    with open(OUTPUT_FILE, 'w') as f_out:
        # Process in batches of 8
        for i in tqdm(range(0, len(selected_data), BATCH_SIZE), desc="Processing Batches"):
            batch_data = selected_data[i : i + BATCH_SIZE]
            batch_messages = []
            
            for data in batch_data:
                img_path = os.path.join(FACEBOOK_DATA_DIR, data['img'])
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": f"file://{img_path}"},
                            {"type": "text", "text": prompt_text},
                        ],
                    }
                ]
                batch_messages.append(messages)

            # Preparation for Vision batch inference
            texts = [
                processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
                for msg in batch_messages
            ]
            image_inputs, video_inputs = process_vision_info(batch_messages)
            
            inputs = processor(
                text=texts,
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            ).to("cuda")

            # Batch Inference (Outputs structured as JSON)
            generated_ids = model.generate(
                **inputs, 
                max_new_tokens=1024,
                temperature=0.2, # Low temp for stable JSON outputs
                do_sample=True
            )
            
            # Trim the prompt tokens from the output
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            
            output_texts = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )

            # Parse outputs and save
            for data, raw_output in zip(batch_data, output_texts):
                try:
                    # Strip markdown blocks to isolate the JSON string
                    clean_json = raw_output.strip().replace("```json", "").replace("```", "")
                    result_json = json.loads(clean_json)
                    
                    final_output = {
                        "category": "HMD-memes",
                        "img_captions": result_json.get("img_captions", []),
                        "meme_captions": result_json.get("meme_captions", []),
                        "title": result_json.get("title", ""),
                        "img_fname": data['img'],
                        "metaphors": result_json.get("metaphors", []),
                        "post_id": str(data['id']),
                        "raw_output": raw_output,
                    }
                    f_out.write(json.dumps(final_output) + "\n")
                except Exception as e:
                    print(f"\nError parsing JSON for ID {data['id']}: {e}\nRaw Output was:\n{raw_output}")
            
            f_out.flush()

if __name__ == "__main__":
    process_memes()