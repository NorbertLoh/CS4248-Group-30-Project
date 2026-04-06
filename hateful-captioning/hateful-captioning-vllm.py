import json
import os
from typing import Any, Dict, List, Literal

from PIL import Image
from pydantic import BaseModel, Field
from tqdm import tqdm
from vllm import LLM, SamplingParams
from vllm.sampling_params import StructuredOutputsParams

# --- 1. CONFIGURATION ---
MODEL_ID = os.environ.get("HATEFUL_CAPTIONING_MODEL_ID", "QuantTrio/Qwen3-VL-32B-Thinking-AWQ")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FACEBOOK_DATA_DIR = os.path.join(BASE_DIR, "..", "facebook-data")
DATASET_JSONL = os.environ.get(
    "HATEFUL_CAPTIONING_DATASET",
    os.path.join(FACEBOOK_DATA_DIR, "train.jsonl"),
)

RUN_SUFFIX = 1
SAMPLES_PER_LABEL = 2000
TARGET_LABELS = (0, 1)
OUTPUT_FILE = os.environ.get(
    "HATEFUL_CAPTIONING_OUTPUT",
    os.path.join(BASE_DIR, f"captions_vllm_output{RUN_SUFFIX}_structured.jsonl"),
)

# --- 2. PURE PYDANTIC SCHEMA ---
class Metaphor(BaseModel):
    metaphor: str
    meaning: str

class MemeAnalysis(BaseModel):
    img_captions: List[str] = Field(description="Literal visual descriptions of the scene")
    ocr_text: str = Field(description="The exact text written on the meme")
    meme_captions: List[str] = Field(description="Explanation of the intended joke or social message")
    title: str = Field(description="A short, descriptive title for the meme")
    hateful_metaphors: List[Metaphor] = Field(default_factory=list, description="Worst-case interpretations targeting specific groups")
    benign_metaphors: List[Metaphor] = Field(default_factory=list, description="Harmless, literal, or ironically comedic interpretations")
    reasoning: str = Field(description="Step-by-step logical analysis of the meme's intent and cultural context")
    reasoning_label: Literal["hateful", "benign"] = Field(description="The final classification based on the reasoning above")

# --- 3. DATA HELPERS ---
def select_balanced_samples(
    dataset_jsonl: str,
    samples_per_label: int,
    target_labels: tuple,
    run_suffix: int,
) -> List[Dict[str, Any]]:
    counts = dict.fromkeys(target_labels, 0)
    skipped = dict.fromkeys(target_labels, 0)
    selected: List[Dict[str, Any]] = []
    skip_target = (run_suffix - 1) * samples_per_label

    with open(dataset_jsonl, "r", encoding="utf-8") as f_in:
        for line in f_in:
            data = json.loads(line)
            try:
                label = int(data.get("label"))
            except (ValueError, TypeError):
                continue

            if label in counts:
                if skipped[label] < skip_target:
                    skipped[label] += 1
                    continue
                if counts[label] < samples_per_label:
                    selected.append(data)
                    counts[label] += 1
                if all(counts[l] >= samples_per_label for l in target_labels):
                    break
    return selected

def resolve_image_path(data_item: Dict[str, Any]) -> str:
    rel = str(data_item.get("img", "")).strip()
    candidates = [
        os.path.join(FACEBOOK_DATA_DIR, rel),
        os.path.join(FACEBOOK_DATA_DIR, "img", os.path.basename(rel)),
        os.path.join(BASE_DIR, "..", rel),
    ]
    for path in candidates:
        if path and os.path.exists(path):
            return path
    return candidates[0]

# --- 4. MAIN PROCESSING ---
def process_memes() -> None:
    print(f"Initializing vLLM with {MODEL_ID}...")
    
    os.environ["VLLM_USE_V1"] = "0"
    
    llm = LLM(
        model=MODEL_ID,
        trust_remote_code=True,
        max_model_len=4096,
        gpu_memory_utilization=0.90,
        enforce_eager=True,
    )

    selected_data = select_balanced_samples(
        DATASET_JSONL, SAMPLES_PER_LABEL, TARGET_LABELS, RUN_SUFFIX
    )

    # 1. Extract JSON Schema from Pydantic
    json_schema = MemeAnalysis.model_json_schema()

    # 2. Setup standard multimodal prompt (No <think> tag needed)
    prompt_text = (
        "Analyze this meme thoroughly. Identify the visual components, extract the OCR text, "
        "explain the underlying joke/message, and provide both hateful and benign "
        "metaphorical interpretations. Conclude with a logical reasoning "
        "paragraph and a final classification label."
    )

    all_inputs = []
    print("Preparing images and prompts...")
    for data in tqdm(selected_data):
        img_path = resolve_image_path(data)
        vllm_prompt = (
            "<|im_start|>system\n"
            "You are a highly precise multimodal data extraction engine specialized in deep visual and cultural analysis.<|im_end|>\n"
            "<|im_start|>user\n"
            "<|vision_start|><|image_pad|><|vision_end|>\n"
            f"{prompt_text}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        all_inputs.append({
            "prompt": vllm_prompt,
            "multi_modal_data": {"image": Image.open(img_path).convert("RGB")},
        })
        break

    # 3. Setup Structured Outputs Parameters
    sampling = SamplingParams(
        temperature=0.3,
        max_tokens=2048,
        structured_outputs=StructuredOutputsParams(json=json_schema) # USING YOUR ORIGINAL CLUSTER API
    )

    print(f"Generating for {len(all_inputs)} memes...")
    outputs = llm.generate(all_inputs, sampling_params=sampling)

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    
    # 4. Save results with direct Pydantic validation
    print(f"Saving results to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f_out:
        for data, out in zip(selected_data, outputs):
            raw_json_str = out.outputs[0].text if out.outputs else "{}"
            
            try:
                # Direct validation from the generated string
                parsed = MemeAnalysis.model_validate_json(raw_json_str)
                parsed_dict = parsed.model_dump()
            except Exception as e:
                print(f"Validation failed for post_id {data.get('id')}: {e}")
                parsed_dict = {
                    "ocr_text": "",
                    "reasoning": f"Validation Error: {str(e)}", 
                    "reasoning_label": "benign",
                    "img_captions": [], "meme_captions": [], "title": "error",
                    "hateful_metaphors": [], "benign_metaphors": []
                }

            final_output = {
                "category": "HMD-memes",
                "img_fname": data.get("img", ""),
                "post_id": str(data.get("id", "")),
                "label": data.get("label", None),
                "raw_output": raw_json_str,
                **parsed_dict 
            }
            f_out.write(json.dumps(final_output, ensure_ascii=False) + "\n")

    print("Processing complete.")

if __name__ == "__main__":
    process_memes()