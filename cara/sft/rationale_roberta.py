import json
import os
import torch
import torch.nn.functional as F
from PIL import Image
from pydantic import BaseModel, Field
from typing import List, Dict, Any
from tqdm import tqdm
from vllm import LLM, SamplingParams
from vllm.sampling_params import StructuredOutputsParams
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# --- 1. CONFIGURATION ---
VLLM_MODEL_ID = os.environ.get("VLLM_MODEL_ID", "Qwen/Qwen3-VL-8B-Thinking")
ROBERTA_MODEL_DIR = os.environ.get("MODEL_DIR", "./metameme_roberta_model")
INPUT_DATA = os.environ.get("INPUT_FILE", "datapreparation/output/facebook-samples-test-roberta.jsonl")
OUTPUT_FILE = os.environ.get("OUTPUT_FILE", "datapreparation/output/final_roberta_predictions.jsonl")
THRESHOLD = float(os.environ.get("THRESHOLD", "0.40"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "16"))

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMG_DIR = os.path.join(BASE_DIR, "facebook-data")

# --- 2. PYDANTIC SCHEMA ---
class VisualEvidenceItem(BaseModel):
    metaphor: str = Field(
        description="Literal visual element(s) in the meme image (e.g., 'A golden retriever dog', 'A historical uniform')."
    )
    meaning: str = Field(
        description="The cultural, social, or contextual meaning inferred when combining the image and text. Must explain the implicit joke, stereotype, or harmless irony."
    )

class DualRationaleOutput(BaseModel):
    # CRITICAL: 'reasoning' must be the first field so the model generates it before the lists.
    reasoning: str = Field(
        description="Step-by-step logical analysis of the meme's intent and cultural context. Provide this detailed thought process first."
    )
    hateful: List[VisualEvidenceItem] = Field(
        description="A list of arguments explaining why this meme relies on harmful stereotypes, slurs, or toxic social contexts."
    )
    benign: List[VisualEvidenceItem] = Field(
        description="A list of arguments explaining why this meme is socially acceptable, literally interpreted, or relying on harmless irony."
    )

# --- 3. HELPERS ---
def resolve_image_path(img_field: str) -> str:
    rel = str(img_field).strip()
    candidates = [
        rel,
        os.path.join(IMG_DIR, rel),
        os.path.join(IMG_DIR, "img", os.path.basename(rel))
    ]
    for c in candidates:
        if os.path.exists(c): return c
    return ""

def flatten_for_roberta(parsed_json: Dict, ocr_text: str) -> str:
    """Flattens the vLLM output into the text format RoBERTa expects."""
    reasoning = parsed_json.get("reasoning", "")
    
    hateful_args = parsed_json.get("hateful", [])
    if hateful_args and isinstance(hateful_args, list):
        hateful_str = " ".join([f"'{m.get('metaphor', '')}' implies {m.get('meaning', '')}" for m in hateful_args])
    else:
        hateful_str = "None."

    benign_args = parsed_json.get("benign", [])
    if benign_args and isinstance(benign_args, list):
        benign_str = " ".join([f"'{m.get('metaphor', '')}' implies {m.get('meaning', '')}" for m in benign_args])
    else:
        benign_str = "None."

    return (
        f"Meme Text: {ocr_text}. "
        f"Contextual Reasoning: {reasoning} "
        f"Hateful Interpretation: {hateful_str} "
        f"Benign Interpretation: {benign_str}"
    )

# --- 4. MAIN PIPELINE ---
def main():
    # ---------------------------------------------------------
    # STAGE 1: LOAD DATA
    # ---------------------------------------------------------
    print(f"Loading input data from {INPUT_DATA}...")
    with open(INPUT_DATA, "r", encoding="utf-8") as f:
        samples = [json.loads(line) for line in f if line.strip()]

    schema = DualRationaleOutput.model_json_schema()
    schema_json = json.dumps(schema)

    # ---------------------------------------------------------
    # STAGE 2: vLLM EXTRACTION (The Miner)
    # ---------------------------------------------------------
    print(f"Initializing vLLM ({VLLM_MODEL_ID})...")
    os.environ["VLLM_USE_V1"] = "0"
    llm = LLM(
        model=VLLM_MODEL_ID,
        trust_remote_code=True,
        max_model_len=4096,
        gpu_memory_utilization=0.85, # Lowered to leave room for RoBERTa
        enforce_eager=True,
    )

    prompts = []
    for sample in samples:
        img_path = resolve_image_path(sample.get("img", ""))
        ocr_text = str(sample.get("text", "")).strip()
        
        task_text = (
            f"### TARGET MEME TEXT:\n'{ocr_text}'\n\n"
            "### TASK:\n"
            "You are analyzing a meme with text and image. You must generate structured JSON containing:\n"
            "1) reasoning: A detailed logical analysis of the meme's visual and cultural context. Do this first.\n"
            "2) hateful: A LIST of visual evidence items showing how elements combine to create a toxic, prejudiced, or offensive message.\n"
            "3) benign: A LIST of visual evidence items showing how elements combine to create an innocent, literal, or harmless message.\n\n"
            "Constraints for EACH item in your lists:\n"
            "- metaphor: State only the literal, objective visual elements you are basing the argument on.\n"
            "- meaning: Explain the social, cultural, or contextual subtext. You MUST interpret the implicit meaning.\n"
            "- Keep each field concise and highly specific.\n"
            "- Output only valid JSON matching the schema.\n"
        )
        vllm_prompt = (
            "<|im_start|>system\n"
            "You are a highly precise multimodal data extraction engine specialized in deep visual and cultural analysis.<|im_end|>\n"
            "<|im_start|>user\n"
            "<|vision_start|><|image_pad|><|vision_end|>\n"
            f"{task_text}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        prompts.append({
            "prompt": vllm_prompt,
            "multi_modal_data": {"image": Image.open(img_path).convert("RGB")} if img_path else None
        })

    sampling = SamplingParams(
        temperature=0.0,
        max_tokens=2048,
        structured_outputs=StructuredOutputsParams(json=schema_json),
    )

    print("Running vLLM Extraction...")
    vllm_outputs = llm.generate(prompts, sampling_params=sampling)
    
    # Optional: Delete LLM from memory if you are extremely tight on VRAM
    # del llm
    # torch.cuda.empty_cache()

    # Parse and flatten the vLLM outputs
    roberta_inputs = []
    for sample, out in zip(samples, vllm_outputs):
        try:
            parsed = json.loads(out.outputs[0].text)
        except Exception:
            parsed = {"reasoning": "Error", "hateful": [], "benign": []}
            
        sample["vllm_raw_parsed"] = parsed
        flat_text = flatten_for_roberta(parsed, str(sample.get("text", "")))
        roberta_inputs.append(flat_text)

    # ---------------------------------------------------------
    # STAGE 3: RoBERTa CLASSIFICATION (The Judge)
    # ---------------------------------------------------------
    print(f"Initializing RoBERTa ({ROBERTA_MODEL_DIR})...")
    tokenizer = AutoTokenizer.from_pretrained(ROBERTA_MODEL_DIR)
    roberta_model = AutoModelForSequenceClassification.from_pretrained(ROBERTA_MODEL_DIR)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    roberta_model.to(device)
    roberta_model.eval()

    print("Running RoBERTa Inference...")
    batch_size = BATCH_SIZE
    final_results = []
    
    for i in tqdm(range(0, len(roberta_inputs), batch_size)):
        batch_texts = roberta_inputs[i : i + batch_size]
        batch_samples = samples[i : i + batch_size]
        
        inputs = tokenizer(batch_texts, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = roberta_model(**inputs)
            probs = F.softmax(outputs.logits, dim=-1)
            hateful_probs = probs[:, 1].cpu().numpy()
            
        for j, sample in enumerate(batch_samples):
            prob = float(hateful_probs[j])
            sample["roberta_hateful_prob"] = prob
            sample["final_prediction"] = 1 if prob >= THRESHOLD else 0
            
            # Clean up the raw parsed dict to match your output format
            sample["reasoning"] = sample["vllm_raw_parsed"].get("reasoning", "")
            sample["hateful"] = sample["vllm_raw_parsed"].get("hateful", [])
            sample["benign"] = sample["vllm_raw_parsed"].get("benign", [])
            del sample["vllm_raw_parsed"]
            
            final_results.append(sample)

    # ---------------------------------------------------------
    # STAGE 4: SAVE OUTPUTS
    # ---------------------------------------------------------
    print(f"Saving final pipeline results to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f_out:
        for item in final_results:
            f_out.write(json.dumps(item, ensure_ascii=False) + "\n")

    print("End-to-End Inference Complete.")

if __name__ == "__main__":
    main()