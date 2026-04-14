import json
import os
import torch
import torch.nn.functional as F
from PIL import Image
from pydantic import BaseModel, Field
from typing import List, Dict, Any
from vllm import LLM, SamplingParams
from vllm.sampling_params import StructuredOutputsParams
from transformers import AutoTokenizer, AutoModelForSequenceClassification

os.environ["VLLM_USE_V1"] = "0"
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["NCCL_IGNORE_DISABLED_P2P"] = "1"

# --- 1. CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))
INDIVIDUAL_IMAGE_DIR = os.path.join(BASE_DIR, "images")

VLLM_MODEL_ID = os.environ.get("VLLM_MODEL_ID", "Qwen/Qwen3-VL-8B-Thinking")
ROBERTA_MODEL_DIR = os.environ.get(
    "ROBERTA_MODEL_DIR",
    os.path.join(PROJECT_ROOT, "metameme_roberta_model", "checkpoint-958"),
)
THRESHOLD = float(os.environ.get("THRESHOLD", "0.25"))


def resolve_roberta_model_dir(model_dir: str) -> str:
    raw = str(model_dir or "").strip()
    if not raw:
        return ""

    if os.path.isabs(raw) and os.path.isdir(raw):
        return raw

    candidates = [
        raw,
        os.path.join(PROJECT_ROOT, raw),
        os.path.join(PROJECT_ROOT, "metameme_roberta_model", "checkpoint-958"),
    ]

    # Common misconfiguration: prefixing local paths with "cara/"
    if raw.startswith("cara/"):
        candidates.append(os.path.join(PROJECT_ROOT, raw[len("cara/") :]))

    seen = set()
    for c in candidates:
        norm = os.path.normpath(c)
        if norm in seen:
            continue
        seen.add(norm)
        if os.path.isdir(norm):
            return norm

    return ""

# --- 2. PYDANTIC SCHEMA ---
class VisualEvidenceItem(BaseModel):
    metaphor: str = Field(
        description="Literal visual element(s) in the meme image (e.g., 'A golden retriever dog', 'A historical uniform')."
    )
    meaning: str = Field(
        description="The cultural, social, or contextual meaning inferred when combining the image and text. Must explain the implicit joke, stereotype, or harmless irony."
    )

class DualRationaleOutput(BaseModel):
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


def resolve_image_path(img_path: str) -> str:
    raw = str(img_path or "").strip()
    if not raw:
        return ""

    if os.path.isabs(raw) and os.path.exists(raw):
        return raw

    candidates = [
        raw,
        os.path.join(INDIVIDUAL_IMAGE_DIR, raw),
        os.path.join(INDIVIDUAL_IMAGE_DIR, os.path.basename(raw)),
        os.path.join(os.getcwd(), raw),
        os.path.join(PROJECT_ROOT, raw),
        os.path.join(PROJECT_ROOT, "images", os.path.basename(raw)),
        os.path.join(PROJECT_ROOT, "facebook-data", "img", os.path.basename(raw)),
    ]

    seen = set()
    for c in candidates:
        norm = os.path.normpath(c)
        if norm in seen:
            continue
        seen.add(norm)
        if os.path.exists(norm):
            return norm
    return ""

# --- 4. MAIN PIPELINE ---
def main():
    # Configuration - list of (image_path, ocr_text, custom_prompt) tuples
    # custom_prompt is optional; leave as None or empty string to use default
    test_cases = [
        ("image.png", "they shot the wrong gorilla", "This meme appears to be a benign joke because"),
        ("image.png", "they shot the wrong gorilla", "This meme appears to be a hateful joke because"),
        # Add more test cases as needed
    ]
    
    schema = DualRationaleOutput.model_json_schema()
    schema_json = json.dumps(schema)

    # ---------------------------------------------------------
    # STAGE 1: INITIALIZE MODELS
    # ---------------------------------------------------------
    print(f"Initializing vLLM ({VLLM_MODEL_ID})...")
    llm = LLM(
        model=VLLM_MODEL_ID,
        trust_remote_code=True,
        max_model_len=4096,
        gpu_memory_utilization=0.85,
        enforce_eager=True,
    )

    resolved_roberta_dir = resolve_roberta_model_dir(ROBERTA_MODEL_DIR)
    if not resolved_roberta_dir:
        raise FileNotFoundError(
            "Could not resolve ROBERTA_MODEL_DIR to a local directory. "
            f"Received: '{ROBERTA_MODEL_DIR}'. "
            "Set ROBERTA_MODEL_DIR to an absolute path or a path relative to the repo root."
        )

    print(f"Initializing RoBERTa ({resolved_roberta_dir})...")
    tokenizer = AutoTokenizer.from_pretrained(resolved_roberta_dir)
    roberta_model = AutoModelForSequenceClassification.from_pretrained(resolved_roberta_dir)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    roberta_model.to(device)
    roberta_model.eval()

    # ---------------------------------------------------------
    # STAGE 2: BUILD vLLM PROMPTS
    # ---------------------------------------------------------
    prompts = []
    for img_path, ocr_text, custom_prompt in test_cases:
        resolved_path = resolve_image_path(img_path)
        if not resolved_path:
            raise FileNotFoundError(
                f"Could not resolve image path '{img_path}'. "
                "Use an absolute path or a path relative to the repo root."
            )

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
            f"{task_text}{custom_prompt}"
            "<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        prompts.append({
            "prompt": vllm_prompt,
            "multi_modal_data": {"image": Image.open(resolved_path).convert("RGB")}
        })

    # ---------------------------------------------------------
    # STAGE 3: vLLM EXTRACTION
    # ---------------------------------------------------------
    sampling = SamplingParams(
        temperature=0.0,
        max_tokens=2048,
        structured_outputs=StructuredOutputsParams(json=schema_json),
    )

    print("Running vLLM Extraction...")
    vllm_outputs = llm.generate(prompts, sampling_params=sampling)
    
    # Parse and flatten the vLLM outputs
    roberta_inputs = []
    vllm_parsed_results = []
    
    for (img_path, ocr_text, custom_prompt), out in zip(test_cases, vllm_outputs):
        try:
            parsed = json.loads(out.outputs[0].text)
        except Exception as e:
            print(f"Error parsing vLLM output for {img_path}: {e}")
            parsed = {"reasoning": "Error", "hateful": [], "benign": []}
        
        vllm_parsed_results.append(parsed)
        flat_text = flatten_for_roberta(parsed, ocr_text)
        roberta_inputs.append(flat_text)

    # ---------------------------------------------------------
    # STAGE 4: RoBERTa CLASSIFICATION
    # ---------------------------------------------------------
    print("Running RoBERTa Inference...")
    inputs = tokenizer(roberta_inputs, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = roberta_model(**inputs)
        probs = F.softmax(outputs.logits, dim=-1)
        hateful_probs = probs[:, 1].cpu().numpy()

    # ---------------------------------------------------------
    # STAGE 5: COMPILE RESULTS
    # ---------------------------------------------------------
    final_results = []
    
    for i, (img_path, ocr_text, custom_prompt) in enumerate(test_cases):
        prob = float(hateful_probs[i])
        result = {
            "img": os.path.basename(img_path),
            "ocr_text": ocr_text,
            "reasoning": vllm_parsed_results[i].get("reasoning", ""),
            "hateful": vllm_parsed_results[i].get("hateful", []),
            "benign": vllm_parsed_results[i].get("benign", []),
            "roberta_hateful_prob": prob,
            "final_prediction": 1 if prob >= THRESHOLD else 0,
            "label_text": "hateful" if prob >= THRESHOLD else "benign",
        }
        final_results.append(result)
        
        print("-" * 50)
        print(f"Image: {os.path.basename(img_path)}")
        print(f"OCR Text: {ocr_text}")
        print(f"RoBERTa Hateful Probability: {prob:.4f}")
        print(f"Final Prediction: {result['label_text']}")
        print(f"Reasoning: {result['reasoning']}")
        print(f"Hateful Arguments: {len(result['hateful'])} items")
        print(f"Benign Arguments: {len(result['benign'])} items")

    print("\n" + "="*50)
    print("ALL RESULTS:")
    print("="*50)
    print(json.dumps(final_results, indent=2, ensure_ascii=False))
    
    return final_results


if __name__ == "__main__":
    main()
