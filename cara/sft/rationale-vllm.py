import json
import os
from typing import List

from PIL import Image
from pydantic import BaseModel, Field
from tqdm import tqdm
from vllm import LLM, SamplingParams
from vllm.sampling_params import StructuredOutputsParams


# --- 1. PYDANTIC SCHEMA ---
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


# --- 2. CONFIGURATION ---
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

DATA_PATH = os.environ.get(
    "RATIONALE_DATA_PATH",
    os.path.join(BASE_DIR, "datapreparation", "output", "facebook-data/train.jsonl"),
)
IMG_DIR = os.environ.get(
    "RATIONALE_IMG_DIR",
    os.path.join(BASE_DIR, "facebook-data", "img"),
)
OUT_PATH = os.environ.get(
    "RATIONALE_OUT_PATH",
    os.path.join(BASE_DIR, "datapreparation", "output", "predictions_rationale_vllm_8B_all.jsonl"),
)

MODEL_ID = os.environ.get("UNSLOTH_QWEN3_VL_MODEL_ID", "Qwen/Qwen3-VL-8B-Thinking")


# --- 3. HELPERS ---
def resolve_image_path(img_field: str) -> str:
    img_value = (img_field or "").strip()
    if not img_value:
        return ""

    return os.path.abspath(os.path.join(IMG_DIR, os.path.basename(img_value)))


def load_rgb_image(img_path: str):
    try:
        with Image.open(img_path) as image:
            return image.convert("RGB")
    except Exception:
        return None


def build_prompt(ocr_text: str) -> str:
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

    return (
        "<|im_start|>system\n"
        "You are a highly precise multimodal data extraction engine specialized in deep visual and cultural analysis.<|im_end|>\n"
        "<|im_start|>user\n"
        "<|vision_start|><|image_pad|><|vision_end|>\n"
        f"{task_text}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


# --- 4. MAIN PROCESSING ---
def main() -> None:
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Input JSONL not found: {DATA_PATH}")

    print(f"Loading input data from {DATA_PATH}...")
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        samples = [json.loads(line) for line in f if line.strip()]

    schema = DualRationaleOutput.model_json_schema()
    schema_json = json.dumps(schema)

    prompts = []
    valid_indices: List[int] = []

    for i, sample in enumerate(tqdm(samples, desc="Preparing inputs")):
        img_path = resolve_image_path(sample.get("img", ""))
        if not img_path:
            # Fallbacks
            sample["reasoning"] = "IMAGE_NOT_FOUND"
            sample["hateful"] = [{"metaphor": "", "meaning": "IMAGE_NOT_FOUND"}]
            sample["benign"] = [{"metaphor": "", "meaning": "IMAGE_NOT_FOUND"}]
            continue

        image = load_rgb_image(img_path)
        if image is None:
            sample["reasoning"] = "IMAGE_UNREADABLE"
            sample["hateful"] = [{"metaphor": "", "meaning": "IMAGE_UNREADABLE"}]
            sample["benign"] = [{"metaphor": "", "meaning": "IMAGE_UNREADABLE"}]
            continue

        ocr_text = str(sample.get("text", "")).strip()
        prompt = build_prompt(ocr_text=ocr_text)

        prompts.append(
            {
                "prompt": prompt,
                "multi_modal_data": {"image": image},
            }
        )
        valid_indices.append(i)

    print(f"Loading model: {MODEL_ID}")
    
    os.environ["VLLM_USE_V1"] = "0"
    
    llm = LLM(
        model=MODEL_ID,
        trust_remote_code=True,
        max_model_len=4096,
        gpu_memory_utilization=0.85,
        enforce_eager=True,
    )

    # Increased max_tokens slightly to accommodate the reasoning paragraph
    sampling = SamplingParams(
        temperature=0.0,
        max_tokens=2048,
        structured_outputs=StructuredOutputsParams(json=schema_json),
    )

    print(f"Generating structured outputs for {len(prompts)} samples...")
    outputs = llm.generate(prompts, sampling_params=sampling)

    for sample_idx, output in zip(valid_indices, outputs):
        try:
            parsed = json.loads(output.outputs[0].text)
            
            # Extract reasoning
            reasoning = str(parsed.get("reasoning", "")).strip()
            
            # Extract lists
            hateful_list = parsed.get("hateful", [])
            benign_list = parsed.get("benign", [])

            cleaned_hateful = [
                {
                    "metaphor": str(item.get("metaphor", "")).strip(),
                    "meaning": str(item.get("meaning", "")).strip()
                } for item in hateful_list if isinstance(item, dict)
            ]
            
            cleaned_benign = [
                {
                    "metaphor": str(item.get("metaphor", "")).strip(),
                    "meaning": str(item.get("meaning", "")).strip()
                } for item in benign_list if isinstance(item, dict)
            ]

            samples[sample_idx]["reasoning"] = reasoning
            samples[sample_idx]["hateful"] = cleaned_hateful
            samples[sample_idx]["benign"] = cleaned_benign

        except (json.JSONDecodeError, IndexError, TypeError, AttributeError):
            samples[sample_idx]["reasoning"] = "PARSING_ERROR"
            samples[sample_idx]["hateful"] = [{"metaphor": "", "meaning": "PARSING_ERROR"}]
            samples[sample_idx]["benign"] = [{"metaphor": "", "meaning": "PARSING_ERROR"}]

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    print(f"Saving outputs to {OUT_PATH}...")
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for sample in samples:
            out_item = {
                "id": sample.get("id"),
                "img": sample.get("img"),
                "ocr_text": sample.get("text", ""),
                "label": sample.get("label"),
                "reasoning": sample.get("reasoning", ""), # Saved reasoning
                "hateful": sample.get("hateful", []),
                "benign": sample.get("benign", []),
            }
            f.write(json.dumps(out_item, ensure_ascii=False) + "\n")

    print("Done: generated rationales and lists with structured vLLM output.")


if __name__ == "__main__":
    main()