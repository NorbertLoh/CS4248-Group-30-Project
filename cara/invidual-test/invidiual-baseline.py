import json
import os
os.environ["VLLM_USE_V1"] = "0"
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["NCCL_IGNORE_DISABLED_P2P"] = "1"

from PIL import Image
from pydantic import BaseModel, Field
from vllm import LLM, SamplingParams
from vllm.sampling_params import StructuredOutputsParams


class HatefulBenignDecision(BaseModel):
    reasoning: str = Field(
        description="Brief explanation for why the meme is predicted as hateful or benign"
    )
    label: int = Field(
        description="Binary meme decision: 1 for hateful, 0 for benign"
    )


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))
INDIVIDUAL_IMAGE_DIR = os.path.join(BASE_DIR, "images")


def normalize_binary(label_text: str) -> int:
    text = str(label_text or "").strip().lower()
    return 1 if text == "hateful" else 0


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


def main() -> list[dict]:
    # Configuration - list of (image_path, ocr_text, custom_prompt) tuples
    # custom_prompt is optional; leave as None or empty string to use default
    test_cases = [
        ("image.png", "they shot the wrong gorilla", "This meme appears to be a benign joke because"),
        ("image.png", "they shot the wrong gorilla", "This meme appears to be a hateful joke because"),
        # Add more test cases as needed
    ]
    
    # Default system prompt
    DEFAULT_SYSTEM_PROMPT = "You are a strict multimodal meme classifier."
    
    MODEL_ID = os.environ.get("BASELINE_MODEL_ID", "Qwen/Qwen3-VL-8B-Thinking")
    
    # Load model once
    print(f"Loading model: {MODEL_ID}")
    llm = LLM(
        model=MODEL_ID,
        trust_remote_code=True,
        max_model_len=4096,
        gpu_memory_utilization=0.85,
    )
    
    schema_dict = HatefulBenignDecision.model_json_schema()
    
    results = []
    
    for img_path, ocr_text, custom_prompt in test_cases:
        print(f"\nProcessing image: {img_path}")
        print(f"OCR text: {ocr_text}")
        if custom_prompt:
            print(f"Custom prompt: {custom_prompt}")

        resolved_path = resolve_image_path(img_path)
        if not resolved_path:
            raise FileNotFoundError(
                f"Could not resolve image path '{img_path}'. "
                "Use an absolute path or a path relative to the repo root."
            )
        
        # Load image
        image = Image.open(resolved_path).convert("RGB")
        
        # Build prompt
        prompt_text = (
            "Classify this meme as hateful or benign.\n"
            "Use both image content and OCR text.\n"
            "Return only JSON that matches the schema.\n\n"
            f"OCR text: {ocr_text}\n"
            f"Schema: {json.dumps(schema_dict)}"
        )
        
        vllm_prompt = (
            "<|im_start|>{DEFAULT_SYSTEM_PROMPT}\n"
            f"<|im_end|>\n"
            "<|im_start|>user\n"
            "<|vision_start|><|image_pad|><|vision_end|>\n"
            f"{prompt_text}\n{custom_prompt}<|im_end|>\n"
            "<|im_start|>assistant\n"
            "<think>\n"
        )
        
        # Generate output
        sampling = SamplingParams(
            temperature=0.0,
            max_tokens=1024,
            structured_outputs=StructuredOutputsParams(json=schema_dict),
        )
        
        outputs = llm.generate(
            [{"prompt": vllm_prompt, "multi_modal_data": {"image": image}}],
            sampling_params=sampling
        )
        
        # Parse and display result
        out = outputs[0]
        parsed = HatefulBenignDecision.model_validate_json(out.outputs[0].text)
        
        raw_text = "<think>\n" + out.outputs[0].text
        
        result = {
            "img": os.path.basename(resolved_path),
            "resolved_img_path": resolved_path,
            "ocr_text": ocr_text,
            "reasoning": parsed.reasoning,
            "label": normalize_binary(parsed.label),
            "label_text": parsed.label,
            "raw_output": raw_text,
        }
        
        results.append(result)
        
        print("-" * 50)
        print(json.dumps(result, indent=2, ensure_ascii=False))
    
    print("\n" + "="*50)
    print("ALL RESULTS:")
    print("="*50)
    print(json.dumps(results, indent=2, ensure_ascii=False))
    
    return results


if __name__ == "__main__":
    main()
