import json
import os
os.environ["VLLM_USE_V1"] = "0"
# 2. BYPASS SLURM SHARED MEMORY LIMITS
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

# 3. PREVENT NCCL TIMEOUTS ON CLUSTERS
os.environ["NCCL_IGNORE_DISABLED_P2P"] = "1"
import re
from typing import Literal

from PIL import Image
from pydantic import BaseModel, Field
from tqdm import tqdm
from vllm import LLM, SamplingParams


class HatefulBenignDecision(BaseModel):
    label: Literal["hateful", "benign"] = Field(
        description="Binary meme decision: hateful or benign"
    )
    reason: str = Field(
        description="Short explanation based on image and OCR text"
    )
    thought_process: str = Field(
        default="", 
        description="The chain of thought extracted from the think tags"
    )

    @classmethod
    def from_raw_text(cls, text: str):
        # 1. Extract the thinking block
        think_match = re.search(r'<think>(.*?)</think>', text, re.DOTALL)
        thought_process = think_match.group(1).strip() if think_match else ""
        
        # 2. Extract everything after the thinking block (this should be the JSON)
        remainder = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
        
        # 3. Safely extract JSON from the remainder (handles ```json ... ``` markdown)
        json_match = re.search(r'\{.*\}', remainder, re.DOTALL)
        json_str = json_match.group(0) if json_match else remainder

        # 4. Parse JSON and instantiate
        try:
            parsed_data = json.loads(json_str)
            label = parsed_data.get("label", "benign")
            reason = parsed_data.get("reason", "")
            
            # Failsafe for invalid literal
            if label not in ["hateful", "benign"]:
                label = "benign"
                
        except (json.JSONDecodeError, TypeError, AttributeError):
            label = "benign"
            reason = "parse_fallback"
            
        return cls(label=label, reason=reason, thought_process=thought_process)


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_PATH = os.environ.get(
    "BASELINE_DATA_PATH",
    os.path.join(BASE_DIR, "datapreparation", "output", "facebook-samples-test-roberta.jsonl"),
)
IMG_DIR = os.environ.get(
    "BASELINE_IMG_DIR", os.path.join(BASE_DIR, "facebook-data", "img")
)
OUT_PATH = os.environ.get(
    "BASELINE_OUT_PATH",
    os.path.join(BASE_DIR, "datapreparation", "output", "predictions_baseline_vllm_8b.jsonl"),
)
# MODEL_ID = os.environ.get("BASELINE_MODEL_ID", "QuantTrio/Qwen3-VL-32B-Thinking-AWQ")
MODEL_ID = os.environ.get("BASELINE_MODEL_ID", "Qwen/Qwen3-VL-8B-Thinking")

def normalize_binary(label_text: str) -> int:
    text = str(label_text or "").strip().lower()
    return 1 if text == "hateful" else 0


def main() -> None:
    print(f"Loading dataset from {DATA_PATH}")
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        samples = [json.loads(line) for line in f]

    print(f"Loading model: {MODEL_ID}")
    llm = LLM(
        model=MODEL_ID,
        trust_remote_code=True,
        max_model_len=4096,
        gpu_memory_utilization=0.85,
    )

    # We exclude thought_process from the schema we pass to the prompt
    # so the model knows the JSON it generates should only be label and reason.
    schema_dict = HatefulBenignDecision.model_json_schema()
    if "thought_process" in schema_dict.get("properties", {}):
        del schema_dict["properties"]["thought_process"]

    prompts = []

    for s in tqdm(samples, desc="Preparing prompts"):
        image_name = os.path.basename(s.get("img", ""))
        img_path = os.path.join(IMG_DIR, image_name)

        ocr_text = s.get("text", "")
        prompt_text = (
            "Classify this meme as hateful or benign.\n"
            "Use both image content and OCR text.\n"
            "Return only JSON that matches the schema.\n\n"
            f"OCR text: {ocr_text}\n"
            f"Schema: {json.dumps(schema_dict)}"
        )

        vllm_prompt = (
            "<|im_start|>system\n"
            "You are a strict multimodal meme classifier.<|im_end|>\n"
            "<|im_start|>user\n"
            "<|vision_start|><|image_pad|><|vision_end|>\n"
            f"{prompt_text}<|im_end|>\n"
            "<|im_start|>assistant\n"
            "<think>\n" # Added newline here for better generation structure
        )

        prompts.append(
            {
                "prompt": vllm_prompt,
                "multi_modal_data": {"image": Image.open(img_path).convert("RGB")},
            }
        )

    sampling = SamplingParams(
        temperature=0.3,
        max_tokens=1024,
        stop=["<|im_end|>"]
    )

    outputs = llm.generate(prompts, sampling_params=sampling)

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    print(f"Saving predictions to {OUT_PATH}")
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for s, out in zip(samples, outputs):
            # Since we ended our prompt with "<think>\n", the model output
            # starts directly with the thought. We prepend the tag to make regex work.
            raw_text = "<think>\n" + out.outputs[0].text
            
            # Parse everything via our custom method
            parsed = HatefulBenignDecision.from_raw_text(raw_text)

            record = {
                "id": s.get("id"),
                "img": s.get("img"),
                "ocr_text": s.get("text", ""),
                "label": normalize_binary(parsed.label),
                "label_text": parsed.label,
                "reason": parsed.reason,
                "thought_process": parsed.thought_process, # Saved the thoughts!
                "raw_output": raw_text,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print("Done.")


if __name__ == "__main__":
    main()