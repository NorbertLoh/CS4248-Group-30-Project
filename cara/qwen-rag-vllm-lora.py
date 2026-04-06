import gc
import json
import os
import re
from typing import Any, Dict, List, Optional

import torch
from PIL import Image
from pydantic import BaseModel, Field
from tqdm import tqdm
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from vllm.sampling_params import StructuredOutputsParams


class VisualEvidenceItem(BaseModel):
    metaphor: str = Field(
        description="Literal visual element(s) in the meme image (e.g., 'A golden retriever dog', 'A historical uniform')."
    )
    meaning: str = Field(
        description="The cultural, social, or contextual meaning inferred when combining the image and text. Must explain the implicit joke, stereotype, or harmless irony."
    )


class DualRationaleOutput(BaseModel):
    hateful: List[VisualEvidenceItem] = Field(
        description="A list of arguments explaining why this meme relies on harmful stereotypes, slurs, or toxic social contexts."
    )
    benign: List[VisualEvidenceItem] = Field(
        description="A list of arguments explaining why this meme is socially acceptable, literally interpreted, or relying on harmless irony."
    )


class JudgeChoice(BaseModel):
    judge_reasoning: str = Field(description="Why the chosen rationale better matches the policy")
    label: str = Field(description="Final verdict word label: 'Hateful' or 'Benign'")


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_PATH = os.environ.get(
    "STAGE2_DATA_PATH",
    os.path.join(BASE_DIR, "datapreparation", "output", "facebook-samples-test-rationale-excluded.jsonl"),
)
IMG_DIR = os.environ.get("RATIONALE_IMG_DIR", os.path.join(BASE_DIR, "facebook-data", "img"))
OUT_PATH = os.environ.get(
    "STAGE2_OUT_PATH",
    os.path.join(BASE_DIR, "datapreparation", "output", "predictions_stage2_vllm_lora.jsonl"),
)

MODEL_ID = os.environ.get("UNSLOTH_QWEN3_VL_MODEL_ID", "Qwen/Qwen3-VL-8B-Thinking")
JUDGE_LORA_PATH = os.environ.get("JUDGE_LORA_PATH", os.path.join(BASE_DIR, "judge-qwen3-lora"))


def _env_flag(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip().lower() in {"1", "true", "yes", "on"}


USE_JUDGE_LORA = _env_flag("USE_JUDGE_LORA", "1")


POLICY_MANUAL = (
    "Decide if the meme is Hateful or Benign.\n"
    "Return Hateful when the meme relies on slurs, dehumanization, harmful stereotypes, or incitement against a protected group.\n"
    "Return Benign when it is neutral, literal, situational humor, or does not target a protected group.\n"
    "Output strict JSON with fields: judge_reasoning and label."
)


def resolve_image_path(img_field: str) -> str:
    img_value = (img_field or "").strip()
    if not img_value:
        return ""

    candidates = [
        img_value,
        os.path.join(BASE_DIR, img_value),
        os.path.join(BASE_DIR, "datapreparation", "output", img_value),
        os.path.join(IMG_DIR, os.path.basename(img_value)),
    ]

    for candidate in candidates:
        abs_candidate = os.path.abspath(candidate)
        if os.path.exists(abs_candidate):
            return abs_candidate

    return ""


def build_stage1_prompt(ocr_text: str, schema_json: str) -> str:
    task_text = (
        f"### TARGET MEME TEXT:\n'{ocr_text}'\n\n"
        "### TASK:\n"
        "You are analyzing a meme with text and image. You must generate exactly two sets of rationales:\n"
        "1) hateful: A LIST of visual evidence items showing how elements combine to create a toxic, prejudiced, or offensive message.\n"
        "2) benign: A LIST of visual evidence items showing how elements combine to create an innocent, literal, or harmless message.\n\n"
        "Constraints for EACH item in your lists:\n"
        "- metaphor: State only the literal, objective visual elements you are basing the argument on.\n"
        "- meaning: Explain the social, cultural, or contextual subtext. You MUST interpret the implicit meaning.\n"
        "- Keep each field concise and highly specific.\n"
        "- Output only valid JSON matching the schema.\n\n"
        f"SCHEMA: {schema_json}"
    )

    return (
        "<|im_start|>system\n"
        "You are an expert sociologist and multimodal meme analyst. You must return strict JSON."
        "<|im_end|>\n"
        "<|im_start|>user\n"
        "<|vision_start|><|image_pad|><|vision_end|>\n"
        f"{task_text}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def build_stage2_prompt(ocr_text: str, hateful: List[Dict[str, str]], benign: List[Dict[str, str]], schema_json: str) -> str:
    hateful_json = json.dumps(hateful, ensure_ascii=False)
    benign_json = json.dumps(benign, ensure_ascii=False)

    task_text = (
        f"### TARGET MEME TEXT:\n'{ocr_text}'\n\n"
        f"### HATEFUL RATIONALES:\n{hateful_json}\n\n"
        f"### BENIGN RATIONALES:\n{benign_json}\n\n"
        f"### POLICY:\n{POLICY_MANUAL}\n\n"
        "### TASK:\n"
        "Choose the better-supported side and output label as a WORD:\n"
        "- label = Hateful if hateful rationale is stronger\n"
        "- label = Benign if benign rationale is stronger\n"
        "Respond only with strict JSON matching the schema.\n"
        f"SCHEMA: {schema_json}"
    )

    return (
        "<|im_start|>system\n"
        "You are a strict multimodal policy judge. Use the image and rationale evidence."
        "<|im_end|>\n"
        "<|im_start|>user\n"
        "<|vision_start|><|image_pad|><|vision_end|>\n"
        f"{task_text}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def clean_evidence_items(raw_items: Any) -> List[Dict[str, str]]:
    cleaned: List[Dict[str, str]] = []
    if not isinstance(raw_items, list):
        return cleaned

    for item in raw_items:
        if not isinstance(item, dict):
            continue
        cleaned.append(
            {
                "metaphor": str(item.get("metaphor", "")).strip(),
                "meaning": str(item.get("meaning", "")).strip(),
            }
        )
    return cleaned


def parse_judge_output(text: str) -> Dict[str, Any]:
    working_text = text
    if "</think>" in working_text:
        working_text = working_text.split("</think>")[-1].strip()

    match = re.search(r"\{.*\}", working_text, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(0))
            raw_label = str(data.get("label", "")).strip().upper()
            reasoning = str(data.get("judge_reasoning", "Parsed from JSON")).strip()
            if raw_label in {"1", "TRUE", "A", "HATEFUL"}:
                return {"label": 1, "judge_choice": "A", "judge_reasoning": reasoning}
            if raw_label in {"0", "FALSE", "B", "BENIGN"}:
                return {"label": 0, "judge_choice": "B", "judge_reasoning": reasoning}
        except json.JSONDecodeError:
            pass

    lowered = working_text.lower()
    if "hateful" in lowered or '"label": 1' in lowered:
        return {"label": 1, "judge_choice": "A", "judge_reasoning": "Parsed by keyword fallback"}
    return {"label": 0, "judge_choice": "B", "judge_reasoning": "Parsed by keyword fallback"}


def unload_model(llm: LLM) -> None:
    del llm
    gc.collect()
    torch.cuda.empty_cache()
    print("VRAM partially cleared (model object deleted)")


def main() -> None:
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Input JSONL not found: {DATA_PATH}")

    print(f"Loading input data from {DATA_PATH}...")
    with open(DATA_PATH, "r", encoding="utf-8") as file_handle:
        samples = [json.loads(line) for line in file_handle if line.strip()]

    enable_judge_lora = USE_JUDGE_LORA and os.path.isdir(JUDGE_LORA_PATH)
    if USE_JUDGE_LORA and not enable_judge_lora:
        print(f"WARNING: USE_JUDGE_LORA=1 but adapter path not found: {JUDGE_LORA_PATH}. Falling back to base model.")

    llm = LLM(
        model=MODEL_ID,
        trust_remote_code=True,
        max_model_len=4096,
        gpu_memory_utilization=0.85,
        enable_lora=enable_judge_lora,
    )

    judge_lora_request: Optional[LoRARequest] = None
    if enable_judge_lora:
        judge_lora_request = LoRARequest("judge_adapter", 1, JUDGE_LORA_PATH)
        print(f"Judge LoRA enabled from: {JUDGE_LORA_PATH}")
    else:
        print("Judge LoRA disabled; using base model for decision stage.")

    stage1_schema = DualRationaleOutput.model_json_schema()
    stage1_schema_json = json.dumps(stage1_schema)
    stage2_schema = JudgeChoice.model_json_schema()
    stage2_schema_json = json.dumps(stage2_schema)

    valid_indices: List[int] = []
    stage1_prompts: List[Dict[str, Any]] = []

    for idx, sample in enumerate(tqdm(samples, desc="Preparing Stage 1 Inputs")):
        img_path = resolve_image_path(str(sample.get("img", "")))
        if not img_path:
            sample["hateful"] = [{"metaphor": "", "meaning": "IMAGE_NOT_FOUND"}]
            sample["benign"] = [{"metaphor": "", "meaning": "IMAGE_NOT_FOUND"}]
            sample["label"] = 0
            sample["judge_choice"] = "B"
            sample["judge_reasoning"] = "IMAGE_NOT_FOUND"
            continue

        sample["abs_img_path"] = img_path
        ocr_text = str(sample.get("text", sample.get("ocr_text", ""))).strip()

        stage1_prompts.append(
            {
                "prompt": build_stage1_prompt(ocr_text=ocr_text, schema_json=stage1_schema_json),
                "multi_modal_data": {"image": Image.open(img_path).convert("RGB")},
            }
        )
        valid_indices.append(idx)

    sampling_stage1 = SamplingParams(
        temperature=0.0,
        max_tokens=1024,
        structured_outputs=StructuredOutputsParams(json=stage1_schema),
    )

    print(f"Generating Stage 1 rationales for {len(stage1_prompts)} samples...")
    stage1_outputs = llm.generate(stage1_prompts, sampling_params=sampling_stage1)

    stage2_prompts: List[Dict[str, Any]] = []
    stage2_indices: List[int] = []

    for sample_idx, output in zip(valid_indices, stage1_outputs):
        sample = samples[sample_idx]
        try:
            parsed = json.loads(output.outputs[0].text)
            hateful = clean_evidence_items(parsed.get("hateful", []))
            benign = clean_evidence_items(parsed.get("benign", []))
            if not hateful:
                hateful = [{"metaphor": "", "meaning": "PARSING_ERROR"}]
            if not benign:
                benign = [{"metaphor": "", "meaning": "PARSING_ERROR"}]
        except (json.JSONDecodeError, IndexError, TypeError, AttributeError):
            hateful = [{"metaphor": "", "meaning": "PARSING_ERROR"}]
            benign = [{"metaphor": "", "meaning": "PARSING_ERROR"}]

        sample["hateful"] = hateful
        sample["benign"] = benign

        ocr_text = str(sample.get("text", sample.get("ocr_text", ""))).strip()
        stage2_prompts.append(
            {
                "prompt": build_stage2_prompt(
                    ocr_text=ocr_text,
                    hateful=hateful,
                    benign=benign,
                    schema_json=stage2_schema_json,
                ),
                "multi_modal_data": {"image": Image.open(sample["abs_img_path"]).convert("RGB")},
            }
        )
        stage2_indices.append(sample_idx)

    sampling_stage2 = SamplingParams(
        temperature=0.0,
        max_tokens=700,
    )

    print(f"Generating Stage 2 judge decisions for {len(stage2_prompts)} samples...")
    stage2_outputs = llm.generate(
        stage2_prompts,
        sampling_params=sampling_stage2,
        lora_request=judge_lora_request,
    )

    for sample_idx, output in zip(stage2_indices, stage2_outputs):
        parsed = parse_judge_output(output.outputs[0].text)
        samples[sample_idx]["label"] = parsed["label"]
        samples[sample_idx]["judge_choice"] = parsed["judge_choice"]
        samples[sample_idx]["judge_reasoning"] = parsed["judge_reasoning"]

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    print(f"Saving results to {OUT_PATH}...")
    with open(OUT_PATH, "w", encoding="utf-8") as out_file:
        for sample in samples:
            out_item = {
                "id": sample.get("id"),
                "img": sample.get("img"),
                "ocr_text": sample.get("text", sample.get("ocr_text", "")),
                "label": sample.get("label", 0),
                "hateful": sample.get("hateful", []),
                "benign": sample.get("benign", []),
                "judge_choice": sample.get("judge_choice", "B"),
                "judge_reasoning": sample.get("judge_reasoning", ""),
            }
            out_file.write(json.dumps(out_item, ensure_ascii=False) + "\n")

    unload_model(llm)
    print("Success! Stage1 rationale extraction + Stage2 LoRA judging complete.")


if __name__ == "__main__":
    main()
