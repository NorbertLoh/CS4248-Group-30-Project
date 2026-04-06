import json
import os
import re
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F
from PIL import Image
from pydantic import BaseModel, Field, ValidationError, field_validator
from transformers import (
    AutoModel,
    AutoProcessor,
    AutoTokenizer,
    Qwen3VLForConditionalGeneration,
)


class Stage2Response(BaseModel):
    rationale_a: str = Field(
        description="Prosecution rationale: why meme is hateful via OCR text + RAG metaphors"
    )
    rationale_b: str = Field(
        description="Defense rationale: why meme may be benign or neutral"
    )

    @field_validator("rationale_a", "rationale_b", mode="before")
    @classmethod
    def normalize_rationale(cls, value: Any) -> str:
        if value is None:
            return ""
        return str(value).strip()


class JudgeResponse(BaseModel):
    chosen_rationale: str = Field(description="Either 'A' or 'B'")
    judge_reasoning: str = Field(description="Why the chosen rationale best matches policy")

    @field_validator("chosen_rationale", mode="before")
    @classmethod
    def normalize_choice(cls, value: Any) -> str:
        text = str(value or "").strip().upper()
        if text in {"A", "RATIONALE A", "PROSECUTION", "1"}:
            return "A"
        if text in {"B", "RATIONALE B", "DEFENSE", "2"}:
            return "B"
        raise ValueError("chosen_rationale must be A or B")

    @field_validator("judge_reasoning", mode="before")
    @classmethod
    def normalize_reasoning(cls, value: Any) -> str:
        return str(value or "").strip()

def _to_stage2_payload(obj: Dict[str, Any]) -> Dict[str, Any]:
    rationale_a = (
        obj.get("rationale_a")
        or obj.get("prosecution")
        or obj.get("argument_a")
        or obj.get("caption_a")
        or obj.get("hateful_rationale")
        or obj.get("reasoning")
        or ""
    )
    rationale_b = (
        obj.get("rationale_b")
        or obj.get("defense")
        or obj.get("argument_b")
        or obj.get("caption_b")
        or obj.get("benign_rationale")
        or ""
    )
    return {
        "rationale_a": rationale_a,
        "rationale_b": rationale_b,
    }


def _extract_json_candidates(text: str) -> List[str]:
    candidates: List[str] = []

    # Parse fenced content first since models often wrap valid JSON there.
    fenced = re.findall(r"```(?:json)?\s*([\s\S]*?)```", text, flags=re.IGNORECASE)
    candidates.extend([chunk.strip() for chunk in fenced if chunk.strip()])

    # Non-greedy extraction of potential object snippets.
    inline_objects = re.findall(r"\{[\s\S]*?\}", text)
    candidates.extend([obj.strip() for obj in inline_objects if obj.strip()])

    # Preserve order while removing duplicates.
    deduped: List[str] = []
    seen = set()
    for candidate in candidates:
        normalized = candidate.rstrip(";").strip()
        if normalized and normalized not in seen:
            seen.add(normalized)
            deduped.append(normalized)
    return deduped


def _parse_candidate(candidate: str) -> Optional[Dict[str, Any]]:
    try:
        parsed = Stage2Response.model_validate_json(candidate)
        return parsed.model_dump()
    except ValidationError:
        try:
            raw_obj = json.loads(candidate)
            if not isinstance(raw_obj, dict):
                return None
            parsed = Stage2Response.model_validate(_to_stage2_payload(raw_obj))
            return parsed.model_dump()
        except Exception:
            return None
    except Exception:
        return None


def _to_judge_payload(obj: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "chosen_rationale": (
            obj.get("chosen_rationale")
            or obj.get("winner")
            or obj.get("choice")
            or obj.get("selected")
            or obj.get("selected_rationale")
            or ""
        ),
        "judge_reasoning": (
            obj.get("judge_reasoning")
            or obj.get("reasoning")
            or obj.get("justification")
            or obj.get("explanation")
            or ""
        ),
    }


def parse_judge_output(text: str) -> Dict[str, Any]:
    text = re.sub(r"[\x00-\x1F\x7F]", " ", text).strip()

    parse_candidates = [text] + _extract_json_candidates(text)
    for candidate in parse_candidates:
        try:
            parsed = JudgeResponse.model_validate_json(candidate)
            return parsed.model_dump()
        except ValidationError:
            try:
                raw_obj = json.loads(candidate)
                if not isinstance(raw_obj, dict):
                    continue
                parsed = JudgeResponse.model_validate(_to_judge_payload(raw_obj))
                return parsed.model_dump()
            except Exception:
                continue
        except Exception:
            continue

    choice_match = re.search(
        r"(?:chosen[_\s-]*rationale|winner|choice|selected)\s*[:=]\s*([AB])",
        text,
        re.IGNORECASE,
    )
    choice = choice_match.group(1).upper() if choice_match else "B"
    return {
        "chosen_rationale": choice,
        "judge_reasoning": "REGEX_FALLBACK: " + text[:1200],
    }


def _regex_fallback_parse(text: str) -> Dict[str, Any]:
    b_match = re.search(r"Rationale\s*B\s*[:\-]\s*([\s\S]+)$", text, re.IGNORECASE)

    a_label_match = re.search(r"Rationale\s*A\s*[:\-]\s*", text, re.IGNORECASE)
    if a_label_match:
        a_start = a_label_match.end()
        b_start = b_match.start() if b_match else len(text)
        rationale_a = text[a_start:b_start].strip()
    else:
        rationale_a = "REGEX_FALLBACK: " + text[:1200]

    rationale_b = b_match.group(1).strip() if b_match else ""
    return {"rationale_a": rationale_a, "rationale_b": rationale_b}


def robust_parse(text: str) -> Dict[str, Any]:
    """Parse model output with Pydantic-first validation and safe fallback."""
    text = re.sub(r"[\x00-\x1F\x7F]", " ", text).strip()

    parse_candidates = [text] + _extract_json_candidates(text)
    for candidate in parse_candidates:
        parsed = _parse_candidate(candidate)
        if parsed is not None:
            return parsed

    return _regex_fallback_parse(text)


BASE_DIR = os.path.abspath(os.path.dirname(__file__) + "/..")
MEMECAP_DATA = os.path.join(BASE_DIR, "memecap-data", "memes-test.json")
DATA_PATH = os.environ.get(
    "STAGE2_DATA_PATH",
    os.path.join(BASE_DIR, "datapreparation", "output", "facebook-samples-test.jsonl"),
)
IMG_DIR = os.path.join(BASE_DIR, "facebook-data", "img")
RESULTS_DIR = os.path.join(BASE_DIR, "api-inference", "results")
OUT_PATH = os.environ.get(
    "STAGE2_OUT_PATH",
    os.path.join(BASE_DIR, "datapreparation", "output", "predictions_stage2_judge_test.jsonl"),
)

MODEL_ID = os.environ.get(
    "UNSLOTH_QWEN3_VL_MODEL_ID",
    "unsloth/Qwen3-VL-8B-Instruct-bnb-4bit",
)
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "512"))
RAG_K = int(os.environ.get("RAG_K", "3"))
HF_LOCAL_ONLY = os.environ.get("HF_LOCAL_ONLY", "0").strip().lower() in {"1", "true", "yes"}
TEXT_EMBED_MODEL_ID = os.environ.get(
    "RAG_TEXT_EMBED_MODEL_ID",
    "sentence-transformers/all-MiniLM-L6-v2",
)
JUDGE_MODEL_ID = os.environ.get(
    "JUDGE_MODEL_ID",
    "unsloth/Qwen3-VL-2B-thinking-bnb-4bit",
)
JUDGE_MAX_NEW_TOKENS = int(os.environ.get("JUDGE_MAX_NEW_TOKENS", "384"))
POLICY_MANUAL = os.environ.get(
    "POLICY_MANUAL",
    "Hate requires both: (1) a protected target group and (2) an attack that dehumanizes, "
    "threatens, or calls for exclusion/violence. Offensive language alone is insufficient.",
)


class MetaphorMeaningRAG:
    """Text-only retriever that embeds only metaphor and meaning fields."""

    def __init__(self, memecap_json_path: str, model_id: str = TEXT_EMBED_MODEL_ID):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=HF_LOCAL_ONLY)
        self.encoder = AutoModel.from_pretrained(model_id, local_files_only=HF_LOCAL_ONLY).to(self.device).eval()
        self.entries = self._build_entries(memecap_json_path)

        if not self.entries:
            raise ValueError("No metaphor/meaning entries found in MemeCap data.")

        self.entry_texts = [entry["text"] for entry in self.entries]
        self.entry_embeddings = self._embed_texts(self.entry_texts)

    def _build_entries(self, memecap_json_path: str) -> List[Dict[str, str]]:
        with open(memecap_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        entries: List[Dict[str, str]] = []
        for item in data:
            for m in item.get("metaphors", []):
                metaphor = str(m.get("metaphor", "")).strip()
                meaning = str(m.get("meaning", "")).strip()
                if not metaphor and not meaning:
                    continue
                text = f"Metaphor: {metaphor}. Meaning: {meaning}."
                entries.append({"text": text, "metaphor": metaphor, "meaning": meaning})
        return entries

    def _mean_pool(self, model_output: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        token_embeddings = model_output
        expanded_mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        pooled = (token_embeddings * expanded_mask).sum(dim=1) / expanded_mask.sum(dim=1).clamp(min=1e-9)
        return F.normalize(pooled, p=2, dim=1)

    def _embed_texts(self, texts: List[str], batch_size: int = 64) -> torch.Tensor:
        all_embeddings: List[torch.Tensor] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors="pt",
            )
            encoded = {k: v.to(self.device) for k, v in encoded.items()}
            with torch.no_grad():
                outputs = self.encoder(**encoded)
                embeddings = self._mean_pool(outputs.last_hidden_state, encoded["attention_mask"])
                all_embeddings.append(embeddings.cpu())
        return torch.cat(all_embeddings, dim=0)

    def query(self, query_text: str, k: int = 3) -> str:
        cleaned = (query_text or "").strip()
        if not cleaned:
            cleaned = "neutral meme text"

        q_embedding = self._embed_texts([cleaned])[0]
        scores = torch.matmul(self.entry_embeddings, q_embedding)
        top_k = min(k, scores.shape[0])
        top_indices = torch.topk(scores, k=top_k).indices.tolist()

        contexts = [self.entry_texts[idx] for idx in top_indices]
        return "\n".join(contexts)


class UnslothQwen3VLRAG:
    """MemeCap RAG + Unsloth Qwen3-VL-8B 4-bit inference pipeline."""

    def __init__(self, model_id: str = MODEL_ID, memecap_json_path: str = MEMECAP_DATA):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.rag = MetaphorMeaningRAG(memecap_json_path)

        print(f"Loading model: {model_id}")
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is required for Qwen3-VL-8B 4-bit inference. "
                "Run on a GPU machine with bitsandbytes support."
            )

        self.processor = AutoProcessor.from_pretrained(
            model_id,
            trust_remote_code=True,
            local_files_only=HF_LOCAL_ONLY,
        )
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_id,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            local_files_only=HF_LOCAL_ONLY,
        )
        self.model.eval()

        print(f"Loading judge model: {JUDGE_MODEL_ID}")
        self.judge_processor = AutoProcessor.from_pretrained(
            JUDGE_MODEL_ID,
            trust_remote_code=True,
            local_files_only=HF_LOCAL_ONLY,
        )
        self.judge_model = Qwen3VLForConditionalGeneration.from_pretrained(
            JUDGE_MODEL_ID,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            local_files_only=HF_LOCAL_ONLY,
        )
        self.judge_model.eval()

    def _prepare_image(self, image_path: str) -> Image.Image:
        image_obj = Image.open(image_path).convert("RGB")
        image_obj.thumbnail((768, 768), Image.Resampling.LANCZOS)
        image = Image.new("RGB", (768, 768), (0, 0, 0))
        image.paste(image_obj, ((768 - image_obj.width) // 2, (768 - image_obj.height) // 2))
        return image

    def query(self, image_path: str, meme_text: str = "") -> Dict[str, Any]:
        memecap_context = self.rag.query(meme_text, k=RAG_K)
        image = self._prepare_image(image_path)

        messages = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "You are a specialist in semiotics and multimodal hate speech detection. "
                            "Your expertise lies in identifying harmful subversions where benign cultural metaphors "
                            "are hijacked to dehumanize protected groups. "
                            "You must generate two opposing rationales grounded in OCR text and retrieved metaphor "
                            "meanings. "
                            "Always respond in valid JSON."
                        ),
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "### TASK: Contrastive Metaphor Analysis\n\n"
                            "### 1. REFERENCE CONTEXT (MemeCap Benign Examples):\n"
                            f"{memecap_context}\n\n"
                            "### 2. TARGET MEME DATA:\n"
                            f"Text: '{meme_text}'\n\n"
                            "### 3. STAGE-2 DUAL CAPTION TASK:\n"
                            "Rationale A (The Prosecution): Argue why this meme is hateful, specifically linking OCR "
                            "text to dehumanizing metaphors retrieved from RAG.\n"
                            "Rationale B (The Defense): Argue why this meme is benign, for example neutral metaphor, "
                            "self-deprecating humor, or literal observation.\n\n"
                            "### 4. OUTPUT RULES:\n"
                            "- Return JSON only.\n"
                            "- Keep both rationales grounded in the retrieved context and target meme text.\n"
                            "- Do not include markdown or extra prose outside JSON.\n"
                            "Return exactly this schema:\n"
                            '{"rationale_a": "...", "rationale_b": "..."}'
                        ),
                    },
                    {"type": "image"},
                ],
            },
        ]

        prompt = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.processor(
            text=[prompt],
            images=[image],
            return_tensors="pt",
        )

        target_device = self.model.device if hasattr(self.model, "device") else "cuda"
        inputs = {k: (v.to(target_device) if hasattr(v, "to") else v) for k, v in inputs.items()}

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                repetition_penalty=1.1,
            )

        input_len = inputs["input_ids"].shape[1]
        generated_ids = output_ids[0, input_len:]
        output_text = self.processor.decode(generated_ids, skip_special_tokens=True)

        parsed = robust_parse(output_text)
        return {
            "rationale_a": parsed.get("rationale_a", ""),
            "rationale_b": parsed.get("rationale_b", ""),
            "ocr_text": meme_text,
            "retrieved_context": memecap_context,
            "raw_output": output_text,
        }

    def judge(self, ocr_text: str, rationale_a: str, rationale_b: str) -> Dict[str, Any]:
        judge_prompt = (
            "You are a strict policy judge. You do not see the image. "
            "You must pick which rationale better matches the policy manual.\n\n"
            "Inputs:\n"
            f"OCR Text: {ocr_text}\n\n"
            f"Rationale A (Prosecution): {rationale_a}\n\n"
            f"Rationale B (Defense): {rationale_b}\n\n"
            f"Policy Manual: {POLICY_MANUAL}\n\n"
            "Task:\n"
            "Select the rationale most consistent with policy.\n"
            "Return JSON only with this schema:\n"
            '{"chosen_rationale": "A", "judge_reasoning": "..."}'
        )

        # Wrap in chat template for Qwen3-VL (text-only, no image)
        messages = [{"role": "user", "content": [{"type": "text", "text": judge_prompt}]}]
        text = self.judge_processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        encoded = self.judge_processor(
            text=[text], return_tensors="pt"
        )
        target_device = self.judge_model.device if hasattr(self.judge_model, "device") else "cuda"
        encoded = {k: (v.to(target_device) if hasattr(v, "to") else v) for k, v in encoded.items()}

        with torch.no_grad():
            output_ids = self.judge_model.generate(
                **encoded,
                max_new_tokens=JUDGE_MAX_NEW_TOKENS,
                do_sample=False,
                repetition_penalty=1.1,
            )

        prompt_len = encoded["input_ids"].shape[1]
        generated_ids = output_ids[0, prompt_len:]
        judge_text = self.judge_processor.decode(generated_ids, skip_special_tokens=True)
        parsed = parse_judge_output(judge_text)
        return {
            "judge_choice": parsed.get("chosen_rationale", "B"),
            "judge_reasoning": parsed.get("judge_reasoning", ""),
            "judge_raw_output": judge_text,
        }


def run_batch() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Data file not found: {DATA_PATH}")

    bot = UnslothQwen3VLRAG()

    with open(DATA_PATH, "r", encoding="utf-8") as f:
        memes = [json.loads(line) for line in f]

    final_results = []
    for meme in memes:
        image_name = os.path.basename(meme["img"])
        img_path = os.path.join(IMG_DIR, image_name)

        if not os.path.exists(img_path):
            final_results.append(
                {
                    "id": meme.get("id"),
                    "ocr_text": meme.get("text", ""),
                    "rationale_a": "",
                    "rationale_b": "",
                    "judge_choice": "",
                    "judge_reasoning": "",
                    "error": f"Image not found: {img_path}",
                }
            )
            continue

        try:
            stage2 = bot.query(
                image_path=img_path,
                meme_text=meme.get("text", ""),
            )
            judge = bot.judge(
                ocr_text=stage2.get("ocr_text", meme.get("text", "")),
                rationale_a=stage2.get("rationale_a", ""),
                rationale_b=stage2.get("rationale_b", ""),
            )
            final_results.append(
                {
                    "id": meme.get("id"),
                    "ocr_text": stage2.get("ocr_text", meme.get("text", "")),
                    "rationale_a": stage2.get("rationale_a", ""),
                    "rationale_b": stage2.get("rationale_b", ""),
                    "judge_choice": judge.get("judge_choice", ""),
                    "judge_reasoning": judge.get("judge_reasoning", ""),
                    "retrieved_context": stage2.get("retrieved_context", ""),
                    "error": "",
                }
            )
        except Exception as exc:
            final_results.append(
                {
                    "id": meme.get("id"),
                    "ocr_text": meme.get("text", ""),
                    "rationale_a": "",
                    "rationale_b": "",
                    "judge_choice": "",
                    "judge_reasoning": "",
                    "error": str(exc),
                }
            )

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for item in final_results:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Saved {len(final_results)} predictions to: {OUT_PATH}")


if __name__ == "__main__":
    run_batch()
