import json
import os
from typing import Any, Dict, List

import torch
import torch.nn.functional as F
from PIL import Image
from pydantic import BaseModel, Field
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.sampling_params import StructuredOutputsParams


class MemeDecision(BaseModel):
	label: int = Field(description="Binary label: 1 for hateful, 0 for benign")
	rationale: str = Field(description="Short reason grounded in meme text, image, and retrieved context")


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

DATA_PATH = os.environ.get(
	"STAGE2_DATA_PATH",
	os.path.join(BASE_DIR, "datapreparation", "output", "facebook-samples-test-roberta.jsonl"),
)
IMG_DIR = os.environ.get("STAGE2_IMG_DIR", os.path.join(BASE_DIR, "facebook-data", "img"))
OUT_PATH = os.environ.get(
	"STAGE2_OUT_PATH",
	os.path.join(BASE_DIR, "datapreparation", "output", "predictions_simple_rag_qwen3vl8b.jsonl"),
)

MEMECAP_DATA = os.environ.get(
	"MEMECAP_DATA",
	os.path.join(BASE_DIR, "memecap-data", "memes-trainval.json"),
)

MODEL_ID = os.environ.get("RAG_VLM_MODEL_ID", "Qwen/Qwen3-VL-8B-Thinking")
EMBED_MODEL_ID = os.environ.get("RAG_TEXT_EMBED_MODEL_ID", "sentence-transformers/all-MiniLM-L6-v2")

RAG_TOP_K = int(os.environ.get("RAG_TOP_K", "5"))
RAG_THRESHOLD = float(os.environ.get("RAG_SCORE_THRESHOLD", "0.0"))
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "512"))


def load_json_or_jsonl(path: str) -> List[Dict[str, Any]]:
	if not os.path.exists(path):
		raise FileNotFoundError(f"File not found: {path}")

	with open(path, "r", encoding="utf-8") as file_handle:
		first_char = file_handle.read(1)
		file_handle.seek(0)
		if first_char == "[":
			data = json.load(file_handle)
			return data if isinstance(data, list) else []

		rows: List[Dict[str, Any]] = []
		for line in file_handle:
			line = line.strip()
			if not line:
				continue
			try:
				parsed = json.loads(line)
			except json.JSONDecodeError:
				continue
			if isinstance(parsed, dict):
				rows.append(parsed)
		return rows


def resolve_image_path(img_field: str) -> str:
	img_value = (img_field or "").strip()
	if not img_value:
		return ""

	candidates = [
		img_value,
		os.path.join(BASE_DIR, img_value),
		os.path.join(IMG_DIR, os.path.basename(img_value)),
	]
	for candidate in candidates:
		abs_candidate = os.path.abspath(candidate)
		if os.path.exists(abs_candidate):
			return abs_candidate
	return ""


def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
	mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
	summed = (last_hidden_state * mask).sum(dim=1)
	denom = mask.sum(dim=1).clamp(min=1e-9)
	return summed / denom


class MemeCapRetriever:
	def __init__(self, memecap_path: str, embed_model_id: str) -> None:
		self.device = "cuda" if torch.cuda.is_available() else "cpu"
		self.tokenizer = AutoTokenizer.from_pretrained(embed_model_id)
		self.encoder = AutoModel.from_pretrained(embed_model_id).to(self.device).eval()

		raw = load_json_or_jsonl(memecap_path)
		self.entries = self._build_entries(raw)
		self.texts = [entry["text"] for entry in self.entries]
		self.embeddings = self._embed_texts(self.texts) if self.texts else None

	@staticmethod
	def _build_entries(raw_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
		entries: List[Dict[str, Any]] = []

		for row in raw_rows:
			title = str(row.get("title", "")).strip()
			meme_captions = row.get("meme_captions", [])
			caption_text = " ; ".join([str(c).strip() for c in meme_captions if str(c).strip()])

			for metaphor_item in row.get("metaphors", []):
				if not isinstance(metaphor_item, dict):
					continue

				metaphor = str(metaphor_item.get("metaphor", "")).strip()
				meaning = str(metaphor_item.get("meaning", "")).strip()
				if not metaphor and not meaning:
					continue

				entry_text = (
					f"Title: {title}. "
					f"Metaphor: {metaphor}. "
					f"Meaning: {meaning}. "
					f"Meme context: {caption_text}."
				)

				entries.append(
					{
						"text": entry_text,
						"title": title,
						"metaphor": metaphor,
						"meaning": meaning,
						"context": caption_text,
					}
				)

		return entries

	def _embed_texts(self, texts: List[str], batch_size: int = 64, show_progress: bool = True) -> torch.Tensor:
		all_embeddings: List[torch.Tensor] = []

		iterator = range(0, len(texts), batch_size)
		if show_progress:
			iterator = tqdm(iterator, desc="Embedding MemeCap KB")

		for start in iterator:
			batch = texts[start : start + batch_size]
			encoded = self.tokenizer(
				batch,
				padding=True,
				truncation=True,
				max_length=256,
				return_tensors="pt",
			).to(self.device)

			with torch.no_grad():
				out = self.encoder(**encoded)
				pooled = mean_pool(out.last_hidden_state, encoded["attention_mask"])
				normalized = F.normalize(pooled, p=2, dim=1).cpu()
				all_embeddings.append(normalized)

		return torch.cat(all_embeddings, dim=0)

	def retrieve(self, query_text: str, k: int, threshold: float) -> List[Dict[str, Any]]:
		if not query_text.strip() or self.embeddings is None or not self.entries:
			return []

		q_emb = self._embed_texts([query_text], batch_size=1, show_progress=False)[0]
		scores = torch.matmul(self.embeddings, q_emb)

		k = min(k, scores.shape[0])
		top_k = torch.topk(scores, k=k)

		results: List[Dict[str, Any]] = []
		for score, idx in zip(top_k.values.tolist(), top_k.indices.tolist()):
			if score < threshold:
				continue
			item = dict(self.entries[idx])
			item["score"] = float(score)
			results.append(item)

		if results:
			return results

		fallback_results: List[Dict[str, Any]] = []
		for score, idx in zip(top_k.values.tolist(), top_k.indices.tolist()):
			item = dict(self.entries[idx])
			item["score"] = float(score)
			fallback_results.append(item)
		return fallback_results


def build_context_block(retrieved: List[Dict[str, Any]]) -> str:
	if not retrieved:
		return "No retrieved metaphor-meaning context available."

	lines: List[str] = []
	for i, item in enumerate(retrieved, start=1):
		lines.append(
			f"[{i}] score={item['score']:.4f}; metaphor={item['metaphor']}; meaning={item['meaning']}; title={item['title']}"
		)
	return "\n".join(lines)


def build_prompt(ocr_text: str, rag_context: str, schema_json: str) -> str:
	instruction = (
		"You are a strict hateful meme classifier.\n"
		"Use both the image and OCR text.\n"
		"The retrieved MemeCap metaphor/meaning context is reference evidence, not ground truth.\n"
		"Classify as 1 only when there is clear hateful targeting of a protected group, slur, dehumanization, or incitement.\n"
		"Otherwise classify as 0.\n"
		"Return only strict JSON matching the schema."
	)

	user_payload = (
		f"OCR text: {ocr_text}\n\n"
		f"Retrieved MemeCap metaphor/meaning context:\n{rag_context}\n\n"
		f"Schema: {schema_json}"
	)

	return (
		"<|im_start|>system\n"
		f"{instruction}<|im_end|>\n"
		"<|im_start|>user\n"
		"<|vision_start|><|image_pad|><|vision_end|>\n"
		f"{user_payload}<|im_end|>\n"
		"<|im_start|>assistant\n"
	)


def parse_label(raw_label: Any) -> int:
	if isinstance(raw_label, bool):
		return int(raw_label)
	text = str(raw_label).strip().lower()
	if text in {"1", "true", "hateful", "yes", "a"}:
		return 1
	if text in {"0", "false", "benign", "no", "b"}:
		return 0
	return 0


def main() -> None:
	print(f"Loading samples from {DATA_PATH}")
	samples = load_json_or_jsonl(DATA_PATH)
	if not samples:
		raise RuntimeError(f"No input samples loaded from: {DATA_PATH}")

	print(f"Building MemeCap retriever from {MEMECAP_DATA}")
	retriever = MemeCapRetriever(memecap_path=MEMECAP_DATA, embed_model_id=EMBED_MODEL_ID)
	print(f"Loaded {len(retriever.entries)} metaphor/meaning KB entries")

	print(f"Loading VLM: {MODEL_ID}")
	llm = LLM(
		model=MODEL_ID,
		trust_remote_code=True,
		max_model_len=4096,
		gpu_memory_utilization=0.85,
	)

	schema = MemeDecision.model_json_schema()
	schema_json = json.dumps(schema)

	valid_samples: List[Dict[str, Any]] = []
	prompts: List[Dict[str, Any]] = []

	for sample in tqdm(samples, desc="Preparing prompts"):
		img_path = resolve_image_path(str(sample.get("img", "")))
		if not img_path:
			sample["pred_label"] = 0
			sample["pred_rationale"] = "IMAGE_NOT_FOUND"
			sample["rag_context"] = ""
			sample["raw_output"] = ""
			continue

		ocr_text = str(sample.get("text", sample.get("ocr_text", ""))).strip()
		retrieved = retriever.retrieve(query_text=ocr_text, k=RAG_TOP_K, threshold=RAG_THRESHOLD)
		rag_context = build_context_block(retrieved)

		prompt = build_prompt(ocr_text=ocr_text, rag_context=rag_context, schema_json=schema_json)

		with Image.open(img_path) as image_file:
			image_rgb = image_file.convert("RGB")

		prompts.append(
			{
				"prompt": prompt,
				"multi_modal_data": {"image": image_rgb},
			}
		)
		valid_samples.append(
			{
				"sample": sample,
				"retrieved": retrieved,
				"rag_context": rag_context,
			}
		)

	sampling = SamplingParams(
		temperature=0.0,
		max_tokens=MAX_NEW_TOKENS,
		structured_outputs=StructuredOutputsParams(json=schema),
	)

	if not prompts:
		print("No valid image samples found; writing fallback output only.")
		outputs = []
	else:
		print(f"Running inference for {len(prompts)} samples")
		outputs = llm.generate(prompts, sampling_params=sampling)

	for meta, out in zip(valid_samples, outputs):
		sample = meta["sample"]
		generated = out.outputs[0].text

		parsed_label = 0
		parsed_rationale = ""
		try:
			parsed = json.loads(generated)
			parsed_label = parse_label(parsed.get("label", 0))
			parsed_rationale = str(parsed.get("rationale", "")).strip()
		except json.JSONDecodeError:
			lowered = generated.lower()
			parsed_label = 1 if "hateful" in lowered else 0
			parsed_rationale = "parse_fallback"

		sample["pred_label"] = parsed_label
		sample["pred_rationale"] = parsed_rationale
		sample["rag_context"] = meta["rag_context"]
		sample["rag_top_k"] = [
			{
				"score": item["score"],
				"metaphor": item["metaphor"],
				"meaning": item["meaning"],
				"title": item["title"],
			}
			for item in meta["retrieved"]
		]
		sample["raw_output"] = generated

	os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
	print(f"Writing output to {OUT_PATH}")
	with open(OUT_PATH, "w", encoding="utf-8") as file_handle:
		for sample in samples:
			label_value = sample.get("pred_label", sample.get("label", 0))
			record = {
				"id": sample.get("id"),
				"img": sample.get("img"),
				"ocr_text": sample.get("text", sample.get("ocr_text", "")),
				"label": int(label_value),
				"reason": sample.get("pred_rationale", ""),
				"rag_context": sample.get("rag_context", ""),
				"rag_top_k": sample.get("rag_top_k", []),
				"raw_output": sample.get("raw_output", ""),
			}
			file_handle.write(json.dumps(record, ensure_ascii=False) + "\n")

	print("Done.")


if __name__ == "__main__":
	main()
