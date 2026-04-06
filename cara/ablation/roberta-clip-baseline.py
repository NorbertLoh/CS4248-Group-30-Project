import json
import os
from typing import Any, Dict, List, Tuple

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoProcessor, AutoTokenizer, CLIPModel


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

DATA_PATH = os.environ.get(
	"BASELINE_DATA_PATH",
	os.path.join(BASE_DIR, "datapreparation", "output", "facebook-samples-test-roberta.jsonl"),
)
IMG_DIR = os.environ.get("BASELINE_IMG_DIR", os.path.join(BASE_DIR, "facebook-data", "img"))
OUT_PATH = os.environ.get(
	"BASELINE_OUT_PATH",
	os.path.join(BASE_DIR, "datapreparation", "output", "predictions_roberta_clip_baseline.jsonl"),
)

ROBERTA_MODEL_DIR = os.environ.get("ROBERTA_MODEL_DIR", os.path.join(BASE_DIR, "metameme_roberta_model"))
CLIP_MODEL_ID = os.environ.get("CLIP_MODEL_ID", "openai/clip-vit-base-patch32")

TEXT_WEIGHT = float(os.environ.get("TEXT_WEIGHT", "0.7"))
IMAGE_WEIGHT = float(os.environ.get("IMAGE_WEIGHT", "0.3"))
THRESHOLD = float(os.environ.get("THRESHOLD", "0.5"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "16"))
CLIP_MAX_TEXT_CHARS = int(os.environ.get("CLIP_MAX_TEXT_CHARS", "120"))


def load_jsonl(path: str) -> List[Dict[str, Any]]:
	with open(path, "r", encoding="utf-8") as file_handle:
		return [json.loads(line) for line in file_handle if line.strip()]


def resolve_image_path(img_field: str) -> str:
	img_value = str(img_field or "").strip()
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


def softmax_pair(logits: torch.Tensor) -> torch.Tensor:
	return torch.softmax(logits, dim=-1)


def compute_roberta_probs(
	texts: List[str],
	tokenizer: AutoTokenizer,
	model: AutoModelForSequenceClassification,
	device: torch.device,
	batch_size: int,
) -> List[float]:
	probs: List[float] = []

	for start in tqdm(range(0, len(texts), batch_size), desc="RoBERTa OCR inference"):
		batch = texts[start : start + batch_size]
		encoded = tokenizer(batch, padding=True, truncation=True, max_length=256, return_tensors="pt").to(device)

		with torch.no_grad():
			outputs = model(**encoded)
			batch_probs = softmax_pair(outputs.logits)[:, 1].detach().cpu().tolist()
			probs.extend(float(p) for p in batch_probs)

	return probs


def build_clip_prompts(ocr_text: str) -> Tuple[str, str]:
	ocr_short = " ".join(str(ocr_text).split())[:CLIP_MAX_TEXT_CHARS]
	hateful_prompt = (
		"hateful meme with targeted harassment, slurs, or dehumanization. "
		f"text: {ocr_short}"
	)
	benign_prompt = (
		"benign meme with neutral or non-targeted humor. "
		f"text: {ocr_short}"
	)
	return hateful_prompt, benign_prompt


def compute_clip_prob(
	image_path: str,
	ocr_text: str,
	clip_model: CLIPModel,
	clip_processor: AutoProcessor,
	device: torch.device,
) -> float:
	if not image_path:
		return 0.5

	hateful_prompt, benign_prompt = build_clip_prompts(ocr_text)
	text_candidates = [benign_prompt, hateful_prompt]

	with Image.open(image_path) as image_file:
		image = image_file.convert("RGB")

	inputs = clip_processor(
		text=text_candidates,
		images=image,
		return_tensors="pt",
		padding=True,
		truncation=True,
		max_length=77,
	).to(device)

	with torch.no_grad():
		outputs = clip_model(**inputs)
		logits_per_image = outputs.logits_per_image
		probs = torch.softmax(logits_per_image, dim=-1)[0]
		hateful_prob = float(probs[1].detach().cpu().item())

	return hateful_prob


def fuse_probs(text_prob: float, image_prob: float) -> float:
	total = TEXT_WEIGHT + IMAGE_WEIGHT
	if total <= 0:
		return text_prob
	tw = TEXT_WEIGHT / total
	iw = IMAGE_WEIGHT / total
	return tw * text_prob + iw * image_prob


def main() -> None:
	if not os.path.exists(DATA_PATH):
		raise FileNotFoundError(f"Input data file not found: {DATA_PATH}")

	print(f"Loading dataset: {DATA_PATH}")
	rows = load_jsonl(DATA_PATH)
	if not rows:
		raise RuntimeError("No rows loaded from input dataset.")

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	print(f"Loading RoBERTa model: {ROBERTA_MODEL_DIR}")
	roberta_tokenizer = AutoTokenizer.from_pretrained(ROBERTA_MODEL_DIR)
	roberta_model = AutoModelForSequenceClassification.from_pretrained(ROBERTA_MODEL_DIR).to(device).eval()

	print(f"Loading CLIP model: {CLIP_MODEL_ID}")
	clip_model = CLIPModel.from_pretrained(CLIP_MODEL_ID).to(device).eval()
	clip_processor = AutoProcessor.from_pretrained(CLIP_MODEL_ID)

	ocr_texts = [str(row.get("text", row.get("ocr_text", ""))).strip() for row in rows]
	roberta_probs = compute_roberta_probs(
		texts=ocr_texts,
		tokenizer=roberta_tokenizer,
		model=roberta_model,
		device=device,
		batch_size=BATCH_SIZE,
	)

	print("Running CLIP image+text inference")
	output_records: List[Dict[str, Any]] = []

	for row, text_prob, ocr_text in tqdm(zip(rows, roberta_probs, ocr_texts), total=len(rows), desc="Fusing outputs"):
		image_path = resolve_image_path(str(row.get("img", "")))
		clip_prob = compute_clip_prob(
			image_path=image_path,
			ocr_text=ocr_text,
			clip_model=clip_model,
			clip_processor=clip_processor,
			device=device,
		)

		fused_prob = fuse_probs(text_prob=float(text_prob), image_prob=float(clip_prob))
		pred_label = 1 if fused_prob >= THRESHOLD else 0

		output_records.append(
			{
				"id": row.get("id"),
				"img": row.get("img"),
				"ocr_text": ocr_text,
				"label": pred_label,
				"final_prediction": pred_label,
				"roberta_hateful_prob": float(text_prob),
				"clip_hateful_prob": float(clip_prob),
				"fused_hateful_prob": float(fused_prob),
				"weights": {"text": TEXT_WEIGHT, "image": IMAGE_WEIGHT},
				"threshold": THRESHOLD,
			}
		)

	os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
	print(f"Writing predictions: {OUT_PATH}")
	with open(OUT_PATH, "w", encoding="utf-8") as file_handle:
		for record in output_records:
			file_handle.write(json.dumps(record, ensure_ascii=False) + "\n")

	print("Done.")


if __name__ == "__main__":
	main()
