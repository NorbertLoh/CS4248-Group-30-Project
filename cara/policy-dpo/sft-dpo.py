"""Prepare and fine-tune a multimodal preference model for contrastive rationale alignment.

This script consumes the repo's JSONL outputs and converts them into a pairwise
preference format that keeps the meme image attached to each example.

Supported source rows:
- raw meme rows with ``img``/``image`` + ``text`` + ``rationale_a`` + ``rationale_b``
- judged rows with ``label`` or ``judge_choice`` selecting the preferred rationale
- ChatML-style rows with ``conversations`` and ``image``

The prepared dataset is saved as JSONL with these fields:
- ``conversations``: ChatML messages with a system prompt, user prompt, and chosen assistant response
- ``image``: resolved image path
- ``prompt``: flattened prompt string for DPO/ORPO
- ``chosen`` / ``rejected``: preference pair
- ``label``: binary label derived from the preferred rationale

The training entrypoint is best-effort across TRL versions. If your local TRL
build exposes ``DPOTrainer`` or ``ORPOTrainer`` for vision models, it will run.
Otherwise, the dataset export still gives you the exact format needed for a follow-up run.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from tqdm.auto import tqdm


BASE_DIR = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_PATH = Path(
	os.environ.get(
		"PREFERENCE_INPUT_PATH",
		str(BASE_DIR / "datapreparation" / "output" / "predictions_stage2_vllm.jsonl"),
	)
)
DEFAULT_IMAGE_ROOT = Path(
	os.environ.get(
		"PREFERENCE_IMAGE_ROOT",
		str(BASE_DIR / "facebook-data"),
	)
)
DEFAULT_OUTPUT_PATH = Path(
	os.environ.get(
		"PREFERENCE_OUTPUT_PATH",
		str(SCRIPT_DIR / "prepared_dpo.jsonl"),
	)
)
DEFAULT_MODEL_ID = os.environ.get(
	"UNSLOTH_VISION_MODEL_ID",
	"unsloth/Qwen3-VL-8B-Instruct-unsloth-bnb-4bit",
)
DEFAULT_TRAINING_MODE = os.environ.get("PREFERENCE_TRAINING_MODE", "dpo").strip().lower()
DEFAULT_RESPONSE_MODE = os.environ.get("PREFERENCE_RESPONSE_MODE", "rationale").strip().lower()

POLICY_MANUAL = (
	"### HATEFUL (1) CRITERIA:\n"
	"- A meme is Hateful if it relies on Harmful Stereotypes to function (e.g., linking a protected group "
	"to violence, terrorism, criminality, greed, or intellectual inferiority).\n"
	"- Dehumanization includes treating a protected group as a collective threat or a punchline for their identity.\n"
	"- IMPORTANT: If the joke requires the audience to believe a negative trait about a race, religion, "
	"or nationality for it to be funny, it is Hateful (1), even if it claims to be dark humor.\n\n"
	"### BENIGN (0) CRITERIA:\n"
	"- A meme is Benign if it mocks situations, absurd behaviors, or non-protected traits (e.g., pizza toppings, "
	"corporate jargon, or self-deprecating cringe humor).\n"
	"- Mocking an individual for their specific actions is Benign, UNLESS the mockery is tied to their protected identity."
)

SYSTEM_PROMPT = (
	"You are a strict policy judge for hateful meme detection. "
	"Read the image and OCR text, compare the two candidate rationales, and prefer the one "
	"that best matches the policy manual.\n\n"
	"### POLICY MANUAL:\n"
	"Use the following criteria exactly when judging the rationales:\n\n"
	 f"{POLICY_MANUAL}"
)


def _normalize_whitespace(value: Any) -> str:
	text = "" if value is None else str(value)
	text = re.sub(r"\s+", " ", text).strip()
	return text


def _normalize_choice(value: Any) -> Optional[int]:
	text = _normalize_whitespace(value).upper()
	if text in {"1", "A", "RATIONALE A", "PROSECUTION", "HATEFUL", "TRUE"}:
		return 1
	if text in {"0", "B", "RATIONALE B", "DEFENSE", "BENIGN", "FALSE"}:
		return 0
	return None


def _first_non_empty(*values: Any) -> str:
	for value in values:
		text = _normalize_whitespace(value)
		if text:
			return text
	return ""


def _resolve_image_path(raw_image: Any, image_root: Path) -> Optional[Path]:
	if raw_image is None:
		return None

	image_text = _normalize_whitespace(raw_image)
	if not image_text:
		return None

	candidate = Path(image_text)
	if candidate.is_file():
		return candidate

	if not candidate.is_absolute():
		rooted_candidate = image_root / candidate
		if rooted_candidate.is_file():
			return rooted_candidate

		if candidate.parts and candidate.parts[0] == image_root.name:
			nested_candidate = BASE_DIR / candidate
			if nested_candidate.is_file():
				return nested_candidate

	return None


def _extract_source_fields(record: Dict[str, Any]) -> Dict[str, str]:
	if isinstance(record.get("conversations"), list):
		conversations = record.get("conversations", [])
		user_text = ""
		assistant_text = ""
		system_text = ""
		for message in conversations:
			role = _normalize_whitespace(message.get("from") or message.get("role")).lower()
			value = _normalize_whitespace(message.get("value") or message.get("content"))
			if role == "system":
				system_text = value
			elif role == "user":
				user_text = value
			elif role == "assistant":
				assistant_text = value

		return {
			"system": system_text or SYSTEM_PROMPT,
			"ocr_text": _first_non_empty(record.get("ocr_text"), record.get("text"), user_text),
			"rationale_a": _first_non_empty(record.get("rationale_a"), record.get("chosen_rationale_a")),
			"rationale_b": _first_non_empty(record.get("rationale_b"), record.get("rejected_rationale_b")),
			"assistant": assistant_text,
		}

	return {
		"system": _first_non_empty(record.get("system"), SYSTEM_PROMPT),
		"ocr_text": _first_non_empty(record.get("ocr_text"), record.get("text")),
		"rationale_a": _first_non_empty(
			record.get("rationale_a"),
			record.get("prosecution"),
			record.get("hateful_rationale"),
			record.get("argument_a"),
			record.get("caption_a"),
		),
		"rationale_b": _first_non_empty(
			record.get("rationale_b"),
			record.get("defense"),
			record.get("benign_rationale"),
			record.get("argument_b"),
			record.get("caption_b"),
		),
		"assistant": _first_non_empty(record.get("assistant"), record.get("judge_reasoning")),
	}


def _infer_label(record: Dict[str, Any]) -> int:
	choice = _normalize_choice(record.get("label"))
	if choice is not None:
		return choice

	choice = _normalize_choice(record.get("judge_choice"))
	if choice is not None:
		return choice

	choice = _normalize_choice(record.get("chosen_rationale"))
	if choice is not None:
		return choice

	if _normalize_whitespace(record.get("chosen_rationale")).upper() == "A":
		return 1
	if _normalize_whitespace(record.get("chosen_rationale")).upper() == "B":
		return 0

	return 0


def _preferred_and_rejected_rationales(record: Dict[str, Any]) -> Tuple[str, str, int]:
	label = _infer_label(record)
	rationale_a = _first_non_empty(record.get("rationale_a"))
	rationale_b = _first_non_empty(record.get("rationale_b"))

	if label == 1:
		preferred = rationale_a
		rejected = rationale_b
	else:
		preferred = rationale_b
		rejected = rationale_a

	return preferred, rejected, label


def _render_prompt(record: Dict[str, Any]) -> str:
	fields = _extract_source_fields(record)
	ocr_text = fields["ocr_text"]
	rationale_a = fields["rationale_a"]
	rationale_b = fields["rationale_b"]

	return (
		"<image>\n"
		"### TASK: Contrastive Rationale Alignment\n\n"
		f"### OCR TEXT:\n{ocr_text}\n\n"
		f"### RATIONALE A (Prosecution):\n{rationale_a}\n\n"
		f"### RATIONALE B (Defense):\n{rationale_b}\n\n"
		"Choose the rationale that best matches the policy manual and explain why."
	)


def _render_judge_response(record: Dict[str, Any], chosen_label: int) -> str:
	fields = _extract_source_fields(record)
	rationale_a = fields["rationale_a"]
	rationale_b = fields["rationale_b"]
	preferred = rationale_a if chosen_label == 1 else rationale_b
	explanation = fields["assistant"]

	if explanation:
		explanation_text = explanation
	else:
		explanation_text = (
			"Rationale A identifies a protected target and a harmful attack."
			if chosen_label == 1
			else "Rationale B better fits the policy because it does not rely on a harmful stereotype."
		)

	return f"### REASONING: {explanation_text}\n### LABEL: {chosen_label}\n### PREFERRED_RATIONALE: {preferred}"


def _render_rationale_response(chosen_rationale: str, label: int) -> str:
	return f"### CHOSEN_RATIONALE: {chosen_rationale}\n### LABEL: {label}"


def _chatml_record(record: Dict[str, Any], image_path: Optional[Path], response_mode: str) -> Dict[str, Any]:
	preferred, rejected, label = _preferred_and_rejected_rationales(record)
	prompt = _render_prompt(record)
	fields = _extract_source_fields(record)

	if response_mode == "judge":
		chosen_text = _render_judge_response(record, label)
		rejected_text = _render_judge_response(record, 1 - label)
	else:
		chosen_text = _render_rationale_response(preferred, label)
		rejected_text = _render_rationale_response(rejected, 1 - label)

	conversations = [
		{"from": "system", "value": fields["system"]},
		{"from": "user", "value": prompt},
		{"from": "assistant", "value": chosen_text},
	]

	output: Dict[str, Any] = {
		"conversations": conversations,
		"prompt": prompt,
		"chosen": chosen_text,
		"rejected": rejected_text,
		"label": label,
		"rationale_a": fields["rationale_a"],
		"rationale_b": fields["rationale_b"],
	}

	if image_path is not None:
		image_value = str(image_path)
		output["image"] = image_value
		output["images"] = image_value

	if record.get("id") is not None:
		output["id"] = record.get("id")

	return output


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
	with path.open("r", encoding="utf-8") as handle:
		return [json.loads(line) for line in handle if line.strip()]


def _prepare_rows(
	records: Sequence[Dict[str, Any]],
	image_root: Path,
	response_mode: str,
) -> List[Dict[str, Any]]:
	rows: List[Dict[str, Any]] = []
	for record in tqdm(records, desc="Preparing preference rows"):
		raw_image = record.get("image", record.get("img"))
		if not raw_image:
			post_id = record.get("id", record.get("post_id"))
			if post_id:
				raw_image = f"img/{str(post_id).zfill(5)}.png"
		image_path = _resolve_image_path(raw_image, image_root)
		if not image_path:
			print(
				f"Warning: Could not resolve image for ID: {record.get('id')} with path {raw_image}"
			)
			continue
		row = _chatml_record(record, image_path, response_mode=response_mode)
		rows.append(row)
	return rows


def _save_jsonl(rows: Iterable[Dict[str, Any]], output_path: Path) -> None:
	output_path.parent.mkdir(parents=True, exist_ok=True)
	with output_path.open("w", encoding="utf-8") as handle:
		for row in rows:
			handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _load_preference_trainer(training_mode: str):
	if training_mode == "orpo":
		from trl import ORPOConfig, ORPOTrainer  # type: ignore

		return ORPOTrainer, ORPOConfig

	from trl import DPOConfig, DPOTrainer  # type: ignore

	return DPOTrainer, DPOConfig


def _load_model(model_id: str):
	try:
		from unsloth import FastVisionModel  # type: ignore
	except ImportError as exc:
		raise ImportError(
			"unsloth is required for training. Install the project requirements before running --train."
		) from exc

	model, tokenizer = FastVisionModel.from_pretrained(
		model_id,
		load_in_4bit=True,
		use_gradient_checkpointing="unsloth",
	)

	model = FastVisionModel.get_peft_model(
		model,
		finetune_vision_layers=True,
		finetune_language_layers=True,
		finetune_attention_modules=True,
		finetune_mlp_modules=True,
		r=16,
		lora_alpha=16,
		lora_dropout=0,
		bias="none",
		random_state=3407,
		use_rslora=False,
		loftq_config=None,
	)

	FastVisionModel.for_training(model)
	return model, tokenizer


def _build_training_dataset(rows: Sequence[Dict[str, Any]]):
	try:
		from datasets import Dataset as HFDataset, Features, Image as HFImage, Sequence as HFSequence, Value  # type: ignore
	except ImportError as exc:
		raise ImportError(
			"datasets is required for training. Install the project requirements before running --train."
		) from exc

	train_rows: List[Dict[str, Any]] = []
	for row in rows:
		image_path = _normalize_whitespace(row.get("images") or row.get("image"))
		if not image_path:
			raise ValueError("Found a training row without image. Every row must contain a valid image path.")

		train_row = {
			"prompt": row["prompt"],
			"chosen": row["chosen"],
			"rejected": row["rejected"],
			"image": image_path,
			"images": [image_path],
		}
		train_rows.append(train_row)

	if not train_rows:
		raise ValueError("No training rows with valid image paths were found.")

	feature_schema = Features(
		{
			"prompt": Value("string"),
			"chosen": Value("string"),
			"rejected": Value("string"),
			"image": HFImage(),
			"images": HFSequence(HFImage()),
		}
	)
	dataset = HFDataset.from_list(train_rows, features=feature_schema)
	return dataset


def _train(
	rows: Sequence[Dict[str, Any]],
	model_id: str,
	output_dir: Path,
	training_mode: str,
	learning_rate: float,
	max_steps: int,
) -> None:
	trainer_cls, config_cls = _load_preference_trainer(training_mode)
	model, tokenizer = _load_model(model_id)
	dataset = _build_training_dataset(rows)
	merged_output_dir = output_dir / "final_judge_model"
	gguf_output_dir = output_dir / "final_judge_model_gguf"

	config_kwargs: Dict[str, Any] = {
		"output_dir": str(output_dir),
		"per_device_train_batch_size": 1,
		"gradient_accumulation_steps": 4,
		"learning_rate": learning_rate,
		"warmup_steps": 5,
		"logging_steps": 1,
		"max_steps": max_steps,
		"report_to": "none",
		"remove_unused_columns": False,
	}

	if training_mode == "dpo":
		config_kwargs.update(
			{
				"beta": 0.1,
				"max_length": 2048,
				"max_prompt_length": 1536,
			}
		)
	else:
		config_kwargs.update(
			{
				"max_length": 2048,
				"max_prompt_length": 1536,
			}
		)

	args = config_cls(**config_kwargs)

	trainer_kwargs = {
		"model": model,
		"args": args,
		"train_dataset": dataset,
	}

	try:
		trainer_kwargs["processing_class"] = tokenizer
		trainer = trainer_cls(**trainer_kwargs)
	except TypeError:
		trainer_kwargs.pop("processing_class", None)
		trainer_kwargs["tokenizer"] = tokenizer
		trainer = trainer_cls(**trainer_kwargs)

	trainer.train()
	model.save_pretrained_merged(
		str(merged_output_dir),
		tokenizer,
		save_method="merged_16bit",
	)
	model.save_pretrained_gguf(
		str(gguf_output_dir),
		tokenizer,
		quantization_method="q4_k_m",
	)
	model.save_pretrained(str(output_dir / "adapter"))


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Prepare and train multimodal preference data.")
	parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH, help="Source JSONL file")
	parser.add_argument(
		"--image-root",
		type=Path,
		default=DEFAULT_IMAGE_ROOT,
		help="Root directory used to resolve relative image paths",
	)
	parser.add_argument(
		"--output",
		type=Path,
		default=DEFAULT_OUTPUT_PATH,
		help="Where to write the prepared JSONL dataset",
	)
	parser.add_argument(
		"--model-id",
		type=str,
		default=DEFAULT_MODEL_ID,
		help="Unsloth vision model to fine-tune",
	)
	parser.add_argument(
		"--training-mode",
		choices=("dpo", "orpo"),
		default=DEFAULT_TRAINING_MODE if DEFAULT_TRAINING_MODE in {"dpo", "orpo"} else "dpo",
		help="Preference optimization algorithm",
	)
	parser.add_argument(
		"--response-mode",
		choices=("rationale", "judge"),
		default=DEFAULT_RESPONSE_MODE if DEFAULT_RESPONSE_MODE in {"rationale", "judge"} else "rationale",
		help="Whether the chosen response should be the preferred rationale or the judge output",
	)
	parser.add_argument(
		"--train",
		action="store_true",
		help="Run the preference trainer after exporting the dataset",
	)
	parser.add_argument(
		"--learning-rate",
		type=float,
		default=float(os.environ.get("PREFERENCE_LR", "2e-5")),
		help="Learning rate for DPO/ORPO",
	)
	parser.add_argument(
		"--max-steps",
		type=int,
		default=int(os.environ.get("PREFERENCE_MAX_STEPS", "30")),
		help="Training steps when --train is enabled",
	)
	parser.add_argument(
		"--prepared-only",
		action="store_true",
		help="Only export the prepared JSONL and skip training",
	)
	return parser.parse_args()


def main() -> None:
	args = parse_args()

	if not args.input.exists():
		raise FileNotFoundError(f"Input JSONL not found: {args.input}")

	records = _load_jsonl(args.input)
	rows = _prepare_rows(records, args.image_root, args.response_mode)
	_save_jsonl(rows, args.output)
	print(f"Saved {len(rows)} prepared preference rows to: {args.output}")

	if args.train and not args.prepared_only:
		output_dir = args.output.parent / f"{args.training_mode}_outputs"
		_train(
			rows=rows,
			model_id=args.model_id,
			output_dir=output_dir,
			training_mode=args.training_mode,
			learning_rate=args.learning_rate,
			max_steps=args.max_steps,
		)
		print(f"Saved adapter weights to: {output_dir / 'adapter'}")


if __name__ == "__main__":
	main()
