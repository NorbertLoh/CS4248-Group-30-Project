from __future__ import annotations

import argparse
import math
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


TYPE_NAMES = [
    "Race/Ethnicity",
    "Religion",
    "Gender/Sexuality",
    "Disability",
    "Nationality",
    "Other",
]


def repo_root() -> Path:
	return Path(__file__).resolve().parents[2]


def default_type_labels_path() -> Path:
	return repo_root() / "cara" / "type" / "target_type_label1_qwen8b.jsonl"


def default_prediction_files() -> List[Path]:
	root = repo_root()
	return [
		root / "datapreparation" / "output" / "cara-results" / "preds_all0,25_final_roberta_predictions_dev.jsonl",
		root / "datapreparation" / "output" / "cara-results" / "preds_2_predictions_baseline_vllm_8b_Thinking.jsonl",
  		root / "datapreparation" / "output" / "cara-results" / "preds_2_predictions_baseline_vllm_8b_Instruct.jsonl",
  		root / "datapreparation" / "output" / "cara-results" / "preds_2_predictions_roberta_clip_baseline.jsonl",

	]


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
	rows: List[Dict[str, Any]] = []
	with path.open("r", encoding="utf-8") as f:
		for line in f:
			line = line.strip()
			if not line:
				continue
			rows.append(json.loads(line))
	return rows


def load_type_map(path: Path) -> Dict[str, str]:
	type_map: Dict[str, str] = {}
	for row in load_jsonl(path):
		if int(row.get("original_label", row.get("label", 0)) or 0) != 1:
			continue
		rid = str(row.get("id", ""))
		category = str(row.get("target_category", ""))
		if rid and category in TYPE_NAMES:
			type_map[rid] = category
	return type_map


def label_to_int(value: Any) -> int:
	if isinstance(value, bool):
		return int(value)
	if isinstance(value, (int, float)):
		return 1 if float(value) >= 0.5 else 0
	text = str(value or "").strip().lower()
	if text in {"1", "true", "hateful", "hate", "positive"}:
		return 1
	if text in {"0", "false", "benign", "normal", "negative"}:
		return 0
	return 0


def prediction_score(row: Dict[str, Any]) -> float:
	for key in ("roberta_hateful_prob", "hateful_prob", "prob", "score"):
		if key in row and row.get(key) is not None:
			try:
				return float(row[key])
			except (TypeError, ValueError):
				pass
	if "final_prediction" in row:
		return float(label_to_int(row.get("final_prediction")))
	if "label_text" in row:
		return float(label_to_int(row.get("label_text")))
	return float(label_to_int(row.get("label")))


def prediction_label(row: Dict[str, Any]) -> int:
	if "final_prediction" in row:
		return label_to_int(row.get("final_prediction"))
	if "label_text" in row:
		return label_to_int(row.get("label_text"))
	if "label" in row and any(k in row for k in ("roberta_hateful_prob", "hateful_prob", "prob", "score")):
		# Some files store the true label and a score. Threshold the score.
		return 1 if prediction_score(row) >= 0.5 else 0
	return label_to_int(row.get("label"))


def format_float(value: float) -> str:
	return "nan" if math.isnan(value) else f"{value:.4f}"


def evaluate_type(
	rows: Iterable[Dict[str, Any]],
	type_map: Dict[str, str],
	target_type: str,
) -> Tuple[int, float, float]:
	"""Evaluate one type using prediction-first labels.

	Type annotations exist only for hateful IDs. For rows mapped to a target type,
	this measures how often the model predicts hateful (using final_prediction
	first, then label fallback) and the average hateful score.
	"""
	predicted_hateful = 0
	scores: List[float] = []
	count = 0

	for row in rows:
		rid = str(row.get("id", ""))
		row_type = type_map.get(rid)
		if row_type != target_type:
			continue

		count += 1
		predicted_hateful += prediction_label(row)
		scores.append(prediction_score(row))

	if count == 0:
		return 0, float("nan"), float("nan")
	return count, predicted_hateful / count, sum(scores) / count


def main() -> None:
	parser = argparse.ArgumentParser(
		description="Print per-type accuracy and AUROC for meme prediction files."
	)
	parser.add_argument(
		"--type-labels",
		default=str(default_type_labels_path()),
		help="JSONL file with target_category labels for label=1 memes.",
	)
	parser.add_argument(
		"--pred-files",
		nargs="*",
		default=[str(path) for path in default_prediction_files()],
		help="Prediction JSONL files to evaluate.",
	)
	args = parser.parse_args()

	type_labels_path = Path(args.type_labels)
	pred_files = [Path(path) for path in args.pred_files]
	type_map = load_type_map(type_labels_path)

	print(f"Type labels: {type_labels_path}")
	print(f"Loaded {len(type_map)} type-labeled hateful memes")

	for pred_path in pred_files:
		rows = load_jsonl(pred_path)
		print()
		print(f"FILE: {pred_path}")
		print("type               n    detect@1  avg_score")
		print("-" * 42)
		for target_type in TYPE_NAMES:
			n, detect_rate, avg_score = evaluate_type(rows, type_map, target_type)
			print(f"{target_type:<18} {n:4d} {format_float(detect_rate):>10} {format_float(avg_score):>10}")


if __name__ == "__main__":
	main()
