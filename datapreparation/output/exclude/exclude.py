import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


LABEL_MAP = {
	"benign": 0,
	"non-hateful": 0,
	"non_hateful": 0,
	"normal": 0,
	"neutral": 0,
	"hateful": 1,
	"hate": 1,
	"offensive": 1,
}


def normalize_label(value: Any) -> Optional[int]:
	if value is None:
		return None

	if isinstance(value, bool):
		return int(value)

	if isinstance(value, (int, float)):
		return int(value)

	if isinstance(value, str):
		text = value.strip().lower()
		if text in LABEL_MAP:
			return LABEL_MAP[text]
		if text.isdigit():
			return int(text)

	return None


def parse_json_object_from_text(text: str) -> Optional[Dict[str, Any]]:
	# Some outputs include wrappers like <think> ... </think> before/around JSON.
	start = text.find("{")
	end = text.rfind("}")
	if start == -1 or end == -1 or end <= start:
		return None

	try:
		parsed = json.loads(text[start : end + 1])
	except json.JSONDecodeError:
		return None

	return parsed if isinstance(parsed, dict) else None


def load_ground_truth_lookup(repo_root: Path) -> Dict[int, int]:
	lookup: Dict[int, int] = {}
	# Compare CARA outputs against the Facebook dev split only.
	for split_name in ("dev.jsonl",):
		split_path = repo_root / "facebook-data" / split_name
		if not split_path.exists():
			continue

		with split_path.open("r", encoding="utf-8") as src:
			for raw_line in src:
				line = raw_line.strip()
				if not line:
					continue

				try:
					row = json.loads(line)
				except json.JSONDecodeError:
					continue

				row_id = row.get("id")
				if row_id is None:
					continue

				try:
					row_id_int = int(row_id)
				except (TypeError, ValueError):
					continue

				gt = normalize_label(row.get("label"))
				if gt not in (0, 1):
					gt = normalize_label(row.get("hateful"))

				if gt in (0, 1):
					lookup[row_id_int] = gt

	return lookup


def get_ground_truth(
	row: Dict[str, Any], gt_lookup: Optional[Dict[int, int]] = None
) -> Tuple[Optional[int], Optional[str]]:
	if gt_lookup:
		row_id = row.get("id")
		if row_id is not None:
			try:
				row_id_int = int(row_id)
			except (TypeError, ValueError):
				row_id_int = None

			if row_id_int is not None and row_id_int in gt_lookup:
				return gt_lookup[row_id_int], "lookup.id"

		# When lookup is provided, only use the lookup as ground truth.
		# This avoids treating model output fields as labels.
		return None, None

	# Prefer explicit GT-like fields before falling back to "label".
	for key in ("ground_truth", "gold_label", "true_label", "target", "hateful"):
		parsed = normalize_label(row.get(key))
		if parsed in (0, 1):
			return parsed, key

	structured = row.get("structured")
	if isinstance(structured, dict):
		parsed = normalize_label(structured.get("hateful"))
		if parsed in (0, 1):
			return parsed, "structured.hateful"

	if "label" in row:
		parsed = normalize_label(row.get("label"))
		if parsed in (0, 1):
			return parsed, "label"

	return None, None


def get_prediction(row: Dict[str, Any], gt_source: Optional[str]) -> Optional[int]:
	# Prefer final_prediction first, then fall back to label.
	for key in ("final_prediction", "label"):
		parsed = normalize_label(row.get(key))
		if parsed in (0, 1):
			return parsed

	# Other prediction aliases as a secondary fallback.
	for key in ("prediction", "pred", "pred_label", "output_label", "label_text"):
		parsed = normalize_label(row.get(key))
		if parsed in (0, 1):
			return parsed

	raw_output = row.get("raw_output")
	if isinstance(raw_output, str):
		parsed_raw = parse_json_object_from_text(raw_output)
		if isinstance(parsed_raw, dict):
			parsed = normalize_label(parsed_raw.get("label"))
			if parsed in (0, 1):
				return parsed

	if isinstance(raw_output, dict):
		parsed = normalize_label(raw_output.get("label"))
		if parsed in (0, 1):
			return parsed

	return None


def filter_incorrect(
	input_path: Path, output_path: Path, gt_lookup: Optional[Dict[int, int]] = None
) -> None:
	total = 0
	incorrect = 0
	skipped = 0
	incorrect_rows = []

	with input_path.open("r", encoding="utf-8") as src:
		for raw_line in src:
			line = raw_line.strip()
			if not line:
				continue

			total += 1

			try:
				row = json.loads(line)
			except json.JSONDecodeError:
				skipped += 1
				continue

			gt, gt_source = get_ground_truth(row, gt_lookup)
			pred = get_prediction(row, gt_source)

			if gt is None or pred is None:
				skipped += 1
				continue

			if gt != pred:
				incorrect += 1
				incorrect_rows.append(json.dumps(row, ensure_ascii=False))

	if incorrect_rows:
		output_path.parent.mkdir(parents=True, exist_ok=True)
		with output_path.open("w", encoding="utf-8") as dst:
			dst.write("\n".join(incorrect_rows) + "\n")
	else:
		if output_path.exists():
			output_path.unlink()

	print(f"Input: {input_path}")
	if incorrect_rows:
		print(f"Output: {output_path}")
	else:
		print("Output: no incorrect rows; no file written")
	print(f"Total rows: {total}")
	print(f"Incorrect rows written: {incorrect}")
	print(f"Skipped rows (missing/invalid labels): {skipped}")


def main() -> None:
	# Automatically process all JSONL files from datapreparation/output/cara-results.
	script_dir = Path(__file__).resolve().parent
	repo_root = script_dir.parents[2]
	results_dir = script_dir.parent / "cara-results"
	output_dir = script_dir.parent / "cara-results-exclude"
	gt_lookup = load_ground_truth_lookup(repo_root)
	print(f"Loaded {len(gt_lookup)} ground-truth labels from facebook-data/dev.jsonl")

	if not results_dir.exists():
		raise FileNotFoundError(f"Results folder not found: {results_dir}")

	result_files = sorted(
		p
		for p in results_dir.glob("*.jsonl")
		if p.is_file() and not p.stem.endswith("_incorrect_only")
	)

	if not result_files:
		print(f"No CARA result files found in {results_dir}")
		return

	print(f"Found {len(result_files)} CARA result file(s) to process")
	for file_path in result_files:
		output_path = output_dir / f"{file_path.stem}_incorrect_only.jsonl"
		filter_incorrect(input_path=file_path, output_path=output_path, gt_lookup=gt_lookup)


if __name__ == "__main__":
	main()
