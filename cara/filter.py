import argparse
import json
from pathlib import Path


def load_ground_truth_labels(dataset_paths):
	"""Load id -> label mapping from Facebook dataset JSONL files."""
	ground_truth = {}
	loaded = 0

	for dataset_path in dataset_paths:
		with dataset_path.open("r", encoding="utf-8") as src:
			for line in src:
				line = line.strip()
				if not line:
					continue

				row = json.loads(line)
				if "label" not in row:
					continue

				try:
					row_id = int(row["id"])
					label = int(row["label"])
				except (KeyError, TypeError, ValueError):
					continue

				ground_truth[row_id] = label
				loaded += 1

	return ground_truth, loaded


def is_mislabeled(row, ground_truth):
	try:
		row_id = int(row.get("id"))
		predicted_label = int(row.get("label"))
	except (TypeError, ValueError):
		return False

	true_label = ground_truth.get(row_id)
	if true_label is None:
		return False

	return predicted_label != true_label


def filter_mislabeled_rows(input_path, output_path, ground_truth):
	kept = 0
	total = 0
	matched_ids = 0

	with input_path.open("r", encoding="utf-8") as src, output_path.open("w", encoding="utf-8") as dst:
		for line in src:
			line = line.strip()
			if not line:
				continue

			total += 1
			row = json.loads(line)
			try:
				row_id = int(row.get("id"))
			except (TypeError, ValueError):
				row_id = None

			if row_id in ground_truth:
				matched_ids += 1

			if is_mislabeled(row, ground_truth):
				dst.write(json.dumps(row, ensure_ascii=False) + "\n")
				kept += 1

	return total, matched_ids, kept


def main():
	parser = argparse.ArgumentParser(description="Filter mislabeled rows from a predictions JSONL file.")
	parser.add_argument(
		"--input",
		default="datapreparation/output/results/preds_2_predictions_stage2_vllm.jsonl",
		help="Path to input JSONL file",
	)
	parser.add_argument(
		"--dataset",
		nargs="+",
		default=["facebook-data/train.jsonl", "facebook-data/dev.jsonl"],
		help="Facebook dataset JSONL files with ground-truth labels",
	)
	parser.add_argument(
		"--output",
		default=str(Path(__file__).with_name("preds_1_predictions_stage2_vllm_mislabeled.jsonl")),
		help="Path to output JSONL file",
	)
	args = parser.parse_args()

	input_path = Path(args.input)
	dataset_paths = [Path(p) for p in args.dataset]
	output_path = Path(args.output)
	ground_truth, loaded = load_ground_truth_labels(dataset_paths)

	total, matched_ids, kept = filter_mislabeled_rows(input_path, output_path, ground_truth)
	print(f"Loaded {loaded} labeled dataset rows from {len(dataset_paths)} file(s)")
	print(f"Processed {total} rows")
	print(f"Matched {matched_ids} rows by id against dataset")
	print(f"Wrote {kept} mislabeled rows to: {output_path}")


if __name__ == "__main__":
	main()
