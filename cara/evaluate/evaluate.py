import argparse
import json
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
	rows: list[dict[str, Any]] = []
	with path.open("r", encoding="utf-8") as f:
		for line in f:
			line = line.strip()
			if not line:
				continue
			rows.append(json.loads(line))
	return rows


def roc_auc_from_scores(y_true: list[int], y_score: list[float]) -> float:
	pairs = sorted(zip(y_score, y_true), key=lambda x: x[0])
	n = len(pairs)

	ranks = [0.0] * n
	i = 0
	while i < n:
		j = i
		while j + 1 < n and pairs[j + 1][0] == pairs[i][0]:
			j += 1

		avg_rank = (i + 1 + j + 1) / 2.0
		for k in range(i, j + 1):
			ranks[k] = avg_rank

		i = j + 1

	n_pos = sum(1 for _, y in pairs if y == 1)
	n_neg = n - n_pos
	if n_pos == 0 or n_neg == 0:
		return float("nan")

	rank_sum_pos = sum(rank for rank, (_, y) in zip(ranks, pairs) if y == 1)
	return (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def derive_labels(
	preds: list[dict[str, Any]],
	fb_records: list[dict[str, Any]],
) -> list[int]:
	if all("label" in r for r in preds):
		return [int(r["label"]) for r in preds]

	fb_label_by_id: dict[int, int] = {}
	for row in fb_records:
		if "id" in row and "label" in row:
			try:
				fb_label_by_id[int(row["id"])] = int(row["label"])
			except (TypeError, ValueError):
				pass

	labels: list[int] = []
	missing_ids = 0
	for r in preds:
		try:
			rid = int(r["id"])
			labels.append(fb_label_by_id[rid])
		except (KeyError, TypeError, ValueError):
			missing_ids += 1
			labels.append(-1)

	if missing_ids:
		print(f"Warning: {missing_ids} prediction rows had no matching FB label by id.")

	return labels


def main() -> None:
	parser = argparse.ArgumentParser(description="Evaluate RoBERTa predictions against Facebook labels.")
	parser.add_argument(
		"--predictions",
		default="datapreparation/output/final_roberta_predictions.jsonl",
		help="Path to prediction JSONL file.",
	)
	parser.add_argument(
		"--fb",
		default="facebook-data/train.jsonl",
		help="Path to Facebook JSONL split used as reference labels.",
	)
	parser.add_argument(
		"--pred-col",
		default="final_prediction",
		help="Column name for binary predictions.",
	)
	parser.add_argument(
		"--score-col",
		default="roberta_hateful_prob",
		help="Column name for probability scores (positive class).",
	)
	args = parser.parse_args()

	pred_path = Path(args.predictions)
	fb_path = Path(args.fb)

	if not pred_path.exists():
		raise FileNotFoundError(f"Prediction file not found: {pred_path}")
	if not fb_path.exists():
		raise FileNotFoundError(f"FB file not found: {fb_path}")

	preds = load_jsonl(pred_path)
	fb_records = load_jsonl(fb_path)

	labels = derive_labels(preds, fb_records)
	y_true: list[int] = []
	y_pred: list[int] = []
	y_score: list[float] = []

	for row, y in zip(preds, labels):
		if y not in (0, 1):
			continue

		if args.pred_col not in row or args.score_col not in row:
			continue

		try:
			p = int(row[args.pred_col])
			s = float(row[args.score_col])
		except (TypeError, ValueError):
			continue

		if p not in (0, 1):
			continue

		y_true.append(y)
		y_pred.append(p)
		y_score.append(s)

	n = len(y_true)
	if n == 0:
		raise ValueError("No valid rows found for evaluation.")

	tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 1)
	tn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 0)
	fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 1)
	fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 0)

	accuracy = (tp + tn) / n
	precision = tp / (tp + fp) if (tp + fp) else 0.0
	recall = tp / (tp + fn) if (tp + fn) else 0.0
	f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
	aucroc = roc_auc_from_scores(y_true, y_score)

	print(f"n={n}")
	print(f"accuracy={accuracy:.6f}")
	print(f"precision={precision:.6f}")
	print(f"recall={recall:.6f}")
	print(f"f1={f1:.6f}")
	print(f"aucroc={aucroc:.6f}")
	print(f"tp={tp} tn={tn} fp={fp} fn={fn}")


if __name__ == "__main__":
	main()
