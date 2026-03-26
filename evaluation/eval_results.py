import json
import glob
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tabulate import tabulate

# Load facebook-samples.jsonl (ground truth)
with open('datapreparation/output/facebook-samples.jsonl', 'r', encoding='utf-8') as f:
    gt = [json.loads(line) for line in f]
    gt_dict = {str(item['id']): item['label'] for item in gt}

# Find all prediction files
result_files = glob.glob('datapreparation/output/results/preds_*.jsonl')

headers = ["Model", "Accuracy", "Precision", "Recall", "F1 Score"]
table = []

for pred_file in result_files:
    preds = []
    with open(pred_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                item = json.loads(line)
                preds.append(item)
            except Exception:
                continue
    true_labels = []
    pred_labels = []
    for item in preds:
        id_str = str(item['id'])
        if id_str in gt_dict:
            true_labels.append(gt_dict[id_str])
            pred_labels.append(item['label'])
    if not true_labels:
        continue
    acc = accuracy_score(true_labels, pred_labels)
    prec = precision_score(true_labels, pred_labels, zero_division=0)
    rec = recall_score(true_labels, pred_labels, zero_division=0)
    f1 = f1_score(true_labels, pred_labels, zero_division=0)
    model_name = pred_file.split('preds_')[1].rsplit('.jsonl', 1)[0]
    table.append([model_name, f"{acc:.4f}", f"{prec:.4f}", f"{rec:.4f}", f"{f1:.4f}"])

print(tabulate(table, headers=headers, tablefmt="github"))
