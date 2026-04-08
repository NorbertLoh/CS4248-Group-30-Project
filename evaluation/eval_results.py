import json
import glob
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from tabulate import tabulate


def pick_pred_label(item):
    for key in ("final_prediction", "pred_label", "prediction", "label"):
        if key in item:
            try:
                value = int(item[key])
            except Exception:
                continue
            if value in (0, 1):
                return value
    return None


def pick_score(item, pred_label):
    for key in ("roberta_hateful_prob", "hateful_prob", "prob", "score", "confidence"):
        if key in item:
            try:
                return float(item[key])
            except Exception:
                pass
    return float(pred_label)

# Load facebook-samples.jsonl (ground truth)
with open('facebook-data/dev.jsonl', 'r', encoding='utf-8') as f:
    gt = [json.loads(line) for line in f]
    gt_dict = {str(item['id']): item['label'] for item in gt}

# Find all prediction files
result_files = glob.glob('datapreparation/output/cara-results/preds_*.jsonl')

headers = ["Model", "Accuracy", "Precision", "Recall", "F1 Score", "AUROC"]
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
    pred_scores = []
    for item in preds:
        if 'id' not in item:
            continue
        id_str = str(item['id'])
        if id_str in gt_dict:
            pred_label = pick_pred_label(item)
            if pred_label is None:
                continue
            true_labels.append(gt_dict[id_str])
            pred_labels.append(pred_label)
            pred_scores.append(pick_score(item, pred_label))
    if not true_labels:
        continue
    acc = accuracy_score(true_labels, pred_labels)
    prec = precision_score(true_labels, pred_labels, zero_division=0)
    rec = recall_score(true_labels, pred_labels, zero_division=0)
    f1 = f1_score(true_labels, pred_labels, zero_division=0)
    try:
        auroc = roc_auc_score(true_labels, pred_scores)
    except Exception:
        auroc = float('nan')
    model_name = pred_file.split('preds_')[1].rsplit('.jsonl', 1)[0]
    table.append([model_name, f"{acc:.4f}", f"{prec:.4f}", f"{rec:.4f}", f"{f1:.4f}", f"{auroc:.4f}"])

print(tabulate(table, headers=headers, tablefmt="github"))
