import argparse
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import (
	AutoImageProcessor,
	AutoModel,
	AutoTokenizer,
	CLIPVisionModel,
	get_linear_schedule_with_warmup,
)


def set_seed(seed: int) -> None:
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)


@dataclass
class Sample:
	post_id: str
	text: str
	img_path: str
	label: int


def _as_list(value) -> List[str]:
	if value is None:
		return []
	if isinstance(value, list):
		return [str(v).strip() for v in value if str(v).strip()]
	if isinstance(value, str):
		v = value.strip()
		return [v] if v else []
	return [str(value).strip()]


def _build_metaphor_text(value) -> str:
	if not isinstance(value, list):
		return ""

	parts: List[str] = []
	for item in value:
		if not isinstance(item, dict):
			continue
		metaphor = str(item.get("metaphor", "")).strip()
		meaning = str(item.get("meaning", "")).strip()
		if metaphor and meaning:
			parts.append(f"{metaphor} means {meaning}")
		elif metaphor:
			parts.append(metaphor)
		elif meaning:
			parts.append(meaning)

	return " ".join(parts)


def load_samples(data_file: str, image_root: str) -> List[Sample]:
	samples: List[Sample] = []
	image_root = str(Path(image_root))

	with open(data_file, "r", encoding="utf-8") as f:
		for line in f:
			line = line.strip()
			if not line:
				continue

			row = json.loads(line)
			label = int(row["label"])
			title = str(row.get("title", "")).strip()

			meme_caps = _as_list(row.get("meme_captions"))
			img_caps = _as_list(row.get("img_captions"))
			metaphor_text = _build_metaphor_text(row.get("metaphors", []))

			# Hybrid strategy: title + meme logic + explicit metaphor mappings.
			caption_text = " ".join(meme_caps) if meme_caps else " ".join(img_caps)
			sep = "</s>"
			if metaphor_text:
				text = (
					f"Title: {title} {sep} Logic: {caption_text} "
					f"{sep} Symbols: {metaphor_text}"
				).strip()
			else:
				text = f"Title: {title} {sep} Logic: {caption_text}".strip()

			img_rel = str(row.get("img_fname", "")).strip()
			if not img_rel:
				continue
			img_path = os.path.join(image_root, img_rel)

			if not os.path.exists(img_path):
				continue

			samples.append(
				Sample(
					post_id=str(row.get("post_id", "")),
					text=text,
					img_path=img_path,
					label=label,
				)
			)

	if not samples:
		raise ValueError(f"No valid samples loaded from {data_file}")
	return samples

class MemeDataset(Dataset):
	def __init__(self, samples: List[Sample]):
		self.samples = samples

	def __len__(self) -> int:
		return len(self.samples)

	def __getitem__(self, idx: int) -> Dict:
		s = self.samples[idx]
		image = Image.open(s.img_path).convert("RGB")
		return {
			"post_id": s.post_id,
			"text": s.text,
			"image": image,
			"label": s.label,
		}


class MultimodalRobertaClassifier(nn.Module):
	def __init__(
		self,
		text_model_name: str,
		image_model_name: str,
		num_labels: int = 2,
		dropout: float = 0.2,
		fusion_hidden_dim: int = 1024,
		image_proj_dim: int = 256,
		freeze_image_encoder: bool = True,
	):
		super().__init__()

		self.text_encoder = AutoModel.from_pretrained(text_model_name)
		text_dim = self.text_encoder.config.hidden_size

		self.image_encoder = CLIPVisionModel.from_pretrained(image_model_name)
		image_dim = self.image_encoder.config.hidden_size

		if freeze_image_encoder:
			for p in self.image_encoder.parameters():
				p.requires_grad = False

		self.image_proj = nn.Linear(image_dim, image_proj_dim)

		# Requested 2-layer MLP head: Dropout -> Linear -> ReLU -> Linear
		self.classifier = nn.Sequential(
			nn.Dropout(dropout),
			nn.Linear(text_dim + image_proj_dim, fusion_hidden_dim),
			nn.ReLU(),
			nn.Linear(fusion_hidden_dim, num_labels),
		)

	def forward(
		self,
		input_ids: torch.Tensor,
		attention_mask: torch.Tensor,
		pixel_values: torch.Tensor,
	) -> torch.Tensor:
		text_outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
		text_feat = text_outputs.last_hidden_state[:, 0]

		image_outputs = self.image_encoder(pixel_values=pixel_values)
		image_feat = image_outputs.pooler_output
		image_feat = self.image_proj(image_feat)

		fused = torch.cat([text_feat, image_feat], dim=-1)
		logits = self.classifier(fused)
		return logits


def make_collate_fn(tokenizer, image_processor, max_length: int):
	def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
		texts = [x["text"] for x in batch]
		images = [x["image"] for x in batch]
		labels = torch.tensor([x["label"] for x in batch], dtype=torch.long)

		tok = tokenizer(
			texts,
			padding=True,
			truncation=True,
			max_length=max_length,
			return_tensors="pt",
		)
		img = image_processor(images=images, return_tensors="pt")
		tok["pixel_values"] = img["pixel_values"]
		tok["labels"] = labels
		return tok

	return collate_fn


def compute_metrics(y_true: List[int], y_pred: List[int]) -> Dict[str, float]:
	return {
		"accuracy": float(accuracy_score(y_true, y_pred)),
		"precision": float(precision_score(y_true, y_pred, zero_division=0)),
		"recall": float(recall_score(y_true, y_pred, zero_division=0)),
		"f1": float(f1_score(y_true, y_pred, zero_division=0)),
	}


def evaluate(model, loader, device) -> Dict[str, float]:
	model.eval()
	y_true, y_pred = [], []
	total_loss = 0.0
	criterion = nn.CrossEntropyLoss()

	with torch.no_grad():
		for batch in loader:
			input_ids = batch["input_ids"].to(device)
			attention_mask = batch["attention_mask"].to(device)
			pixel_values = batch["pixel_values"].to(device)
			labels = batch["labels"].to(device)

			logits = model(
				input_ids=input_ids,
				attention_mask=attention_mask,
				pixel_values=pixel_values,
			)
			loss = criterion(logits, labels)
			total_loss += loss.item()

			preds = torch.argmax(logits, dim=-1)
			y_true.extend(labels.cpu().tolist())
			y_pred.extend(preds.cpu().tolist())

	metrics = compute_metrics(y_true, y_pred)
	metrics["loss"] = total_loss / max(1, len(loader))
	return metrics


def write_predictions(model, samples: List[Sample], tokenizer, image_processor, args, device) -> str:
	pred_ds = MemeDataset(samples)
	pred_loader = DataLoader(
		pred_ds,
		batch_size=args.batch_size,
		shuffle=False,
		num_workers=args.num_workers,
		collate_fn=make_collate_fn(tokenizer, image_processor, args.max_length),
		pin_memory=torch.cuda.is_available(),
	)

	model.eval()
	all_rows = []
	start_idx = 0
	with torch.no_grad():
		for batch in pred_loader:
			input_ids = batch["input_ids"].to(device)
			attention_mask = batch["attention_mask"].to(device)
			pixel_values = batch["pixel_values"].to(device)
			labels = batch["labels"].cpu().tolist()

			logits = model(
				input_ids=input_ids,
				attention_mask=attention_mask,
				pixel_values=pixel_values,
			)
			probs = torch.softmax(logits, dim=-1).cpu().numpy()
			preds = np.argmax(probs, axis=-1)

			batch_size = len(labels)
			batch_samples = samples[start_idx:start_idx + batch_size]
			for i, s in enumerate(batch_samples):
				all_rows.append(
					{
						"post_id": s.post_id,
						"img_fname": s.img_path,
						"gold_label": int(labels[i]),
						"pred_label": int(preds[i]),
						"prob_0": float(probs[i][0]),
						"prob_1": float(probs[i][1]),
					}
				)
			start_idx += batch_size

	pred_path = os.path.join(args.output_dir, args.predictions_file)
	with open(pred_path, "w", encoding="utf-8") as f:
		for row in all_rows:
			f.write(json.dumps(row, ensure_ascii=False) + "\n")

	return pred_path


def train(args):
	set_seed(args.seed)
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	samples = load_samples(args.data_file, args.image_root)
	labels = [s.label for s in samples]
	train_idx, val_idx = train_test_split(
		np.arange(len(samples)),
		test_size=args.val_size,
		random_state=args.seed,
		stratify=labels,
	)

	train_samples = [samples[i] for i in train_idx]
	val_samples = [samples[i] for i in val_idx]

	tokenizer = AutoTokenizer.from_pretrained(args.text_model)
	image_processor = AutoImageProcessor.from_pretrained(args.image_model)

	train_ds = MemeDataset(train_samples)
	val_ds = MemeDataset(val_samples)

	collate_fn = make_collate_fn(tokenizer, image_processor, args.max_length)

	train_loader = DataLoader(
		train_ds,
		batch_size=args.batch_size,
		shuffle=True,
		num_workers=args.num_workers,
		collate_fn=collate_fn,
		pin_memory=torch.cuda.is_available(),
	)
	val_loader = DataLoader(
		val_ds,
		batch_size=args.batch_size,
		shuffle=False,
		num_workers=args.num_workers,
		collate_fn=collate_fn,
		pin_memory=torch.cuda.is_available(),
	)

	model = MultimodalRobertaClassifier(
		text_model_name=args.text_model,
		image_model_name=args.image_model,
		num_labels=2,
		dropout=args.dropout,
		fusion_hidden_dim=args.hidden_dim,
		image_proj_dim=args.image_proj_dim,
		freeze_image_encoder=args.freeze_image_encoder,
	).to(device)

	optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
	total_steps = len(train_loader) * args.epochs
	warmup_steps = int(args.warmup_ratio * total_steps)
	scheduler = get_linear_schedule_with_warmup(
		optimizer,
		num_warmup_steps=warmup_steps,
		num_training_steps=total_steps,
	)

	criterion = nn.CrossEntropyLoss()
	best_f1 = -1.0
	best_path = os.path.join(args.output_dir, "best_multimodal_roberta.pt")

	os.makedirs(args.output_dir, exist_ok=True)

	for epoch in range(1, args.epochs + 1):
		model.train()
		running_loss = 0.0

		for batch in train_loader:
			input_ids = batch["input_ids"].to(device)
			attention_mask = batch["attention_mask"].to(device)
			pixel_values = batch["pixel_values"].to(device)
			labels = batch["labels"].to(device)

			logits = model(
				input_ids=input_ids,
				attention_mask=attention_mask,
				pixel_values=pixel_values,
			)
			loss = criterion(logits, labels)

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			scheduler.step()

			running_loss += loss.item()

		train_loss = running_loss / max(1, len(train_loader))
		val_metrics = evaluate(model, val_loader, device)

		print(
			f"Epoch {epoch}/{args.epochs} | "
			f"train_loss={train_loss:.4f} | "
			f"val_loss={val_metrics['loss']:.4f} | "
			f"val_acc={val_metrics['accuracy']:.4f} | "
			f"val_f1={val_metrics['f1']:.4f}"
		)

		if val_metrics["f1"] > best_f1:
			best_f1 = val_metrics["f1"]
			torch.save(
				{
					"model_state_dict": model.state_dict(),
					"args": vars(args),
					"best_val_metrics": val_metrics,
				},
				best_path,
			)
			print(f"Saved best checkpoint to: {best_path}")

	final_metrics = evaluate(model, val_loader, device)
	metrics_path = os.path.join(args.output_dir, "final_val_metrics.json")
	with open(metrics_path, "w", encoding="utf-8") as f:
		json.dump(final_metrics, f, indent=2)
	print(f"Final validation metrics: {final_metrics}")
	print(f"Saved final metrics to: {metrics_path}")

	if os.path.exists(best_path):
		ckpt = torch.load(best_path, map_location=device)
		model.load_state_dict(ckpt["model_state_dict"])

	pred_path = write_predictions(
		model=model,
		samples=samples,
		tokenizer=tokenizer,
		image_processor=image_processor,
		args=args,
		device=device,
	)
	print(f"Saved predictions to: {pred_path}")


def parse_args():
	parser = argparse.ArgumentParser(
		description="Train a multimodal RoBERTa classifier on captioned Facebook memes."
	)
	parser.add_argument(
		"--data-file",
		type=str,
		default="hateful-captioning/captions_vllm_output1.jsonl",
		help="Path to captioned JSONL file.",
	)
	parser.add_argument(
		"--image-root",
		type=str,
		default="facebook-data",
		help="Image root folder; img_fname is resolved relative to this.",
	)
	parser.add_argument("--output-dir", type=str, default="cara/classification/outputs")
	parser.add_argument("--text-model", type=str, default="roberta-large")
	parser.add_argument("--image-model", type=str, default="openai/clip-vit-base-patch32")
	parser.add_argument("--max-length", type=int, default=256)
	parser.add_argument("--batch-size", type=int, default=8)
	parser.add_argument("--epochs", type=int, default=4)
	parser.add_argument("--lr", type=float, default=2e-5)
	parser.add_argument("--weight-decay", type=float, default=0.01)
	parser.add_argument("--warmup-ratio", type=float, default=0.1)
	parser.add_argument("--dropout", type=float, default=0.2)
	parser.add_argument("--hidden-dim", type=int, default=1024)
	parser.add_argument("--image-proj-dim", type=int, default=256)
	parser.add_argument("--predictions-file", type=str, default="predictions.jsonl")
	parser.add_argument("--val-size", type=float, default=0.15)
	parser.add_argument("--seed", type=int, default=42)
	parser.add_argument("--num-workers", type=int, default=2)
	parser.add_argument(
		"--freeze-image-encoder",
		action="store_true",
		help="Freeze pretrained image encoder (enabled by default).",
		default=True,
	)
	parser.add_argument(
		"--unfreeze-image-encoder",
		action="store_false",
		dest="freeze_image_encoder",
		help="Unfreeze image encoder for end-to-end fine-tuning.",
	)
	return parser.parse_args()


if __name__ == "__main__":
	args = parse_args()
	train(args)
