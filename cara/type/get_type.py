import argparse
import json
import os
from typing import Any, Dict, List, Literal, Optional, Set

from PIL import Image
from pydantic import BaseModel, Field, ValidationError
from tqdm import tqdm
from vllm import LLM, SamplingParams
from vllm.sampling_params import StructuredOutputsParams


# Keep compatibility with existing repository vLLM setup.
os.environ.setdefault("VLLM_USE_V1", "0")


TargetCategory = Literal[
	"Race/Ethnicity",
	"Religion",
	"Gender/Sexuality",
	"Disability",
	"Nationality",
	"Other",
]


class TargetTypeDecision(BaseModel):
	target_category: TargetCategory = Field(
		description=(
			"Primary demographic target category of hate/stereotype in the meme. "
			"Must be exactly one of the allowed categories."
		)
	)


def parse_args() -> argparse.Namespace:
	base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
	default_data_path = os.path.join(base_dir, "facebook-data", "dev.jsonl")
	default_img_dir = os.path.join(base_dir, "facebook-data", "img")
	default_out_path = os.path.join(os.path.dirname(__file__), "target_type_label1_qwen8b.jsonl")

	parser = argparse.ArgumentParser(
		description="Label memes with original label=1 into target demographic categories using Qwen-VL via vLLM."
	)
	parser.add_argument(
		"--data-path",
		default=os.environ.get("GET_TYPE_DATA_PATH", default_data_path),
		help="Input JSONL dataset path.",
	)
	parser.add_argument(
		"--img-dir",
		default=os.environ.get("GET_TYPE_IMG_DIR", default_img_dir),
		help="Directory containing meme image files.",
	)
	parser.add_argument(
		"--out-path",
		default=os.environ.get("GET_TYPE_OUT_PATH", default_out_path),
		help="Output JSONL file path.",
	)
	parser.add_argument(
		"--model-id",
		default=os.environ.get("GET_TYPE_MODEL_ID", "Qwen/Qwen3-VL-8B-Thinking"),
		help="Hugging Face model id for vLLM.",
	)
	parser.add_argument(
		"--batch-size",
		type=int,
		default=int(os.environ.get("GET_TYPE_BATCH_SIZE", "32")),
		help="Batch size for generation.",
	)
	parser.add_argument(
		"--max-samples",
		type=int,
		default=None,
		help="Optional cap on number of label=1 samples to process.",
	)
	return parser.parse_args()


def load_label1_samples(data_path: str, max_samples: Optional[int] = None) -> List[Dict[str, Any]]:
	samples: List[Dict[str, Any]] = []
	with open(data_path, "r", encoding="utf-8") as f:
		for line in f:
			if not line.strip():
				continue
			row = json.loads(line)
			try:
				label_val = int(row.get("label"))
			except (TypeError, ValueError):
				continue
			if label_val != 1:
				continue
			samples.append(row)
			if max_samples is not None and len(samples) >= max_samples:
				break
	return samples


def resolve_image_path(sample: Dict[str, Any], img_dir: str) -> str:
	raw_img = str(sample.get("img", "")).strip()
	image_name = os.path.basename(raw_img)
	candidates = [
		raw_img,
		os.path.join(img_dir, raw_img),
		os.path.join(img_dir, image_name),
	]
	for path in candidates:
		if path and os.path.exists(path):
			return path
	return os.path.join(img_dir, image_name)


def build_vllm_prompt() -> str:
	instruction = (
		"Analyze this hateful meme. What is the primary demographic target of the hate or stereotype? "
		"Choose exactly ONE category from the following list: "
		"[Race/Ethnicity, Religion, Gender/Sexuality, Disability, Nationality, Other]. "
		"Output only the category name."
	)
	return (
		"<|im_start|>system\n"
		"You are a strict multimodal classifier. Follow the user instruction exactly and return valid JSON for the schema.<|im_end|>\n"
		"<|im_start|>user\n"
		"<|vision_start|><|image_pad|><|vision_end|>\n"
		f"{instruction}<|im_end|>\n"
		"<|im_start|>assistant\n"
	)


def load_done_ids(out_path: str) -> Set[str]:
	if not os.path.exists(out_path):
		return set()
	done: Set[str] = set()
	with open(out_path, "r", encoding="utf-8") as f:
		for line in f:
			if not line.strip():
				continue
			try:
				row = json.loads(line)
			except json.JSONDecodeError:
				continue
			done.add(str(row.get("id", "")))
	return done


def batched(items: List[Dict[str, Any]], batch_size: int) -> List[List[Dict[str, Any]]]:
	return [items[i : i + batch_size] for i in range(0, len(items), batch_size)]


def build_batch_inputs(batch: List[Dict[str, Any]], img_dir: str, prompt_template: str) -> tuple[List[Dict[str, Any]], List[Image.Image]]:
	inputs: List[Dict[str, Any]] = []
	opened_images: List[Image.Image] = []
	for sample in batch:
		img_path = resolve_image_path(sample, img_dir)
		image = Image.open(img_path).convert("RGB")
		opened_images.append(image)
		inputs.append(
			{
				"prompt": prompt_template,
				"multi_modal_data": {"image": image},
			}
		)
	return inputs, opened_images


def make_record(sample: Dict[str, Any], raw_text: str) -> Dict[str, Any]:
	try:
		parsed = TargetTypeDecision.model_validate_json(raw_text)
		target = parsed.target_category
		err = None
	except ValidationError as exc:
		target = "Other"
		err = str(exc)

	record = {
		"id": sample.get("id"),
		"img": sample.get("img"),
		"original_label": sample.get("label"),
		"target_category": target,
		"raw_output": raw_text,
	}
	if err:
		record["error"] = err
	return record


def process_batch(
	llm: LLM,
	batch: List[Dict[str, Any]],
	sampling: SamplingParams,
	img_dir: str,
	prompt_template: str,
	f_out: Any,
) -> None:
	inputs, opened_images = build_batch_inputs(batch, img_dir, prompt_template)
	outputs = llm.generate(inputs, sampling_params=sampling)

	for sample, out in zip(batch, outputs):
		raw_text = out.outputs[0].text if out.outputs else "{}"
		record = make_record(sample, raw_text)
		f_out.write(json.dumps(record, ensure_ascii=False) + "\n")

	for image in opened_images:
		image.close()


def main() -> None:
	args = parse_args()

	print(f"Loading label=1 samples from: {args.data_path}")
	samples = load_label1_samples(args.data_path, max_samples=args.max_samples)
	print(f"Found {len(samples)} samples with label=1")

	done_ids = load_done_ids(args.out_path)
	if done_ids:
		before = len(samples)
		samples = [s for s in samples if str(s.get("id", "")) not in done_ids]
		print(f"Skipping {before - len(samples)} already processed rows from existing output")

	if not samples:
		print("No samples to process. Exiting.")
		return

	print(f"Loading model with vLLM: {args.model_id}")
	llm = LLM(
		model=args.model_id,
		trust_remote_code=True,
		max_model_len=4096,
		gpu_memory_utilization=0.90,
	)

	schema_dict = TargetTypeDecision.model_json_schema()
	sampling = SamplingParams(
		temperature=0.0,
		max_tokens=128,
		structured_outputs=StructuredOutputsParams(json=schema_dict),
	)

	prompt_template = build_vllm_prompt()
	os.makedirs(os.path.dirname(args.out_path) or ".", exist_ok=True)

	with open(args.out_path, "a", encoding="utf-8") as f_out:
		for batch in tqdm(batched(samples, args.batch_size), desc="Batches"):
			process_batch(
				llm=llm,
				batch=batch,
				sampling=sampling,
				img_dir=args.img_dir,
				prompt_template=prompt_template,
				f_out=f_out,
			)

	print(f"Done. Wrote predictions to: {args.out_path}")


if __name__ == "__main__":
	main()
