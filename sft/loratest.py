from unsloth import FastVisionModel # FastLanguageModel for LLMs
import torch
from pathlib import Path
from PIL import Image

try:
    import pytesseract
except ImportError:
    pytesseract = None

# 4bit pre quantized models we support for 4x faster downloading + no OOMs.
fourbit_models = [
    "unsloth/Llama-3.2-11B-Vision-Instruct-bnb-4bit", # Llama 3.2 vision support
    "unsloth/Llama-3.2-11B-Vision-bnb-4bit",
    "unsloth/Llama-3.2-90B-Vision-Instruct-bnb-4bit", # Can fit in a 80GB card!
    "unsloth/Llama-3.2-90B-Vision-bnb-4bit",

    "unsloth/Pixtral-12B-2409-bnb-4bit",              # Pixtral fits in 16GB!
    "unsloth/Pixtral-12B-Base-2409-bnb-4bit",         # Pixtral base model

    "unsloth/Qwen2-VL-2B-Instruct-bnb-4bit",          # Qwen2 VL support
    "unsloth/Qwen2-VL-7B-Instruct-bnb-4bit",
    "unsloth/Qwen2-VL-72B-Instruct-bnb-4bit",

    "unsloth/llava-v1.6-mistral-7b-hf-bnb-4bit",      # Any Llava variant works!
    "unsloth/llava-1.5-7b-hf-bnb-4bit",
] # More models at https://huggingface.co/unsloth

from unsloth import FastVisionModel
# model, tokenizer = FastVisionModel.from_pretrained(
#     "unsloth/Qwen3-VL-8B-Instruct-unsloth-bnb-4bit",
#     load_in_4bit = True, # Use 4bit to reduce memory use. False for 16bit LoRA.
#     use_gradient_checkpointing = "unsloth", # True or "unsloth" for long context
# )

model, tokenizer = FastVisionModel.from_pretrained(
    model_name = "qwen_lora", # YOUR MODEL YOU USED FOR TRAINING
    load_in_4bit = True, # Set to False for 16bit LoRA
)
FastVisionModel.for_inference(model) # Enable for inference!

project_root = Path(__file__).resolve().parents[1]
images_dir = project_root / "datapreparation" / "output" / "facebook-images"
image_paths = sorted(
    p for p in images_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
)[:10]

if not image_paths:
    raise FileNotFoundError(f"No images found in {images_dir}")

instruction = """
Analyze this meme and output the meaning in JSON format. Identify specific metaphors where visual elements represent abstract concepts.
"""

system_prompt = "You are a specialized Multimodal Content Safety Analyst. Your task is to analyze memes for harmful content, hate speech, and demeaning stereotypes."

def extract_ocr_text(image):
    if pytesseract is None:
        return ""
    text = pytesseract.image_to_string(image).strip()
    return text

for image_path in image_paths:
    image = Image.open(image_path).convert("RGB")
    ocr_text = extract_ocr_text(image)
    ocr_block = ocr_text if ocr_text else "[No OCR text detected]"
    user_prompt = f"{instruction}\n\nOCR text from image:\n{ocr_block}"

    messages = [
        # {"role": "system", "content": system_prompt},
        {"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": user_prompt}
        ]}
    ]
    input_text = tokenizer.apply_chat_template(messages, add_generation_prompt = True)

    inputs = tokenizer(
        image,
        input_text,
        add_special_tokens = False,
        return_tensors = "pt",
    ).to("cuda")

    outputs = model.generate(
        **inputs,
        max_new_tokens = 1024,
        use_cache = True,
        temperature = 0.3, min_p = 0.1,
    )

    generated_tokens = outputs[:, inputs["input_ids"].shape[1]:]
    response = tokenizer.batch_decode(generated_tokens, skip_special_tokens = True)[0].strip()
    print(f"\n=== {image_path.name} ===")
    print(response)