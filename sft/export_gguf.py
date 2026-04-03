from unsloth import FastVisionModel
import torch

# Load the LoRA you just trained
model, tokenizer = FastVisionModel.from_pretrained(
    model_name = "qwen_lora", # This points to the folder created by your SFT script
    load_in_4bit = True,
)

# Attempt the GGUF export again
# Unsloth will see the folder already exists and skip the 'git clone' step
model.save_pretrained_gguf(
    "qwen_finetune", 
    tokenizer, 
    quantization_method = "q4_k_m"
)