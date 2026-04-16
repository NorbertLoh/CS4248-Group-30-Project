import base64
import mimetypes
from pathlib import Path
from llama_cpp import Llama

# 1. Initialize the model
GGUF_MODEL_PATH = "qwen_finetune_gguf/qwen3-vl-8b-instruct.Q4_K_M.gguf" 

print(f"Loading {GGUF_MODEL_PATH}...")
llm = Llama(
    model_path=GGUF_MODEL_PATH,
    n_ctx=8192,       # Keep context low to avoid OOM and speed up inference
    n_gpu_layers=-1,  # Ensure TITAN RTX is doing the work
    verbose=False 
)

# 2. Helper function to encode image
def encode_image(image_path):
    mime_type = mimetypes.guess_type(image_path)[0] or "application/octet-stream"
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
        return f"data:{mime_type};base64,{encoded_string}"

# 3. Setup the payload
project_root = Path(__file__).resolve().parents[1]
test_image_path = str(project_root / "memecap-data" / "memes" / "memes_bpet7l.png")
base64_image = encode_image(test_image_path)

# 4. CRITICAL: Use Raw ChatML Formatting
# We avoid the word 'poster' in instructions to reduce logit bias.
instruction = "JSON only. Provide: 'title', 'metaphors' (list), and 'explanation'."

# We PRE-FILL the '{' to lock it into JSON mode immediately.
raw_prompt = f"""<|im_start|>system
You are a meme analyst. Output strict JSON.<|im_end|>
<|im_start|>user
<|vision_start|><|image_pad|><|vision_end|>{instruction}<|im_end|>
<|im_start|>assistant
{{"""

# 5. Run inference with High Repetition Penalty
print("\nRunning inference...")
output = llm(
    prompt=raw_prompt,
    max_tokens=256,        # Short limit to prevent infinite yapping
    temperature=0.1,       # Low temperature = less hallucination/looping
    repeat_penalty=1.8,    # AGGRESSIVE penalty to kill the "poster" loop
    top_p=0.1,             # Only allow the most certain tokens
    stop=["<|im_end|>", "}"], # Kill process once JSON or turn ends
    echo=False
)

# 6. Output the result
print("\n--- Output ---")
generated_text = output["choices"][0]["text"]
# Re-attach the brace we pre-filled
full_json = "{" + generated_text
if not full_json.endswith("}"):
    full_json += "}"
print(full_json)