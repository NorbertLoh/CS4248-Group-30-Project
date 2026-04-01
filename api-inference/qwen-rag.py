import os
import json
import torch
import importlib
from PIL import Image

# Fix Protobuf/Tensorflow conflict common in these environments
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

# Import the existing RAG class from api-inference-local
api_local = importlib.import_module('api-inference-local')
MemeCapRAG = api_local.MemeCapRAG
import re
from pydantic import BaseModel, Field

class MemeResponse(BaseModel):
    reasoning: str = Field(description="Step-by-step contrastive analysis")
    hateful: int = Field(description="0 if benign, 1 if dehumanizing subversion")

def robust_parse(text: str) -> dict:
    """Robust parsing to extract JSON even if Qwen outputs free-form text."""
    text = re.sub(r"[\x00-\x1F\x7F]", " ", text).strip()
    
    if "```json" in text:
        match = re.search(r"```json\s+(.*?)\s+```", text, re.DOTALL)
        if match:
            text = match.group(1)
            
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            parsed = MemeResponse.model_validate_json(match.group(0))
            return {
                "hateful": parsed.hateful,
                "reasoning": parsed.reasoning
            }
        except Exception:
            pass

    # Fallback extraction 
    h_match = re.search(r'"hateful"\s*:\s*([01])', text, re.IGNORECASE) or \
              re.search(r"'hateful'\s*:\s*([01])", text, re.IGNORECASE) or \
              re.search(r"(?:label|hateful).*?(?:\sis\s|:\s*|\s)([01])(?!\d)", text, re.IGNORECASE)

    hateful = int(h_match.group(1)) if h_match else 0
    return {"hateful": hateful, "reasoning": "REGEX_FALLBACK: " + text}

BASE_DIR = os.path.abspath(os.path.dirname(__file__) + "/..")
MEMECAP_DATA = os.path.join(BASE_DIR, 'memecap-data/memes-test.json')
QWEN_VL_MODEL_PATH = os.environ.get('QWEN_VL_MODEL_PATH', os.path.join(BASE_DIR, 'models', 'Qwen3-VL-2B-Instruct'))
DATA_PATH = os.path.join(BASE_DIR, 'datapreparation/output/facebook-samples_50.jsonl')
IMG_DIR = os.path.join(BASE_DIR, 'facebook-data/img')
RESULTS_DIR = os.path.join(BASE_DIR, 'api-inference/results')
OUT_PATH = os.path.join(RESULTS_DIR, 'qwen3_vl_2b_local_results.jsonl')
MAX_NEW_TOKENS = 1024

class Qwen3VLRAG:
    """
    A unified pipeline combining SigLIP-based MemeCap RAG and Qwen3-VL-2B inference.
    """
    def __init__(self, model_path: str = QWEN_VL_MODEL_PATH, memecap_json_path: str = MEMECAP_DATA):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # 1. Initialize RAG system
        print("Initializing SigLIP MemeCap RAG system...")
        self.rag = MemeCapRAG(memecap_json_path)
        
        # 2. Initialize Qwen Model
        print(f"Loading Qwen-VL model from {model_path}...")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model path not found: {model_path}. Please set QWEN_VL_MODEL_PATH.")
            
        self.processor = AutoProcessor.from_pretrained(
            model_path, 
            local_files_only=True
        )
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path,
            dtype="auto",
            device_map="auto" if torch.cuda.is_available() else None,
            local_files_only=True,
        )
        
        if not torch.cuda.is_available():
            self.model = self.model.to("cpu")
            
        self.model.eval()
        print("Model loaded successfully!")

    def query(self, image_path: str, meme_text: str = "") -> dict:
        """
        Analyzes a single image using MemeCap context.
        """
        print(f"Retrieving context for image: {image_path}")
        memecap_context = self.rag.query(image_path, k=3)
        
        image_obj = Image.open(image_path).convert("RGB")
        image_obj.thumbnail((768, 768), Image.Resampling.LANCZOS)
        
        # Consistent padding to avoid dimension mismatches in Qwen VL attention setup
        image = Image.new("RGB", (768, 768), (0, 0, 0))
        image.paste(image_obj, ((768 - image_obj.width) // 2, (768 - image_obj.height) // 2))

        messages = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "You are a specialist in semiotics and multimodal hate speech detection. "
                            "Your expertise lies in identifying 'Harmful Subversions'—where benign cultural metaphors "
                            "are hijacked to dehumanize protected groups. You must distinguish between "
                            "harsh satire (0) and genuine dehumanizing subversion (1). "
                            "Always respond in valid JSON strictly formatted:\n"
                            '{"reasoning": "your reasoning", "hateful": 0}'
                        )
                    }
                ]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "### TASK: Contrastive Metaphor Analysis\n\n"
                            "### 1. REFERENCE CONTEXT (MemeCap Benign Examples):\n"
                            f"{memecap_context}\n\n"
                            "### 2. TARGET MEME DATA:\n"
                            f"Text: '{meme_text}'\n\n"
                            "### 3. REASONING PROTOCOL:\n"
                            "A. IDENTIFY the visual metaphor in the Target Meme.\n"
                            "B. COMPARE with Reference Context: Is the target using the metaphor in a standard way, "
                            "or is it a 'Delta' (a subversion)?\n"
                            "C. EVALUATE DEHUMANIZATION: Does the text target a protected group (race, religion, etc.) "
                            "by using the metaphor to imply they are sub-human, a plague, or a threat?\n\n"
                            "### 4. OUTPUT RULE:\n"
                            "If the metaphor is used for general political satire or offensive humor without "
                            "dehumanizing a protected group, label as 0.\n"
                            "Respond EXCLUSIVELY with a valid JSON object strictly using double quotes:\n"
                            '{\n  "reasoning": "Step-by-step contrastive analysis...",\n  "hateful": 0\n}'
                        )
                    },
                    {"type": "image"}
                ]
            }
        ]

        prompt = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.processor(
            text=[prompt],
            images=[image],
            return_tensors="pt",
        )

        target_device = self.model.device if hasattr(self.model, "device") else "cpu"
        inputs = {k: (v.to(target_device) if hasattr(v, "to") else v) for k, v in inputs.items()}

        print("Generating analysis...")
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                repetition_penalty=1.1,
            )

        input_len = inputs["input_ids"].shape[1]
        generated_ids = output_ids[0, input_len:]
        out_text = self.processor.decode(generated_ids, skip_special_tokens=True)
        
        try:
            parsed = robust_parse(out_text)
            return {
                "label": parsed["hateful"],
                "reasoning": parsed["reasoning"],
                "retrieved_context": memecap_context,
                "raw_output": out_text
            }
        except Exception as e:
            return {
                "error": str(e),
                "label": -1,
                "retrieved_context": memecap_context,
                "raw_output": out_text
            }


if __name__ == "__main__":
    from tqdm import tqdm
    
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    print("--- Starting Qwen3-VL-2B RAG Batch Inference ---")
    bot = Qwen3VLRAG()
    
    if not os.path.exists(DATA_PATH):
        print(f"Data path not found: {DATA_PATH}")
        exit(1)
        
    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        memes = [json.loads(line) for line in f]
        
    print(f"Loaded {len(memes)} memes for inference.")
    
    final_results = []
    # Using sequential processing to avoid batch padding dimension bugs with Qwen VL
    for meme in tqdm(memes, desc="Processing Memes"):
        img_path = os.path.join(IMG_DIR, os.path.basename(meme['img']))
        if not os.path.exists(img_path):
            print(f"Warning: Image not found {img_path}")
            continue
            
        try:
            res = bot.query(
                image_path=img_path,
                meme_text=meme.get('text', '')
            )
            final_results.append({
                'id': meme['id'],
                'label': res['label'],
                'reasoning': res.get('reasoning', ''),
                'retrieved_context': res.get('retrieved_context', ''),
                'error': res.get('error', '')
            })
        except Exception as e:
            final_results.append({'id': meme['id'], 'error': str(e), 'label': -1})
            
    with open(OUT_PATH, 'w', encoding='utf-8') as f:
        for item in final_results:
            f.write(json.dumps(item) + '\n')
            
    print(f"\nDone! Saved {len(final_results)} results to '{OUT_PATH}'.")
