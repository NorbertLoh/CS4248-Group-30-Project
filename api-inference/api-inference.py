import os
import json
import base64
import time
from typing import List, Dict, Optional
from dotenv import load_dotenv
from groq import Groq
from pydantic import BaseModel, Field
from tqdm import tqdm

load_dotenv()

# --- CONFIGURATION ---
DATA_PATH = os.path.join(os.path.dirname(__file__), '../datapreparation/output/facebook-samples_50.jsonl')
IMG_DIR = os.path.join(os.path.dirname(__file__), '../facebook-data/img')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')

class MemeResponse(BaseModel):
    """Schema for structured JSON output from LLM"""
    reasoning: str = Field(description="Brief analysis of the image + text relationship")
    hateful: int = Field(description="1 if hateful, 0 if not")

def encode_image(image_path: str) -> str:
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def load_memes(jsonl_path: str) -> List[Dict]:
    memes = []
    if not os.path.exists(jsonl_path):
        print(f"Error: {jsonl_path} not found.")
        return []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            memes.append(json.loads(line))
    return memes

def run_inference(memes: List[Dict], max_samples: Optional[int] = None) -> List[Dict]:
    client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    results = []
    
    for i, meme in enumerate(tqdm(memes, desc="Groq Inference")):
        if max_samples and i >= max_samples:
            break
            
        img_path = os.path.join(IMG_DIR, os.path.basename(meme['img']))
        if not os.path.exists(img_path):
            continue

        img_data_url = f"data:image/jpeg;base64,{encode_image(img_path)}"
        
        # System instructions set the context for "benign confounders"
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an expert AI moderator. Analyze multimodal memes for hate speech. "
                    "Hate speech targets protected groups (race, religion, etc.). "
                    "Focus on how the text and image interact—often one is harmless without the other. "
                    "You must respond in valid JSON."
                )
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text", 
                        "text": (
                            f"Meme Text: '{meme['text']}'\n\n"
                            "Task: Describe the image content, explain the interaction between "
                            "the text and image, and then classify as hateful (1) or not (0).\n"
                            "Format: {'reasoning': '...', 'hateful': 0 or 1}"
                        )
                    },
                    {"type": "image_url", "image_url": {"url": img_data_url}}
                ]
            }
        ]

        try:
            completion = client.chat.completions.create(
                model="meta-llama/llama-4-scout-17b-16e-instruct",
                messages=messages,
                temperature=1,
                max_completion_tokens=300,
                response_format={"type": "json_object"}, # Strict JSON enforcement
                stream=False
            )
            
            raw_output = completion.choices[0].message.content
            parsed = MemeResponse.model_validate_json(raw_output)
            
            result = {
                'id': meme['id'],
                'text': meme['text'],
                'label': parsed.hateful,
                'reasoning': parsed.reasoning,
                'error': None
            }
        except Exception as e:
            print(f"Error on id={meme['id']}: {e}")
            result = {'id': meme['id'], 'error': str(e), 'label': -1}

        results.append(result)
        
        # Respect Rate Limits (Approx 30 RPM for Groq preview)
        time.sleep(2.2) 

    return results

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    memes = load_memes(DATA_PATH)
    
    if not memes: return

    final_results = run_inference(memes)
    
    out_path = os.path.join(RESULTS_DIR, 'inference_results.jsonl')
    with open(out_path, 'w', encoding='utf-8') as f:
        for item in final_results:
            f.write(json.dumps(item) + '\n')
            
    print(f"\nDone! Results saved to {out_path}")

if __name__ == '__main__':
    main()