import os
import json
import base64
import time
import shutil
import requests
import numpy as np
import faiss
from typing import List, Dict, Optional
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from tqdm import tqdm
import torch
from transformers import AutoProcessor, AutoModel
from PIL import Image

load_dotenv()

# --- CONFIGURATION ---
BASE_DIR = os.path.abspath(os.path.dirname(__file__) + "/..")

DATA_PATH = os.path.join(BASE_DIR, 'datapreparation/output/facebook-samples_50.jsonl')
IMG_DIR = os.path.join(BASE_DIR, 'facebook-data/img')
MEMECAP_DATA = os.path.join(BASE_DIR, 'memecap-data/memes-test.json')
MEMECAP_IMG_DIR = os.path.join(BASE_DIR, 'memecap-data/memes') 

RESULTS_DIR = os.path.join(BASE_DIR, 'api-inference/results')
RETRIEVED_IMG_DIR = os.path.join(RESULTS_DIR, 'retrieved_images')
RAG_CACHE_DIR = os.path.join(RESULTS_DIR, 'rag_cache')

OLLAMA_API_URL = os.getenv('OLLAMA_API_URL', 'http://25.15.129.222:11434/api/chat')
OLLAMA_MODEL = os.getenv('OLLAMA_MODEL', 'qwen3-vl:8b-thinking-q4_K_M')

MEME_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "reasoning": {"type": "string"},
        "hateful": {"type": "integer", "enum": [0, 1]}
    },
    "required": ["reasoning", "hateful"],
    "additionalProperties": False
}

class MemeResponse(BaseModel):
    reasoning: str = Field(description="Deep analysis using visual context and retrieved metaphors")
    hateful: int = Field(description="1 if hateful, 0 if not")


class MemeCapRAG:
    def __init__(self, memecap_json_path: str):
        print("Loading SigLIP model and processor...")
        model_id = "google/siglip-so400m-patch14-384"
        self.model_id = model_id
        self.memecap_json_path = memecap_json_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id).to(self.device)
        self.model.eval()
        self.data = self.load_memecap(memecap_json_path)
        self.index = None
        self.valid_data_indices = []
        self.build_index()

    def _cache_paths(self):
        index_path = os.path.join(RAG_CACHE_DIR, 'memecap_siglip.index')
        valid_indices_path = os.path.join(RAG_CACHE_DIR, 'memecap_valid_indices.json')
        metadata_path = os.path.join(RAG_CACHE_DIR, 'memecap_cache_metadata.json')
        return index_path, valid_indices_path, metadata_path

    def _load_cached_index(self) -> bool:
        index_path, valid_indices_path, metadata_path = self._cache_paths()
        if not (os.path.exists(index_path) and os.path.exists(valid_indices_path) and os.path.exists(metadata_path)):
            return False

        try:
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)

            expected = {
                'model_id': self.model_id,
                'memecap_json_path': os.path.abspath(self.memecap_json_path),
                'memecap_json_mtime': os.path.getmtime(self.memecap_json_path),
                'memecap_size': len(self.data),
            }
            for key, value in expected.items():
                if metadata.get(key) != value:
                    return False

            self.index = faiss.read_index(index_path)
            with open(valid_indices_path, 'r', encoding='utf-8') as f:
                self.valid_data_indices = json.load(f)
            print("Loaded cached MemeCap SigLIP index.")
            return True
        except Exception:
            return False

    def _save_cached_index(self):
        index_path, valid_indices_path, metadata_path = self._cache_paths()
        os.makedirs(RAG_CACHE_DIR, exist_ok=True)
        faiss.write_index(self.index, index_path)
        with open(valid_indices_path, 'w', encoding='utf-8') as f:
            json.dump(self.valid_data_indices, f)
        metadata = {
            'model_id': self.model_id,
            'memecap_json_path': os.path.abspath(self.memecap_json_path),
            'memecap_json_mtime': os.path.getmtime(self.memecap_json_path),
            'memecap_size': len(self.data),
        }
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f)

    def load_memecap(self, path: str) -> List[Dict]:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def get_image_embedding(self, PIL_image):
        inputs = self.processor(images=PIL_image.convert("RGB"), return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model.get_image_features(**inputs)
            if hasattr(outputs, "pooler_output"):
                features = outputs.pooler_output
            elif hasattr(outputs, "image_embeds"):
                features = outputs.image_embeds
            else:
                features = outputs
        return features.cpu().numpy()

    def build_index(self):
        if self._load_cached_index():
            return

        print(f"Indexing MemeCap VISUALLY with SigLIP...")
        all_embeddings = []
        for i, entry in enumerate(tqdm(self.data, desc="Encoding MemeCap")):
            img_fname = entry.get('img_fname')
            if not img_fname:
                continue
            img_path = os.path.join(MEMECAP_IMG_DIR, img_fname)
            # Case-sensitivity fallback
            if not os.path.exists(img_path):
                img_path = os.path.join(MEMECAP_IMG_DIR, img_fname.lower())
            if os.path.exists(img_path):
                try:
                    img = Image.open(img_path)
                    embedding = self.get_image_embedding(img)
                    all_embeddings.append(embedding)
                    self.valid_data_indices.append(i)
                except Exception:
                    continue
        if not all_embeddings:
            raise ValueError("No valid images found to index! Check your MEMECAP_IMG_DIR.")
        embeddings_np = np.vstack(all_embeddings).astype('float32')
        self.index = faiss.IndexFlatL2(embeddings_np.shape[1])
        self.index.add(embeddings_np)
        self._save_cached_index()

    def query(self, image_path: str, meme_id: int, k: int = 3) -> str:
        if not os.path.exists(image_path):
            return "Visual context unavailable."
        img = Image.open(image_path)
        query_vec = self.get_image_embedding(img)
        distances, indices = self.index.search(query_vec, k)
        context_strings = []
        for rank, idx in enumerate(indices[0]):
            data_idx = self.valid_data_indices[idx]
            match = self.data[data_idx]
            # Save matched image for visual confirmation
            img_fname = match.get('img_fname')
            if img_fname:
                src_path = os.path.join(MEMECAP_IMG_DIR, img_fname)
                if not os.path.exists(src_path):
                    src_path = os.path.join(MEMECAP_IMG_DIR, img_fname.lower())
                if os.path.exists(src_path):
                    dest_name = f"fb_{meme_id}_match_{rank}_{img_fname}"
                    shutil.copy(src_path, os.path.join(RETRIEVED_IMG_DIR, dest_name))
            metaphors = ", ".join([f"'{m['metaphor']}' means '{m['meaning']}'" for m in match.get('metaphors', [])])
            context_strings.append(
                f"Reference {rank}:\n"
                f"- Image Content: {match.get('img_captions', [''])[0]}\n"
                f"- Metaphors: {metaphors if metaphors else 'None'}"
            )
        return "\n\n".join(context_strings)


def encode_image(image_path: str) -> str:
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def ollama_chat(messages: List[Dict]) -> str:
    payload = {
        "model": OLLAMA_MODEL,
        "messages": messages,
        "stream": False,
        "format": MEME_RESPONSE_SCHEMA,
        "options": {"temperature": 0.2}
    }
    response = requests.post(OLLAMA_API_URL, json=payload, timeout=180)
    response.raise_for_status()
    data = response.json()
    return data["message"]["content"]


def parse_structured_response(raw_output: str) -> MemeResponse:
    parsed_json = None
    try:
        parsed_json = json.loads(raw_output)
    except json.JSONDecodeError:
        # Fallback: some models wrap JSON in extra text, so extract the first JSON object.
        start = raw_output.find("{")
        end = raw_output.rfind("}")
        if start != -1 and end != -1 and end > start:
            parsed_json = json.loads(raw_output[start:end + 1])

    if not isinstance(parsed_json, dict):
        raise ValueError(f"Model did not return a JSON object. Raw output: {raw_output}")

    if isinstance(parsed_json.get('hateful'), bool):
        parsed_json['hateful'] = int(parsed_json['hateful'])

    return MemeResponse.model_validate(parsed_json)


def run_inference(memes: List[Dict], rag: MemeCapRAG) -> List[Dict]:
    results = []
    for meme in tqdm(memes, desc="Ollama Qwen3-VL Inference"):
        img_path = os.path.join(IMG_DIR, os.path.basename(meme['img']))
        if not os.path.exists(img_path):
            continue
        # Retrieval
        memecap_context = rag.query(img_path, meme['id'])
        encoded_image = encode_image(img_path)
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a specialist in semiotics and multimodal hate speech detection. "
                    "Your expertise lies in identifying 'Harmful Subversions'—where benign cultural metaphors "
                    "are hijacked to dehumanize protected groups. You must distinguish between "
                    "harsh satire (0) and genuine dehumanizing subversion (1). "
                    "Always respond in valid JSON format."
                )
            },
            {
                "role": "user",
                "content": (
                    "/no_think"
                    "### TASK: Contrastive Metaphor Analysis\n\n"
                    "### 1. REFERENCE CONTEXT (MemeCap Benign Examples):\n"
                    f"{memecap_context}\n\n"
                    # "No reference context available.\n\n"
                    "### 2. TARGET MEME DATA:\n"
                    f"Text: '{meme['text']}'\n\n"
                    "### 3. REASONING PROTOCOL:\n"
                    "A. IDENTIFY the visual metaphor in the Target Meme.\n"
                    "B. COMPARE with Reference Context: Is the target using the metaphor in a standard way, "
                    "or is it a 'Delta' (a subversion)?\n"
                    "C. EVALUATE DEHUMANIZATION: Does the text target a protected group (race, religion, etc.) "
                    "by using the metaphor to imply they are sub-human, a plague, or a threat?\n\n"
                    "### 4. OUTPUT RULE:\n"
                    "If the metaphor is used for general political satire or offensive humor without "
                    "dehumanizing a protected group, label as 0.\n"
                    "Respond with JSON: {'reasoning': 'Step-by-step contrastive analysis...', 'hateful': 0 or 1}"
                ),
                "images": [encoded_image]
            }
        ]
        try:
            raw_output = ollama_chat(messages)
            parsed = parse_structured_response(raw_output)
            results.append({
                'id': meme['id'],
                'label': parsed.hateful,
                'reasoning': parsed.reasoning,
                'retrieved_context': memecap_context
            })
        except Exception as e:
            results.append({'id': meme['id'], 'error': str(e), 'label': -1})
        time.sleep(2.0)
    return results


def main():
    os.makedirs(RETRIEVED_IMG_DIR, exist_ok=True)
    rag_system = MemeCapRAG(MEMECAP_DATA)
    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        memes = [json.loads(line) for line in f]
    final_results = run_inference(memes, rag_system)
    out_path = os.path.join(RESULTS_DIR, 'llama4_siglip_results.jsonl')
    with open(out_path, 'w', encoding='utf-8') as f:
        for item in final_results:
            f.write(json.dumps(item) + '\n')
    print(f"\nDone! Saved {len(final_results)} results. Check '{RETRIEVED_IMG_DIR}' for matched visuals.")


if __name__ == '__main__':
    main()