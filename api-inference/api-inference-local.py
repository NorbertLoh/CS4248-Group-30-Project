import os
import json
import shutil
import re
import glob
import numpy as np
import faiss
from typing import List, Dict
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from tqdm import tqdm
import torch
from transformers import AutoProcessor, AutoModel, Qwen3VLForConditionalGeneration
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
OUT_PATH = os.path.join(RESULTS_DIR, 'qwen3_vl_2b_local_results.jsonl')

# Set this in your shell to the local model directory, for example:
# set QWEN_VL_MODEL_PATH=C:\path\to\Qwen3-VL-2B
QWEN_VL_MODEL_PATH = os.environ.get('QWEN_VL_MODEL_PATH', os.path.join(BASE_DIR, 'models', 'Qwen3-VL-2B-Instruct'))
MAX_NEW_TOKENS = 1024
BATCH_SIZE = int(os.environ.get('LOCAL_INFERENCE_BATCH_SIZE', '2'))

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INDEX_PATH = os.path.join(SCRIPT_DIR, "memecap_siglip.index")
MAP_PATH = os.path.join(SCRIPT_DIR, "memecap_map.json")

class MemeResponse(BaseModel):
    reasoning: str = Field(description="Deep analysis using visual context and retrieved metaphors")
    hateful: int = Field(description="1 if hateful, 0 if not")


class MemeCapRAG:
    def __init__(self, memecap_json_path: str):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        model_id = "google/siglip-so400m-patch14-384"
        print(f"Loading SigLIP on {self.device}...")
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id).to(self.device).eval()
        
        self.data = self.load_memecap(memecap_json_path)
        self.valid_data_indices = []
        self.index = None

        if os.path.exists(INDEX_PATH) and os.path.exists(MAP_PATH):
            print("Found existing index. Loading...")
            self.index = faiss.read_index(INDEX_PATH)
            with open(MAP_PATH, 'r') as f:
                self.valid_data_indices = json.load(f)
        else:
            self.build_index()

    def load_memecap(self, path: str) -> List[Dict]:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def get_embedding(self, pil_image: Image.Image):
        inputs = self.processor(images=pil_image.convert("RGB"), return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model.get_image_features(**inputs)
            if hasattr(outputs, "pooler_output"):
                features = outputs.pooler_output
            elif hasattr(outputs, "image_embeds"):
                features = outputs.image_embeds
            else:
                features = outputs
        return features.cpu().numpy()

    def _resolve_image_path(self, img_fname: str) -> str:
        if not img_fname:
            return ""

        basename = os.path.basename(img_fname)
        candidates = [
            img_fname,
            img_fname.lower(),
            img_fname.upper(),
            basename,
            basename.lower(),
            basename.upper(),
        ]

        for cand in candidates:
            cand_path = os.path.join(MEMECAP_IMG_DIR, cand)
            if os.path.exists(cand_path):
                return cand_path

        # Fallback: match by stem regardless of extension/case.
        stem = os.path.splitext(basename)[0]
        wildcard = os.path.join(MEMECAP_IMG_DIR, f"{stem}.*")
        matches = glob.glob(wildcard)
        return matches[0] if matches else ""

    def build_index(self):
        print("Building fresh index (This will only happen once)...")
        all_embeddings = []
        failed = 0
        for i, entry in enumerate(tqdm(self.data)):
            img_path = self._resolve_image_path(entry.get('img_fname', ''))
            if os.path.exists(img_path):
                try:
                    img = Image.open(img_path)
                    emb = self.get_embedding(img)
                    all_embeddings.append(emb)
                    self.valid_data_indices.append(i)
                except Exception:
                    failed += 1
                    continue

        if not all_embeddings:
            raise ValueError(
                "No MemeCap images were indexed. "
                f"Check MEMECAP_IMG_DIR='{MEMECAP_IMG_DIR}' and memecap img_fname values."
            )
        
        embeddings_np = np.vstack(all_embeddings).astype('float32')
        self.index = faiss.IndexFlatL2(embeddings_np.shape[1])
        self.index.add(embeddings_np)

        print(f"Indexed {len(all_embeddings)} MemeCap images ({failed} failed).")
        
        # Save to scratch
        faiss.write_index(self.index, INDEX_PATH)
        with open(MAP_PATH, 'w') as f:
            json.dump(self.valid_data_indices, f)

    def query(self, image_path: str, k: int = 3) -> str:
        img = Image.open(image_path)
        vec = self.get_embedding(img)
        _, indices = self.index.search(vec, k)
        
        contexts = []
        for idx in indices[0]:
            match = self.data[self.valid_data_indices[idx]]
            caps = match.get('img_captions', ['N/A'])[0]
            metas = [f"{m['metaphor']}:{m['meaning']}" for m in match.get('metaphors', [])]
            contexts.append(f"Similar Meme: {caps} | Metaphors: {', '.join(metas)}")
        return "\n".join(contexts)
    
def parse_safe_json(text: str) -> MemeResponse:
    # 1. Clean control characters and invisible garbage
    text = re.sub(r"[\x00-\x1F\x7F]", " ", text)
    
    # 2. Extract JSON block using regex
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        # Emergency recovery: check for "hateful": 1 in text
        h = 1 if '"hateful": 1' in text or '"hateful":1' in text else 0
        return MemeResponse(reasoning="Regex extraction failed, fallback used", hateful=h)
    
    try:
        return MemeResponse.model_validate_json(match.group(0))
    except Exception:
        # Final attempt: manual field extraction
        h_match = re.search(r'"hateful":\s*(\d)', text)
        h = int(h_match.group(1)) if h_match else 0
        return MemeResponse(reasoning="Pydantic validation failed, manual field recovery", hateful=h)
    
def load_local_qwen_model(model_path: str):
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Local Qwen model path not found: {model_path}. "
            "Set QWEN_VL_MODEL_PATH to your downloaded model directory."
        )

    device_map = "auto" if torch.cuda.is_available() else None

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path,
        dtype="auto",
        device_map=device_map,
        local_files_only=True,
    )

    processor = AutoProcessor.from_pretrained(
        model_path,
        local_files_only=True,
    )

    if not torch.cuda.is_available():
        model = model.to("cpu")
    model.eval()
    return model, processor


def parse_model_json(raw_output: str) -> MemeResponse:
    try:
        return MemeResponse.model_validate_json(raw_output)
    except Exception:
        match = re.search(r"\{.*\}", raw_output, re.DOTALL)
        if not match:
            raise ValueError("Model output did not contain a valid JSON object.")
        return MemeResponse.model_validate_json(match.group(0))


def infer_with_local_qwen(
    model,
    processor,
    image_path: str,
    meme_text: str,
    memecap_context: str,
) -> MemeResponse:
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
                        "Always respond in valid JSON format."
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
                        "Respond with JSON: {'reasoning': 'Step-by-step contrastive analysis...', 'hateful': 0 or 1}"
                    )
                },
                {"type": "image", "image": image_path}
            ]
        }
    ]

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )

    target_device = model.device if hasattr(model, "device") else "cpu"
    inputs = inputs.to(target_device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
        )

    generated_ids = [out[len(ins):] for ins, out in zip(inputs.input_ids, output_ids)]
    return processor.batch_decode(generated_ids, skip_special_tokens=True)


def infer_with_local_qwen_batch(
    model,
    processor,
    batch_items: List[Dict],
) -> List[str]:
    # Process sequentially to avoid Qwen3-VL padding bugs with varying image grid sizes
    # that cause INT_MAX nonzero() errors and profound CUDA OOMs.
    results = []

    for item in batch_items:
        image_path = item['image_path']
        meme_text = item['meme_text']
        memecap_context = item['memecap_context']

        image_obj = Image.open(image_path).convert("RGB")
        image_obj.thumbnail((768, 768), Image.Resampling.LANCZOS)
        
        # Pad with black bars to exactly 768x768 to avoid INT_MAX grid logic arithmetic bugs
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
                            "Always respond in valid JSON format."
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
                            "Respond with JSON: {'reasoning': 'Step-by-step contrastive analysis...', 'hateful': 0 or 1}"
                        )
                    },
                    {"type": "image"}
                ]
            }
        ]

        prompt = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = processor(
            text=[prompt],
            images=[image],
            return_tensors="pt",
        )

        target_device = model.device if hasattr(model, "device") else "cpu"
        inputs = {k: (v.to(target_device) if hasattr(v, "to") else v) for k, v in inputs.items()}

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
            )

        input_len = inputs["input_ids"].shape[1]
        generated_ids = output_ids[0, input_len:]
        out_text = processor.decode(generated_ids, skip_special_tokens=True)
        results.append(out_text)

    return results


def run_inference(memes: List[Dict], rag: MemeCapRAG) -> List[Dict]:
    model, processor = load_local_qwen_model(QWEN_VL_MODEL_PATH)
    results = []

    prepared_items = []
    for meme in tqdm(memes, desc="Preparing RAG context"):
        img_path = os.path.join(IMG_DIR, os.path.basename(meme['img']))
        if not os.path.exists(img_path):
            continue
        memecap_context = rag.query(img_path, meme['id'])

        prepared_items.append({
            'id': meme['id'],
            'image_path': img_path,
            'meme_text': meme.get('text', ''),
            'memecap_context': memecap_context,
        })

    for i in tqdm(range(0, len(prepared_items), BATCH_SIZE), desc="Qwen3-VL-2B Batched Inference"):
        batch_items = prepared_items[i:i + BATCH_SIZE]

        try:
            raw_outputs = infer_with_local_qwen_batch(
                model=model,
                processor=processor,
                batch_items=batch_items,
            )

            for item, raw_output in zip(batch_items, raw_outputs):
                try:
                    parsed = parse_safe_json(raw_output.strip())
                    results.append({
                        'id': item['id'],
                        'label': parsed.hateful,
                        'reasoning': parsed.reasoning,
                        'retrieved_context': item['memecap_context']
                    })
                except Exception as parse_err:
                    results.append({'id': item['id'], 'error': str(parse_err), 'label': -1})
        except Exception as e:
            for item in batch_items:
                results.append({'id': item['id'], 'error': str(e), 'label': -1})
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    del model
    del processor
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return results


def main():
    os.makedirs(RETRIEVED_IMG_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    rag_system = MemeCapRAG(MEMECAP_DATA)
    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        memes = [json.loads(line) for line in f]
    final_results = run_inference(memes, rag_system)
    with open(OUT_PATH, 'w', encoding='utf-8') as f:
        for item in final_results:
            f.write(json.dumps(item) + '\n')
    print(f"\nDone! Saved {len(final_results)} results to '{OUT_PATH}'.")


if __name__ == '__main__':
    main()