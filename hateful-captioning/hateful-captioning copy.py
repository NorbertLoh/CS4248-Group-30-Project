import os
import json
import base64
from io import BytesIO
from PIL import Image
from huggingface_hub import hf_hub_download
import pydantic
from llama_cpp import Llama
from llama_cpp.llama_chat_format import Qwen25VLChatHandler
from pydantic import BaseModel
from tqdm import tqdm
from typing import List

# --- 1. CONFIGURATION ---
LLM_REPO_ID = "unsloth/Qwen3-VL-8B-Instruct-GGUF"
MODEL_FILE = "Qwen3-VL-8B-Instruct-Q4_K_M.gguf"
MMPROJ_REPO_ID = "Qwen/Qwen3-VL-8B-Instruct-GGUF"
MMPROJ_FILE = "mmproj-Qwen3VL-8B-Instruct-F16.gguf"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FACEBOOK_DATA_DIR = os.path.join(BASE_DIR, "..", "facebook-data")
DATASET_JSONL = os.path.join(FACEBOOK_DATA_DIR, "train.jsonl")  # Path to your FHM dataset file
RUN_SUFFIX = 1  # Change this for a different run output file suffix
SAMPLES_PER_LABEL = 200
TARGET_LABELS = (0, 1)
MAX_SUCCESSFUL_PROCESSES = None  # Set to None to process all selected samples; counts attempts
OUTPUT_FILE = os.path.join(BASE_DIR, f"captions_output{RUN_SUFFIX}.jsonl")

# --- 2. DOWNLOAD MODELS ---
def download_models():
    # Download LLM from Unsloth
    if not os.path.exists(MODEL_FILE):
        print(f"Downloading {MODEL_FILE} from Unsloth...")
        hf_hub_download(repo_id=LLM_REPO_ID, filename=MODEL_FILE, local_dir="./")
    else:
        print(f"{MODEL_FILE} already exists.")

    # Download Vision Projector from Official Qwen
    if not os.path.exists(MMPROJ_FILE):
        print(f"Downloading {MMPROJ_FILE} from Official Qwen...")
        hf_hub_download(repo_id=MMPROJ_REPO_ID, filename=MMPROJ_FILE, local_dir="./")
    else:
        print(f"{MMPROJ_FILE} already exists.")

# --- 3. STRUCTURED OUTPUT SCHEMA ---

class Metaphor(BaseModel):
    metaphor: str
    meaning: str

# We only ask the LLM for things it can actually deduce from the image
class MemeAnalysis(BaseModel):
    img_captions: List[str]
    meme_captions: List[str]
    title: str
    metaphors: List[Metaphor]


def select_balanced_samples_chunked(dataset_jsonl, samples_per_label, target_labels, run_suffix):
    counts = dict.fromkeys(target_labels, 0)
    skipped = dict.fromkeys(target_labels, 0)
    selected = []
    
    # Calculate how many of each label we need to skip based on the run number.
    # Run 1 skips 0. Run 2 skips 200. Run 3 skips 400.
    skip_target = (run_suffix - 1) * samples_per_label

    with open(dataset_jsonl, 'r') as f_in:
        for line in f_in:
            data = json.loads(line)
            
            # Safely extract and cast the label
            raw_label = data.get('label')
            if raw_label is None: continue
            try:
                label = int(raw_label)
            except ValueError: continue

            if label in counts:
                # 1. Skip Phase: Ignore this line if we haven't skipped enough for this run yet
                if skipped[label] < skip_target:
                    skipped[label] += 1
                    continue
                
                # 2. Collect Phase: Keep this line if we still need it for this run
                if counts[label] < samples_per_label:
                    selected.append(data)
                    counts[label] += 1
                    
                # 3. Stop Phase: Break the loop once we've collected the requested amount for all labels
                if all(counts[l] >= samples_per_label for l in target_labels):
                    break

    print(f"Run {run_suffix} | Skipped first {skip_target} per label | Collected: {counts}")
    return selected

# --- 4. PROCESSING LOGIC ---
def process_memes():
    download_models()

    # Initialize Vision Handler & Model
    chat_handler = Qwen25VLChatHandler(clip_model_path=MMPROJ_FILE)
    llm = Llama(
        model_path=MODEL_FILE,
        chat_handler=chat_handler,
        n_gpu_layers=-1, # Offload all to GPU
        n_ctx=1024,
        logits_all=True
    )

    selected_data = select_balanced_samples_chunked(
        DATASET_JSONL,
        SAMPLES_PER_LABEL,
        TARGET_LABELS,
        RUN_SUFFIX
    )

    with open(OUTPUT_FILE, 'w') as f_out:
        progress = tqdm(selected_data, desc="Captioning memes", unit="meme")
        attempted_processes = 0
        for data in progress:
            img_path = os.path.join(FACEBOOK_DATA_DIR, data['img'])
            
            # Convert image to data URL for llama-cpp
            with open(img_path, "rb") as image_file:
                base64_image = base64.b64encode(image_file.read()).decode('utf-8')
                data_url = f"data:image/jpeg;base64,{base64_image}"

            print(f"Processing ID: {data['id']}...")
            progress.set_postfix_str(f"id={data['id']}")
            attempted_processes += 1

            try:
                # The 'Thinking' prompt for Qwen3
                prompt_text = (
                    "Analyze this meme and extract its components. "
                    "1. Provide all literal 'img_captions' describing exactly what is visually seen. "
                    "2. Provide 1 'meme_captions' explaining the underlying joke or message of the meme. "
                    "3. Give the meme a short 'title'. "
                    "4. Identify 'metaphors' where a visual element represents a broader meaning. "
                    "You MUST output exactly in JSON format."
                )
                
                response = llm.create_chat_completion(
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt_text},
                                {"type": "image_url", "image_url": {"url": data_url}}
                            ]
                        }
                    ],
                    # Force structured output using our Pydantic model
                    response_format={
                        "type": "json_object",
                        "schema": MemeAnalysis.model_json_schema(),
                    },
                    temperature=0.2
                )

                result_text = response["choices"][0]["message"]["content"]
                print(f"Raw model output:\n{result_text}\n")
                result_json = json.loads(result_text)
                
                # Save progress incrementally
                final_memecap_layout = {
                    "category": "HMD-memes", # Static
                    "img_captions": result_json.get("img_captions", []),
                    "meme_captions": result_json.get("meme_captions", []),
                    "title": result_json.get("title", ""),
                    "url": "HMD", # Placeholder or derive from your dataset
                    "img_fname": data['img'], # From your data object
                    "metaphors": result_json.get("metaphors", []),
                    "post_id": str(data['id']),
                    "raw_output": result_text, # Store raw output for debugging
                }
                
                f_out.write(json.dumps(final_memecap_layout) + "\n")
                f_out.flush()
            except Exception as e:
                print(f"Error processing {data['id']}: {e}")

            if (
                MAX_SUCCESSFUL_PROCESSES is not None
                and attempted_processes >= MAX_SUCCESSFUL_PROCESSES
            ):
                print(f"Reached test stop limit: {MAX_SUCCESSFUL_PROCESSES}")
                break

if __name__ == "__main__":
    process_memes()