import json
import os
from PIL import Image
from datasets import Dataset

# Configuration
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

IMAGE_FOLDER = os.path.join(PROJECT_ROOT, "memecap-data", "memes")
INPUT_JSON = os.path.join(PROJECT_ROOT, "memecap-data", "memes-trainval.json")
OUTPUT_PARQUET = os.path.join(SCRIPT_DIR, "memecap_train.parquet")

def gen():
    """Generator function to yield rows one by one"""
    if not os.path.exists(INPUT_JSON):
        raise FileNotFoundError(f"Input JSON not found: {INPUT_JSON}")

    with open(INPUT_JSON, 'r') as f:
        data = json.load(f)

    for entry in data:
        img_path = os.path.join(IMAGE_FOLDER, entry['img_fname'])
        
        if not os.path.exists(img_path):
            continue

        try:
            # .convert("RGB") is essential for Parquet consistency
            img = Image.open(img_path).convert("RGB")
            
            output_data = {
                "title": entry.get("title"),
                "metaphors": entry.get("metaphors", []),
                "interpretation": entry.get("meme_captions", [""])[0]
            }
            json_output = json.dumps(output_data, indent=2)

            # Yielding prevents the 'Killed' error by not storing the whole list
            yield {
                "image": img,
                "text": f"{json_output}"
            }
        except Exception as e:
            print(f"Error processing {entry['img_fname']}: {e}")
            continue

# Create dataset from generator
print("Starting stream to Parquet...")
dataset = Dataset.from_generator(gen)

# Save to Parquet
dataset.to_parquet(OUTPUT_PARQUET)
print(f"✅ Success! Saved to {OUTPUT_PARQUET}")