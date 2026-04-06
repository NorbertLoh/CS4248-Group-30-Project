"""
Build SigLIP-based RAG vector store from MemeCap test data and Facebook Hateful Memes data.
This script creates multimodal embeddings using SigLIP for both images and text.
"""
import json
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import pickle
import argparse
import glob

# Patch for missing torch int types
if not hasattr(torch, 'int1'):
    for i in range(1, 8):
        setattr(torch, f'int{i}', torch.int8)

from transformers.models.siglip import SiglipProcessor, SiglipModel

# Configuration
BASE = Path(__file__).resolve().parents[1]
MEMECAP_TEST = BASE / "memecap-data" / "memes-test.json"
MEMECAP_IMAGES = BASE / "memecap-data" / "memes"
FACEBOOK_DATA_DIR = BASE / "facebook-data"
HATEFUL_CAPTIONING_DIR = BASE / "hateful-captioning"
OUTPUT_DIR = BASE / "inference" / "rag_data"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SIGLIP_MODEL_NAME = "google/siglip-base-patch16-224"

def load_memecap_data():
    """Load memecap test data."""
    with open(MEMECAP_TEST, "r") as f:
        data = json.load(f)
    print(f"Loaded {len(data)} memecap test samples")
    for item in data:
        item["source"] = "memecap"
    return data

def load_facebook_data():
    """Load facebook data from hateful-captioning jsonl output files."""
    data = []
    glob_pattern = str(HATEFUL_CAPTIONING_DIR / "captions_output*.jsonl")
    for file_path in glob.glob(glob_pattern):
        with open(file_path, "r") as f:
            for line in f:
                item = json.loads(line)
                item["source"] = "facebook"
                data.append(item)
    print(f"Loaded {len(data)} facebook dataset samples")
    return data

def create_siglip_embeddings(data, model, processor, device, version):
    """
    Create SigLIP embeddings for multimodal data.
    Each entry will have both image and text embeddings.
    """
    embeddings = []
    metadata = []

    print(f"Creating SigLIP embeddings for version: {version}...")
    for item in tqdm(data):
        img_fname = item.get("img_fname", "")
        
        if item.get("source") == "facebook":
            # For facebook data, img_fname is formatted like 'img/42953.png'
            img_path = FACEBOOK_DATA_DIR / img_fname
        else:
            img_path = MEMECAP_IMAGES / img_fname

        if not img_path.exists():
            print(f"Warning: Image not found: {img_path}")
            continue

        try:
            # Load and process image
            image = Image.open(img_path).convert("RGB")

            combined_text = ""
            
            if version != "no_context":
                # Create combined text representation
                # Combine title, meme captions, img captions, and metaphors for rich context
                text_parts = []
                if item.get("title"):
                    text_parts.append(f"Title: {item['title']}")

                if item.get("meme_captions"):
                    text_parts.append("Meaning: " + " ".join(item["meme_captions"]))

                if item.get("img_captions"):
                    text_parts.append("Description: " + " ".join(item["img_captions"]))

                # Add metaphors if present
                if item.get("metaphors") and isinstance(item["metaphors"], list) and len(item["metaphors"]) > 0:
                    metaphor_texts = []
                    for m in item["metaphors"]:
                        # Some entries may only have 'meaning', some may have 'metaphor' and 'meaning'
                        if isinstance(m, dict):
                            parts = []
                            if m.get("metaphor"):
                                parts.append(f"Metaphor: {m['metaphor']}")
                            if m.get("meaning"):
                                parts.append(f"Meaning: {m['meaning']}")
                            if parts:
                                metaphor_texts.append("; ".join(parts))
                    if metaphor_texts:
                        text_parts.append("Metaphors: " + " | ".join(metaphor_texts))

                combined_text = " | ".join(text_parts)

            # Process with SigLIP
            inputs = processor(
                text=[combined_text] if combined_text else [""], # Empty string for no context
                images=[image],
                return_tensors="pt",
                padding="max_length",
                truncation=True,
            ).to(device)

            with torch.no_grad():
                outputs = model(**inputs)

                # Get normalized embeddings
                image_embedding = outputs.image_embeds[0].cpu().numpy()
                text_embedding = outputs.text_embeds[0].cpu().numpy()

                # Combine image and text embeddings (concatenate)
                combined_embedding = np.concatenate([image_embedding, text_embedding])

            embeddings.append(combined_embedding)

            # Store metadata for retrieval
            metadata.append({
                "post_id": item.get("post_id", "unknown"),
                "img_fname": img_fname,
                "title": item.get("title", ""),
                "meme_captions": item.get("meme_captions", []),
                "img_captions": item.get("img_captions", []),
                "metaphors": item.get("metaphors", []),
                "combined_text": combined_text,
                "img_path": str(img_path),
                "source": item.get("source", "unknown")
            })

        except Exception as e:
            print(f"Error processing {img_fname}: {e}")
            continue

    return np.array(embeddings), metadata

def save_embeddings(embeddings, metadata, version):
    """Save embeddings and metadata to disk."""
    embedding_file = OUTPUT_DIR / f"siglip_embeddings_{version}.npy"
    metadata_file = OUTPUT_DIR / f"siglip_metadata_{version}.pkl"

    np.save(embedding_file, embeddings)

    with open(metadata_file, "wb") as f:
        pickle.dump(metadata, f)

    print(f"Saved {len(embeddings)} embeddings to {embedding_file}")
    print(f"Saved metadata to {metadata_file}")

def main():
    parser = argparse.ArgumentParser(description="Build SigLIP embeddings for RAG")
    parser.add_argument(
        "--version", 
        type=str, 
        choices=["no_context", "memecap_only", "memecap_facebook", "all"],
        default="all",
        help="Which version of embeddings to build"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Building SigLIP-based RAG from Data")
    print("=" * 60)

    # Load SigLIP model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nLoading SigLIP model: {SIGLIP_MODEL_NAME} on {device}")

    model = SiglipModel.from_pretrained(SIGLIP_MODEL_NAME).to(device)
    processor = SiglipProcessor.from_pretrained(SIGLIP_MODEL_NAME)
    
    versions_to_run = ["no_context", "memecap_only", "memecap_facebook"] if args.version == "all" else [args.version]
    
    memecap_data = load_memecap_data()
    facebook_data = None # Lazy load
    
    for version in versions_to_run:
        data = memecap_data.copy()
        
        if version == "memecap_facebook":
            if facebook_data is None:
                facebook_data = load_facebook_data()
            data.extend(facebook_data)
            
        embeddings, metadata = create_siglip_embeddings(data, model, processor, device, version)
        save_embeddings(embeddings, metadata, version)

    print("\n" + "=" * 60)
    print(f"Successfully finished processing versions: {versions_to_run}")
    print("=" * 60)

if __name__ == "__main__":
    main()
