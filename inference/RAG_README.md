# CLIP-based RAG Pipeline for Meme Analysis

This directory contains a Retrieval-Augmented Generation (RAG) pipeline that uses CLIP embeddings from the MemeCap test dataset to provide relevant meme examples during inference.

## Overview

The RAG pipeline enhances meme classification by:
1. **Embedding Creation**: Using CLIP to create multimodal embeddings of memes (both image and text)
2. **Retrieval**: Finding semantically similar meme examples during inference
3. **Context**: Providing these examples to the VLM to improve understanding

## Setup

### 1. Build CLIP Embeddings

First, generate CLIP embeddings from the MemeCap test dataset:

```bash
cd inference
python build_clip_rag.py
```

This will:
- Load all memes from `memecap-data/memes-test.json`
- Process each meme's image and text with CLIP
- Save embeddings to `inference/rag_data/clip_embeddings.npy`
- Save metadata to `inference/rag_data/clip_metadata.pkl`

**Time**: ~5-10 minutes with GPU (depends on test set size)

### 2. Run Inference with CLIP RAG

The main inference script will automatically use CLIP-based RAG:

```bash
python inference.py
```

## Configuration

Edit `inference/inference.py` to configure RAG behavior:

```python
# RAG Configuration
USE_CLIP_RAG = True   # Set to False to disable CLIP retrieval
RAG_K = 3            # Number of similar examples to retrieve
```

## How It Works

### Embedding Process

For each meme in the test set:
1. **Image Processing**: Image is encoded with CLIP vision encoder
2. **Text Processing**: Combined text (title + captions + descriptions) encoded with CLIP text encoder
3. **Combined Embedding**: Both embeddings are concatenated for multimodal representation

### Retrieval Process

During inference:
1. Query text is encoded with CLIP
2. Cosine similarity computed against all test embeddings
3. Top-K most similar memes are retrieved
4. Their metadata (title, meaning, description) is formatted as context
5. Context is injected into the system prompt

### Example Context Format

```
Example Meme:
Title: He did it
Meaning: Husband feels great after having their wife fall in love with him again after getting amnesia.
Description: A woman shows off her engagement ring which Thor approves of.

Example Meme:
Title: One apple a year keeps your wallet empty
Meaning: Meme poster is not interested in paying for a new phone just for an added camera.
Description: cartoon character is very surprised by what they are seeing
```

## Files

- **`build_clip_rag.py`**: Script to build CLIP embeddings from MemeCap data
- **`clip_retriever.py`**: CLIPRetriever class for semantic search
- **`inference.py`**: Main inference script (updated to use CLIP RAG)
- **`rag_data/`**: Directory containing embeddings and metadata (created after running build script)

## Architecture

```
┌─────────────────────┐
│  MemeCap Test Data  │
│  (images + text)    │
└──────────┬──────────┘
           │
           ▼
    ┌──────────────┐
    │ CLIP Encoder │
    └──────┬───────┘
           │
           ▼
  ┌─────────────────┐
  │   Embeddings    │
  │   + Metadata    │
  └────────┬────────┘
           │
           ▼
  ┌─────────────────────┐
  │  Inference Query    │
  └────────┬────────────┘
           │
           ▼
  ┌─────────────────────┐
  │  Semantic Search    │
  │  (Cosine Similarity)│
  └────────┬────────────┘
           │
           ▼
  ┌─────────────────────┐
  │  Retrieved Context  │
  │  (Top-K Examples)   │
  └────────┬────────────┘
           │
           ▼
  ┌─────────────────────┐
  │   VLM Inference     │
  │  (with examples)    │
  └─────────────────────┘
```

## Requirements

- `transformers` (for CLIP model)
- `torch`
- `Pillow`
- `numpy`
- `tqdm`

All requirements are in the main `requirements.txt`.

## Troubleshooting

### "CLIP embeddings not found"
Run `python build_clip_rag.py` first to generate embeddings.

### Out of Memory
- Reduce batch size in `build_clip_rag.py`
- Use CPU instead of GPU: Set `device = "cpu"`

### Missing Images
Ensure `memecap-data/memes/` contains all image files referenced in `memes-test.json`.

## Performance Notes

- **CLIP Model**: Uses `openai/clip-vit-base-patch32` (faster, lighter)
- **Embedding Size**: 1024 dimensions (512 image + 512 text)
- **Search**: Fast cosine similarity with NumPy
- **Storage**: ~500KB for ~1000 embeddings

## Future Improvements

- [ ] Use larger CLIP model (ViT-L/14) for better embeddings
- [ ] Add image-to-image retrieval during inference
- [ ] Implement re-ranking with cross-attention
- [ ] Cache embeddings for training set too
- [ ] Add FAISS index for faster search on large datasets
