import json
import re
import os
import sys  # Added for stdout flushing
import torch
import numpy as np
import pickle
from tqdm import tqdm
from pathlib import Path
from typing import Any, List, Optional, Dict
from PIL import Image
from pydantic import BaseModel, Field
from dataclasses import dataclass

# Patch for missing torch int types
if not hasattr(torch, 'int1'):
    for i in range(1, 8):
        setattr(torch, f'int{i}', torch.int8)

from unsloth import FastVisionModel
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, AIMessage
from langchain_core.outputs import ChatResult, ChatGeneration
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# SigLIP imports
try:
    from transformers import SiglipProcessor, SiglipModel
    SIGLIP_AVAILABLE = True
except Exception as e:
    print(f"Warning: SigLIP not available: {e}")
    SIGLIP_AVAILABLE = False


# =============================
# 1. SIGLIP RETRIEVER
# =============================
@dataclass
class RetrievedDocument:
    """A retrieved document with content and metadata."""
    content: str
    metadata: Dict
    score: float

class SigLIPRetriever:
    """
    SigLIP-based retriever that searches over multimodal memecap embeddings.
    Supports both text and image queries.
    """

    def __init__(
        self,
        embeddings_path: str,
        metadata_path: str,
        model_name: str = "google/siglip-base-patch16-224",
        device: Optional[str] = None
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Load precomputed embeddings
        print(f"Loading SigLIP embeddings from {embeddings_path}")
        self.embeddings = np.load(embeddings_path)

        print(f"Loading metadata from {metadata_path}")
        with open(metadata_path, "rb") as f:
            self.metadata = pickle.load(f)

        print(f"Loaded {len(self.embeddings)} embeddings")

        # Load SigLIP model for query encoding
        print(f"Loading SigLIP model: {model_name}")
        self.model = SiglipModel.from_pretrained(model_name).to(self.device)
        self.processor = SiglipProcessor.from_pretrained(model_name)

        # Normalize embeddings for cosine similarity
        self.embeddings = self.embeddings / np.linalg.norm(
            self.embeddings, axis=1, keepdims=True
        )

    def encode_text_query(self, query):
        """
        Encode a text query using SigLIP.
        Uses get_text_features() to get projected embeddings and fills both halves
        to enable matching against both image and text components in the database.
        """
        inputs = self.processor(
            text=query,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=64
        ).to(self.device)

        with torch.no_grad():
            # Use get_text_features for text-only encoding
            text_feat = self.model.get_text_features(**inputs)
            # Normalize the features
            text_feat = text_feat / text_feat.norm(p=2, dim=-1, keepdim=True)
            text_np = text_feat.cpu().numpy().flatten()

        # Fill BOTH halves to ensure strong matching across the multimodal DB
        # This allows text queries to match against both visual and semantic aspects
        combined = np.concatenate([text_np, text_np])
        # Normalize the combined vector to match stored embeddings
        return combined / np.linalg.norm(combined)

    def encode_image_query(self, image_path: str) -> np.ndarray:
        """
        Encode an image query using SigLIP.
        Uses get_image_features() to get projected embeddings and fills both halves
        to enable matching against both image and text components in the database.
        """
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=[image], return_tensors="pt").to(self.device)

        with torch.no_grad():
            # Use get_image_features for image-only encoding
            image_feat = self.model.get_image_features(**inputs)
            # Normalize the features
            image_feat = image_feat / image_feat.norm(p=2, dim=-1, keepdim=True)
            image_np = image_feat.cpu().numpy().flatten()

        # Filling both halves allows an image to match
        # against BOTH the stored image and the stored text description
        combined = np.concatenate([image_np, image_np])
        # Normalize the combined vector to match stored embeddings
        return combined / np.linalg.norm(combined)

    def retrieve(
        self,
        query: str,
        k: int = 3,
        is_image: bool = False
    ) -> List[RetrievedDocument]:
        """
        Retrieve top-k most relevant documents.

        Args:
            query: Text query or path to image
            k: Number of documents to retrieve
            is_image: Whether query is an image path

        Returns:
            List of RetrievedDocument objects
        """
        # Encode query
        if is_image:
            query_embedding = self.encode_image_query(query)
        else:
            query_embedding = self.encode_text_query(query)

        # Compute cosine similarities
        query_embedding_1d = query_embedding.flatten()
        similarities = np.dot(self.embeddings, query_embedding_1d)
        # Get top-k indices
        top_k_indices = np.argsort(similarities)[-k:][::-1]

        # Build results
        results = []
        for idx in top_k_indices:
            meta = self.metadata[idx]
            score = float(similarities[idx])

            # Format content for context
            content = f"""
Example Meme:
Title: {meta['title']}
Meaning: {meta['meme_captions'][0] if meta['meme_captions'] else 'N/A'}
Description: {meta['img_captions'][0] if meta['img_captions'] else 'N/A'}
""".strip()

            results.append(RetrievedDocument(
                content=content,
                metadata=meta,
                score=score
            ))

        return results

    def retrieve_multimodal(self, text_query, image_path, k=3):
        """
        Retrieve using both text and image as a multimodal query.
        Combines normalized image and text features to match the database format.
        """
        # 1. Load the image
        image = Image.open(image_path).convert("RGB")
        # 2. Process BOTH modalities with truncation for the text
        inputs = self.processor(
            text=text_query,
            images=image,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=64
        ).to(self.device)
        # 3. Extract both features and normalize them
        with torch.no_grad():
            outputs = self.model(**inputs)
            image_features = outputs.image_embeds
            text_features = outputs.text_embeds
            # Normalize each modality separately
            image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
            # 4. Concatenate them to match your 1536-dimensional index
            combined_features = torch.cat((image_features, text_features), dim=1)
        # 5. Flatten to a 1D array and normalize the combined vector
        query_embedding_1d = combined_features.cpu().numpy().flatten()
        query_embedding_1d = query_embedding_1d / np.linalg.norm(query_embedding_1d)
        # 6. Compute cosine similarities
        similarities = np.dot(self.embeddings, query_embedding_1d)
        top_k_indices = np.argsort(similarities)[-k:][::-1]
        results = []
        for idx in top_k_indices:
            meta = self.metadata[idx]
            score = float(similarities[idx])
            content = f"""
Example Meme:
Title: {meta['title']}
Meaning: {meta['meme_captions'][0] if meta['meme_captions'] else 'N/A'}
Description: {meta['img_captions'][0] if meta['img_captions'] else 'N/A'}
""".strip()
            results.append(RetrievedDocument(
                content=content,
                metadata=meta,
                score=score
            ))
        return results

    def invoke(self, query: str, **kwargs) -> List[RetrievedDocument]:
        """
        LangChain-compatible invoke method.
        Retrieves relevant documents for a text query.
        """
        k = kwargs.get("k", 3)
        return self.retrieve(query, k=k, is_image=False)


def get_siglip_retriever(k: int = 3):
    """
    Factory function to create a SigLIP retriever.
    Returns a retriever configured for the memecap dataset.
    """
    base_path = Path(__file__).resolve().parent
    embeddings_path = base_path / "rag_data" / "siglip_embeddings.npy"
    metadata_path = base_path / "rag_data" / "siglip_metadata.pkl"

    if not embeddings_path.exists():
        raise FileNotFoundError(
            f"SigLIP embeddings not found at {embeddings_path}. "
            "Please run build_clip_rag.py first."
        )

    retriever = SigLIPRetriever(
        embeddings_path=str(embeddings_path),
        metadata_path=str(metadata_path)
    )

    # Wrap to use k by default

    class ConfiguredRetriever:
        def __init__(self, base_retriever, default_k):
            self.base = base_retriever
            self.k = default_k

        def invoke(self, text, image_path):
            # If both text and image_path are provided, use multimodal
            # Use multimodal if image exists (even with empty text for Facebook dataset)
            if image_path:
                # Default to empty string if text is None
                text = text if text is not None else ""
                return self.base.retrieve_multimodal(text, image_path, k=self.k)
            else:
                return self.base.retrieve(text, k=self.k, is_image=False)

    return ConfiguredRetriever(retriever, k)

# =============================
# 2. CONFIGURATION
# =============================
# RAG Configuration
USE_SIGLIP_RAG = True  # Set to False to use basic text retriever
RAG_K = 3  # Number of examples to retrieve

MODELS_TO_RUN = [
    "unsloth/Qwen3-VL-2B-Instruct-unsloth-bnb-4bit",
    # "unsloth/Qwen3-VL-2B-Thinking-unsloth-bnb-4bit",
    "unsloth/Qwen3-VL-8B-Instruct-unsloth-bnb-4bit",
    # "unsloth/Qwen3-VL-8B-Thinking-unsloth-bnb-4bit",
]

BASE = Path(__file__).resolve().parents[1]
SAMPLES_JSONL = BASE / "datapreparation" / "output" / "facebook-samples.jsonl"
OUTPUT_DIR = BASE / "datapreparation" / "output" / "results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Safe batch size for 40GB card with resized images
BATCH_SIZE = 8

class HateSpeechPrediction(BaseModel):
    label: int = Field(description="0 for non-hateful, 1 for hateful", ge=0, le=1)

# =============================
# 3. BATCH-ENABLED WRAPPER
# =============================
class ChatQwen3(BaseChatModel):
    model: Any
    processor: Any
    is_thinking: bool = False

    def _generate(self, messages: List[BaseMessage], **kwargs: Any) -> ChatResult:
        results = self.generate_batch([messages], **kwargs)
        return results[0]

    def generate_batch(self, messages_list: List[List[BaseMessage]], **kwargs: Any) -> List[ChatResult]:
        all_prompts = []
        all_images = []
        role_map = {"human": "user", "ai": "assistant", "system": "system"}

        for messages in messages_list:
            images = []
            qwen_messages = []
            for msg in messages:
                role = role_map.get(msg.type, msg.type)
                content_list = []
                if isinstance(msg.content, list):
                    for item in msg.content:
                        if item.get("type") == "image":
                            img_path = item.get("image")
                            pil_img = Image.open(img_path).convert("RGB")
                            
                            # MEMORY FIX: Resize to max 512x512 to prevent OOM
                            pil_img.thumbnail((512, 512))
                            
                            images.append(pil_img)
                            content_list.append({"type": "image", "image": pil_img})
                        else:
                            content_list.append(item)
                    qwen_messages.append({"role": role, "content": content_list})
                else:
                    qwen_messages.append({"role": role, "content": [{"type": "text", "text": msg.content}]})
            
            prompt_text = self.processor.apply_chat_template(qwen_messages, tokenize=False, add_generation_prompt=True)
            all_prompts.append(prompt_text)
            all_images.extend(images)

        inputs = self.processor(
            text=all_prompts, 
            images=all_images if all_images else None, 
            return_tensors="pt",
            padding=True
        ).to(self.model.device)

        max_tokens = 1024 if self.is_thinking else 512

        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,
                repetition_penalty=1.2,
                use_cache=True
            )

        input_len = inputs.input_ids.shape[1]
        decoded_outputs = self.processor.batch_decode(
            generated_ids[:, input_len:], 
            skip_special_tokens=True
        )

        return [ChatResult(generations=[ChatGeneration(message=AIMessage(content=out))]) for out in decoded_outputs]

    @property
    def _llm_type(self) -> str:
        return "qwen3-vl-unsloth"

# =============================
# 4. UTILS & RAG
# =============================
def get_rag_retriever(use_siglip=True, k=3):
    """
    Get RAG retriever - either SigLIP-based (memecap) or fallback text-based.

    Args:
        use_siglip: If True, use SigLIP-based retriever with memecap data
        k: Number of examples to retrieve
    """
    if use_siglip and SIGLIP_AVAILABLE:
        try:
            print(f"Using SigLIP-based retriever with k={k}")
            return get_siglip_retriever(k=k)
        except FileNotFoundError as e:
            print(f"SigLIP embeddings not found: {e}")
            print("Falling back to basic text retriever")

    # Fallback to basic text-based retriever
    print("Using fallback text-based retriever")
    texts = ["Hate speech is defined as..."]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2", model_kwargs={'device': device})
    vectorstore = FAISS.from_texts(texts, embeddings)
    return vectorstore.as_retriever(search_kwargs={"k": k})

def extract_output(ai_message, is_thinking):
    raw = ai_message.content
    reasoning = "N/A (Instruct Model)"

    if is_thinking:
        thought_match = re.search(r'<think>(.*?)</think>', raw, re.DOTALL)
        reasoning = thought_match.group(1).strip() if thought_match else "Thinking truncated."
        text_to_parse = raw.split("</think>")[-1]
    else:
        text_to_parse = raw

    json_match = re.search(r'\{.*\}', text_to_parse, re.DOTALL)
    label = 0
    if json_match:
        try:
            label = json.loads(json_match.group(0)).get("label", 0)
        except:
            pass
    return {"label": label, "reasoning": reasoning, "raw_output": raw}

# =============================
# 5. MAIN INFERENCE LOOP
# =============================
def main():
    if not SAMPLES_JSONL.exists():
        print(f"Error: Could not find {SAMPLES_JSONL}")
        return

    with open(SAMPLES_JSONL, "r") as f:
        samples = [json.loads(line) for line in f if line.strip()]

    print("\n>>> Phase 1: Pre-calculating RAG context and resolving paths...")
    sys.stdout.flush() # Force print to show up

    retriever = get_rag_retriever(use_siglip=USE_SIGLIP_RAG, k=RAG_K)

    # Updated TQDM for real-time output
    for rec in tqdm(samples, desc="Pre-calculating", file=sys.stdout, mininterval=0, miniters=1):
        # Resolve image path first
        raw_path = str(rec.get("img", "")).replace("\\", "/")
        resolved_img_path = str((SAMPLES_JSONL.parent / raw_path).resolve())
        rec["pre_img_path"] = resolved_img_path

        # Get retrieval inputs (both text and image for multimodal RAG)
        text_query = rec.get("text", "")
        context_docs = retriever.invoke(text_query, resolved_img_path)

        # Handle different retriever return types
        if hasattr(context_docs[0], 'content'):
            # CLIP retriever returns RetrievedDocument with .content
            rec["pre_context"] = "\n\n".join([doc.content for doc in context_docs])
        else:
            # Basic retriever returns documents with .page_content
            rec["pre_context"] = "\n\n".join([doc.page_content for doc in context_docs])

        sys.stdout.flush() # Force flush

    parser = PydanticOutputParser(pydantic_object=HateSpeechPrediction)
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", (
            "You are an objective content moderator. Use the provided examples of similar memes to help classify this meme.\n"
            "These examples show how other memes convey meaning through metaphors and visual elements.\n"
            "0 = Non-hateful, 1 = Hateful.\n\n"
            "Example Context:\n{context}\n\n{format_instructions}"
        )),
        ("user", [
            {"type": "text", "text": "Analyze this: {text}"},
            {"type": "image", "image": "{image}"}
        ])
    ])

    for model_path in MODELS_TO_RUN:
        model_id = model_path.split("/")[-1]
        output_file = OUTPUT_DIR / f"preds_{model_id}.jsonl"
        
        print(f"\n>>> Loading Model: {model_id}")
        sys.stdout.flush()

        model, processor = FastVisionModel.from_pretrained(
            model_name=model_path,
            load_in_4bit=False,   # TURN THIS OFF: 4-bit is slowing down your generation
            dtype=torch.bfloat16, # TURN THIS ON: Native A100 speed format
            use_safetensors=True
        )
        FastVisionModel.for_inference(model)
        llm = ChatQwen3(model=model, processor=processor, is_thinking=("Thinking" in model_path))

        with open(output_file, "w") as out_f:
            # Updated TQDM for real-time output
            for i in tqdm(range(0, len(samples), BATCH_SIZE), desc=f"A100 Batch Inference", file=sys.stdout, mininterval=0, miniters=1):
                batch_samples = samples[i : i + BATCH_SIZE]
                batch_messages = []

                for s in batch_samples:
                    msgs = prompt_template.format_messages(
                        context=s["pre_context"],
                        text=s.get("text", ""),
                        image=s["pre_img_path"],
                        format_instructions=parser.get_format_instructions()
                    )
                    batch_messages.append(msgs)

                try:
                    results = llm.generate_batch(batch_messages)
                    for s, res in zip(batch_samples, results):
                        data = extract_output(res.generations[0].message, llm.is_thinking)
                        out_f.write(json.dumps({"id": s["id"], **data}) + "\n")
                    out_f.flush()
                except Exception as e:
                    print(f"Batch Error: {e}")
                
                # Force the terminal log to update
                sys.stdout.flush()

        del model
        del processor
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()