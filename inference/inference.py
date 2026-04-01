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
            content_parts = [
                "Example Meme:",
                f"Title: {meta['title']}"
            ]

            if meta['meme_captions']:
                content_parts.append(f"Meaning: {meta['meme_captions'][0]}")

            if meta['img_captions']:
                content_parts.append(f"Description: {meta['img_captions'][0]}")

            # Add metaphors if present
            if meta.get('metaphors'):
                metaphor_texts = []
                for m in meta['metaphors']:
                    if isinstance(m, dict):
                        parts = []
                        if m.get("metaphor"):
                            parts.append(f"Metaphor: {m['metaphor']}")
                        if m.get("meaning"):
                            parts.append(f"Meaning: {m['meaning']}")
                        if parts:
                            metaphor_texts.append("; ".join(parts))
                if metaphor_texts:
                    content_parts.append("Metaphorical Elements: " + " | ".join(metaphor_texts))

            content = "\n".join(content_parts)

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

            # Format content for context (same as retrieve method)
            content_parts = [
                "Example Meme:",
                f"Title: {meta['title']}"
            ]

            if meta['meme_captions']:
                content_parts.append(f"Meaning: {meta['meme_captions'][0]}")

            if meta['img_captions']:
                content_parts.append(f"Description: {meta['img_captions'][0]}")

            # Add metaphors if present
            if meta.get('metaphors'):
                metaphor_texts = []
                for m in meta['metaphors']:
                    if isinstance(m, dict):
                        parts = []
                        if m.get("metaphor"):
                            parts.append(f"Metaphor: {m['metaphor']}")
                        if m.get("meaning"):
                            parts.append(f"Meaning: {m['meaning']}")
                        if parts:
                            metaphor_texts.append("; ".join(parts))
                if metaphor_texts:
                    content_parts.append("Metaphorical Elements: " + " | ".join(metaphor_texts))

            content = "\n".join(content_parts)

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
# 2. QWEN3 RERANKER
# =============================
class Qwen3Reranker:
    """
    Reranks retrieval results using Qwen3-VL to score relevance.
    Takes top-k candidates from SigLIP and returns top-k' most relevant.
    """

    def __init__(
        self,
        model_name: str = "unsloth/Qwen3-VL-2B-Instruct-unsloth-bnb-4bit",
        device: Optional[str] = None,
        initial_k: int = 10,
        final_k: int = 3
    ):
        """
        Args:
            model_name: Qwen3-VL model to use for reranking
            device: Device to run on
            initial_k: Number of candidates to retrieve from SigLIP
            final_k: Number of results to return after reranking
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.initial_k = initial_k
        self.final_k = final_k

        print(f"Loading Qwen3 reranker: {model_name}")
        self.model, self.processor = FastVisionModel.from_pretrained(
            model_name=model_name,
            load_in_4bit=False,
            dtype=torch.bfloat16,
            use_safetensors=True
        )
        FastVisionModel.for_inference(self.model)

    def score_relevance_batch(
        self,
        query_texts: List[str],
        query_image_paths: List[str],
        retrieved_docs: List[RetrievedDocument]
    ) -> List[float]:
        """
        Score relevance for multiple query-document pairs in batch.

        Args:
            query_texts: List of query texts
            query_image_paths: List of query image paths
            retrieved_docs: List of retrieved documents to score

        Returns:
            List of relevance scores (0-10)
        """
        all_prompts = []
        all_images = []

        # Build prompts and load images
        for query_text, query_image_path, doc in zip(query_texts, query_image_paths, retrieved_docs):
            prompt = f"""Compare the query meme with this retrieved example and rate relevance from 0-10.
Query text: {query_text}

Retrieved Example:
{doc.content}

Rate how similar these memes are in meaning and context (0=unrelated, 10=highly similar).
Respond with ONLY a number 0-10."""

            # Load and resize query image
            query_img = Image.open(query_image_path).convert("RGB")
            query_img.thumbnail((512, 512))
            all_images.append(query_img)

            # Prepare messages
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": query_img},
                    {"type": "text", "text": prompt}
                ]
            }]

            # Apply chat template
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            all_prompts.append(text)

        # Process all inputs at once
        inputs = self.processor(
            text=all_prompts,
            images=all_images,
            return_tensors="pt",
            padding=True
        ).to(self.device)

        # Generate scores
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=10,
                do_sample=False,
                temperature=0.0
            )

        input_len = inputs.input_ids.shape[1]
        outputs = self.processor.batch_decode(
            generated_ids[:, input_len:],
            skip_special_tokens=True
        )

        # Parse all scores
        scores = []
        for output in outputs:
            try:
                score = float(re.search(r'\d+', output.strip()).group())
                scores.append(min(max(score, 0.0), 10.0))  # Clamp to 0-10
            except:
                scores.append(5.0)  # Default middle score if parsing fails

        return scores

    def rerank(
        self,
        query_text: str,
        query_image_path: str,
        retrieved_docs: List[RetrievedDocument]
    ) -> List[RetrievedDocument]:
        """
        Rerank retrieved documents using Qwen3-VL scoring.

        Args:
            query_text: The query text
            query_image_path: Path to query image
            retrieved_docs: Initial retrieval results

        Returns:
            Reranked documents (top final_k)
        """
        if not retrieved_docs:
            return []

        # Prepare batch inputs for all documents
        query_texts = [query_text] * len(retrieved_docs)
        query_image_paths = [query_image_path] * len(retrieved_docs)

        # Score all documents in batch
        relevance_scores = self.score_relevance_batch(query_texts, query_image_paths, retrieved_docs)

        # Combine SigLIP similarity with Qwen3 relevance (weighted average)
        scored_docs = []
        for doc, relevance_score in zip(retrieved_docs, relevance_scores):
            combined_score = 0.3 * doc.score + 0.7 * (relevance_score / 10.0)

            # Create new document with updated score
            scored_docs.append(RetrievedDocument(
                content=doc.content,
                metadata=doc.metadata,
                score=combined_score
            ))

        # Sort by combined score and return top-k
        scored_docs.sort(key=lambda x: x.score, reverse=True)
        return scored_docs[:self.final_k]

    def rerank_batch(
        self,
        query_texts: List[str],
        query_image_paths: List[str],
        retrieved_docs_list: List[List[RetrievedDocument]]
    ) -> List[List[RetrievedDocument]]:
        """
        Rerank multiple sets of retrieved documents in batch.

        Args:
            query_texts: List of query texts
            query_image_paths: List of query image paths
            retrieved_docs_list: List of retrieval results for each query

        Returns:
            List of reranked documents for each query
        """
        # Flatten all query-document pairs
        all_query_texts = []
        all_query_image_paths = []
        all_docs = []
        doc_counts = []

        for query_text, query_image_path, retrieved_docs in zip(
            query_texts, query_image_paths, retrieved_docs_list
        ):
            doc_counts.append(len(retrieved_docs))
            for doc in retrieved_docs:
                all_query_texts.append(query_text)
                all_query_image_paths.append(query_image_path)
                all_docs.append(doc)

        if not all_docs:
            return [[] for _ in query_texts]

        # Score all documents in batch
        all_relevance_scores = self.score_relevance_batch(
            all_query_texts, all_query_image_paths, all_docs
        )

        # Reconstruct results for each query
        results = []
        score_idx = 0

        for count, retrieved_docs in zip(doc_counts, retrieved_docs_list):
            relevance_scores = all_relevance_scores[score_idx:score_idx + count]
            score_idx += count

            # Combine scores
            scored_docs = []
            for doc, relevance_score in zip(retrieved_docs, relevance_scores):
                combined_score = 0.3 * doc.score + 0.7 * (relevance_score / 10.0)
                scored_docs.append(RetrievedDocument(
                    content=doc.content,
                    metadata=doc.metadata,
                    score=combined_score
                ))

            # Sort and take top-k
            scored_docs.sort(key=lambda x: x.score, reverse=True)
            results.append(scored_docs[:self.final_k])

        return results


class RerankingRetriever:
    """Wrapper that combines SigLIP retrieval with Qwen3 reranking."""

    def __init__(self, base_retriever, reranker: Qwen3Reranker):
        self.base = base_retriever
        self.reranker = reranker

    def invoke(self, text: str, image_path: str) -> List[RetrievedDocument]:
        """
        Retrieve and rerank documents.

        Args:
            text: Query text
            image_path: Query image path

        Returns:
            Reranked documents
        """
        # Get initial candidates using SigLIP (more than final k)
        initial_results = self.base.invoke(text, image_path)

        # Rerank using Qwen3
        reranked_results = self.reranker.rerank(text, image_path, initial_results)

        return reranked_results

    def invoke_batch(
        self,
        texts: List[str],
        image_paths: List[str]
    ) -> List[List[RetrievedDocument]]:
        """
        Retrieve and rerank documents for multiple queries in batch.

        Args:
            texts: List of query texts
            image_paths: List of query image paths

        Returns:
            List of reranked documents for each query
        """
        # Get initial candidates for all queries
        all_initial_results = []
        for text, image_path in zip(texts, image_paths):
            initial_results = self.base.invoke(text, image_path)
            all_initial_results.append(initial_results)

        # Batch rerank using Qwen3
        reranked_results = self.reranker.rerank_batch(
            texts, image_paths, all_initial_results
        )

        return reranked_results


# =============================
# 3. CONFIGURATION
# =============================
# RAG Configuration
USE_SIGLIP_RAG = True  # Set to False to use basic text retriever
USE_RERANKING = False   # Set to True to enable Qwen3 reranking
RAG_K = 3              # Number of examples to retrieve (final k)
RERANK_INITIAL_K = 10  # Number of candidates to retrieve before reranking
RERANK_MODEL = "unsloth/Qwen3-VL-2B-Instruct-unsloth-bnb-4bit"  # Reranker model
RERANK_BATCH_SIZE = 4  # Batch size for reranking (number of queries to process together)

MODELS_TO_RUN = [
    # "unsloth/Qwen3-VL-2B-Instruct-unsloth-bnb-4bit",
    "unsloth/Qwen3-VL-2B-Thinking-unsloth-bnb-4bit",
    # "unsloth/Qwen3-VL-8B-Instruct-unsloth-bnb-4bit",
    "unsloth/Qwen3-VL-8B-Thinking-unsloth-bnb-4bit",
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
# 4. BATCH-ENABLED WRAPPER
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
# 5. UTILS & RAG
# =============================
def get_rag_retriever(use_siglip=True, k=3, use_reranking=False, rerank_initial_k=10, rerank_model=None):
    """
    Get RAG retriever - either SigLIP-based (memecap) or fallback text-based.
    Optionally wraps with Qwen3 reranker for improved relevance.

    Args:
        use_siglip: If True, use SigLIP-based retriever with memecap data
        k: Number of examples to retrieve (final k)
        use_reranking: If True, use Qwen3 reranking
        rerank_initial_k: Number of candidates to retrieve before reranking
        rerank_model: Model name for reranker

    Returns:
        Configured retriever (with or without reranking)
    """
    base_retriever = None

    if use_siglip and SIGLIP_AVAILABLE:
        try:
            if use_reranking:
                print(f"Using SigLIP retriever with Qwen3 reranking (initial_k={rerank_initial_k}, final_k={k})")
                # Get more candidates for reranking
                base_retriever = get_siglip_retriever(k=rerank_initial_k)
            else:
                print(f"Using SigLIP-based retriever with k={k}")
                base_retriever = get_siglip_retriever(k=k)
        except FileNotFoundError as e:
            print(f"SigLIP embeddings not found: {e}")
            print("Falling back to basic text retriever")

    if base_retriever is None:
        # Fallback to basic text-based retriever
        print("Using fallback text-based retriever")
        texts = ["Hate speech is defined as..."]
        device = "cuda" if torch.cuda.is_available() else "cpu"
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2", model_kwargs={'device': device})
        vectorstore = FAISS.from_texts(texts, embeddings)
        return vectorstore.as_retriever(search_kwargs={"k": k})

    # Optionally wrap with reranker
    if use_reranking:
        reranker = Qwen3Reranker(
            model_name=rerank_model or "unsloth/Qwen3-VL-2B-Instruct-unsloth-bnb-4bit",
            initial_k=rerank_initial_k,
            final_k=k
        )
        return RerankingRetriever(base_retriever, reranker)

    return base_retriever

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
# 6. MAIN INFERENCE LOOP
# =============================
def main():
    if not SAMPLES_JSONL.exists():
        print(f"Error: Could not find {SAMPLES_JSONL}")
        return

    with open(SAMPLES_JSONL, "r") as f:
        samples = [json.loads(line) for line in f if line.strip()]

    print("\n>>> Phase 1: Pre-calculating RAG context and resolving paths...")
    sys.stdout.flush() # Force print to show up

    retriever = get_rag_retriever(
        use_siglip=USE_SIGLIP_RAG,
        k=RAG_K,
        use_reranking=USE_RERANKING,
        rerank_initial_k=RERANK_INITIAL_K,
        rerank_model=RERANK_MODEL
    )

    # Pre-resolve all image paths first
    for rec in samples:
        raw_path = str(rec.get("img", "")).replace("\\", "/")
        resolved_img_path = str((SAMPLES_JSONL.parent / raw_path).resolve())
        rec["pre_img_path"] = resolved_img_path

    # Batch pre-calculation if using reranking, otherwise sequential
    if USE_RERANKING and isinstance(retriever, RerankingRetriever):
        print(f"Using batch reranking with batch_size={RERANK_BATCH_SIZE}")
        # Process in batches for efficient reranking
        for i in tqdm(range(0, len(samples), RERANK_BATCH_SIZE), desc="Pre-calculating (batch)", file=sys.stdout, mininterval=0, miniters=1):
            batch_samples = samples[i:i + RERANK_BATCH_SIZE]

            # Prepare batch inputs
            batch_texts = [rec.get("text", "") for rec in batch_samples]
            batch_image_paths = [rec["pre_img_path"] for rec in batch_samples]

            # Batch retrieve and rerank
            batch_results = retriever.invoke_batch(batch_texts, batch_image_paths)

            # Store results
            for rec, context_docs in zip(batch_samples, batch_results):
                if hasattr(context_docs[0], 'content'):
                    rec["pre_context"] = "\n\n".join([doc.content for doc in context_docs])
                else:
                    rec["pre_context"] = "\n\n".join([doc.page_content for doc in context_docs])

            sys.stdout.flush()
    else:
        # Sequential processing (no batch reranking)
        for rec in tqdm(samples, desc="Pre-calculating", file=sys.stdout, mininterval=0, miniters=1):
            text_query = rec.get("text", "")
            context_docs = retriever.invoke(text_query, rec["pre_img_path"])

            # Handle different retriever return types
            if hasattr(context_docs[0], 'content'):
                # CLIP retriever returns RetrievedDocument with .content
                rec["pre_context"] = "\n\n".join([doc.content for doc in context_docs])
            else:
                # Basic retriever returns documents with .page_content
                rec["pre_context"] = "\n\n".join([doc.page_content for doc in context_docs])

            sys.stdout.flush()

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