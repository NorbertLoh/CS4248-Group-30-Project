import json
import os
import re
import torch
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from collections import Counter
from enum import Enum
from pydantic import BaseModel, Field
from PIL import Image
from tqdm import tqdm
import gc
from vllm import LLM, SamplingParams
from vllm.sampling_params import StructuredOutputsParams

from transformers import AutoModel, AutoTokenizer
import torch.nn.functional as F

# =============================
# 1. PYDANTIC MODELS
# =============================

class VisualEvidenceItem(BaseModel):
    metaphor: str = Field(description="Literal visual element(s) in the meme image")
    meaning: str = Field(description="Objective visible description of what is happening; no social interpretation")

class MemeRationales(BaseModel):
    visual_evidence: List[VisualEvidenceItem] = Field(
        description="List of objective visual evidence entries in metaphor/meaning structure"
    )
    target_group: str = Field(
        description="The protected group targeted by the meme, if any. Use 'none' if no protected group is targeted. Examples: 'Black people', 'Muslims', 'Jews', 'women', 'LGBTQ+', 'immigrants', 'disabled people', 'none'"
    )
    attack_type: str = Field(
        description="The type of attack used, if any. Use 'none' if no attack. Examples: 'stereotype', 'dehumanization', 'slur', 'incitement_to_violence', 'coded_metaphor', 'trivialization_of_atrocity', 'none'"
    )
    rationale_a: str = Field(description="Prosecution: Why the meme is hateful based on retrieved metaphor/meaning context and OCR text")
    rationale_b: str = Field(description="Defense: Why the meme is benign or neutral (dark humor, literal, etc.)")

class JudgeChoice(BaseModel):
    judge_reasoning: str = Field(description="Why the chosen rationale better matches the policy")
    label: int = Field(description="Binary label: 1 for hateful, 0 for benign")

# =============================
# CHANGE 1: Robust multi-strategy parser with hateful-biased default
# =============================

def parse_thinking_output(output_text: str, knn_fallback_label: Optional[int] = None, knn_confidence: float = 0.0) -> dict:
    """Extracts JSON and label from an output string containing <think> blocks.
    
    Uses 4 extraction strategies in order:
    1. JSON from after </think>
    2. Aggressive regex for label patterns
    3. Keyword scanning for 'hateful'/'benign' 
    4. Default to kNN vote (if high confidence) or BENIGN (label=0) (precision-biased)
    """
    # Isolate the text after the thinking block
    if "</think>" in output_text:
        text_to_parse = output_text.split("</think>")[-1].strip()
        think_text = output_text.split("</think>")[0]
    else:
        text_to_parse = output_text.strip()
        think_text = ""
    
    # ---- Strategy 1: Extract JSON ----
    match = re.search(r'\{.*\}', text_to_parse, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group())
            raw_label = data.get("label", None)
            reasoning = data.get("judge_reasoning", "Parsed from JSON")
            
            if raw_label is not None:
                if str(raw_label).strip().upper() in {"1", "A", "TRUE", "HATEFUL"}:
                    return {"label": 1, "judge_choice": "A", "judge_reasoning": reasoning}
                elif str(raw_label).strip().upper() in {"0", "B", "FALSE", "BENIGN"}:
                    return {"label": 0, "judge_choice": "B", "judge_reasoning": reasoning}
        except json.JSONDecodeError:
            pass

    # ---- Strategy 2: Regex for label patterns in post-think text ----
    combined_text = text_to_parse + " " + think_text
    
    # Check for explicit label=1 indicators
    hateful_patterns = [
        r'"label"\s*:\s*1',
        r'"chosen_rationale"\s*:\s*"A"',
        r'"judge_choice"\s*:\s*"A"',
        r'\blabel\s*[:=]\s*1\b',
        r'\bhateful\s*\(1\)',
        r'\bchoose\s+1\b',
        r'\bRationale\s+A\s+is\s+better\b',
        r'\bProsecution\s+.*\b(?:succeed|correct|right|convincing|stronger)\b',
    ]
    
    benign_patterns = [
        r'"label"\s*:\s*0',
        r'"chosen_rationale"\s*:\s*"B"',
        r'"judge_choice"\s*:\s*"B"',
        r'\blabel\s*[:=]\s*0\b',
        r'\bbenign\s*\(0\)',
        r'\bchoose\s+0\b',
        r'\bRationale\s+B\s+is\s+better\b',
        r'\bDefense\s+.*\b(?:succeed|correct|right|convincing|stronger)\b',
    ]
    
    hateful_score = sum(1 for p in hateful_patterns if re.search(p, combined_text, re.IGNORECASE))
    benign_score = sum(1 for p in benign_patterns if re.search(p, combined_text, re.IGNORECASE))
    
    if hateful_score > benign_score:
        return {"label": 1, "judge_choice": "A", "judge_reasoning": f"REGEX MULTI-STRATEGY (hateful={hateful_score}, benign={benign_score})"}
    elif benign_score > hateful_score:
        return {"label": 0, "judge_choice": "B", "judge_reasoning": f"REGEX MULTI-STRATEGY (hateful={hateful_score}, benign={benign_score})"}
    
    # ---- Strategy 3: Keyword dominance in the thinking trace ----
    hateful_keywords = len(re.findall(r'\b(?:hateful|harmful|stereotype|dehumaniz|slur|offensive|discriminat|bigot)\b', think_text, re.IGNORECASE))
    benign_keywords = len(re.findall(r'\b(?:benign|innocent|harmless|satire|self-deprecat|literal|non-protected)\b', think_text, re.IGNORECASE))
    
    if hateful_keywords > benign_keywords + 3: # Increased delta for precision
        return {"label": 1, "judge_choice": "A", "judge_reasoning": f"KEYWORD SCAN (hateful_kw={hateful_keywords}, benign_kw={benign_keywords})"}
    elif benign_keywords > hateful_keywords + 2:
        return {"label": 0, "judge_choice": "B", "judge_reasoning": f"KEYWORD SCAN (hateful_kw={hateful_keywords}, benign_kw={benign_keywords})"}
    
    # ---- Strategy 4: PRECISION-BIASED FALLBACK ----
    if knn_fallback_label is not None and knn_confidence >= 0.8:
        return {"label": knn_fallback_label, "judge_choice": "A" if knn_fallback_label == 1 else "B", 
                "judge_reasoning": f"AMBIGUOUS FALLBACK → kNN HIGH CONfidence ({knn_confidence})"}
    
    return {"label": 0, "judge_choice": "B", "judge_reasoning": "AMBIGUOUS FALLBACK → BENIGN (precision-biased)"}


def normalize_label_output(raw_label: Any, fallback_choice: str = "B") -> int:
    """Normalize model output to strict binary label (1 hateful, 0 benign)."""
    text = str(raw_label or "").strip().upper()
    if text in {"1", "TRUE", "HATEFUL", "A", "RATIONALE A", "PROSECUTION"}:
        return 1
    if text in {"0", "FALSE", "BENIGN", "B", "RATIONALE B", "DEFENSE"}:
        return 0
    return 1 if str(fallback_choice or "").strip().upper() in {"A", "RATIONALE A", "PROSECUTION", "1"} else 0

# =============================
# 2. CONFIGURATION
# =============================

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Path to the rich captioned Knowledge Base
CAPTIONS_KB_PATH = os.path.join(BASE_DIR, "hateful-captioning", "captions_vllm_output1.jsonl")

# If the rich captions file exists, we FORCE its use as it's the target KB (ignoring potentially stale cluster env vars).
if os.path.exists(CAPTIONS_KB_PATH):
    MEMECAP_DATA = CAPTIONS_KB_PATH
    TRAIN_DATA_PATH = CAPTIONS_KB_PATH
    print(f"DEBUG: Using Rich Captions KB ({CAPTIONS_KB_PATH})")
else:
    # Use environment variables if set, otherwise fallback to original defaults
    MEMECAP_DATA = os.environ.get("MEMECAP_DATA", os.path.join(BASE_DIR, "memecap-data", "memes-test.json"))
    TRAIN_DATA_PATH = os.environ.get("KNN_TRAIN_PATH", os.path.join(BASE_DIR, "facebook-data", "train.jsonl"))
    print(f"DEBUG: Rich KB not found. Fallback: RAG={MEMECAP_DATA}, kNN={TRAIN_DATA_PATH}")

DATA_PATH = os.environ.get("STAGE2_DATA_PATH", os.path.join(BASE_DIR, "datapreparation", "output", "facebook-samples-test.jsonl"))
IMG_DIR = os.path.join(BASE_DIR, "facebook-data", "img")
OUT_PATH = os.environ.get("STAGE2_OUT_PATH", os.path.join(BASE_DIR, "datapreparation", "output", "predictions_stage2_vllm.jsonl"))

KNN_K = int(os.environ.get("KNN_K", "11"))

MODEL_8B_ID = os.environ.get("UNSLOTH_QWEN3_VL_MODEL_ID", "Qwen/Qwen3-VL-8B-Thinking")
RAG_TEXT_EMBED_MODEL_ID = os.environ.get("RAG_TEXT_EMBED_MODEL_ID", "sentence-transformers/all-MiniLM-L6-v2")
RAG_TOP_K = int(os.environ.get("RAG_TOP_K", "3"))
RAG_SCORE_THRESHOLD = float(os.environ.get("RAG_SCORE_THRESHOLD", "0.0"))
STAGE1_NO_RESPONSE = "ERROR: No response"
STAGE1_PARSE_ERROR = "PARSING ERROR"

# Self-consistency voting
SC_N = int(os.environ.get("SC_N", "3"))  # Number of Stage 2 generations per sample

# =============================
# CHANGE 3: Restructured POLICY_MANUAL with concrete rules
# =============================

POLICY_MANUAL = (
    "### DECISION RULES (apply strictly in priority order):\n\n"
    "RULE 1 — SLURS: If the meme uses a slur against a protected group "
    "(e.g., n***er, f***ot, k*ke, tranny used as insult, retard targeting disabled) → HATEFUL (1)\n\n"
    "RULE 2 — STEREOTYPE-DEPENDENT HUMOR: If the joke REQUIRES the audience to believe "
    "a negative stereotype about a race, religion, gender, sexuality, disability, or nationality "
    "for it to be 'funny' → HATEFUL (1)\n\n"
    "RULE 3 — CODED DEHUMANIZING METAPHORS: The following are KNOWN coded hate metaphors:\n"
    "  • 'dishwasher' / 'sandwich maker' / 'coffee maker' = woman (sexist dehumanization)\n"
    "  • 'farm equipment' / 'property' / 'livestock' = Black person (racist dehumanization)\n"
    "  • 'goat fucker' / 'goat lover' = Muslim (Islamophobic slur)\n"
    "  • 'oven' / 'gas' / 'shower' in Holocaust context = trivializing genocide\n"
    "  • 'monkey' / 'ape' / 'chimp' directed at Black people = racist dehumanization\n"
    "  • 'cockroach' / 'parasite' / 'vermin' for any group = dehumanization\n"
    "  If any of these coded metaphors are used to refer to a protected group → HATEFUL (1)\n\n"
    "RULE 4 — ATROCITY TRIVIALIZATION: If the meme uses the Holocaust, slavery, genocide, "
    "or mass shootings as punchlines that trivialize victims → HATEFUL (1)\n\n"
    "RULE 5 — INCITEMENT: If the meme calls for violence against, deportation of, or "
    "elimination of a protected group → HATEFUL (1)\n\n"
    "RULE 6 — BENIGN ONLY IF: The meme mocks a SITUATION, a NON-PROTECTED TRAIT "
    "(e.g., pizza toppings, corporate jargon), is genuinely self-deprecating about the poster's OWN group, "
    "or criticizes a specific individual's ACTIONS (not their identity) → BENIGN (0)\n\n"
    "### CRITICAL OVERRIDE:\n"
    "• 'Dark humor' / 'satire' / 'just a joke' is NOT a defense if the humor depends on "
    "a harmful stereotype about a protected group.\n"
    "• WHEN IN DOUBT: If a protected group is the punchline → HATEFUL (1)"
)

# =============================
# 3. RAG MODULE (MemeCap metaphors)
# =============================

class MetaphorMeaningRAG:
    def __init__(self, memecap_json_path: str, model_id: str = RAG_TEXT_EMBED_MODEL_ID):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.encoder = AutoModel.from_pretrained(model_id).to(self.device).eval()
        self.entries = self._build_entries(memecap_json_path)
        self.entry_texts = [e["text"] for e in self.entries]
        self.entry_embeddings = self._embed_texts(self.entry_texts) if self.entry_texts else None
        print(f"RAG entries loaded: {len(self.entry_texts)}")

    def _build_entries(self, path):
        # Support both JSON array and JSONL formats
        data = []
        with open(path, "r", encoding="utf-8") as f:
            first_char = f.read(1).strip()
            f.seek(0)
            if first_char == '[':
                # JSON array format (original MemeCap)
                data = json.load(f)
            else:
                # JSONL format (captions_vllm_output1.jsonl)
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            data.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
        
        entries = []
        for item in data:
            raw_label = item.get("label", item.get("gold_label", item.get("class_label", "unknown")))
            try:
                label_value = int(raw_label)
            except (TypeError, ValueError):
                label_value = str(raw_label).strip() if raw_label is not None else "unknown"

            # Build entries from metaphors
            for m in item.get("metaphors", []):
                metaphor = str(m.get("metaphor", "")).strip()
                meaning = str(m.get("meaning", "")).strip()
                if metaphor or meaning:
                    # Include meme_captions context for richer semantic matching
                    meme_context = "; ".join(item.get("meme_captions", []))[:200]
                    text = f"Metaphor: {metaphor}. Meaning: {meaning}. Context: {meme_context}. Label: {label_value}."
                    entries.append(
                        {
                            "text": text,
                            "metaphor": metaphor,
                            "meaning": meaning,
                            "label": label_value,
                        }
                    )
            
            # Also index meme-level captions (even if no metaphors) for broader coverage
            meme_captions = item.get("meme_captions", [])
            title = item.get("title", "")
            if meme_captions and not item.get("metaphors"):
                caption_text = f"Title: {title}. Caption: {'; '.join(meme_captions)[:300]}. Label: {label_value}."
                entries.append(
                    {
                        "text": caption_text,
                        "metaphor": title,
                        "meaning": "; ".join(meme_captions)[:300],
                        "label": label_value,
                    }
                )
        return entries

    def _embed_texts(self, texts, batch_size=64):
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            encoded = self.tokenizer(batch, padding=True, truncation=True, max_length=256, return_tensors="pt").to(self.device)
            with torch.no_grad():
                out = self.encoder(**encoded)
                mask = encoded["attention_mask"].unsqueeze(-1).expand(out.last_hidden_state.size()).float()
                pooled = (out.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
                all_embeddings.append(F.normalize(pooled, p=2, dim=1).cpu())
        return torch.cat(all_embeddings, dim=0)

    def query(self, query_text, k=3, threshold=0.0):
        if not query_text.strip():
            return "No relevant metaphors found."

        if not self.entry_texts or self.entry_embeddings is None:
            return "No relevant metaphors found."
        
        q_embed = self._embed_texts([query_text])[0]
        scores = torch.matmul(self.entry_embeddings, q_embed)
        
        top_k = torch.topk(scores, k=min(k, scores.shape[0]))
        top_scores = top_k.values.tolist()
        top_indices = top_k.indices.tolist()
        
        valid_entries = []
        for score, idx in zip(top_scores, top_indices):
            if score >= threshold:
                valid_entries.append(self.entry_texts[idx])
                
        if not valid_entries:
            valid_entries = [self.entry_texts[idx] for idx in top_indices]
            
        return "\n".join(valid_entries)

# =============================
# CHANGE 4: kNN Classifier from Training Data
# =============================

class KNNClassifier:
    """kNN classifier using rich captioned embeddings from the HatefulMemes training set."""
    
    def __init__(self, train_jsonl_path: str, model_id: str = RAG_TEXT_EMBED_MODEL_ID, k: int = KNN_K):
        self.k = k
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.encoder = AutoModel.from_pretrained(model_id).to(self.device).eval()
        
        self.texts, self.labels = self._load_training_data(train_jsonl_path)
        print(f"kNN: Building index from {len(self.texts)} training samples...")
        self.embeddings = self._embed_texts(self.texts)
        
        n_hateful = sum(1 for l in self.labels if l == 1)
        n_benign = sum(1 for l in self.labels if l == 0)
        print(f"kNN index built: {n_hateful} hateful, {n_benign} benign")
    
    def _load_training_data(self, path: str) -> Tuple[List[str], List[int]]:
        """Load training data from either rich captioned JSONL or plain text JSONL."""
        texts = []
        labels = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                
                label = int(row.get("label", 0))
                
                # Rich captioned format (captions_vllm_output1.jsonl)
                if "meme_captions" in row or "img_captions" in row:
                    parts = []
                    title = row.get("title", "")
                    if title:
                        parts.append(f"Title: {title}")
                    
                    # Meme-level captions (most informative for hate detection)
                    meme_caps = row.get("meme_captions", [])
                    if meme_caps:
                        parts.append(f"Interpretation: {'; '.join(meme_caps)[:300]}")
                    
                    # Image descriptions
                    img_caps = row.get("img_captions", [])
                    if img_caps:
                        parts.append(f"Visual: {'; '.join(img_caps)[:200]}")
                    
                    # Metaphors
                    metaphors = row.get("metaphors", [])
                    if metaphors:
                        meta_strs = [f"{m.get('metaphor','')}: {m.get('meaning','')}" for m in metaphors]
                        parts.append(f"Metaphors: {'; '.join(meta_strs)[:200]}")
                    
                    text = ". ".join(parts)
                else:
                    # Fallback: plain text format (train.jsonl)
                    text = row.get("text", "").strip()
                
                if text:
                    texts.append(text)
                    labels.append(label)
        return texts, labels
    
    def _embed_texts(self, texts, batch_size=64):
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            encoded = self.tokenizer(batch, padding=True, truncation=True, max_length=128, return_tensors="pt").to(self.device)
            with torch.no_grad():
                out = self.encoder(**encoded)
                mask = encoded["attention_mask"].unsqueeze(-1).expand(out.last_hidden_state.size()).float()
                pooled = (out.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
                all_embeddings.append(F.normalize(pooled, p=2, dim=1).cpu())
        return torch.cat(all_embeddings, dim=0)
    
    def classify(self, query_text: str) -> Dict[str, Any]:
        """Returns kNN vote with details."""
        if not query_text.strip():
            return {"knn_label": 0, "n_hateful": 0, "n_benign": 0, "confidence": 0.0, "similar_examples": []}
        
        q_embed = self._embed_texts([query_text])[0]
        scores = torch.matmul(self.embeddings, q_embed)
        
        top_k = torch.topk(scores, k=min(self.k, scores.shape[0]))
        top_scores = top_k.values.tolist()
        top_indices = top_k.indices.tolist()
        
        neighbor_labels = [self.labels[idx] for idx in top_indices]
        n_hateful = sum(1 for l in neighbor_labels if l == 1)
        n_benign = sum(1 for l in neighbor_labels if l == 0)
        
        # Weighted vote: closer neighbors count more
        weighted_hateful = sum(s for s, l in zip(top_scores, neighbor_labels) if l == 1)
        weighted_benign = sum(s for s, l in zip(top_scores, neighbor_labels) if l == 0)
        
        knn_label = 1 if weighted_hateful > weighted_benign else 0
        total_weight = weighted_hateful + weighted_benign
        confidence = max(weighted_hateful, weighted_benign) / total_weight if total_weight > 0 else 0.5
        
        similar_examples = []
        for idx, score in zip(top_indices[:3], top_scores[:3]):
            similar_examples.append({
                "text": self.texts[idx][:80],
                "label": self.labels[idx],
                "score": round(score, 3)
            })
        
        return {
            "knn_label": knn_label,
            "n_hateful": n_hateful,
            "n_benign": n_benign,
            "confidence": round(confidence, 3),
            "similar_examples": similar_examples
        }

# =============================
# CHANGE 5: Self-Consistency Majority Voting
# =============================

def majority_vote_from_outputs(output_list: list, knn_result: dict = {}) -> dict:
    """Parse multiple Stage 2 outputs and return majority vote."""
    knn_label = knn_result.get("knn_label", -1)
    knn_conf = knn_result.get("confidence", 0.0)
    
    parsed_results = []
    for output_text in output_list:
        parsed = parse_thinking_output(output_text, knn_fallback_label=knn_label, knn_confidence=knn_conf)
        parsed_results.append(parsed)
    
    labels = [p["label"] for p in parsed_results]
    vote_counts = Counter(labels)
    majority_label = vote_counts.most_common(1)[0][0]
    
    # Precision Gate: If kNN disagrees and confidence is high, and LLM vote is split, favor kNN or Benign
    # But only if it's a tight vote (e.g. 2 vs 1)
    if majority_label == 1 and knn_label == 0 and knn_conf > 0.85 and vote_counts.get(1) < len(output_list):
        majority_label = 0 # Favor Precision
        majority_reasoning = f"OVERRIDDEN by high-confidence kNN Benign ({knn_conf})"
    else:
        # Pick the reasoning from the first result that matches the majority
        majority_reasoning = "Majority vote"
        for p in parsed_results:
            if p["label"] == majority_label:
                majority_reasoning = p["judge_reasoning"]
                break
    
    return {
        "label": majority_label,
        "judge_choice": "A" if majority_label == 1 else "B",
        "judge_reasoning": f"SELF-CONSISTENCY ({vote_counts.get(1,0)}H/{vote_counts.get(0,0)}B): {majority_reasoning}",
        "vote_detail": dict(vote_counts),
    }

# =============================
# 4. MAIN BATCH LOGIC
# =============================

def unload_model(llm):
    del llm
    gc.collect()
    torch.cuda.empty_cache()
    print("VRAM partially cleared (model object deleted)")

def main():
    rag = MetaphorMeaningRAG(MEMECAP_DATA)
    
    # CHANGE 4: Build kNN index from training data
    knn = None
    if os.path.exists(TRAIN_DATA_PATH):
        knn = KNNClassifier(TRAIN_DATA_PATH, k=KNN_K)
    else:
        print(f"WARNING: Training data not found at {TRAIN_DATA_PATH}, skipping kNN")

    print(f"Loading input data from {DATA_PATH}...")
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        samples = [json.loads(line) for line in f]
    
    # --- STAGE 1: ANALYSIS (8B Model) ---
    print(f"\n--- STAGE 1: Generating Dual Rationales using {MODEL_8B_ID} ---")
    
    llm = LLM(
        model=MODEL_8B_ID,
        trust_remote_code=True,
        max_model_len=4096,
        gpu_memory_utilization=0.85
    )

    stage1_prompts = []
    stage1_schema = MemeRationales.model_json_schema()

    for s in tqdm(samples, desc="Preparing Stage 1 Inputs"):
        image_name = os.path.basename(s["img"])
        img_path = os.path.join(IMG_DIR, image_name)
        s["abs_img_path"] = img_path
        
        context = rag.query(s.get("text", ""), k=RAG_TOP_K, threshold=RAG_SCORE_THRESHOLD)
        s["rag_context"] = context
        
        # CHANGE 4: kNN classification
        if knn:
            knn_result = knn.classify(s.get("text", ""))
            s["knn_result"] = knn_result
        else:
            s["knn_result"] = {"knn_label": -1, "n_hateful": 0, "n_benign": 0, "confidence": 0}
        
        # CHANGE 2: Updated Stage 1 prompt with target_group/attack_type extraction
        prompt_text = (
            f"### RETRIEVED CONTEXT:\n{context}\n\n"
            f"### TARGET MEME TEXT:\n'{s.get('text', '')}'\n\n"
            "### TASK: Analyze this meme contrastively.\n"
            "1. Visual Evidence: Describe exactly what is in the image objectively.\n"
            "   Output as a JSON list named visual_evidence, where each item has:\n"
            "   - metaphor: literal scene element(s)\n"
            "   - meaning: objective visible description only\n"
            "2. Target Group: Identify the PROTECTED GROUP targeted, if any.\n"
            "   Use 'none' if no protected group is targeted.\n"
            "   Examples: 'Black people', 'Muslims', 'Jews', 'women', 'LGBTQ+', 'immigrants', 'disabled people'\n"
            "3. Attack Type: Identify the TYPE OF ATTACK, if any.\n"
            "   Use 'none' if no attack. Examples: 'stereotype', 'dehumanization', 'slur', \n"
            "   'incitement_to_violence', 'coded_metaphor', 'trivialization_of_atrocity'\n"
            "4. Generate Rationale A (Prosecution): Argue why it is Hateful (1).\n"
            "5. Generate Rationale B (Defense): Argue why it is Benign (0).\n\n"
            "Respond ONLY with a JSON object matching this schema:\n"
            f"SCHEMA: {json.dumps(stage1_schema)}"
        )
        
        vllm_prompt = (
            "<|im_start|>system\n"
            "You are an expert in semiotics and multimodal hate speech detection.<|im_end|>\n"
            "<|im_start|>user\n"
            "<|vision_start|><|image_pad|><|vision_end|>\n"
            f"{prompt_text}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        
        stage1_prompts.append({
            "prompt": vllm_prompt,
            "multi_modal_data": {"image": Image.open(img_path).convert("RGB")}
        })

    sampling_8b = SamplingParams(
        temperature=0.0,
        max_tokens=700,  # Increased for target_group/attack_type fields
        structured_outputs=StructuredOutputsParams(json=stage1_schema)
    )

    outputs_8b = llm.generate(stage1_prompts, sampling_params=sampling_8b)
    
    for i, out in enumerate(outputs_8b):
        try:
            res = json.loads(out.outputs[0].text)
            samples[i]["visual_evidence"] = res.get("visual_evidence", [])
            samples[i]["target_group"] = res.get("target_group", "unknown")
            samples[i]["attack_type"] = res.get("attack_type", "unknown")
            samples[i]["rationale_a"] = res.get("rationale_a", STAGE1_NO_RESPONSE)
            samples[i]["rationale_b"] = res.get("rationale_b", STAGE1_NO_RESPONSE)
        except (json.JSONDecodeError, KeyError, IndexError, TypeError):
            samples[i]["visual_evidence"] = []
            samples[i]["target_group"] = "unknown"
            samples[i]["attack_type"] = "unknown"
            samples[i]["rationale_a"] = STAGE1_PARSE_ERROR
            samples[i]["rationale_b"] = STAGE1_PARSE_ERROR

    # --- STAGE 2: JUDGING (8B Thinking Model with Self-Consistency) ---
    print(f"\n--- STAGE 2: Judging Policy using {MODEL_8B_ID} (SC n={SC_N}) ---")

    stage2_prompts = []
    stage2_schema = JudgeChoice.model_json_schema()

    for s in tqdm(samples, desc="Preparing Stage 2 Inputs"):
        visual_evidence_text = json.dumps(s.get("visual_evidence", []), ensure_ascii=False)
        
        # CHANGE 2 + 4: Include target_group, attack_type, and kNN prior in prompt
        target_group = s.get("target_group", "unknown")
        attack_type = s.get("attack_type", "unknown")
        knn_result = s.get("knn_result", {})
        
        # Build kNN context string
        knn_context = ""
        if knn_result.get("knn_label", -1) >= 0:
            knn_label_str = "HATEFUL" if knn_result["knn_label"] == 1 else "BENIGN"
            knn_context = (
                f"\n### kNN PRIOR (from {KNN_K} most similar training memes):\n"
                f"Vote: {knn_result['n_hateful']} hateful, {knn_result['n_benign']} benign → {knn_label_str} "
                f"(confidence: {knn_result['confidence']})\n"
            )
        
        judge_prompt_text = (
            f"### TARGET MEME TEXT: '{s.get('text', '')}'\n\n"
            f"### VISUAL EVIDENCE (OBJECTIVE): {visual_evidence_text}\n\n"
            f"### STAGE 1 ANALYSIS:\n"
            f"- Identified Target Group: {target_group}\n"
            f"- Identified Attack Type: {attack_type}\n\n"
            f"### RATIONALE A (Prosecution): {s.get('rationale_a', '')}\n\n"
            f"### RATIONALE B (Defense): {s.get('rationale_b', '')}\n\n"
            f"{knn_context}\n"
            f"### POLICY MANUAL:\n{POLICY_MANUAL}\n\n"
            "### EVALUATION RULE:\n"
            "You are a STRICT NEUTRAL POLICY JUDGE. Act as a skeptical observer.\n\n"
            "1. AUDIT FOR HALLUCINATION: Check if the Prosecution (Rationale A) is imagining "
            "hateful intent that isn't supported by the objective visual evidence. "
            "Identifying a target group is NOT enough for a 'Hateful' verdict; there must be "
            "a concrete, dehumanizing attack against that group.\n\n"
            "2. VERIFY THE ATTACK: If the meme just describes a situation or uses 'dark humor' "
            "without relying on a harmful stereotype of a protected group, it is BENIGN (0).\n\n"
            "3. COUNTER-SPEECH: Is the meme mocking HATE itself? Is it an anti-hate PSA? "
            "If so, it is BENIGN (0).\n\n"
            "4. JUXTAPOSITION: Does the combination of text and image create a dehumanizing metaphor? "
            "If yes, proceed to the Slur/Stereotype rules.\n\n"
            "### TASK: Output binary label. Only choose 1 if the evidence for a policy violation is OVERWHELMING.\n"
            "- A is better → label = 1\n"
            "- B is better → label = 0\n"
            f"SCHEMA: {json.dumps(stage2_schema)}"
        )
        
        vllm_prompt = (
            "<|im_start|>system\n"
            "You are a strict multimodal policy judge. Use both the meme image and text."
            "<|im_end|>\n"
            "<|im_start|>user\n"
            "<|vision_start|><|image_pad|><|vision_end|>\n"
            f"{judge_prompt_text}<|im_end|>\n"
            "<|im_start|>assistant\n"
            "<think>\n"
        )
        stage2_prompts.append({
            "prompt": vllm_prompt,
            "multi_modal_data": {"image": Image.open(s["abs_img_path"]).convert("RGB")},
        })

    # CHANGE 5: Self-consistency with n=SC_N generations
    sampling_2b = SamplingParams(
        temperature=0.6,  # Non-zero for diversity in SC voting
        max_tokens=1500,
        n=SC_N,  # Generate SC_N responses per prompt
    )

    outputs_2b = llm.generate(stage2_prompts, sampling_params=sampling_2b)

    for i, out in enumerate(outputs_2b):
        # Collect all SC_N output texts
        output_texts = [o.text for o in out.outputs]
        
        if len(output_texts) == 1:
            # Single output, use normal parsing
            knn_res = s.get("knn_result", {})
            parsed_data = parse_thinking_output(output_texts[0], 
                                                knn_fallback_label=knn_res.get("knn_label"), 
                                                knn_confidence=knn_res.get("confidence", 0.0))
            samples[i]["label"] = parsed_data["label"]
            samples[i]["judge_choice"] = parsed_data["judge_choice"]
            samples[i]["judge_reasoning"] = parsed_data["judge_reasoning"]
        else:
            # CHANGE 5: Majority vote across SC_N outputs
            voted = majority_vote_from_outputs(output_texts, knn_result=s.get("knn_result", {}))
            samples[i]["label"] = voted["label"]
            samples[i]["judge_choice"] = voted["judge_choice"]
            samples[i]["judge_reasoning"] = voted["judge_reasoning"]
            samples[i]["sc_vote_detail"] = voted.get("vote_detail", {})

    # --- FINAL SAVING ---
    print(f"\nSaving results to {OUT_PATH}...")
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for s in samples:
            out_item = {
                "id": s.get("id"),
                "ocr_text": s.get("text", ""),
                "visual_evidence": s.get("visual_evidence"),
                "target_group": s.get("target_group", "unknown"),
                "attack_type": s.get("attack_type", "unknown"),
                "rationale_a": s.get("rationale_a"),
                "rationale_b": s.get("rationale_b"),
                "judge_choice": s.get("judge_choice"),
                "label": s.get("label", 0),
                "judge_reasoning": s.get("judge_reasoning"),
                "retrieved_context": s.get("rag_context"),
                "knn_result": s.get("knn_result"),
                "sc_vote_detail": s.get("sc_vote_detail", {}),
            }
            f.write(json.dumps(out_item, ensure_ascii=False) + "\n")

    unload_model(llm)

    print("Success! vLLM batch inference complete.")

if __name__ == "__main__":
    main()
