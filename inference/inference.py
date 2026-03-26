import json
import re
import os
import sys  # Added for stdout flushing
import torch
from tqdm import tqdm
from pathlib import Path
from typing import Any, List, Optional, Dict
from PIL import Image
from pydantic import BaseModel, Field

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

# =============================
# 1. CONFIGURATION
# =============================
MODELS_TO_RUN = [
    "unsloth/Qwen3-VL-2B-Instruct-unsloth-bnb-4bit",
    # "unsloth/Qwen3-VL-2B-Thinking-unsloth-bnb-4bit",
    # "unsloth/Qwen3-VL-8B-Instruct-unsloth-bnb-4bit",
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
# 2. BATCH-ENABLED WRAPPER
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
# 3. UTILS & RAG
# =============================
def get_rag_retriever():
    texts = ["Hate speech is defined as..."] 
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2", model_kwargs={'device': device})
    vectorstore = FAISS.from_texts(texts, embeddings)
    return vectorstore.as_retriever(search_kwargs={"k": 2})

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
# 4. MAIN INFERENCE LOOP
# =============================
def main():
    if not SAMPLES_JSONL.exists():
        print(f"Error: Could not find {SAMPLES_JSONL}")
        return

    with open(SAMPLES_JSONL, "r") as f:
        samples = [json.loads(line) for line in f if line.strip()]

    print("\n>>> Phase 1: Pre-calculating RAG context and resolving paths...")
    sys.stdout.flush() # Force print to show up
    
    retriever = get_rag_retriever()
    
    # Updated TQDM for real-time output
    for rec in tqdm(samples, desc="Pre-calculating", file=sys.stdout, mininterval=0, miniters=1):
        context_docs = retriever.invoke(rec.get("text", ""))
        rec["pre_context"] = "\n".join([doc.page_content for doc in context_docs])
        raw_path = str(rec.get("img", "")).replace("\\", "/")
        rec["pre_img_path"] = str((SAMPLES_JSONL.parent / raw_path).resolve())
        sys.stdout.flush() # Force flush

    parser = PydanticOutputParser(pydantic_object=HateSpeechPrediction)
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", (
            "You are an objective content moderator. Use the provided Policy Context to classify the meme.\n"
            "0 = Non-hateful, 1 = Hateful.\n"
            "Policy Context: {context}\n\n{format_instructions}"
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