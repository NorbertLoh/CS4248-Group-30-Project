import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq
from sentence_transformers import SentenceTransformer, util
from qwen_vl_utils import process_vision_info

# ==========================================
# 1. SETUP RETRIEVER (Vector Database)
# ==========================================
print("Loading Retriever Model...")
# CLIP bridges the gap between text queries and image embeddings
retriever_model = SentenceTransformer('clip-ViT-B-32')

# Mock Knowledge Base (Replace these with paths to your actual local images)
knowledge_base_paths = [
    "architecture_diagram.jpg",
    "financial_report_2025.png",
    "server_rack_config.jpg"
]

print("Indexing Knowledge Base...")
# Load images and pre-compute their embeddings for fast searching
kb_images = [Image.open(img).convert("RGB") for img in knowledge_base_paths]
image_embeddings = retriever_model.encode(kb_images)


# ==========================================
# 2. SETUP GENERATOR (Qwen3-VL)
# ==========================================
print("Loading Qwen3-VL Model...")
model_id = "Qwen/Qwen3-VL-8B-Instruct"

# Load the processor and the model
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForVision2Seq.from_pretrained(
    model_id, 
    torch_dtype=torch.float16, 
    device_map="auto" # Automatically uses GPU if available
)


# ==========================================
# 3. CORE RAG FUNCTION
# ==========================================
def query_qwen3_rag(query: str, top_k: int = 1):
    # --- Step A: Retrieve Context ---
    # Convert user text query into an embedding
    query_embedding = retriever_model.encode(query)
    
    # Perform semantic search to find the closest image
    hits = util.semantic_search(query_embedding, image_embeddings, top_k=top_k)[0]
    best_hit_idx = hits[0]['corpus_id']
    
    retrieved_image_path = knowledge_base_paths[best_hit_idx]
    print(f"\n[Retriever] Found relevant context: {retrieved_image_path} (Confidence: {hits[0]['score']:.4f})")

    # --- Step B: Prepare Prompt for Qwen3-VL ---
    # Qwen3 uses a structured chat template for multimodal inputs
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": retrieved_image_path},
                {"type": "text", "text": f"Based strictly on the provided image context, answer the following query: {query}"}
            ]
        }
    ]
    
    # Parse the vision info using Qwen's utility
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    
    inputs = processor(
        text=[text], 
        images=image_inputs, 
        videos=video_inputs, 
        padding=True, 
        return_tensors="pt"
    ).to(model.device)

    # --- Step C: Generate Answer ---
    print("[Generator] Thinking...")
    generated_ids = model.generate(**inputs, max_new_tokens=512)
    
    # Slice the output to remove the input prompt tokens
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    
    return output_text[0]


# ==========================================
# 4. EXECUTION
# ==========================================
if __name__ == "__main__":
    # Ensure you have actual images matching the names in `knowledge_base_paths` before running
    user_query = "What is the total revenue shown in the financial chart?"
    
    try:
        final_answer = query_qwen3_rag(user_query)
        print("\n--- Final Answer ---")
        print(final_answer)
    except Exception as e:
        print(f"Error running pipeline: {e}")