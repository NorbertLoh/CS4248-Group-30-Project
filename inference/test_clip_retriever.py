"""
Test script to verify CLIP retriever is working correctly.
Run this after building embeddings to test retrieval.
"""
from clip_retriever import get_clip_retriever
from pathlib import Path

def test_retriever():
    print("=" * 60)
    print("Testing CLIP Retriever")
    print("=" * 60)

    # Create retriever
    try:
        retriever = get_clip_retriever(k=3)
        print("✓ Retriever loaded successfully\n")
    except FileNotFoundError as e:
        print(f"✗ Error: {e}")
        print("\nPlease run: python build_clip_rag.py")
        return

    # Test queries
    test_queries = [
        "A funny meme about marriage and relationships",
        "A meme criticizing Apple products",
        "Surprised person looking at something shocking",
        "Political satire about world events"
    ]

    for i, query in enumerate(test_queries, 1):
        print(f"\n{'=' * 60}")
        print(f"Query {i}: {query}")
        print(f"{'=' * 60}\n")

        results = retriever.invoke(query)

        for j, doc in enumerate(results, 1):
            print(f"Result {j} (Score: {doc.score:.3f}):")
            print(doc.content)
            print(f"Image: {doc.metadata['img_fname']}")
            print()

    print("=" * 60)
    print("✓ Test completed successfully!")
    print("=" * 60)

if __name__ == "__main__":
    test_retriever()
