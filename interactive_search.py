"""
Interactive Text-to-Image Search CLI
====================================
Allows you to type any description and instantly retrieve matching images
from the indexed Flickr30k gallery using EmbeddingGemma 2.
"""

import os
import numpy as np
from PIL import Image
from download_dataset import load_flickr30k
from embedding_gemma import EmbeddingGemma2
from image_search import build_image_index, search_images


def run_interactive():
    print("=" * 65)
    print("  EmbeddingGemma 2: Interactive Text-to-Image Search")
    print("=" * 65)
    
    # 1. Load Model
    model = EmbeddingGemma2()
    
    # 2. Load or Build Index (100 images already cached in flickr30k_index.npz)
    image_embeddings, indices = build_image_index(model, num_images=100)
    print(f"\n Ready! {len(image_embeddings)} images indexed.")
    print("Type your search query (or 'exit' / 'q' to quit):\n")
    
    while True:
        try:
            query = input("\n🔎 Enter search query: ").strip()
            if not query:
                continue
            if query.lower() in ("exit", "quit", "q"):
                print("Goodbye!")
                break
                
            search_images(query, model, image_embeddings, indices, top_k=3)
        except (KeyboardInterrupt, EOFError):
            print("\nExiting...")
            break


if __name__ == "__main__":
    run_interactive()
