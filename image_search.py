"""
Text-to-Image Semantic Search Engine using EmbeddingGemma 2 & Flickr30k
========================================================================
Demonstrates cross-modal retrieval:
1. Indexes a gallery of images into a vector store.
2. Encodes natural language queries into the same vector space.
3. Finds the most semantically relevant images using cosine similarity.
"""

import os
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
from download_dataset import load_flickr30k
from embedding_gemma import EmbeddingGemma2

INDEX_FILE = "flickr30k_index.npz"


def build_image_index(model: EmbeddingGemma2, num_images: int = 100):
    """
    Builds or loads an embedding index for the first `num_images` of Flickr30k.
    Caches the vectors so future searches run in milliseconds.
    """
    if os.path.exists(INDEX_FILE):
        data = np.load(INDEX_FILE, allow_pickle=True)
        cached_vectors = data["embeddings"]
        cached_indices = data["indices"]
        if len(cached_vectors) >= num_images:
            print(f" Loaded cached vector index ({len(cached_vectors)} images) from {INDEX_FILE}")
            return cached_vectors[:num_images], cached_indices[:num_images]
    
    print(f"\nBuilding vector index for {num_images} images from Flickr30k...")
    ds = load_flickr30k()
    
    embeddings = []
    indices = []
    
    for i in tqdm(range(min(num_images, len(ds))), desc="Embedding images"):
        img = ds[i]["image"]
        emb = model.embed_image(img)
        embeddings.append(emb)
        indices.append(i)
        
    embeddings = np.array(embeddings, dtype=np.float32)
    indices = np.array(indices, dtype=np.int32)
    
    # Save cache
    np.savez_compressed(INDEX_FILE, embeddings=embeddings, indices=indices)
    print(f" Saved index of {len(embeddings)} images to {INDEX_FILE}")
    return embeddings, indices


def search_images(query: str, model: EmbeddingGemma2, image_embeddings: np.ndarray, indices: np.ndarray, top_k: int = 3):
    """
    Performs text-to-image semantic search.
    
    1. Embeds text query -> 768-dim vector
    2. Computes dot products (cosine similarities) against all indexed image vectors
    3. Returns top-k matching image indices and scores
    """
    print(f"\n Searching for: \"{query}\"...")
    
    # Encode text query
    query_emb = model.embed_text(query)
    
    # Fast vectorized cosine similarity: (N, 768) dot (768,) -> (N,)
    scores = np.dot(image_embeddings, query_emb)
    
    # Sort descending
    top_indices_local = np.argsort(scores)[::-1][:top_k]
    
    ds = load_flickr30k()
    results = []
    
    print("\n" + "=" * 65)
    print(f" TOP {top_k} RESULTS FOR: \"{query}\"")
    print("=" * 65)
    
    output_dir = "search_results"
    os.makedirs(output_dir, exist_ok=True)
    
    for rank, idx_local in enumerate(top_indices_local, 1):
        global_idx = int(indices[idx_local])
        score = float(scores[idx_local])
        sample = ds[global_idx]
        img = sample["image"]
        captions = sample["captions"]
        
        # Save matching image
        clean_q = "".join(c for c in query if c.isalnum() or c in (" ", "_")).replace(" ", "_")[:20]
        out_path = os.path.join(output_dir, f"rank{rank}_{clean_q}_sample{global_idx}.jpg")
        img.save(out_path)
        
        print(f"\n[Rank {rank}] Similarity Score: {score:.4f} (Sample #{global_idx})")
        print(f"Saved Image: {out_path}")
        print("Dataset Annotations:")
        for cap in captions[:3]:
            print(f"  • {cap}")
            
        results.append({
            "rank": rank,
            "global_idx": global_idx,
            "score": score,
            "captions": captions,
            "image_path": out_path
        })
        
    print("=" * 65 + "\n")
    return results


def main():
    parser = argparse.ArgumentParser(description="EmbeddingGemma 2 Text-to-Image Search")
    parser.add_argument("--query", type=str, default="a dog running on the grass", help="Search text query")
    parser.add_argument("--num_images", type=int, default=100, help="Number of gallery images to index")
    parser.add_argument("--top_k", type=int, default=3, help="Number of results to return")
    args = parser.parse_args()
    
    # 1. Initialize local model
    model = EmbeddingGemma2()
    
    # 2. Build or load image index
    image_embeddings, indices = build_image_index(model, num_images=args.num_images)
    
    # 3. Perform search
    search_images(args.query, model, image_embeddings, indices, top_k=args.top_k)


if __name__ == "__main__":
    main()
