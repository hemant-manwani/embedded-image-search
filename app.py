"""
EmbeddingGemma 2: Text-to-Image Search API Server
=================================================
FastAPI backend powered by:
- Google's EmbeddingGemma 2 (768-D local ONNX model)
- LanceDB embedded columnar vector database (Apache Arrow format)
"""

import os
import io
import time
import threading
from typing import Optional
import numpy as np
from PIL import Image

# Ensure offline mode if models already downloaded
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from fastapi import FastAPI, Query, HTTPException, Response
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

from download_dataset import load_flickr30k
from embedding_gemma import EmbeddingGemma2
from lancedb_manager import LanceDBManager, migrate_from_npz_if_needed

app = FastAPI(
    title="EmbeddingGemma 2 Multimodal Search API (LanceDB)",
    description="Local Text-to-Image Cross-Modal Retrieval Engine backed by LanceDB",
    version="2.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global in-memory resources
model: Optional[EmbeddingGemma2] = None
lance_mgr: Optional[LanceDBManager] = None
dataset = None
image_cache = {}

# Background Indexing Worker State
index_lock = threading.Lock()
indexing_thread: Optional[threading.Thread] = None

indexing_state = {
    "is_indexing": False,
    "cancel_requested": False,
    "current_count": 0,
    "target_count": 0,
    "current_sample_idx": None,
    "current_caption": "",
    "start_time": 0.0,
    "speed_sec_per_img": 0.0,
    "message": "Idle"
}


def background_indexer(target_count: int):
    """Worker thread that embeds images into LanceDB with O(1) instant appends."""
    global indexing_state
    
    print(f"\n[LanceDB Indexer] Starting indexing towards target {target_count} images...")
    t_start = time.time()
    
    with index_lock:
        indexed_ids = lance_mgr.get_indexed_ids()
        total_in_dataset = len(dataset)
        target = min(target_count, total_in_dataset)
        
        indexing_state["is_indexing"] = True
        indexing_state["cancel_requested"] = False
        indexing_state["target_count"] = target
        indexing_state["current_count"] = len(indexed_ids)
        indexing_state["start_time"] = t_start
        indexing_state["message"] = f"Indexing images up to {target} in LanceDB..."
        
        batch_records = []
        
        for sample_i in range(target):
            if indexing_state["cancel_requested"]:
                print(f"[LanceDB Indexer] Stop requested by user at sample #{sample_i}.")
                indexing_state["message"] = "Indexing paused by user."
                break
                
            if sample_i in indexed_ids:
                continue
                
            step_start = time.time()
            sample = dataset[sample_i]
            img = sample["image"]
            captions = sample["captions"]
            
            indexing_state["current_sample_idx"] = sample_i
            indexing_state["current_caption"] = captions[0] if captions else ""
            
            # Embed image via EmbeddingGemma 2 ONNX model
            vector = model.embed_image(img)
            
            record = {
                "vector": vector.astype(np.float32).tolist(),
                "sample_idx": sample_i,
                "caption": captions[0] if captions else "",
                "captions": captions
            }
            batch_records.append(record)
            indexed_ids.add(sample_i)
            
            # O(1) Append to LanceDB immediately or in micro-batches
            lance_mgr.add_records([record])
            batch_records.clear()
            
            # Update live state
            step_duration = round(time.time() - step_start, 2)
            total_indexed = lance_mgr.count()
            indexing_state["current_count"] = total_indexed
            indexing_state["speed_sec_per_img"] = step_duration
            indexing_state["message"] = f"Indexed image #{sample_i} ({total_indexed}/{target})"
            
        total_time = round(time.time() - t_start, 1)
        if not indexing_state["cancel_requested"]:
            indexing_state["message"] = f"Finished! {lance_mgr.count()} images indexed in {total_time}s."
            
        indexing_state["is_indexing"] = False
        indexing_state["cancel_requested"] = False
        print(f"[LanceDB Indexer] Finished. Total in LanceDB: {lance_mgr.count()} records.")


@app.on_event("startup")
def startup_event():
    global model, lance_mgr, dataset, indexing_state
    print("=" * 60)
    print("Starting EmbeddingGemma 2 API Server with LanceDB...")
    print("=" * 60)
    
    # 1. Initialize local embedding model
    model = EmbeddingGemma2()
    
    # 2. Initialize LanceDB manager
    lance_mgr = LanceDBManager()
    
    # 3. Load dataset
    print("Loading Flickr30k dataset from local cache...")
    dataset = load_flickr30k()
    
    # 4. Migrate any legacy NPZ vectors into LanceDB
    migrate_from_npz_if_needed(lance_mgr, dataset=dataset)
    
    indexing_state["current_count"] = lance_mgr.count()
    print(f"LanceDB ready! Total records: {lance_mgr.count()} (Size: {lance_mgr.get_disk_size_mb()} MB)")


@app.get("/")
def serve_index():
    """Serve the web search interface."""
    return FileResponse("static/index.html")


@app.get("/manage")
def serve_manage():
    """Serve the dedicated Vector Index Management interface."""
    return FileResponse("static/manage.html")


@app.get("/api/index/records")
def get_recent_records(limit: int = 12):
    """Return the most recently indexed LanceDB records for the inspector."""
    if not lance_mgr:
        return {"records": []}
    records = lance_mgr.get_recent_records(limit=min(50, max(1, limit)))
    return {"total_in_db": lance_mgr.count(), "records": records}



@app.get("/api/stats")
def get_stats():
    """Return system, model, and LanceDB statistics."""
    count = lance_mgr.count() if lance_mgr else 0
    size_mb = lance_mgr.get_disk_size_mb() if lance_mgr else 0.0
    
    return {
        "status": "ready",
        "storage_engine": "LanceDB (Embedded Arrow Vector DB)",
        "model": "google/embeddinggemma-2 (quantized ONNX)",
        "embedding_dim": 768,
        "indexed_images": count,
        "total_dataset_images": len(dataset) if dataset is not None else 0,
        "db_size_mb": size_mb,
        "inference_engine": "ONNX Runtime (Local CPU)",
        "offline": os.environ.get("HF_HUB_OFFLINE", "0") == "1"
    }


@app.get("/api/index/status")
def get_index_status():
    """Get live progress of the image indexing worker."""
    curr = lance_mgr.count() if lance_mgr else 0
    target = indexing_state["target_count"]
    pct = round((curr / target * 100), 1) if target > 0 else 0.0
    size_mb = lance_mgr.get_disk_size_mb() if lance_mgr else 0.0
    
    return {
        "is_indexing": indexing_state["is_indexing"],
        "cancel_requested": indexing_state["cancel_requested"],
        "current_count": curr,
        "target_count": target,
        "total_available": len(dataset) if dataset is not None else 0,
        "progress_percent": pct,
        "current_sample_idx": indexing_state["current_sample_idx"],
        "current_caption": indexing_state["current_caption"],
        "speed_sec_per_img": indexing_state["speed_sec_per_img"],
        "message": indexing_state["message"],
        "index_file_size_mb": size_mb,
        "storage_engine": "LanceDB"
    }


@app.post("/api/index/start")
def start_indexing(target: int = Query(150, ge=1, le=31783, description="Target total number of images to index")):
    """Start or expand background indexing in LanceDB up to target images."""
    global indexing_thread
    if indexing_state["is_indexing"]:
        raise HTTPException(status_code=400, detail="Indexing is already in progress")
        
    curr_count = lance_mgr.count() if lance_mgr else 0
    if target <= curr_count:
        raise HTTPException(
            status_code=400, 
            detail=f"Target ({target}) must be greater than current indexed count ({curr_count})"
        )
        
    indexing_thread = threading.Thread(target=background_indexer, args=(target,), daemon=True)
    indexing_thread.start()
    return {"status": "started", "target": target}


@app.post("/api/index/stop")
def stop_indexing():
    """Request graceful stop of the background indexing worker."""
    if not indexing_state["is_indexing"]:
        return {"status": "not_running", "message": "No indexing currently in progress"}
        
    indexing_state["cancel_requested"] = True
    indexing_state["message"] = "Stopping indexing (saving current image to LanceDB)..."
    return {"status": "stopping", "message": "Cancellation requested. Process will stop shortly."}


@app.api_route("/api/image/{sample_idx}", methods=["GET", "HEAD"])
def get_image(sample_idx: int):
    """Retrieve and serve image JPEG bytes by dataset sample index."""
    if dataset is None or sample_idx < 0 or sample_idx >= len(dataset):
        raise HTTPException(status_code=404, detail="Image not found")
        
    if sample_idx in image_cache:
        jpeg_bytes = image_cache[sample_idx]
    else:
        sample = dataset[sample_idx]
        img = sample["image"]
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        jpeg_bytes = buf.getvalue()
        if len(image_cache) < 300:
            image_cache[sample_idx] = jpeg_bytes
            
    return Response(
        content=jpeg_bytes,
        media_type="image/jpeg",
        headers={"Cache-Control": "public, max-age=86400"}
    )


@app.get("/api/search")
def search(
    q: str = Query(..., description="Text query to search for images", min_length=1),
    top_k: int = Query(20, ge=1, le=100, description="Number of results to return")
):
    """Perform text-to-image semantic search using LanceDB vector search."""
    if model is None or lance_mgr is None:
        raise HTTPException(status_code=503, detail="Search engine not ready")
        
    t0 = time.perf_counter()
    
    # 1. Embed user text query via EmbeddingGemma 2 -> 768-dim normalized vector
    query_vector = model.embed_text(q)
    t_embed = (time.perf_counter() - t0) * 1000
    
    # 2. LanceDB vector search (zero-copy memory-mapped search)
    t_search_start = time.perf_counter()
    results = lance_mgr.search(query_vector, top_k=top_k)
    t_search = (time.perf_counter() - t_search_start) * 1000
    
    total_time = (time.perf_counter() - t0) * 1000
    
    return {
        "query": q,
        "timing_ms": {
            "query_embedding": round(t_embed, 2),
            "vector_search": round(t_search, 2),
            "total_latency": round(total_time, 2)
        },
        "storage_engine": "LanceDB",
        "total_indexed": lance_mgr.count(),
        "results": results
    }


@app.get("/api/gallery")
def get_gallery_preview(count: int = Query(30, ge=1, le=100)):
    """Return a preview of indexed images directly from LanceDB."""
    if dataset is None or lance_mgr is None:
        raise HTTPException(status_code=503, detail="Dataset not ready")
        
    indexed_ids = sorted(list(lance_mgr.get_indexed_ids()))[:count]
    items = []
    for idx in indexed_ids:
        sample = dataset[int(idx)]
        items.append({
            "sample_idx": int(idx),
            "image_url": f"/api/image/{int(idx)}",
            "caption": sample["captions"][0],
            "captions": sample["captions"]
        })
    return {"gallery": items}


# Mount static assets
app.mount("/static", StaticFiles(directory="static"), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
