"""
LanceDB Storage & Vector Index Manager for Flickr30k
=====================================================
Embedded, zero-copy, Arrow-based vector database.
Stores 768-D image embeddings alongside metadata in a single table.
"""

import os
import shutil
import lancedb
import pyarrow as pa
import numpy as np
from typing import List, Dict, Set, Optional

LANCEDB_DIR = "flickr30k_lancedb"
TABLE_NAME = "images"
VECTOR_DIM = 768

# Define PyArrow schema
SCHEMA = pa.schema([
    pa.field("vector", pa.list_(pa.float32(), VECTOR_DIM)),
    pa.field("sample_idx", pa.int64()),
    pa.field("caption", pa.string()),
    pa.field("captions", pa.list_(pa.string())),
])


class LanceDBManager:
    def __init__(self, db_path: str = LANCEDB_DIR):
        self.db_path = db_path
        self.db = lancedb.connect(db_path)
        self.table = self._init_table()
        
    def _init_table(self):
        """Opens existing table or creates a new one."""
        if TABLE_NAME in self.db.table_names():
            return self.db.open_table(TABLE_NAME)
        else:
            return self.db.create_table(TABLE_NAME, schema=SCHEMA)

    def count(self) -> int:
        """Return total number of indexed vectors."""
        return len(self.table)

    def get_indexed_ids(self) -> Set[int]:
        """Return set of already-indexed Flickr30k sample indices."""
        if len(self.table) == 0:
            return set()
        return set(self.table.to_arrow().column("sample_idx").to_pylist())

    def add_records(self, records: List[Dict]):
        """
        Instant O(1) append of new vector + metadata records.
        Each record must contain: vector, sample_idx, caption, captions
        """
        if not records:
            return
        self.table.add(records)

    def search(self, query_vector: np.ndarray, top_k: int = 6) -> List[Dict]:
        """
        Perform vector similarity search.
        LanceDB uses cosine distance by default with metric='cosine'.
        Cosine similarity = 1 - (cosine distance).
        """
        if len(self.table) == 0:
            return []
            
        # Ensure query is float32 list
        q_vec = query_vector.astype(np.float32).tolist()
        
        # LanceDB vector search
        raw_results = (
            self.table.search(q_vec)
            .metric("cosine")
            .limit(top_k)
            .to_list()
        )
        
        results = []
        for rank, row in enumerate(raw_results, start=1):
            # _distance is cosine distance: range [0, 2]
            # similarity = 1 - distance
            dist = row.get("_distance", 0.0)
            sim_score = max(0.0, 1.0 - dist)
            
            results.append({
                "rank": rank,
                "sample_idx": int(row["sample_idx"]),
                "similarity_score": round(float(sim_score), 4),
                "match_percentage": round(float(sim_score) * 100, 1),
                "image_url": f"/api/image/{int(row['sample_idx'])}",
                "caption": row.get("caption", ""),
                "captions": row.get("captions", [])
            })
            
        return results

    def get_disk_size_mb(self) -> float:
        """Calculate total directory size of the LanceDB database."""
        total_bytes = 0
        if os.path.exists(self.db_path):
            for dirpath, _, filenames in os.walk(self.db_path):
                for f in filenames:
                    fp = os.path.join(dirpath, f)
                    if not os.path.islink(fp):
                        total_bytes += os.path.getsize(fp)
        return round(total_bytes / (1024 * 1024), 2)

    def get_recent_records(self, limit: int = 12) -> List[Dict]:
        """Fetch the most recently indexed records from the LanceDB table."""
        if len(self.table) == 0:
            return []
        arr = self.table.to_arrow()
        total = len(arr)
        count = min(limit, total)
        slice_tbl = arr.slice(total - count, count)
        rows = slice_tbl.to_pylist()
        records = []
        for row in reversed(rows):
            vec = row.get("vector") or []
            preview = [round(float(v), 4) for v in vec[:5]] if vec else []
            records.append({
                "sample_idx": int(row["sample_idx"]),
                "caption": row.get("caption", ""),
                "captions_count": len(row.get("captions", [])),
                "image_url": f"/api/image/{int(row['sample_idx'])}",
                "vector_dim": len(vec),
                "vector_preview": preview
            })
        return records



def migrate_from_npz_if_needed(lance_mgr: LanceDBManager, npz_path: str = "flickr30k_index.npz", dataset=None):
    """
    One-time migration: imports any existing vectors from flickr30k_index.npz
    into LanceDB so no prior computations are lost.
    """
    if not os.path.exists(npz_path):
        return
        
    indexed_in_lance = lance_mgr.get_indexed_ids()
    
    data = np.load(npz_path, allow_pickle=True)
    npz_embeddings = data["embeddings"]
    npz_indices = data["indices"]
    
    new_records = []
    for emb, s_idx in zip(npz_embeddings, npz_indices):
        s_idx = int(s_idx)
        if s_idx in indexed_in_lance:
            continue
            
        captions = dataset[s_idx]["captions"] if dataset else [""]
        new_records.append({
            "vector": emb.astype(np.float32).tolist(),
            "sample_idx": s_idx,
            "caption": captions[0] if captions else "",
            "captions": captions
        })
        
    if new_records:
        print(f"[LanceDB] Migrating {len(new_records)} vectors from {npz_path} to LanceDB...")
        lance_mgr.add_records(new_records)
        print(f"[LanceDB] Migration complete! LanceDB now has {lance_mgr.count()} records.")
