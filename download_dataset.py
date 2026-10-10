import os
import glob
from pathlib import Path

# Enable fast transfer if available
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

# Compatibility patch for Hugging Face datasets legacy '_type: List' schema
import datasets.features.features as ff
_orig_generate = ff.generate_from_dict
def _patched_generate_from_dict(obj):
    if isinstance(obj, dict) and obj.get("_type") == "List":
        return [_orig_generate(obj["feature"])]
    return _orig_generate(obj)
ff.generate_from_dict = _patched_generate_from_dict

from huggingface_hub import snapshot_download
from datasets import load_dataset


def load_flickr30k():
    """Load the full downloaded Flickr30k dataset."""
    repo_id = "AminDehnavi/flickr30k"
    return load_dataset(repo_id, split="train")


def main():
    repo_id = "AminDehnavi/flickr30k"
    print(f"Downloading {repo_id} (~4.2 GB total across 9 parquet shards)...")
    
    snapshot_path = snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        resume_download=True
    )
    print(f"\nFiles downloaded to: {snapshot_path}")
    
    print("\nLoading dataset into memory...")
    ds = load_flickr30k()
    print(f" Dataset fully ready! Total samples: {len(ds)}")
    print(f"Features: {list(ds.features.keys())}")
    print(f"\nSample 0:")
    print(f"  Captions: {ds[0]['captions']}")
    print(f"  Image size: {ds[0]['image'].size}")


if __name__ == "__main__":
    main()
