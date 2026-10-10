# 🔍 Local Multimodal Image Search with EmbeddingGemma 2 & LanceDB

A blazing-fast, 100% on-device text-to-image semantic search engine powered by **Google's EmbeddingGemma 2** (quantized ONNX) and **LanceDB** (embedded columnar vector database).

Runs entirely local on CPU — zero cloud API calls, full privacy, and sub-200ms retrieval latencies.

---

## ✨ Features

- **⚡️ 100% Offline & Private**: Zero external API calls or telemetry. Runs fully locally on CPU via ONNX Runtime.
- **🧠 Google EmbeddingGemma 2**: Projects natural language queries and raw images into a shared **768-dimensional** multimodal embedding space.
- **🚀 LanceDB Embedded Vector Store**: Zero-copy, memory-mapped vector search built on Apache Arrow format with $O(1)$ fast appends.
- **💻 Sleek Web UI**: Clean, responsive search interface with real-time similarity scores and latency counters.
- **📊 Real-Time Management Dashboard**: Monitor index growth, vector previews, disk usage, and trigger or pause background indexing on the fly.
- **⌨️ Interactive CLI**: Search your local dataset directly from the terminal.

---

## 🏗️ Architecture

```
User Query ("dog running in snow")
       │
       ▼
 EmbeddingGemma 2 (ONNX CPU)  ──▶  768-D Query Vector
       │
       ▼
 LanceDB Vector Table (Arrow) ──▶  Cosine Similarity Nearest-Neighbors
       │
       ▼
 Top-K Images + Metadata (< 200 ms total latency)
```

---

## 🚀 Quickstart

### 1. Clone & Set Up Environment

```bash
git clone https://github.com/hemant-manwani/embedded-image-search.git
cd embedded-image-search

# Optional: Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the Web Application

```bash
python app.py
```

Open your browser to:
- **Search UI**: [http://localhost:8000](http://localhost:8000)
- **Index Manager**: [http://localhost:8000/manage](http://localhost:8000/manage)

---

## 🖥️ Command-Line Interface (CLI)

### Interactive Search
Type any search query in real-time:
```bash
python interactive_search.py
```

### Single Query CLI
```bash
python image_search.py --query "a cat sleeping on a sofa" --top_k 5
```

---

## 📂 Project Structure

```
├── app.py                   # FastAPI backend server & API routes
├── embedding_gemma.py       # EmbeddingGemma 2 ONNX Runtime wrapper (Text & Vision)
├── lancedb_manager.py       # LanceDB vector storage, schema, and search operations
├── download_dataset.py      # Flickr30k dataset loader & compatibility patch
├── image_search.py          # CLI vector search utility
├── interactive_search.py    # REPL CLI for continuous queries
├── flickr30k_index.npz      # Starter pre-indexed embedding cache (384 vectors)
├── static/
│   ├── index.html           # Modern search engine web interface
│   ├── app.js               # Search client logic & animations
│   ├── manage.html          # Vector DB indexing and inspection dashboard
│   ├── manage.js            # Live indexing progress & metrics controller
│   └── style.css            # Dark-mode styling and UI components
├── requirements.txt         # Python package dependencies
└── README.md                # Project documentation
```

---

## 🛠️ Tech Stack

- **Model**: `onnx-community/embeddinggemma-2-ONNX` (Google EmbeddingGemma 2, 4-bit quantized)
- **Vector Database**: [LanceDB](https://lancedb.github.io/lancedb/) (Apache Arrow format)
- **Inference**: ONNX Runtime (CPUExecutionProvider)
- **Backend**: FastAPI + Uvicorn
- **Dataset**: Flickr30k (31,783 images)

---

## 📄 License

MIT License. See [LICENSE](LICENSE) for details.
