# GradusRAG

**Multi-level Query Understanding and Hybrid Enhanced Retrieval-Augmented Generation**

> [中文版 README](README.zh.md)

GradusRAG is a domain-specific intelligent question answering system that automatically classifies user queries into four complexity levels (L1-L4) and dynamically routes each level to the optimal retrieval and generation strategy.

---

## Features

- **Four-Level Query Classification** — L1 Explicit Facts / L2 Implicit Facts / L3 Interpretable Principles / L4 Hidden Principles, with automatic classification and dynamic routing
- **Hybrid Retrieval** — Dense (BGE-M3 embeddings) + Sparse (BM25 keywords) + Graph (knowledge graph multi-hop traversal), combined via RRF fusion, with level-based routing
- **Adaptive Generation** — L1 direct answer / L2 structured analysis / L3 Chain-of-Thought reasoning / L4 Self-RAG iterative critique
- **Knowledge Graph Enhancement** — LLM-based entity and relation extraction, NetworkX graph storage, multi-hop graph retrieval for complex queries
- **Full Observability** — Real-time RAG step visualization, structured trace logging, Streamlit management dashboard
- **Multi-Format Document Support** — PDF, Word (.docx), Excel (.xlsx), TXT, HTML, Markdown
- **Bilingual UI** — Chinese / English toggle, light / dark theme support

---

## Architecture

```
Query → L1-L4 Classification → Dynamic Routing
  │
  ├─ L1: Dense + Sparse → RRF → Direct Generation
  ├─ L2: Dense + Sparse → RRF → Rerank → Structured Analysis
  ├─ L3: Dense + Sparse + Graph(2-hop) → RRF → Rerank → CoT Reasoning
  └─ L4: Dense + Sparse + Graph(1-hop) → RRF → Rerank → Self-RAG Iteration
                                                  ↑ draft→critique→re-retrieve→refine
```

```
GradusRAG_V0/
├── src/
│   ├── core/                    # Types, config, tracing
│   ├── query_classifier/        # L1-L4 classifier (rule + LLM hybrid)
│   ├── ingestion/               # Document loaders, chunking, graph builder
│   ├── retrieval/               # Dense, Sparse, Graph, Hybrid search, Reranker
│   ├── generation/              # Grader, Rewriter, Response generator, Self-RAG
│   ├── evaluation/              # Metrics, test set manager, evaluator
│   ├── libs/                    # LLM service, Embedding service
│   └── api/                     # FastAPI backend
├── frontend/
│   └── app.py                   # Streamlit frontend
├── scripts/                     # CLI tools
├── tests/                       # Unit tests
├── config/
│   ├── settings.yaml.example    # Configuration template
│   └── prompts/                 # Prompt templates
├── data/
│   ├── documents/               # Source documents
│   └── test_sets/               # Golden test set (23 cases)
└── docker-compose.yml           # Milvus + PostgreSQL + Redis
```

---

## Installation

### Prerequisites

- Python 3.10+
- Docker Desktop (for Milvus vector database)
- An OpenAI-compatible API key

### Install from Source

```bash
git clone https://github.com/your-username/GradusRAG.git
cd GradusRAG_V0

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux

# Install dependencies
pip install -e .
pip install pymupdf jieba pyvis streamlit

# Configure
cp config/settings.yaml.example config/settings.yaml
# Edit config/settings.yaml — fill in your API key and base URL
```

### Start Dependencies

```bash
docker compose up -d
```

This starts Milvus (vector database), PostgreSQL, and Redis.

---

## Quick Start

### 1. Ingest Documents

```bash
# Ingest a single file
python scripts/ingest.py -i "data/documents/your_document.pdf"

# Build knowledge graph
python scripts/build_graph.py -i "data/documents/your_document.pdf"
```

### 2. Query via CLI

```bash
python scripts/query.py "什么是机器学习？"
python scripts/query.py "监督学习和无监督学习有什么区别？"
python scripts/query.py "为什么会出现梯度消失？"
python scripts/query.py "如果用ReLU完全替代Sigmoid会怎样？"
```

### 3. Launch Web UI

```bash
# Terminal 1: Backend
python -m uvicorn src.api.app:app --host 127.0.0.1 --port 8001

# Terminal 2: Frontend
streamlit run frontend/app.py --server.port 8501
```

Open `http://localhost:8501` in your browser.

---

## Configuration

All configuration is in `config/settings.yaml`. Copy the template first:

```bash
cp config/settings.yaml.example config/settings.yaml
```

Key configuration sections:

| Section | Description |
|---------|-------------|
| `llm` | LLM provider, model, API key, base URL |
| `embedding` | Embedding model (local BGE-M3 or API) |
| `vector_store` | Milvus host/port/collection |
| `retrieval` | Top-K, RRF fusion parameters |
| `rerank` | Cross-encoder or LLM reranking |
| `ingestion` | Chunk sizes for three-level splitting |
| `query_classifier` | Classification mode (rule/llm/hybrid) |
| `graph` | Knowledge graph settings |
| `generation` | Strategy per query level |

**Important**: Never commit `config/settings.yaml` — it contains your API keys. The `.gitignore` excludes it by default.

---

## API Reference

### Query

```
POST /api/query
Content-Type: application/json

{"question": "什么是机器学习？"}
```

Response includes `answer`, `citations`, `query_level`, `query_type`, `confidence`, and `retrieval_trace`.

### Query (Streaming)

```
POST /api/query/stream
```

Returns Server-Sent Events with real-time RAG process steps.

### Documents

```
POST /api/documents/upload    — Upload and ingest a file
GET  /api/documents           — List ingested documents
DELETE /api/documents/{name}  — Delete document and all associated data
```

### Knowledge Graph

```
GET /api/graph/stats              — Graph statistics
GET /api/graph/data               — Full graph data (for visualization)
GET /api/graph/entities?keyword=X — Search entities
GET /api/graph/neighbors/{name}   — Get entity neighbors
```

### Health

```
GET /api/health
```

---

## Evaluation

GradusRAG includes 7 evaluation metrics aligned with RAGAS:

| Metric | Description |
|--------|-------------|
| Classification Accuracy | L1-L4 classification accuracy |
| Hit Rate@K | Whether correct document appears in top-K |
| MRR | Mean Reciprocal Rank |
| Faithfulness | Whether answer is grounded in retrieved context |
| Answer Relevance | Whether answer addresses the question |
| Context Recall | Whether all relevant information was retrieved |
| Context Precision | Whether retrieved context is clean |

Run evaluation:

```bash
# Full evaluation
python scripts/evaluate.py --output results/full.json

# By query level
python scripts/evaluate.py --by-level --output results/by_level.json

# Quick test (first N cases)
python scripts/evaluate.py --max-cases 4
```

The Golden Test Set (`data/test_sets/golden_test_set.json`) contains 23 annotated cases across all four levels.

---

## Query Level Examples

| Level | Example | Retrieval | Generation |
|-------|---------|-----------|------------|
| L1 Explicit | "什么是机器学习？" | Dense + Sparse | Direct answer |
| L2 Implicit | "监督学习和无监督学习有什么区别？" | Dense + Sparse + Rerank | Structured analysis |
| L3 Causal | "为什么会出现梯度消失？" | Dense + Sparse + Graph + Rerank | CoT reasoning |
| L4 Hypothetical | "如果用ReLU完全替代Sigmoid会怎样？" | Dense + Sparse + Graph + Rerank | Self-RAG iteration |

---

## Troubleshooting

**Milvus connection refused**
Make sure Docker is running and Milvus is healthy: `docker compose ps`

**HuggingFace download timeout**
Set the mirror: `$env:HF_ENDPOINT = "https://hf-mirror.com"` (Windows PowerShell)

**PyTorch crash on import**
Downgrade PyTorch: `pip install torch==2.11.0 --extra-index-url https://download.pytorch.org/whl/cpu`

**Embedding model slow on CPU**
Use a smaller model in `config/settings.yaml`: change `model` to `BAAI/bge-small-zh-v1.5` and `dimensions` to `512`.

---

## Citation

```bibtex
@software{gradusrag2026,
  title = {GradusRAG: Multi-level Query Understanding and Hybrid Enhanced RAG},
  year = {2026},
  url = {https://github.com/your-username/GradusRAG}
}
```

---

## License

MIT
