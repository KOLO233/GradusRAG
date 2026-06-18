# GradusRAG

**Graduated Adaptive Retrieval-Augmented Generation: Multi-level Query Understanding and Hybrid Enhanced RAG for Domain-Specific Question Answering**

> [中文版 README](README.zh.md)

GradusRAG is a domain-specific intelligent QA system that automatically classifies user queries into four complexity levels (L1-L4) and dynamically routes each to the optimal retrieval and generation strategy. The name "Gradus" (Latin for "step/level") reflects the system's core design philosophy: graduated, level-adaptive processing.

---

## Key Features

- **Four-Level Query Classification** — L1 Explicit Facts / L2 Implicit Facts / L3 Interpretable Principles / L4 Hidden Principles, using a hybrid rule-LLM classifier with weighted scoring
- **Level-Adaptive Hybrid Retrieval** — Dense (BGE-M3) + Sparse (BM25) + Knowledge Graph (multi-hop traversal) + Cross-Encoder Reranking, dynamically composed per query level via RRF fusion
- **Tiered Generation Strategies** — L1 direct answer / L2 structured analysis / L3 Chain-of-Thought reasoning / L4 Self-RAG iterative critique (draft → critique → re-retrieve → refine)
- **Knowledge Graph Enhancement** — LLM-based entity and relation extraction, NetworkX graph storage, multi-hop graph retrieval for complex queries
- **Cross-Encoder Reranking** — bge-reranker-v2-m3 with 70% CE + 30% RRF hybrid scoring
- **Bilingual Support** — Chinese and English query classification and generation
- **Full Observability** — Real-time SSE streaming, structured trace logging, FastAPI Swagger docs

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

### Project Structure

```
GradusRAG/
├── src/
│   ├── core/                    # Types, config, tracing
│   ├── query_classifier/        # L1-L4 classifier (rule + LLM hybrid)
│   ├── ingestion/               # Document loaders, chunking, graph builder
│   ├── retrieval/               # Dense, Sparse, Graph, Hybrid search, Reranker
│   ├── generation/              # Grader, Rewriter, Response generator, Self-RAG
│   ├── evaluation/              # Metrics, evaluator, pairwise evaluation
│   ├── libs/                    # LLM service, Embedding service
│   └── api/                     # FastAPI backend
├── frontend/
│   └── app.py                   # Streamlit frontend
├── scripts/                     # CLI tools (ingest, query, evaluate, ablation, etc.)
├── tests/                       # Unit tests
├── config/
│   ├── settings.yaml.example    # Configuration template
│   └── prompts/                 # Prompt templates (7 files)
├── data/
│   └── documents/               # Knowledge base documents (not tracked)
├── docker-compose.yml           # Milvus + PostgreSQL + Redis
└── pyproject.toml               # Project dependencies
```

---

## Installation

### Prerequisites

- Python 3.10+ (3.11 recommended)
- Docker Desktop (for Milvus vector database)
- An OpenAI-compatible API key (DashScope, DeepSeek, OpenAI, etc.)

### Install from Source

```bash
git clone https://github.com/KOLO233/GradusRAG.git
cd GradusRAG

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

This starts Milvus (vector database on port 19530), etcd, and MinIO.

---

## Quick Start

### 1. Ingest Documents

```bash
# Ingest a single file
python scripts/ingest.py --input data/documents/your_document.pdf

# Ingest an entire directory
python scripts/ingest.py --input data/documents/

# Build knowledge graph (after ingestion)
python scripts/build_graph.py --input data/documents/
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

| Section | Description |
|---------|-------------|
| `llm` | LLM provider, model, API key, base URL |
| `embedding` | Embedding model (local BGE-M3 or API) |
| `vector_store` | Milvus host/port/collection |
| `retrieval` | Top-K, RRF fusion parameters |
| `rerank` | Cross-encoder reranking settings |
| `ingestion` | Chunk sizes for three-level splitting |
| `query_classifier` | Classification mode (rule/llm/hybrid) |
| `graph` | Knowledge graph settings |
| `generation` | Strategy per query level (L1-L4) |

**Important**: Never commit `config/settings.yaml` — it contains your API keys.

---

## Evaluation

GradusRAG includes a comprehensive evaluation framework with 7 metrics (aligned with RAGAS) plus pairwise comparison:

| Metric | Type | Description |
|--------|------|-------------|
| Classification Accuracy | Deterministic | L1-L4 classification accuracy |
| Hit Rate@K | Deterministic | Whether correct document appears in top-K |
| MRR | Deterministic | Mean Reciprocal Rank |
| Faithfulness | LLM-evaluated | Whether answer is grounded in retrieved context |
| Answer Relevance | LLM-evaluated | Whether answer addresses the question |
| Context Recall | LLM-evaluated | Whether all relevant information was retrieved |
| Context Precision | LLM-evaluated | Whether retrieved context is clean |
| Pairwise Win Rate | LLM-as-Judge | Head-to-head comparison (4 dimensions) |

Run evaluation:

```bash
# Full evaluation
python scripts/run_evaluation.py --test-set formal_test_set.json --output results/main.json

# Ablation study (6 configurations)
python scripts/ablation.py --test-set formal_test_set.json --output results/ablation.json

# Pairwise comparison (GradusRAG vs baseline)
python scripts/run_pairwise_eval.py --test-set formal_test_set.json --output results/pairwise.json

# Public dataset evaluation (HotpotQA, SQuAD, PubMedQA)
python scripts/eval_public_datasets.py --dataset all --sample 50
```

---

## Query Level Examples

| Level | Example | Retrieval | Generation |
|-------|---------|-----------|------------|
| L1 Explicit | "什么是机器学习？" | Dense + Sparse | Direct answer |
| L2 Implicit | "监督学习和无监督学习有什么区别？" | Dense + Sparse + Rerank | Structured analysis |
| L3 Causal | "为什么会出现梯度消失？" | Dense + Sparse + Graph(2-hop) + Rerank | CoT reasoning |
| L4 Hypothetical | "如果用ReLU完全替代Sigmoid会怎样？" | Dense + Sparse + Graph(1-hop) + Rerank | Self-RAG iteration |

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

---

## Troubleshooting

**Milvus connection refused**: Make sure Docker is running: `docker compose ps`

**HuggingFace download timeout**: Set mirror: `$env:HF_ENDPOINT = "https://hf-mirror.com"` (Windows PowerShell)

**Embedding model slow on CPU**: Use a smaller model in `config/settings.yaml`: `model: "BAAI/bge-small-zh-v1.5"`, `dimensions: 512`

---

## Citation

```bibtex
@software{gradusrag2026,
  title = {GradusRAG: Graduated Adaptive Retrieval-Augmented Generation},
  author = {KOLO233},
  year = {2026},
  url = {https://github.com/KOLO233/GradusRAG}
}
```

---

## License

MIT
