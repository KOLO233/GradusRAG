# GradusRAG

**分级自适应检索增强生成：面向专业领域的多级查询理解与混合增强 RAG 问答系统**

[English README](README.md)

GradusRAG 是一个面向专业领域（AI/ML、医学、教育、法律、金融等）的智能问答系统，能够自动将用户问题分为四个复杂度级别（L1-L4），并为每个级别动态路由到最优的检索策略和生成策略。系统名称"Gradus"源自拉丁语，意为"阶梯/级别"，体现了分级递进处理的核心设计理念。

---

## 核心特性

- **四级查询分类** — L1 显性事实 / L2 隐性事实 / L3 可解释原理 / L4 隐藏原理，采用加权评分规则 + LLM 确认的混合分类策略
- **级别自适应混合检索** — Dense（BGE-M3 语义向量）+ Sparse（BM25 关键词）+ Graph（知识图谱多跳遍历）+ Cross-Encoder 重排序，通过 RRF 融合，按查询级别动态组合
- **分级生成策略** — L1 直接回答 / L2 结构化分析 / L3 思维链推理 / L4 Self-RAG 迭代批判（初稿→批判→重检索→精炼）
- **知识图谱增强** — 基于 LLM 的实体与关系自动抽取，NetworkX 图存储，多跳图检索增强复杂查询推理
- **Cross-Encoder 重排序** — bge-reranker-v2-m3，70% CE + 30% RRF 混合评分策略
- **中英双语支持** — 中英文查询分类与生成
- **全链路可观测** — SSE 实时流式输出，结构化 Trace 追踪，FastAPI Swagger 文档

---

## 系统架构

```
用户提问 → L1-L4 分类 → 动态路由
  │
  ├─ L1: Dense + Sparse → RRF → 直接生成
  ├─ L2: Dense + Sparse → RRF → Rerank → 结构化分析
  ├─ L3: Dense + Sparse + Graph(2跳) → RRF → Rerank → CoT 推理
  └─ L4: Dense + Sparse + Graph(1跳) → RRF → Rerank → Self-RAG 迭代
                                                ↑ 初稿→批判→重检索→精炼
```

### 项目结构

```
GradusRAG/
├── src/
│   ├── core/                    # 核心类型、配置、追踪
│   ├── query_classifier/        # 四级查询分类器（规则+LLM混合）
│   ├── ingestion/               # 文档加载、三级分块、图谱构建
│   ├── retrieval/               # Dense、Sparse、Graph、混合检索、Reranker
│   ├── generation/              # 评分、重写、分级生成、Self-RAG
│   ├── evaluation/              # 指标、评估器、Pairwise 评估
│   ├── libs/                    # LLM 服务、嵌入服务
│   └── api/                     # FastAPI 后端
├── frontend/
│   └── app.py                   # Streamlit 前端
├── scripts/                     # 命令行工具
├── tests/                       # 单元测试
├── config/
│   ├── settings.yaml.example    # 配置模板
│   └── prompts/                 # Prompt 模板（7个）
├── data/
│   └── documents/               # 知识库文档（不入库）
├── docker-compose.yml           # Milvus + etcd + MinIO
└── pyproject.toml               # 项目依赖
```

---

## 安装

### 环境要求

- Python 3.10+（推荐 3.11）
- Docker Desktop（用于 Milvus 向量数据库）
- OpenAI 兼容 API Key（DashScope、DeepSeek、OpenAI 等均可）

### 从源码安装

```bash
git clone https://github.com/KOLO233/GradusRAG.git
cd GradusRAG

# 创建虚拟环境
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux

# 安装依赖
pip install -e .
pip install pymupdf jieba pyvis streamlit

# 配置
cp config/settings.yaml.example config/settings.yaml
# 编辑 config/settings.yaml，填入你的 API Key 和 Base URL
```

### 启动依赖服务

```bash
docker compose up -d
```

启动 Milvus（向量数据库，端口 19530）、etcd 和 MinIO。

---

## 快速开始

### 1. 文档入库

```bash
# 入库单个文件
python scripts/ingest.py --input data/documents/your_document.pdf

# 入库整个目录
python scripts/ingest.py --input data/documents/

# 构建知识图谱（入库后执行）
python scripts/build_graph.py --input data/documents/
```

### 2. 命令行查询

```bash
python scripts/query.py "什么是机器学习？"
python scripts/query.py "监督学习和无监督学习有什么区别？"
python scripts/query.py "为什么会出现梯度消失？"
python scripts/query.py "如果用ReLU完全替代Sigmoid会怎样？"
```

### 3. 启动 Web 界面

```bash
# 终端 1：后端
python -m uvicorn src.api.app:app --host 127.0.0.1 --port 8001

# 终端 2：前端
streamlit run frontend/app.py --server.port 8501
```

浏览器打开 `http://localhost:8501`。

---

## 配置说明

所有配置在 `config/settings.yaml` 中。先复制模板：

```bash
cp config/settings.yaml.example config/settings.yaml
```

| 配置项 | 说明 |
|--------|------|
| `llm` | LLM 提供商、模型名、API Key、Base URL |
| `embedding` | 嵌入模型（本地 BGE-M3 或 API） |
| `vector_store` | Milvus 地址和集合名 |
| `retrieval` | Top-K、RRF 融合参数 |
| `rerank` | Cross-encoder 重排序配置 |
| `ingestion` | 三级分块大小（L1: 3000, L2: 1500, L3: 800 tokens） |
| `query_classifier` | 分类模式（rule/llm/hybrid） |
| `graph` | 知识图谱配置 |
| `generation` | 各级别生成策略（L1-L4） |

**注意**：`config/settings.yaml` 包含 API 密钥，不会提交到仓库。

---

## 评估框架

GradusRAG 包含 7 个评估指标（与 RAGAS 对齐）+ 四维度 Pairwise 对比：

| 指标 | 类型 | 说明 |
|------|------|------|
| Classification Accuracy | 确定性 | L1-L4 分类准确率 |
| Hit Rate@K | 确定性 | Top-K 中是否包含正确文档 |
| MRR | 确定性 | 平均倒数排名 |
| Faithfulness | LLM 评估 | 回答是否忠于检索内容 |
| Answer Relevance | LLM 评估 | 回答与问题的相关性 |
| Context Recall | LLM 评估 | 检索是否覆盖所有相关信息 |
| Context Precision | LLM 评估 | 检索结果是否干净无噪声 |
| Pairwise Win Rate | LLM-as-Judge | 四维度对比（全面性/多样性/启发性/综合质量） |

运行评估：

```bash
# 全量评估
python scripts/run_evaluation.py --test-set formal_test_set.json --output results/main.json

# 消融实验（6 组配置）
python scripts/ablation.py --test-set formal_test_set.json --output results/ablation.json --max-cases 200

# Pairwise 对比（GradusRAG vs 基线）
python scripts/run_pairwise_eval.py --test-set formal_test_set.json --output results/pairwise.json

# 公开数据集域外评估（HotpotQA、SQuAD、PubMedQA）
python scripts/eval_public_datasets.py --dataset all --sample 50
```

---

## 查询级别示例

| 级别 | 示例问题 | 检索策略 | 生成策略 |
|------|----------|----------|----------|
| L1 显性事实 | "什么是机器学习？" | Dense + Sparse | 直接回答 |
| L2 隐性事实 | "监督学习和无监督学习有什么区别？" | Dense + Sparse + Rerank | 结构化分析 |
| L3 可解释原理 | "为什么会出现梯度消失？" | Dense + Sparse + Graph(2跳) + Rerank | CoT 推理 |
| L4 隐藏原理 | "如果用ReLU完全替代Sigmoid会怎样？" | Dense + Sparse + Graph(1跳) + Rerank | Self-RAG 迭代 |

---

## API 接口

### 查询

```
POST /api/query
Content-Type: application/json

{"question": "什么是机器学习？"}
```

返回 `answer`、`citations`、`query_level`、`query_type`、`confidence`、`retrieval_trace`。

### 流式查询

```
POST /api/query/stream
```

返回 SSE 事件流，实时推送 RAG 处理的每个步骤（分类→检索→评分→生成）。

### 文档管理

```
POST /api/documents/upload    — 上传并入库文件
GET  /api/documents           — 列出已入库文档
DELETE /api/documents/{name}  — 删除文档及所有关联数据（向量、图谱、BM25）
```

### 知识图谱

```
GET /api/graph/stats              — 图谱统计（实体数、关系数）
GET /api/graph/data               — 完整图数据（用于可视化）
GET /api/graph/entities?keyword=X — 搜索实体
GET /api/graph/neighbors/{name}   — 获取实体邻居（多跳）
```

---

## 常见问题

**Milvus 连接失败**：确认 Docker 已启动：`docker compose ps`

**HuggingFace 下载超时**：设置镜像：`$env:HF_ENDPOINT = "https://hf-mirror.com"`（Windows PowerShell）

**嵌入模型在 CPU 上太慢**：在 `config/settings.yaml` 中换用小模型：`model: "BAAI/bge-small-zh-v1.5"`，`dimensions: 512`

---

## 引用

```bibtex
@software{gradusrag2026,
  title = {GradusRAG: Graduated Adaptive Retrieval-Augmented Generation},
  author = {KOLO233},
  year = {2026},
  url = {https://github.com/KOLO233/GradusRAG}
}
```

---

## 许可证

MIT
