# GradusRAG

**面向专业领域的多级查询理解与混合增强检索增强生成系统**

GradusRAG 是一个领域智能问答系统，能够自动将用户问题分为四个复杂度级别（L1-L4），并为每个级别动态路由到最优的检索策略和生成策略。

---

## 核心特性

- **四级查询分类** — L1 显性事实 / L2 隐性事实 / L3 可解释原理 / L4 隐藏原理，自动分类与动态路由
- **混合检索增强** — Dense（BGE-M3 语义向量）+ Sparse（BM25 关键词）+ Graph（知识图谱多跳遍历），通过 RRF 融合，按查询级别动态组合
- **动态生成策略** — L1 直接回答 / L2 结构化分析 / L3 Chain-of-Thought 推理 / L4 Self-RAG 迭代批判
- **知识图谱增强** — LLM 实体与关系抽取，NetworkX 图存储，多跳图检索增强复杂查询
- **全链路可观测** — 实时 RAG 步骤可视化，结构化 Trace 追踪，Streamlit 管理面板
- **多格式文档支持** — PDF、Word（.docx）、Excel（.xlsx）、TXT、HTML、Markdown
- **双语界面** — 中文 / 英文切换，深色 / 浅色主题适配

---

## 系统架构

```
用户提问 → L1-L4 分类 → 动态路由
  │
  ├─ L1: Dense + Sparse → RRF → 直接生成
  ├─ L2: Dense + Sparse → RRF → Rerank → 结构化分析
  ├─ L3: Dense + Sparse + Graph(2跳) → RRF → Rerank → CoT 推理
  └─ L4: Dense + Sparse + Graph(1跳) → RRF → Rerank → Self-RAG 迭代
                                                ↑ 初稿→批判→重检索→改进→再批判
```

### 项目结构

```
GradusRAG_V0/
├── ARCHITECTURE.md              # 系统架构设计文档
├── DESIGN_ANALYSIS.md           # 设计分析与改进计划
├── SETUP_GUIDE.md               # 环境搭建指南
├── pyproject.toml               # 项目依赖
├── docker-compose.yml           # Docker 依赖服务
├── config/
│   ├── settings.yaml.example    # 配置模板
│   └── prompts/                 # Prompt 模板（7个）
├── src/
│   ├── core/                    # 核心类型、配置、追踪
│   ├── query_classifier/        # 四级查询分类器
│   ├── ingestion/               # 文档加载、分块、图谱构建
│   ├── retrieval/               # Dense、Sparse、Graph、混合检索、Reranker
│   ├── generation/              # 评分、重写、分级生成、Self-RAG
│   ├── evaluation/              # 指标、测试集、评估器
│   ├── libs/                    # LLM 服务、嵌入服务
│   └── api/                     # FastAPI 后端
├── frontend/
│   └── app.py                   # Streamlit 前端
├── scripts/                     # 命令行工具
├── tests/                       # 单元测试
├── data/
│   ├── documents/               # 原始文档
│   └── test_sets/               # Golden Test Set（23条）
└── results/                     # 评估结果输出
```

---

## 安装

### 环境要求

- Python 3.10+
- Docker Desktop（用于 Milvus 向量数据库）
- OpenAI 兼容 API Key

### 从源码安装

```bash
git clone https://github.com/your-username/GradusRAG.git
cd GradusRAG_V0

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

将启动 Milvus（向量数据库）、PostgreSQL 和 Redis。

---

## 快速开始

### 1. 摄取文档

```bash
# 摄取单个文件
python scripts/ingest.py -i "data/documents/your_document.pdf"

# 构建知识图谱
python scripts/build_graph.py -i "data/documents/your_document.pdf"
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
| `rerank` | Cross-encoder 或 LLM 重排序 |
| `ingestion` | 三级分块大小 |
| `query_classifier` | 分类模式（rule/llm/hybrid） |
| `graph` | 知识图谱配置 |
| `generation` | 各级别生成策略 |

**注意**：`config/settings.yaml` 包含 API 密钥，不会提交到仓库。`.gitignore` 已默认排除。

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

返回 SSE 事件流，实时推送 RAG 过程步骤。

### 文档管理

```
POST /api/documents/upload    — 上传并摄取文件
GET  /api/documents           — 列出已入库文档
DELETE /api/documents/{name}  — 删除文档及所有关联数据（向量、图谱、BM25）
```

### 知识图谱

```
GET /api/graph/stats              — 图谱统计
GET /api/graph/data               — 完整图数据（用于可视化）
GET /api/graph/entities?keyword=X — 搜索实体
GET /api/graph/neighbors/{name}   — 获取实体邻居
```

### 健康检查

```
GET /api/health
```

---

## 评估指标

GradusRAG 包含 7 个评估指标，与 RAGAS 对齐：

| 指标 | 说明 |
|------|------|
| Classification Accuracy | L1-L4 分类准确率 |
| Hit Rate@K | Top-K 中是否包含正确文档 |
| MRR | 平均倒数排名 |
| Faithfulness | 回答是否忠于检索内容 |
| Answer Relevance | 回答与问题的相关性 |
| Context Recall | 检索是否覆盖所有相关信息 |
| Context Precision | 检索结果是否干净无噪声 |

运行评估：

```bash
# 全量评估
python scripts/evaluate.py --output results/full.json

# 按级别评估
python scripts/evaluate.py --by-level --output results/by_level.json

# 快速测试（前 N 条）
python scripts/evaluate.py --max-cases 4
```

Golden Test Set（`data/test_sets/golden_test_set.json`）包含 23 条覆盖四级的标注数据。

---

## 查询级别示例

| 级别 | 示例问题 | 检索策略 | 生成策略 |
|------|----------|----------|----------|
| L1 显性事实 | "什么是机器学习？" | Dense + Sparse | 直接回答 |
| L2 隐性事实 | "监督学习和无监督学习有什么区别？" | Dense + Sparse + Rerank | 结构化分析 |
| L3 可解释原理 | "为什么会出现梯度消失？" | Dense + Sparse + Graph + Rerank | CoT 推理 |
| L4 隐藏原理 | "如果用ReLU完全替代Sigmoid会怎样？" | Dense + Sparse + Graph + Rerank | Self-RAG 迭代 |

---

## 常见问题

**Milvus 连接失败**
确认 Docker 已启动且 Milvus 健康：`docker compose ps`

**HuggingFace 下载超时**
设置镜像：`$env:HF_ENDPOINT = "https://hf-mirror.com"`（Windows PowerShell）

**PyTorch 导入崩溃**
降级 PyTorch：`pip install torch==2.11.0 --extra-index-url https://download.pytorch.org/whl/cpu`

**CPU 上嵌入模型太慢**
在 `config/settings.yaml` 中换用小模型：`model: "BAAI/bge-small-zh-v1.5"`，`dimensions: 512`

---

## 引用

```bibtex
@software{gradusrag2026,
  title = {GradusRAG: Multi-level Query Understanding and Hybrid Enhanced RAG},
  year = {2026},
  url = {https://github.com/your-username/GradusRAG}
}
```

---

## 许可证

MIT
