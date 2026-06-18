"""环境检查脚本。

检查所有模型、依赖、服务是否就绪。
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

def check(name, func):
    try:
        result = func()
        status = "OK" if result else "WARN"
        print(f"  [{status}] {name}: {result}")
    except Exception as e:
        print(f"  [FAIL] {name}: {e}")

print("=" * 60)
print("GradusRAG Environment Check")
print("=" * 60)

# 1. Python 依赖
print("\n--- Python Dependencies ---")
check("pymilvus", lambda: __import__("pymilvus").__version__)
check("sentence_transformers", lambda: __import__("sentence_transformers").__version__)
check("jieba", lambda: __import__("jieba").__version__ if hasattr(__import__("jieba"), "__version__") else "installed")
check("networkx", lambda: __import__("networkx").__version__)
check("fitz (PyMuPDF)", lambda: __import__("fitz").version)
check("python-docx", lambda: __import__("docx").__version__ if hasattr(__import__("docx"), "__version__") else "installed")
check("openpyxl", lambda: __import__("openpyxl").__version__)
check("langchain_openai", lambda: __import__("langchain_openai").__version__)
check("httpx", lambda: __import__("httpx").__version__)

# 2. 模型检查
print("\n--- Models ---")

def check_embedding():
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("BAAI/bge-m3")
    dim = model.encode("test").shape[0]
    return f"loaded, dim={dim}"

def check_cross_encoder():
    from sentence_transformers import CrossEncoder
    model = CrossEncoder("BAAI/bge-reranker-v2-m3")
    score = model.predict([("test query", "test document")])
    return f"loaded, sample_score={score[0]:.3f}"

check("BGE-M3 (embedding)", check_embedding)
check("bge-reranker-v2-m3 (cross-encoder)", check_cross_encoder)

# 3. Milvus 连接
print("\n--- Milvus ---")

def check_milvus():
    from src.core.settings import load_settings
    from src.retrieval.milvus_store import MilvusStore
    s = load_settings()
    store = MilvusStore(host=s.vector_store.host, port=s.vector_store.port, collection=s.vector_store.collection)
    count = store.count()
    return f"connected, {count} records"

check("Milvus connection", check_milvus)

# 4. 知识图谱
print("\n--- Knowledge Graph ---")

def check_graph():
    from src.ingestion.graph_builder.graph_store import GraphStore
    gs = GraphStore(persist_path="data/knowledge_graph.json")
    entities = gs.get_all_entities()
    relations = gs.get_all_relations()
    return f"{len(entities)} entities, {len(relations)} relations"

check("Knowledge graph", check_graph)

# 5. LLM API
print("\n--- LLM API ---")

def check_llm():
    from src.core.settings import load_settings
    from src.libs.llm_service import LLMService
    s = load_settings()
    llm = LLMService.from_settings(s)
    response = llm.invoke("Say 'OK' in one word.")
    return f"model={s.llm.model}, response='{response[:50]}'"

check("LLM API", check_llm)

# 6. 测试集
print("\n--- Test Set ---")

def check_testset():
    import json
    path = Path("data/test_sets/formal_test_set.json")
    if not path.exists():
        return "NOT FOUND"
    data = json.loads(path.read_text(encoding="utf-8"))
    from collections import Counter
    levels = Counter(d.get("expected_level", "?") for d in data)
    domains = Counter(d.get("_source_domain", "?") for d in data)
    return f"{len(data)} cases, levels={dict(levels)}, domains={dict(domains)}"

check("Test set", check_testset)

print("\n" + "=" * 60)
print("Check complete.")
