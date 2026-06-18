"""GradusRAG 配置加载与验证。

从 YAML 文件加载配置，映射为 frozen dataclass，确保类型安全。
设计参考 MODULAR-RAG-MCP-SERVER 的 settings 模块。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml


# ---------------------------------------------------------------------------
# 路径解析
# ---------------------------------------------------------------------------
REPO_ROOT: Path = Path(__file__).resolve().parents[2]
DEFAULT_SETTINGS_PATH: Path = REPO_ROOT / "config" / "settings.yaml"


def resolve_path(relative: Union[str, Path]) -> Path:
    """将相对路径解析为绝对路径（基于 REPO_ROOT）。"""
    p = Path(relative)
    if p.is_absolute():
        return p
    return (REPO_ROOT / p).resolve()


class SettingsError(ValueError):
    """配置验证失败时抛出。"""


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------
def _get(data: Dict[str, Any], key: str, path: str) -> Any:
    if key not in data or data.get(key) is None:
        raise SettingsError(f"Missing required field: {path}.{key}")
    return data[key]


def _get_str(data: Dict[str, Any], key: str, path: str) -> str:
    val = _get(data, key, path)
    if not isinstance(val, str) or not val.strip():
        raise SettingsError(f"Expected non-empty string: {path}.{key}")
    return val


def _get_int(data: Dict[str, Any], key: str, path: str) -> int:
    val = _get(data, key, path)
    if not isinstance(val, int):
        raise SettingsError(f"Expected integer: {path}.{key}")
    return val


def _get_float(data: Dict[str, Any], key: str, path: str) -> float:
    val = _get(data, key, path)
    if not isinstance(val, (int, float)):
        raise SettingsError(f"Expected number: {path}.{key}")
    return float(val)


def _get_bool(data: Dict[str, Any], key: str, path: str) -> bool:
    val = _get(data, key, path)
    if not isinstance(val, bool):
        raise SettingsError(f"Expected boolean: {path}.{key}")
    return val


def _get_list(data: Dict[str, Any], key: str, path: str) -> List[Any]:
    val = _get(data, key, path)
    if not isinstance(val, list):
        raise SettingsError(f"Expected list: {path}.{key}")
    return val


# ---------------------------------------------------------------------------
# 配置数据类
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LLMSettings:
    provider: str
    model: str
    temperature: float
    max_tokens: int
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    api_version: Optional[str] = None
    azure_endpoint: Optional[str] = None
    deployment_name: Optional[str] = None


@dataclass(frozen=True)
class EmbeddingSettings:
    provider: str
    model: str
    dimensions: int
    device: str = "cpu"
    api_key: Optional[str] = None
    api_base_url: Optional[str] = None


@dataclass(frozen=True)
class VectorStoreSettings:
    provider: str
    host: str = "127.0.0.1"
    port: int = 19530
    collection: str = "gradusrag"
    persist_directory: str = "./data/db/chroma"


@dataclass(frozen=True)
class RetrievalSettings:
    dense_top_k: int
    sparse_top_k: int
    fusion_top_k: int
    rrf_k: int


@dataclass(frozen=True)
class RerankSettings:
    enabled: bool
    provider: str
    model: str
    top_k: int


@dataclass(frozen=True)
class IngestionSettings:
    chunk_size_l1: int
    chunk_size_l2: int
    chunk_size_l3: int
    chunk_overlap: int
    splitter: str


@dataclass(frozen=True)
class QueryClassifierSettings:
    mode: str
    llm_threshold: float


@dataclass(frozen=True)
class GraphSettings:
    enabled: bool
    storage: str
    entity_types: List[str]
    max_hops: int


@dataclass(frozen=True)
class GenerationSettings:
    l1_strategy: str
    l2_strategy: str
    l3_strategy: str
    l4_strategy: str
    max_self_rag_iterations: int
    faithfulness_verify: bool = True


@dataclass(frozen=True)
class EvaluationSettings:
    enabled: bool
    framework: str
    metrics: List[str]


@dataclass(frozen=True)
class ObservabilitySettings:
    log_level: str
    trace_enabled: bool
    trace_file: str


# ---------------------------------------------------------------------------
# 主配置
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Settings:
    """GradusRAG 全局配置。"""
    llm: LLMSettings
    embedding: EmbeddingSettings
    vector_store: VectorStoreSettings
    retrieval: RetrievalSettings
    rerank: RerankSettings
    ingestion: IngestionSettings
    query_classifier: QueryClassifierSettings
    graph: GraphSettings
    generation: GenerationSettings
    evaluation: EvaluationSettings
    observability: ObservabilitySettings

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Settings":
        """从字典构建 Settings 实例。"""
        llm = data.get("llm", {})
        emb = data.get("embedding", {})
        vs = data.get("vector_store", {})
        ret = data.get("retrieval", {})
        rr = data.get("rerank", {})
        ing = data.get("ingestion", {})
        qc = data.get("query_classifier", {})
        graph = data.get("graph", {})
        gen = data.get("generation", {})
        evl = data.get("evaluation", {})
        obs = data.get("observability", {})

        return cls(
            llm=LLMSettings(
                provider=_get_str(llm, "provider", "llm"),
                model=_get_str(llm, "model", "llm"),
                temperature=_get_float(llm, "temperature", "llm"),
                max_tokens=_get_int(llm, "max_tokens", "llm"),
                api_key=llm.get("api_key"),
                base_url=llm.get("base_url"),
                api_version=llm.get("api_version"),
                azure_endpoint=llm.get("azure_endpoint"),
                deployment_name=llm.get("deployment_name"),
            ),
            embedding=EmbeddingSettings(
                provider=_get_str(emb, "provider", "embedding"),
                model=_get_str(emb, "model", "embedding"),
                dimensions=_get_int(emb, "dimensions", "embedding"),
                device=emb.get("device", "cpu"),
                api_key=emb.get("api_key"),
                api_base_url=emb.get("api_base_url"),
            ),
            vector_store=VectorStoreSettings(
                provider=_get_str(vs, "provider", "vector_store"),
                host=vs.get("host", "127.0.0.1"),
                port=vs.get("port", 19530),
                collection=vs.get("collection", "gradusrag"),
                persist_directory=vs.get("persist_directory", "./data/db/chroma"),
            ),
            retrieval=RetrievalSettings(
                dense_top_k=_get_int(ret, "dense_top_k", "retrieval"),
                sparse_top_k=_get_int(ret, "sparse_top_k", "retrieval"),
                fusion_top_k=_get_int(ret, "fusion_top_k", "retrieval"),
                rrf_k=_get_int(ret, "rrf_k", "retrieval"),
            ),
            rerank=RerankSettings(
                enabled=_get_bool(rr, "enabled", "rerank"),
                provider=_get_str(rr, "provider", "rerank"),
                model=_get_str(rr, "model", "rerank"),
                top_k=_get_int(rr, "top_k", "rerank"),
            ),
            ingestion=IngestionSettings(
                chunk_size_l1=_get_int(ing, "chunk_size_l1", "ingestion"),
                chunk_size_l2=_get_int(ing, "chunk_size_l2", "ingestion"),
                chunk_size_l3=_get_int(ing, "chunk_size_l3", "ingestion"),
                chunk_overlap=_get_int(ing, "chunk_overlap", "ingestion"),
                splitter=_get_str(ing, "splitter", "ingestion"),
            ),
            query_classifier=QueryClassifierSettings(
                mode=_get_str(qc, "mode", "query_classifier"),
                llm_threshold=_get_float(qc, "llm_threshold", "query_classifier"),
            ),
            graph=GraphSettings(
                enabled=_get_bool(graph, "enabled", "graph"),
                storage=_get_str(graph, "storage", "graph"),
                entity_types=[str(t) for t in _get_list(graph, "entity_types", "graph")],
                max_hops=_get_int(graph, "max_hops", "graph"),
            ),
            generation=GenerationSettings(
                l1_strategy=_get_str(gen, "l1_strategy", "generation"),
                l2_strategy=_get_str(gen, "l2_strategy", "generation"),
                l3_strategy=_get_str(gen, "l3_strategy", "generation"),
                l4_strategy=_get_str(gen, "l4_strategy", "generation"),
                max_self_rag_iterations=_get_int(gen, "max_self_rag_iterations", "generation"),
                faithfulness_verify=gen.get("faithfulness_verify", True),
            ),
            evaluation=EvaluationSettings(
                enabled=_get_bool(evl, "enabled", "evaluation"),
                framework=_get_str(evl, "framework", "evaluation"),
                metrics=[str(m) for m in _get_list(evl, "metrics", "evaluation")],
            ),
            observability=ObservabilitySettings(
                log_level=_get_str(obs, "log_level", "observability"),
                trace_enabled=_get_bool(obs, "trace_enabled", "observability"),
                trace_file=_get_str(obs, "trace_file", "observability"),
            ),
        )


def load_settings(path: Optional[Union[str, Path]] = None) -> Settings:
    """从 YAML 文件加载配置。

    Args:
        path: 配置文件路径，默认为 <repo>/config/settings.yaml

    Returns:
        验证后的 Settings 实例
    """
    settings_path = Path(path) if path is not None else DEFAULT_SETTINGS_PATH
    if not settings_path.is_absolute():
        settings_path = resolve_path(settings_path)
    if not settings_path.exists():
        raise SettingsError(f"Settings file not found: {settings_path}")

    with settings_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    return Settings.from_dict(data or {})
