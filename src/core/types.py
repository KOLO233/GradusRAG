"""GradusRAG 核心类型定义

全链路数据流的统一类型契约。所有模块通过这些类型通信，实现松耦合。

设计原则：
- 集中定义：所有阶段使用同一套类型，避免耦合
- 可序列化：所有类型支持 dict/JSON 转换
- 可扩展元数据：最少必填字段 + 灵活扩展
- 类型安全：完整类型注解
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional


# ===========================================================================
# 数据层类型 (Data Layer Types)
# ===========================================================================

@dataclass
class Document:
    """原始文档，由 Loader 输出。

    Attributes:
        id: 唯一标识（文件哈希或路径生成）
        text: 标准化 Markdown 格式的文档内容
        metadata: 文档级元数据，必须包含 source_path
    """
    id: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if "source_path" not in self.metadata:
            raise ValueError("Document metadata must contain 'source_path'")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Document":
        return cls(**data)


@dataclass
class Chunk:
    """文本分块，由 Splitter 输出。

    Attributes:
        id: 唯一 chunk 标识
        text: 分块文本内容
        metadata: 分块元数据，必须包含 source_path
        start_offset: 在原文档中的起始字符位置
        end_offset: 在原文档中的结束字符位置
        source_ref: 父文档 ID
    """
    id: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    start_offset: Optional[int] = None
    end_offset: Optional[int] = None
    source_ref: Optional[str] = None

    def __post_init__(self):
        if "source_path" not in self.metadata:
            raise ValueError("Chunk metadata must contain 'source_path'")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Chunk":
        return cls(**data)


@dataclass
class ChunkRecord:
    """带向量的完整 chunk 记录，存储到向量库。

    Attributes:
        id: 唯一标识（与 Chunk.id 一致，用于幂等 upsert）
        text: 文本内容
        metadata: 扩展元数据
        dense_vector: 稠密嵌入向量
        sparse_vector: 稀疏向量（BM25）
    """
    id: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    dense_vector: Optional[List[float]] = None
    sparse_vector: Optional[Dict[str, float]] = None

    def __post_init__(self):
        if "source_path" not in self.metadata:
            raise ValueError("ChunkRecord metadata must contain 'source_path'")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_chunk(
        cls,
        chunk: Chunk,
        dense_vector: Optional[List[float]] = None,
        sparse_vector: Optional[Dict[str, float]] = None,
    ) -> "ChunkRecord":
        return cls(
            id=chunk.id,
            text=chunk.text,
            metadata=chunk.metadata.copy(),
            dense_vector=dense_vector,
            sparse_vector=sparse_vector,
        )


# ===========================================================================
# 查询层类型 (Query Layer Types)
# ===========================================================================

@dataclass
class QueryClassification:
    """查询分类结果。

    Attributes:
        level: 查询级别 L1/L2/L3/L4
        confidence: 分类置信度 (0.0 ~ 1.0)
        query_type: 查询类型 (factual/comparative/causal/hypothetical)
        reasoning: 分类理由
        route_config: 路由策略配置
    """
    level: str  # L1, L2, L3, L4
    confidence: float
    query_type: str = ""
    reasoning: str = ""
    route_config: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ProcessedQuery:
    """处理后的查询，传递给检索引擎。

    Attributes:
        original_query: 原始用户查询
        classified_level: 分类后的级别
        keywords: 提取的关键词
        expanded_terms: 扩展词（同义词等）
        filters: 过滤条件
        rewritten_query: 重写后的查询（如有，Step-Back 退步问题）
        hyde_doc: HyDE 假设性文档（如有，用于 Dense 向量检索）
    """
    original_query: str
    classified_level: str = "L1"
    keywords: List[str] = field(default_factory=list)
    expanded_terms: List[str] = field(default_factory=list)
    filters: Dict[str, Any] = field(default_factory=dict)
    rewritten_query: Optional[str] = None
    hyde_doc: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ===========================================================================
# 检索层类型 (Retrieval Layer Types)
# ===========================================================================

@dataclass
class RetrievalResult:
    """单条检索结果。

    Attributes:
        chunk_id: 检索到的 chunk 标识
        score: 相关性分数 (越高越相关)
        text: 文本内容
        metadata: 关联元数据
        retrieval_source: 来源 (dense/sparse/graph)
    """
    chunk_id: str
    score: float
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    retrieval_source: str = "unknown"

    def __post_init__(self):
        if not self.chunk_id:
            raise ValueError("chunk_id cannot be empty")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RetrievalResult":
        return cls(**data)


@dataclass
class HybridSearchResult:
    """混合检索完整结果。

    Attributes:
        results: 最终排序后的结果列表
        dense_results: Dense 检索结果
        sparse_results: Sparse 检索结果
        graph_results: Graph 检索结果
        fusion_method: 融合方法 (rrf/weighted_rrf)
        rerank_applied: 是否应用了重排序
    """
    results: List[RetrievalResult] = field(default_factory=list)
    dense_results: Optional[List[RetrievalResult]] = None
    sparse_results: Optional[List[RetrievalResult]] = None
    graph_results: Optional[List[RetrievalResult]] = None
    fusion_method: str = "rrf"
    rerank_applied: bool = False


# ===========================================================================
# 图谱类型 (Knowledge Graph Types)
# ===========================================================================

@dataclass
class Entity:
    """知识图谱中的实体。

    Attributes:
        name: 实体名称
        entity_type: 实体类型 (疾病/药物/症状/概念等)
        description: 实体描述
        source_chunks: 来源 chunk ID 列表
        properties: 附加属性
    """
    name: str
    entity_type: str
    description: str = ""
    source_chunks: List[str] = field(default_factory=list)
    properties: Dict[str, Any] = field(default_factory=dict)

    @property
    def id(self) -> str:
        return f"{self.entity_type}::{self.name}"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Relation:
    """知识图谱中的关系。

    Attributes:
        source: 源实体名称
        target: 目标实体名称
        relation_type: 关系类型
        description: 关系描述
        weight: 关系权重
        source_chunks: 来源 chunk ID 列表
    """
    source: str
    target: str
    relation_type: str
    description: str = ""
    weight: float = 1.0
    source_chunks: List[str] = field(default_factory=list)

    @property
    def id(self) -> str:
        return f"{self.source}--{self.relation_type}-->{self.target}"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ===========================================================================
# 生成层类型 (Generation Layer Types)
# ===========================================================================

@dataclass
class Citation:
    """引用溯源信息。

    Attributes:
        index: 引用编号 (1, 2, 3...)
        source: 来源文件路径
        page: 页码
        score: 相关性分数
        text_snippet: 引用文本片段
    """
    index: int
    source: str
    chunk_id: str = ""
    page: Optional[int] = None
    score: float = 0.0
    text_snippet: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RAGResponse:
    """RAG 系统最终响应。

    Attributes:
        answer: 生成的答案文本
        citations: 引用列表
        query_level: 查询级别
        query_classification: 查询分类详情
        retrieval_trace: 检索过程追踪
        generation_trace: 生成过程追踪
    """
    answer: str
    citations: List[Citation] = field(default_factory=list)
    query_level: str = "L1"
    query_classification: Optional[QueryClassification] = None
    retrieval_trace: Dict[str, Any] = field(default_factory=dict)
    generation_trace: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        if self.query_classification:
            result["query_classification"] = self.query_classification.to_dict()
        return result


# ===========================================================================
# 评估类型 (Evaluation Types)
# ===========================================================================

@dataclass
class EvalTestCase:
    """评估测试用例。

    Attributes:
        question: 测试问题
        expected_answer: 期望答案
        expected_level: 期望分类级别
        ground_truth_chunks: 标注的相关 chunk ID 列表
        category: 测试类别
    """
    question: str
    expected_answer: str = ""
    expected_level: str = "L1"
    ground_truth_chunks: List[str] = field(default_factory=list)
    category: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EvaluationResult:
    """评估结果。

    Attributes:
        hit_rate: Hit Rate@K
        mrr: Mean Reciprocal Rank
        faithfulness: 忠实度
        relevancy: 相关性
        classification_accuracy: 分类准确率
        total_cases: 总测试用例数
        details: 详细结果
    """
    hit_rate: float = 0.0
    mrr: float = 0.0
    faithfulness: float = 0.0
    answer_relevance: float = 0.0
    context_recall: float = 0.0
    context_precision: float = 0.0
    classification_accuracy: float = 0.0
    total_cases: int = 0
    details: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# 类型别名
Metadata = Dict[str, Any]
Vector = List[float]
SparseVector = Dict[str, float]
