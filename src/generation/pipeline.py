"""GradusRAG 主流水线编排器。

使用 LangGraph 状态图编排完整的 RAG 流程：
  查询分类 → 查询预处理 → 检索 → 评分 → (可选重写) → 分级生成 → 引用溯源

设计参考 SuperMew 的 LangGraph RAG Pipeline。
"""

from __future__ import annotations

import logging
import uuid
from typing import Any, Dict, List, Literal, Optional, TypedDict

from src.core.types import (
    QueryClassification,
    ProcessedQuery,
    RetrievalResult,
    Citation,
    RAGResponse,
)
from src.core.settings import Settings
from src.core.trace import TraceContext, TraceCollector

logger = logging.getLogger(__name__)


# ===========================================================================
# 状态定义 (LangGraph State)
# ===========================================================================

class RAGState(TypedDict, total=False):
    """RAG 流水线状态。"""
    # 输入
    question: str
    query_id: str

    # 查询分类
    classification: QueryClassification
    query_level: str  # L1/L2/L3/L4

    # 查询预处理
    processed_query: ProcessedQuery
    rewritten_query: Optional[str]
    step_back_question: Optional[str]
    hypothetical_doc: Optional[str]

    # 检索
    retrieved_docs: List[dict]
    context: str

    # 评分与路由
    grade_score: str  # yes/no/forced_pass
    route: str  # generate_answer / rewrite_question
    rewrite_count: int  # 已重写次数（防循环）

    # 生成
    answer: str
    citations: List[dict]

    # 追踪
    rag_trace: dict
    query_classification_trace: dict


# ===========================================================================
# Pipeline 编排器
# ===========================================================================

class RAGPipeline:
    """GradusRAG 主流水线。

    协调所有模块完成从查询到响应的完整流程。
    支持按查询级别动态调整处理策略。

    Args:
        mode: 运行模式。
            - "full"（默认）: 完整 GradusRAG（分类 + 分级检索 + 分级生成 + Self-RAG）
            - "naive": NaiveRAG 基线（无分类、仅 Dense 检索、固定 Prompt，无迭代）
    """

    def __init__(
        self,
        settings: Settings,
        classifier=None,
        hybrid_search=None,
        grader=None,
        rewriter=None,
        generator=None,
        citation_generator=None,
        trace_collector: Optional[TraceCollector] = None,
        event_callback=None,
        mode: Literal["full", "naive"] = "full",
    ):
        self._settings = settings
        self._classifier = classifier
        self._search = hybrid_search
        self._grader = grader
        self._rewriter = rewriter
        self._generator = generator
        self._citation_gen = citation_generator
        self._trace_collector = trace_collector or TraceCollector()
        self._event_callback = event_callback
        self._mode = mode
        self._graph = self._build_graph()

        if mode == "naive":
            logger.info("RAGPipeline initialized in NaiveRAG mode (no classification, dense-only, fixed prompt)")

    def _emit(self, icon: str, label: str, detail: str = ""):
        """推送实时事件到前端。"""
        if self._event_callback:
            try:
                self._event_callback({"icon": icon, "label": label, "detail": detail})
            except Exception:
                pass

    def _build_graph(self):
        """构建 LangGraph 状态图。

        流程:
        1. classify_query - 查询分类
        2. preprocess_query - 查询预处理（可选重写）
        3. retrieve - 混合检索
        4. grade_documents - 文档相关性评分
        5. 条件路由：通过 → generate / 不通过 → rewrite
        6. rewrite_query - 查询重写（Step-Back / HyDE）
        7. retrieve_again - 重新检索
        8. generate_answer - 分级生成
        9. add_citations - 引用标注
        """
        try:
            from langgraph.graph import StateGraph, END

            graph = StateGraph(RAGState)

            # 添加节点
            graph.add_node("classify_query", self._classify_query)
            graph.add_node("preprocess_query", self._preprocess_query)
            graph.add_node("retrieve", self._retrieve)
            graph.add_node("grade_documents", self._grade_documents)
            graph.add_node("rewrite_query", self._rewrite_query)
            graph.add_node("generate_answer", self._generate_answer)
            graph.add_node("add_citations", self._add_citations)

            # 定义边
            graph.set_entry_point("classify_query")
            graph.add_edge("classify_query", "preprocess_query")
            graph.add_edge("preprocess_query", "retrieve")
            graph.add_edge("retrieve", "grade_documents")

            # 条件路由：评分通过直接生成，不通过则重写后重新检索
            graph.add_conditional_edges(
                "grade_documents",
                lambda state: state.get("route", "generate_answer"),
                {
                    "generate_answer": "generate_answer",
                    "rewrite_question": "rewrite_query",
                },
            )

            graph.add_edge("rewrite_query", "retrieve")  # 重写后重新检索
            graph.add_edge("generate_answer", "add_citations")
            graph.add_edge("add_citations", END)

            return graph.compile()

        except ImportError:
            logger.warning("LangGraph not available, using sequential pipeline")
            return None

    async def run(self, question: str) -> RAGResponse:
        """执行完整的 RAG 流水线。

        Args:
            question: 用户问题

        Returns:
            RAGResponse 包含答案、引用和完整追踪
        """
        query_id = f"q_{uuid.uuid4().hex[:8]}"
        trace = self._trace_collector.start_trace(query_id, question)

        logger.info(f"[{query_id}] RAG Pipeline: '{question[:80]}...'")

        if self._graph is not None:
            # 使用 LangGraph 执行
            initial_state: RAGState = {
                "question": question,
                "query_id": query_id,
                "context": "",
                "retrieved_docs": [],
                "route": "",
                "grade_score": "",
                "rewrite_count": 0,
                "answer": "",
                "citations": [],
                "rag_trace": {},
                "query_classification_trace": {},
            }
            final_state = await self._graph.ainvoke(initial_state)
        else:
            # 降级为顺序执行
            final_state = await self._run_sequential(question, query_id, trace)

        trace.query_level = final_state.get("query_level", "L1")
        trace_data = self._trace_collector.complete_trace(query_id)

        return RAGResponse(
            answer=final_state.get("answer", ""),
            citations=[
                Citation(**c) for c in final_state.get("citations", [])
            ],
            query_level=final_state.get("query_level", "L1"),
            query_classification=final_state.get("classification"),
            retrieval_trace=final_state.get("rag_trace", {}),
            generation_trace={"trace": trace_data},
        )

    async def _run_sequential(
        self, question: str, query_id: str, trace: TraceContext
    ) -> Dict[str, Any]:
        """无 LangGraph 时的顺序执行降级方案。"""
        state: Dict[str, Any] = {
            "question": question,
            "query_id": query_id,
            "context": "",
            "retrieved_docs": [],
            "route": "",
            "grade_score": "",
            "rewrite_count": 0,
            "answer": "",
            "citations": [],
            "rag_trace": {},
            "query_classification_trace": {},
        }

        # 1. 分类
        state = await self._classify_query(state)

        # 2. 预处理
        state = await self._preprocess_query(state)

        # 3. 检索
        state = await self._retrieve(state)

        # 4. 评分
        state = await self._grade_documents(state)

        # 5. 如果需要重写，重写后重新检索（最多 1 次）
        if state.get("route") == "rewrite_question":
            state = await self._rewrite_query(state)
            state = await self._retrieve(state)

        # 6. 生成
        state = await self._generate_answer(state)

        # 7. 引用
        state = await self._add_citations(state)

        return state

    # -----------------------------------------------------------------------
    # Pipeline 节点实现
    # -----------------------------------------------------------------------

    async def _classify_query(self, state: RAGState) -> Dict[str, Any]:
        """节点 1: 查询分类。NaiveRAG 模式下固定返回 L1。"""
        question = state["question"]

        # NaiveRAG 模式：跳过分类，固定 L1
        if self._mode == "naive":
            self._emit("⏭️", "NaiveRAG 模式：跳过查询分类")
            classification = QueryClassification(
                level="L1", confidence=1.0, query_type="naive"
            )
            return {
                "classification": classification,
                "query_level": "L1",
                "query_classification_trace": {"level": "L1", "mode": "naive"},
            }

        self._emit("🔍", "正在分析问题...", f"查询: {question[:50]}")

        if self._classifier:
            classification = await self._classifier.classify(question)
        else:
            classification = QueryClassification(
                level="L1", confidence=0.5, query_type="factual"
            )

        self._emit(
            "🏷️", f"分类完成: {classification.level}",
            f"类型: {classification.query_type}, 置信度: {classification.confidence:.0%}"
        )

        return {
            "classification": classification,
            "query_level": classification.level,
            "query_classification_trace": classification.to_dict(),
        }

    async def _preprocess_query(self, state: RAGState) -> Dict[str, Any]:
        """节点 2: 查询预处理（提取关键词等）。"""
        from src.libs.text_utils import extract_keywords

        question = state["question"]
        classification = state.get("classification")

        # 使用 jieba 分词提取关键词（中英文双语支持）
        keywords = extract_keywords(question, min_length=2)

        processed = ProcessedQuery(
            original_query=question,
            classified_level=classification.level if classification else "L1",
            keywords=keywords,
        )

        return {"processed_query": processed}

    async def _retrieve(self, state: RAGState) -> Dict[str, Any]:
        """节点 3: 混合检索。NaiveRAG 模式下仅使用 Dense 检索。"""
        processed_query = state.get("processed_query")
        if not processed_query:
            return {"retrieved_docs": [], "context": ""}

        if self._mode == "naive":
            self._emit("🔍", "NaiveRAG 模式：仅 Dense 检索")
            # NaiveRAG: 强制 level=None 使 HybridSearch 仅启用 Dense
            processed_query.classified_level = "NAIVE"
        else:
            self._emit("🔍", "正在检索知识库...", f"Dense + Sparse + Graph")

        if self._search:
            result = await self._search.search(query=processed_query)
            docs = [r.to_dict() for r in result.results]
            context = self._format_context(result.results)
        else:
            docs = []
            context = ""

        sources = set(d.get("metadata", {}).get("filename", "") for d in docs)
        self._emit(
            "✅", f"检索完成，找到 {len(docs)} 个相关片段",
            f"来源: {', '.join(s for s in sources if s)}"
        )

        return {
            "retrieved_docs": docs,
            "context": context,
            "rag_trace": {
                **state.get("rag_trace", {}),
                "retrieved_count": len(docs),
                "fusion_method": "rrf",
            },
        }

    async def _grade_documents(self, state: RAGState) -> Dict[str, Any]:
        """节点 4: 文档相关性评分。NaiveRAG 模式下直接通过。"""
        docs = state.get("retrieved_docs", [])
        rewrite_count = state.get("rewrite_count", 0)

        # NaiveRAG 模式：跳过评分，直接生成
        if self._mode == "naive":
            self._emit("⏭️", "NaiveRAG 模式：跳过文档评分")
            return {
                "grade_score": "forced_pass",
                "route": "generate_answer",
                "rag_trace": {
                    **state.get("rag_trace", {}),
                    "grade_score": "forced_pass",
                    "grade_route": "generate_answer",
                    "grade_reason": "naive_mode_skip",
                },
            }

        self._emit("📊", "正在评估文档相关性...")

        # 已经重写过一次就不再重写，防止无限循环
        if rewrite_count >= 1:
            self._emit("⏭️", "已重写过，跳过评分直接生成")
            return {
                "grade_score": "forced_pass",
                "route": "generate_answer",
                "rag_trace": {
                    **state.get("rag_trace", {}),
                    "grade_score": "forced_pass",
                    "grade_route": "generate_answer",
                    "grade_reason": "already_rewritten_once",
                },
            }

        if not docs:
            self._emit("⚠️", "未找到相关文档，将重写查询")
            return {
                "grade_score": "no",
                "route": "rewrite_question",
                "rewrite_count": rewrite_count + 1,
            }

        if self._grader:
            score = await self._grader.grade(
                question=state["question"],
                context=state.get("context", ""),
            )
        else:
            score = "yes"

        route = "generate_answer" if score == "yes" else "rewrite_question"

        if route == "generate_answer":
            self._emit("✅", "文档相关性评估通过")
        else:
            self._emit("🔄", "文档相关性不足，将重写查询")

        return {
            "grade_score": score,
            "route": route,
            "rewrite_count": rewrite_count + (1 if route == "rewrite_question" else 0),
            "rag_trace": {
                **state.get("rag_trace", {}),
                "grade_score": score,
                "grade_route": route,
            },
        }

    async def _rewrite_query(self, state: RAGState) -> Dict[str, Any]:
        """节点 5: 查询重写（Step-Back / HyDE）。"""
        self._emit("✏️", "正在重写查询...")
        if self._rewriter:
            level = state.get("query_level", "L1")
            rewrite_result = await self._rewriter.rewrite(
                state["question"], level=level
            )
            strategy = rewrite_result.get("strategy", "")
            self._emit("🧠", f"查询重写完成", f"策略: {strategy}, 级别: {level}")
            processed = state.get("processed_query")
            if processed:
                processed.rewritten_query = rewrite_result.get("rewritten_query")
                processed.hyde_doc = rewrite_result.get("hypothetical_doc") or None
            return {
                "processed_query": processed,
                "hypothetical_doc": rewrite_result.get("hypothetical_doc", ""),
                "rag_trace": {
                    **state.get("rag_trace", {}),
                    "rewrite_strategy": strategy,
                    "rewritten_query": rewrite_result.get("rewritten_query"),
                    "step_back_question": rewrite_result.get("step_back_question"),
                    "hypothetical_doc": rewrite_result.get("hypothetical_doc", "")[:200],
                },
            }
        return {}

    async def _generate_answer(self, state: RAGState) -> Dict[str, Any]:
        """节点 6: 生成答案。NaiveRAG 模式下统一使用 L1 直接回答策略。"""
        # NaiveRAG 模式：强制使用 L1 策略，忽略真实级别
        if self._mode == "naive":
            level = "L1"
            self._emit("🤖", "NaiveRAG 模式：直接回答")
        else:
            level = state.get("query_level", "L1")
            strategy_name = {"L1": "直接生成", "L2": "结构化分析", "L3": "CoT推理", "L4": "Self-RAG迭代"}
            self._emit("🤖", f"正在生成回答...", f"级别: {level}, 策略: {strategy_name.get(level, 'direct')}")
        level = state.get("query_level", "L1")
        context = state.get("context", "")
        question = state["question"]

        if self._generator:
            answer = await self._generator.generate(
                question=question,
                context=context,
                level=level,
                query=state.get("processed_query"),
            )
        else:
            answer = self._fallback_answer(question, context, level)

        return {
            "answer": answer,
            "rag_trace": {
                **state.get("rag_trace", {}),
                "generation_level": level,
                "generation_strategy": self._settings.generation.__dict__.get(
                    f"{level.lower()}_strategy", "direct"
                ),
            },
        }

    async def _add_citations(self, state: RAGState) -> Dict[str, Any]:
        """节点 7: 引用标注。"""
        self._emit("📎", "正在添加引用溯源...")
        docs = state.get("retrieved_docs", [])
        citations = []
        for i, doc in enumerate(docs[:5], 1):
            citations.append({
                "index": i,
                "chunk_id": doc.get("chunk_id", ""),
                "source": doc.get("metadata", {}).get("source_path", "Unknown"),
                "page": doc.get("metadata", {}).get("page"),
                "score": doc.get("score", 0.0),
                "text_snippet": doc.get("text", "")[:200],
            })

        return {"citations": citations}

    @staticmethod
    def _format_context(results: List[RetrievalResult]) -> str:
        """将检索结果格式化为上下文字符串。"""
        if not results:
            return ""
        chunks = []
        for i, r in enumerate(results, 1):
            source = r.metadata.get("source_path", "Unknown")
            page = r.metadata.get("page", "N/A")
            chunks.append(f"[{i}] {source} (Page {page}):\n{r.text}")
        return "\n\n---\n\n".join(chunks)

    @staticmethod
    def _fallback_answer(question: str, context: str, level: str) -> str:
        """无 LLM 时的降级回答。"""
        if not context:
            return f"[{level}] 未找到与问题相关的参考资料，无法回答。"
        snippet = context[:500] + ("..." if len(context) > 500 else "")
        return (
            f"[{level}] 查询: {question}\n\n"
            f"基于检索到的参考资料，以下是相关内容：\n\n{snippet}\n\n"
            f"（注：LLM 未配置，以上为检索结果摘要，非生成答案）"
        )
