"""LightRAG 基线对比脚本。

实现一个简化的 LightRAG 基线：
- 知识图谱 + 混合检索（Dense + Sparse + Graph）
- 无 L1-L4 查询分级路由（所有查询走同一策略）
- 无分级生成策略（所有查询用同一 Prompt）
- 无 Self-RAG 迭代循环

对比公平性保证：
- 相同知识库（同一批文档入库）
- 相同测试集（同一份 golden_test_set）
- 相同评估指标

用法：
    python scripts/compare_lightrag.py
    python scripts/compare_lightrag.py --test-set data/test_sets/formal_test_set.json
    python scripts/compare_lightrag.py --output results/lightrag_comparison.json
"""

import argparse
import asyncio
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.settings import load_settings
from src.core.types import ProcessedQuery, RAGResponse, RetrievalResult
from src.core.trace import TraceCollector
from src.libs.llm_service import LLMService
from src.libs.embedding_service import EmbeddingService
from src.retrieval.milvus_store import MilvusStore
from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.sparse_retriever import SparseRetriever
from src.retrieval.reranker import Reranker
from src.retrieval.hybrid_search import HybridSearch
from src.retrieval.fusion import rrf_fuse
from src.evaluation.metrics import (
    hit_rate_at_k, mrr, faithfulness_llm, relevancy_llm, context_recall,
)
from src.evaluation.test_set import TestSetManager
from src.ingestion.graph_builder.graph_store import GraphStore
from src.retrieval.graph_retriever import GraphRetriever

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# ===========================================================================
# LightRAG 基线：简化版（无分级路由）
# ===========================================================================

# LightRAG 风格的统一 Prompt（不区分级别）
LIGHTRAG_PROMPT = """你是一个知识问答助手。请根据以下参考资料回答用户的问题。

参考资料：
{context}

用户问题：{query}

要求：
- 只使用参考资料中的信息回答，不要编造
- 如果参考资料不足以回答，请明确说明"根据现有资料无法回答"
- 引用来源时使用 [1], [2] 等标注
- 回答简洁准确，重点突出"""


class LightRAGBaseline:
    """LightRAG 基线系统。

    与 GradusRAG 的区别：
    1. 无 L1-L4 查询分级：所有查询走 Dense + Sparse + Graph 混合检索
    2. 无分级生成：所有查询使用同一 Prompt
    3. 无 Self-RAG 迭代：生成一次即返回
    4. 无 Rerank：使用 RRF 融合后的原始排序
    """

    def __init__(
        self,
        settings,
        llm_service,
        dense_retriever,
        sparse_retriever,
        graph_retriever,
    ):
        self._settings = settings
        self._llm = llm_service
        self._dense = dense_retriever
        self._sparse = sparse_retriever
        self._graph = graph_retriever

    async def query(self, question: str) -> RAGResponse:
        """执行查询（LightRAG 风格，无分级路由）。"""
        # Step 1: 提取关键词（简单分词）
        keywords = self._extract_keywords(question)

        # Step 2: Dense + Sparse + Graph 三路检索
        dense_results = []
        sparse_results = []
        graph_results = []

        try:
            dense_results = await self._dense.retrieve(
                query=question,
                top_k=self._settings.retrieval.dense_top_k,
            )
        except Exception as e:
            logger.error(f"Dense retrieval failed: {e}")

        try:
            sparse_results = await self._sparse.retrieve(
                keywords=keywords,
                top_k=self._settings.retrieval.sparse_top_k,
                query_text=question,
            )
        except Exception as e:
            logger.error(f"Sparse retrieval failed: {e}")

        if self._graph:
            try:
                graph_results = await self._graph.retrieve(
                    query=question,
                    top_k=5,
                    hops=2,
                )
            except Exception as e:
                logger.error(f"Graph retrieval failed: {e}")

        # Step 3: RRF 融合（无 Rerank）
        ranking_lists = []
        if dense_results:
            ranking_lists.append(dense_results)
        if sparse_results:
            ranking_lists.append(sparse_results)
        if graph_results:
            ranking_lists.append(graph_results)

        if not ranking_lists:
            return RAGResponse(
                answer="无法检索到相关信息。",
                query_level="L1",
                citations=[],
            )

        fused = rrf_fuse(
            ranking_lists,
            k=self._settings.retrieval.rrf_k,
            top_k=self._settings.retrieval.fusion_top_k,
        )

        # Step 4: 生成回答（统一 Prompt，不区分级别）
        context = "\n\n".join([
            f"[{i+1}] {r.text[:800]}" for i, r in enumerate(fused[:5])
        ])

        prompt = LIGHTRAG_PROMPT.format(
            context=context,
            query=question,
        )

        try:
            answer = await self._llm.ainvoke(prompt)
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            answer = "生成失败，请重试。"

        # Step 5: 构建 Citation
        from src.core.types import Citation
        sources = []
        for i, r in enumerate(fused[:5]):
            sources.append(Citation(
                index=i + 1,
                source=r.metadata.get("filename", ""),
                page=r.metadata.get("page_number"),
                score=r.score,
                chunk_id=r.chunk_id,
                text_snippet=r.text[:500],
            ))

        return RAGResponse(
            answer=answer,
            query_level="L1",
            citations=sources,
        )

    @staticmethod
    def _extract_keywords(query: str) -> List[str]:
        """简单关键词提取。"""
        import re
        keywords = []
        # 英文
        en_stop = {"the", "is", "are", "was", "were", "what", "which", "how", "why",
                    "and", "or", "but", "in", "on", "at", "to", "for", "of", "with"}
        for w in re.findall(r'[a-zA-Z]{3,}', query):
            if w.lower() not in en_stop:
                keywords.append(w)
        # 中文
        try:
            import jieba
            cn_stop = {"的", "了", "是", "在", "和", "有", "为", "这", "那", "什么", "怎么", "为什么"}
            for w in jieba.cut(query):
                w = w.strip()
                if len(w) >= 2 and w not in cn_stop:
                    keywords.append(w)
        except ImportError:
            pass
        return keywords or query.split()


# ===========================================================================
# GradusRAG 完整系统
# ===========================================================================

async def run_gradusrag(pipeline, question: str) -> RAGResponse:
    """运行 GradusRAG 完整系统。"""
    return await pipeline.run(question)


# ===========================================================================
# 评估逻辑
# ===========================================================================

def evaluate_response(
    response: RAGResponse,
    test_case: dict,
    llm_service,
) -> dict:
    """评估单个响应。"""
    retrieved_ids = [c.chunk_id for c in response.citations if c.chunk_id]
    ground_truth = test_case.get("ground_truth_chunks", [])

    # 确定性指标
    hr = hit_rate_at_k(retrieved_ids, ground_truth, k=5)
    mrr_score = mrr(retrieved_ids, ground_truth, k=10)

    # LLM 评估指标
    context_text = "\n".join([
        c.text_snippet[:500] for c in (response.citations or [])[:5] if c.text_snippet
    ])

    faith = faithfulness_llm(response.answer, context_text, llm_service)
    relevancy = relevancy_llm(test_case["question"], response.answer, llm_service)

    return {
        "question": test_case["question"],
        "expected_level": test_case.get("expected_level", "L1"),
        "hit_rate": hr,
        "mrr": mrr_score,
        "faithfulness": faith,
        "answer_relevance": relevancy,
    }


def aggregate(details: List[Dict]) -> Dict:
    """汇总评估结果。"""
    valid = [d for d in details if "error" not in d]
    if not valid:
        return {"total": 0}

    def avg(key):
        vals = [d[key] for d in valid if d.get(key) is not None]
        return sum(vals) / len(vals) if vals else 0.0

    return {
        "total": len(valid),
        "hit_rate": avg("hit_rate"),
        "mrr": avg("mrr"),
        "faithfulness": avg("faithfulness"),
        "answer_relevance": avg("answer_relevance"),
    }


# ===========================================================================
# Pairwise 对比评估
# ===========================================================================

PAIRWISE_PROMPT = """请比较以下两个回答的质量。不要考虑回答的长度，只关注质量。

问题：{question}

回答A：
{answer_a}

回答B：
{answer_b}

请从以下三个维度判断哪个更好：
1. 全面性：哪个回答更完整，覆盖了更多相关信息
2. 准确性：哪个回答的事实更准确，没有编造
3. 实用性：哪个回答对用户更有帮助

只输出 JSON，不要输出其他内容：
{{"winner": "A", "reason": "A更全面且准确"}}
或
{{"winner": "B", "reason": "B更简洁准确"}}
或
{{"winner": "Tie", "reason": "两者质量相当"}}"""


async def pairwise_compare(
    llm_service,
    question: str,
    answer_a: str,
    answer_b: str,
) -> dict:
    """对比两个回答的质量。answer_a=GradusRAG, answer_b=LightRAG。"""
    import re
    prompt = PAIRWISE_PROMPT.format(
        question=question,
        answer_a=answer_a[:1500],
        answer_b=answer_b[:1500],
    )
    try:
        response = await llm_service.ainvoke(prompt)
        # 提取 winner
        match = re.search(r'"winner"\s*:\s*"(A|B|Tie)"', response)
        reason_match = re.search(r'"reason"\s*:\s*"([^"]+)"', response)
        winner = match.group(1) if match else "Tie"
        reason = reason_match.group(1) if reason_match else ""
        return {"winner": winner, "reason": reason}
    except Exception as e:
        logger.error(f"Pairwise comparison failed: {e}")
        return {"winner": "Tie", "reason": "evaluation failed"}


async def run_pairwise_evaluation(
    llm_service,
    test_cases: list,
    gradusrag_answers: dict,
    lightrag_answers: dict,
) -> dict:
    """对所有题目运行 pairwise 对比。"""
    import random
    results = {"GradusRAG_wins": 0, "LightRAG_wins": 0, "Tie": 0, "details": []}

    for i, tc in enumerate(test_cases):
        q = tc["question"]
        r_answer = gradusrag_answers.get(q, "")
        l_answer = lightrag_answers.get(q, "")

        if not r_answer or not l_answer:
            continue

        # 随机打乱 A/B 顺序，消除位置偏差
        if random.random() > 0.5:
            comparison = await pairwise_compare(llm_service, q, r_answer, l_answer)
            if comparison["winner"] == "A":
                actual_winner = "GradusRAG"
            elif comparison["winner"] == "B":
                actual_winner = "LightRAG"
            else:
                actual_winner = "Tie"
        else:
            comparison = await pairwise_compare(llm_service, q, l_answer, r_answer)
            if comparison["winner"] == "A":
                actual_winner = "LightRAG"
            elif comparison["winner"] == "B":
                actual_winner = "GradusRAG"
            else:
                actual_winner = "Tie"

        results[actual_winner + "_wins" if actual_winner != "Tie" else "Tie"] = results.get(actual_winner + "_wins" if actual_winner != "Tie" else "Tie", 0) + 1

        detail = {
            "question": q[:50],
            "expected_level": tc.get("expected_level"),
            "winner": actual_winner,
            "reason": comparison["reason"],
        }
        results["details"].append(detail)

        if (i + 1) % 10 == 0:
            logger.info(f"  Pairwise: {i+1}/{len(test_cases)}")

    return results


# ===========================================================================
# 主流程
# ===========================================================================

async def main():
    parser = argparse.ArgumentParser(description="LightRAG 基线对比")
    parser.add_argument("--test-set", "-t", default="data/test_sets/formal_test_set.json", help="测试集路径")
    parser.add_argument("--max-cases", type=int, default=0, help="最大测试数（0=全部）")
    parser.add_argument("--output", "-o", default="results/lightrag_comparison.json", help="输出路径")
    parser.add_argument("--save-path", default="results/comparison_progress.json", help="增量保存路径")
    parser.add_argument("--skip-gradusrag", action="store_true", help="跳过 GradusRAG 评估（只跑基线）")
    parser.add_argument("--judge-model", default="", help="Pairwise 评判模型（如 gpt-4o, qwen-max）。留空则用主模型")
    parser.add_argument("--judge-api-key", default="", help="评判模型 API Key")
    parser.add_argument("--judge-base-url", default="", help="评判模型 API 地址")
    args = parser.parse_args()

    settings = load_settings()
    llm_service = LLMService.from_settings(settings)

    # 创建独立的评判 LLM（如果指定了）
    judge_llm = llm_service  # 默认用主模型
    if args.judge_model and args.judge_api_key:
        from langchain_openai import ChatOpenAI
        judge_llm = ChatOpenAI(
            model=args.judge_model,
            api_key=args.judge_api_key,
            base_url=args.judge_base_url or None,
            temperature=0,
        )
        logger.info(f"Judge LLM: {args.judge_model} (separate from generation LLM)")
    else:
        logger.info(f"Judge LLM: same as generation LLM")

    # 加载测试集
    test_cases = json.loads(Path(args.test_set).read_text(encoding="utf-8"))
    if args.max_cases > 0:
        # 按级别均匀采样
        from collections import defaultdict
        by_level = defaultdict(list)
        for tc in test_cases:
            by_level[tc.get("expected_level", "L1")].append(tc)
        per_level = max(args.max_cases // 4, 1)
        sampled = []
        for level in ["L1", "L2", "L3", "L4"]:
            sampled.extend(by_level.get(level, [])[:per_level])
        test_cases = sampled
        logger.info(f"Sampled {len(test_cases)} cases: {per_level} per level")

    # 初始化组件
    embedding_service = EmbeddingService(
        model_name=settings.embedding.model,
        device=settings.embedding.device,
        dimensions=settings.embedding.dimensions,
        api_key=settings.embedding.api_key,
        api_base_url=settings.embedding.api_base_url,
    )
    milvus_store = MilvusStore(
        host=settings.vector_store.host,
        port=settings.vector_store.port,
        collection=settings.vector_store.collection,
    )
    dense = DenseRetriever(embedding_service, milvus_store)
    sparse = SparseRetriever(embedding_service, milvus_store)

    graph_store = GraphStore(persist_path="data/knowledge_graph.json")
    graph_retriever = GraphRetriever(graph_store) if settings.graph.enabled else None

    reranker = Reranker(llm_service=llm_service)

    all_results = {}

    # ---- LightRAG 基线 ----
    logger.info(f"\n{'='*60}")
    logger.info("Running LightRAG Baseline (no routing, no rerank, no Self-RAG)")
    logger.info(f"{'='*60}")

    lightrag = LightRAGBaseline(
        settings=settings,
        llm_service=llm_service,
        dense_retriever=dense,
        sparse_retriever=sparse,
        graph_retriever=graph_retriever,
    )

    lightrag_details = []
    lightrag_answers = {}  # 收集回答用于 pairwise 对比
    for i, tc in enumerate(test_cases):
        logger.info(f"  [{i+1}/{len(test_cases)}] [{tc.get('expected_level')}] {tc['question'][:50]}...")
        try:
            t0 = time.monotonic()
            response = await lightrag.query(tc["question"])
            elapsed = (time.monotonic() - t0) * 1000

            result = evaluate_response(response, tc, llm_service)
            result["elapsed_ms"] = elapsed
            lightrag_details.append(result)
            lightrag_answers[tc["question"]] = response.answer
            logger.info(f"    → {elapsed:.0f}ms, faith={result['faithfulness']:.2f}, rel={result['answer_relevance']:.2f}")
        except Exception as e:
            logger.error(f"    → Failed: {e}")
            lightrag_details.append({"question": tc["question"], "error": str(e)})

        # 增量保存
        Path(args.save_path).parent.mkdir(parents=True, exist_ok=True)
        Path(args.save_path).write_text(json.dumps({
            "lightrag_details": lightrag_details,
            "lightrag_answers": {k: v[:500] for k, v in lightrag_answers.items()},
        }, ensure_ascii=False, indent=2), encoding="utf-8")

    lightrag_summary = aggregate(lightrag_details)
    all_results["LightRAG_Baseline"] = {"summary": lightrag_summary, "details": lightrag_details}

    # ---- GradusRAG ----
    if not args.skip_gradusrag:
        logger.info(f"\n{'='*60}")
        logger.info("Running GradusRAG (full system with routing)")
        logger.info(f"{'='*60}")

        from src.query_classifier.classifier import create_classifier
        from src.generation.pipeline import RAGPipeline
        from src.generation.document_grader import DocumentGrader
        from src.generation.query_rewriter import QueryRewriter
        from src.generation.response_generator import ResponseGenerator

        hybrid_search = HybridSearch(
            settings=settings,
            dense_retriever=dense,
            sparse_retriever=sparse,
            graph_retriever=graph_retriever,
            reranker=reranker,
        )
        classifier = create_classifier(settings.query_classifier)
        grader = DocumentGrader(llm_service)
        rewriter = QueryRewriter(llm_service)
        generator = ResponseGenerator(llm_service, hybrid_search)

        pipeline = RAGPipeline(
            settings=settings,
            classifier=classifier,
            hybrid_search=hybrid_search,
            grader=grader,
            rewriter=rewriter,
            generator=generator,
        )

        gradusrag_details = []
        gradusrag_answers = {}  # 收集回答用于 pairwise 对比
        for i, tc in enumerate(test_cases):
            logger.info(f"  [{i+1}/{len(test_cases)}] [{tc.get('expected_level')}] {tc['question'][:50]}...")
            try:
                t0 = time.monotonic()
                response = await pipeline.run(tc["question"])
                elapsed = (time.monotonic() - t0) * 1000

                result = evaluate_response(response, tc, llm_service)
                result["elapsed_ms"] = elapsed
                result["predicted_level"] = response.query_level
                result["level_correct"] = response.query_level == tc.get("expected_level")
                gradusrag_details.append(result)
                gradusrag_answers[tc["question"]] = response.answer
                logger.info(f"    → {response.query_level}, {elapsed:.0f}ms, faith={result['faithfulness']:.2f}")
            except Exception as e:
                logger.error(f"    → Failed: {e}")
                gradusrag_details.append({"question": tc["question"], "error": str(e)})

            # 增量保存
            Path(args.save_path).write_text(json.dumps({
                "lightrag_details": lightrag_details,
                "gradusrag_details": gradusrag_details,
                "lightrag_answers": {k: v[:500] for k, v in lightrag_answers.items()},
                "gradusrag_answers": {k: v[:500] for k, v in gradusrag_answers.items()},
            }, ensure_ascii=False, indent=2), encoding="utf-8")

        gradusrag_summary = aggregate(gradusrag_details)
        cls_correct = sum(1 for d in gradusrag_details if d.get("level_correct"))
        gradusrag_summary["classification_accuracy"] = cls_correct / len(test_cases) if test_cases else 0
        all_results["GradusRAG"] = {"summary": gradusrag_summary, "details": gradusrag_details}

    # ---- Pairwise 对比 ----
    pairwise_results = None
    if not args.skip_gradusrag and gradusrag_answers and lightrag_answers:
        logger.info(f"\n{'='*60}")
        logger.info("Running Pairwise Comparison (GradusRAG vs LightRAG)")
        logger.info(f"{'='*60}")

        pairwise_results = await run_pairwise_evaluation(
            judge_llm, test_cases, gradusrag_answers, lightrag_answers
        )
        all_results["pairwise"] = pairwise_results

    # ---- 保存 ----
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(all_results, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---- 打印对比表格 ----
    print(f"\n{'='*90}")
    print(f"COMPARISON: GradusRAG vs LightRAG Baseline ({len(test_cases)} cases)")
    print(f"{'='*90}")
    print(f"{'System':<25} {'Hit@5':>8} {'MRR':>8} {'Faith':>8} {'Rel':>8} {'Cls.Acc':>8}")
    print(f"{'-'*90}")

    for name, data in all_results.items():
        if "summary" not in data:
            continue
        s = data["summary"]
        cls_acc = s.get("classification_accuracy", "N/A")
        cls_str = f"{cls_acc:.2%}" if isinstance(cls_acc, float) else "N/A"
        print(
            f"{name:<25} "
            f"{s['hit_rate']:>7.2%} "
            f"{s['mrr']:>7.2%} "
            f"{s['faithfulness']:>7.2%} "
            f"{s['answer_relevance']:>7.2%} "
            f"{cls_str:>8}"
        )
    print(f"{'='*90}")

    # 按级别细分
    if not args.skip_gradusrag:
        print(f"\n{'='*90}")
        print("PER-LEVEL COMPARISON")
        print(f"{'='*90}")
        for level in ["L1", "L2", "L3", "L4"]:
            print(f"\n  {level}:")
            for name, data in all_results.items():
                if "details" not in data or name == "pairwise":
                    continue
                level_cases = [d for d in data["details"] if d.get("expected_level") == level and "error" not in d and "faithfulness" in d]
                if level_cases:
                    avg_faith = sum(d["faithfulness"] for d in level_cases) / len(level_cases)
                    avg_rel = sum(d.get("answer_relevance", 0) for d in level_cases) / len(level_cases)
                    avg_hr = sum(d.get("hit_rate", 0) for d in level_cases) / len(level_cases)
                    print(f"    {name:<23} Hit={avg_hr:.2%} Faith={avg_faith:.2%} Rel={avg_rel:.2%} ({len(level_cases)} cases)")
        print(f"{'='*90}")

    # Pairwise 对比结果
    if pairwise_results:
        total_pw = pairwise_results["GradusRAG_wins"] + pairwise_results["LightRAG_wins"] + pairwise_results["Tie"]
        print(f"\n{'='*90}")
        print(f"PAIRWISE COMPARISON (GradusRAG vs LightRAG)")
        print(f"{'='*90}")
        print(f"  GradusRAG wins:  {pairwise_results['GradusRAG_wins']} ({pairwise_results['GradusRAG_wins']/max(total_pw,1):.1%})")
        print(f"  LightRAG wins:  {pairwise_results['LightRAG_wins']} ({pairwise_results['LightRAG_wins']/max(total_pw,1):.1%})")
        print(f"  Tie:            {pairwise_results['Tie']} ({pairwise_results['Tie']/max(total_pw,1):.1%})")

        # 按级别细分
        print(f"\n  Per-level breakdown:")
        for level in ["L1", "L2", "L3", "L4"]:
            level_details = [d for d in pairwise_results["details"] if d.get("expected_level") == level]
            if level_details:
                r_wins = sum(1 for d in level_details if d["winner"] == "GradusRAG")
                l_wins = sum(1 for d in level_details if d["winner"] == "LightRAG")
                ties = sum(1 for d in level_details if d["winner"] == "Tie")
                print(f"    {level}: GradusRAG={r_wins} LightRAG={l_wins} Tie={ties} ({len(level_details)} cases)")
        print(f"{'='*90}")

    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
