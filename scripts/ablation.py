"""消融实验脚本。

逐模块关闭，量化每个模块对系统性能的贡献。

实验配置：
1. Full — 完整 GradusRAG 系统
2. w/o Graph — 关闭知识图谱检索
3. w/o Sparse — 关闭 BM25 稀疏检索
4. w/o Rerank — 关闭重排序
5. w/o Self-RAG — L4 使用直接生成代替 Self-RAG
6. w/o Classification — 所有查询使用 L1 策略（不分级）

用法：
    python scripts/ablation.py                    # 跑全部 23 case × 6 config
    python scripts/ablation.py --max-cases 4      # 快速验证
    python scripts/ablation.py --resume           # 从上次中断处继续
    python scripts/ablation.py --output results/ablation.json
"""

import argparse
import asyncio
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.settings import load_settings, Settings
from src.core.types import ProcessedQuery, RAGResponse
from src.core.trace import TraceCollector
from src.query_classifier.classifier import create_classifier
from src.generation.pipeline import RAGPipeline
from src.generation.document_grader import DocumentGrader
from src.generation.query_rewriter import QueryRewriter
from src.generation.response_generator import ResponseGenerator
from src.libs.llm_service import LLMService
from src.libs.embedding_service import EmbeddingService
from src.retrieval.milvus_store import MilvusStore
from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.sparse_retriever import SparseRetriever
from src.retrieval.reranker import Reranker
from src.retrieval.hybrid_search import HybridSearch
from src.evaluation.metrics import (
    hit_rate_at_k, mrr, classification_accuracy,
    faithfulness_llm, relevancy_llm, context_recall, context_precision,
)
from src.evaluation.test_set import TestSetManager

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# ===========================================================================
# Ablation 配置
# ===========================================================================

ABLATION_CONFIGS = {
    "Full": {
        "graph_enabled": True,
        "sparse_enabled": True,
        "rerank_enabled": True,
        "self_rag_enabled": True,
        "classification_enabled": True,
    },
    "w/o Graph": {
        "graph_enabled": False,
        "sparse_enabled": True,
        "rerank_enabled": True,
        "self_rag_enabled": True,
        "classification_enabled": True,
    },
    "w/o Sparse": {
        "graph_enabled": True,
        "sparse_enabled": False,
        "rerank_enabled": True,
        "self_rag_enabled": True,
        "classification_enabled": True,
    },
    "w/o Rerank": {
        "graph_enabled": True,
        "sparse_enabled": True,
        "rerank_enabled": False,
        "self_rag_enabled": True,
        "classification_enabled": True,
    },
    "w/o Self-RAG": {
        "graph_enabled": True,
        "sparse_enabled": True,
        "rerank_enabled": True,
        "self_rag_enabled": False,
        "classification_enabled": True,
    },
    "w/o Classification": {
        "graph_enabled": True,
        "sparse_enabled": True,
        "rerank_enabled": True,
        "self_rag_enabled": True,
        "classification_enabled": False,
    },
}


def build_ablation_pipeline(
    settings: Settings,
    llm_service,
    embedding_service,
    milvus_store,
    config: Dict,
) -> RAGPipeline:
    """根据消融配置构建 Pipeline。"""
    from src.ingestion.graph_builder.graph_store import GraphStore

    # 检索器
    dense_retriever = DenseRetriever(embedding_service, milvus_store)
    sparse_retriever = SparseRetriever(embedding_service, milvus_store) if config["sparse_enabled"] else None
    graph_retriever = None
    if config["graph_enabled"]:
        graph_store = GraphStore(persist_path="data/knowledge_graph.json")
        from src.retrieval.graph_retriever import GraphRetriever
        graph_retriever = GraphRetriever(graph_store)

    reranker = Reranker(llm_service=llm_service) if config["rerank_enabled"] else None

    hybrid_search = HybridSearch(
        settings=settings,
        dense_retriever=dense_retriever,
        sparse_retriever=sparse_retriever,
        graph_retriever=graph_retriever,
        reranker=reranker,
    )

    # 分类器
    classifier = create_classifier(settings.query_classifier) if config["classification_enabled"] else None

    # 生成组件
    grader = DocumentGrader(llm_service)
    rewriter = QueryRewriter(llm_service)

    # Self-RAG 控制
    generator = ResponseGenerator(llm_service, hybrid_search)
    if not config["self_rag_enabled"]:
        # 临时修改配置，强制 L4 使用直接生成
        import dataclasses
        gen_settings = dataclasses.replace(settings.generation, l4_strategy="direct")
        settings_copy = dataclasses.replace(settings, generation=gen_settings)
    else:
        settings_copy = settings

    return RAGPipeline(
        settings=settings_copy,
        classifier=classifier,
        hybrid_search=hybrid_search,
        grader=grader,
        rewriter=rewriter,
        generator=generator,
    )


async def run_single_test(pipeline, question: str) -> RAGResponse:
    """运行单条测试。"""
    return await pipeline.run(question)


def evaluate_response(
    response: RAGResponse,
    test_case,
    llm_service,
) -> Dict:
    """评估单条响应。"""
    # 检索指标
    retrieved_ids = [c.chunk_id for c in response.citations if c.chunk_id]
    hr = hit_rate_at_k(retrieved_ids, test_case.ground_truth_chunks, k=5) if test_case.ground_truth_chunks else None
    mrr_score = mrr(retrieved_ids, test_case.ground_truth_chunks, k=10) if test_case.ground_truth_chunks else None

    # 生成指标
    context = "\n".join([c.text_snippet for c in response.citations])
    faith = faithfulness_llm(response.answer, context, llm_service)
    relev = relevancy_llm(test_case.question, response.answer, llm_service)
    ctx_recall = context_recall(context, test_case.expected_answer, llm_service)
    ctx_prec = context_precision(context, test_case.question, test_case.expected_answer, llm_service)

    return {
        "question": test_case.question,
        "expected_level": test_case.expected_level,
        "predicted_level": response.query_level,
        "level_correct": response.query_level == test_case.expected_level,
        "hit_rate": hr,
        "mrr": mrr_score,
        "faithfulness": faith,
        "answer_relevance": relev,
        "context_recall": ctx_recall,
        "context_precision": ctx_prec,
    }


def aggregate_results(details: List[Dict]) -> Dict:
    """汇总评估结果，含按级别细分。"""
    valid = [d for d in details if "error" not in d]
    if not valid:
        return {"total_cases": len(details), "error_cases": len(details)}

    def avg(key, items=None):
        items = items or valid
        vals = [d[key] for d in items if d.get(key) is not None]
        return sum(vals) / len(vals) if vals else 0.0

    levels_true = sum(1 for d in valid if d.get("level_correct"))
    total = len(valid)

    summary = {
        "total_cases": total,
        "error_cases": len(details) - total,
        "classification_accuracy": levels_true / total if total else 0.0,
        "hit_rate": avg("hit_rate"),
        "mrr": avg("mrr"),
        "faithfulness": avg("faithfulness"),
        "answer_relevance": avg("answer_relevance"),
        "context_recall": avg("context_recall"),
        "context_precision": avg("context_precision"),
    }

    # 按级别细分
    per_level = {}
    for level in ["L1", "L2", "L3", "L4"]:
        level_items = [d for d in valid if d.get("expected_level") == level]
        if level_items:
            lt = sum(1 for d in level_items if d.get("level_correct"))
            per_level[level] = {
                "count": len(level_items),
                "classification_accuracy": lt / len(level_items),
                "hit_rate": avg("hit_rate", level_items),
                "mrr": avg("mrr", level_items),
                "faithfulness": avg("faithfulness", level_items),
                "answer_relevance": avg("answer_relevance", level_items),
                "context_recall": avg("context_recall", level_items),
                "context_precision": avg("context_precision", level_items),
            }
    summary["per_level"] = per_level
    return summary


def save_incremental(output_path: str, all_results: Dict):
    """增量保存结果到文件。"""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(all_results, ensure_ascii=False, indent=2), encoding="utf-8")


def load_existing_results(output_path: str) -> Dict:
    """加载已有结果（用于 --resume）。"""
    p = Path(output_path)
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


async def main():
    parser = argparse.ArgumentParser(description="GradusRAG 消融实验")
    parser.add_argument("--max-cases", type=int, default=0, help="最大测试数（0=全部）")
    parser.add_argument("--output", "-o", default="results/ablation.json", help="输出路径")
    parser.add_argument("--test-set", "-t", default="formal_test_set.json", help="测试集文件名")
    parser.add_argument("--configs", nargs="+", default=None, help="指定要跑的配置（默认全部）")
    parser.add_argument("--resume", action="store_true", help="从上次中断处继续")
    parser.add_argument("--fast", action="store_true", help="快速模式（跳过 LLM 评估，用规则回退）")
    args = parser.parse_args()

    settings = load_settings()
    test_mgr = TestSetManager()
    test_cases = test_mgr.load(args.test_set)

    if not test_cases:
        logger.error("No test cases found")
        return

    if args.max_cases > 0:
        # 按级别均匀采样，而不是取前 N 个
        from collections import defaultdict
        by_level = defaultdict(list)
        for tc in test_cases:
            by_level[tc.expected_level if hasattr(tc, 'expected_level') else tc.get("expected_level", "L1")].append(tc)
        per_level = max(args.max_cases // 4, 1)
        sampled = []
        for level in ["L1", "L2", "L3", "L4"]:
            sampled.extend(by_level.get(level, [])[:per_level])
        test_cases = sampled
        logger.info(f"Sampled {len(test_cases)} cases: {per_level} per level")

    logger.info(f"Test cases: {len(test_cases)}")

    # 初始化共享组件
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
    llm_service = LLMService.from_settings(settings)
    eval_llm = None if args.fast else llm_service  # fast 模式不传 LLM，用规则回退

    # 选择要跑的配置
    configs_to_run = args.configs if args.configs else list(ABLATION_CONFIGS.keys())

    # 支持断点续跑
    all_results = {}
    if args.resume:
        all_results = load_existing_results(args.output)
        if all_results:
            completed = [k for k in all_results if "summary" in all_results[k]]
            logger.info(f"Resuming: found {len(completed)} completed configs: {completed}")
            configs_to_run = [c for c in configs_to_run if c not in completed]

    for config_name in configs_to_run:
        if config_name not in ABLATION_CONFIGS:
            logger.warning(f"Unknown config: {config_name}")
            continue

        config = ABLATION_CONFIGS[config_name]
        logger.info(f"\n{'='*60}")
        logger.info(f"Running ablation: {config_name}")
        logger.info(f"Config: {config}")
        logger.info(f"{'='*60}")

        # 构建 pipeline
        pipeline = build_ablation_pipeline(settings, llm_service, embedding_service, milvus_store, config)

        details = []
        for i, tc in enumerate(test_cases):
            logger.info(f"  [{i+1}/{len(test_cases)}] [{tc.expected_level}] {tc.question[:50]}...")
            try:
                t0 = time.monotonic()
                response = await run_single_test(pipeline, tc.question)
                elapsed = (time.monotonic() - t0) * 1000

                result = evaluate_response(response, tc, eval_llm)
                result["elapsed_ms"] = elapsed
                details.append(result)
                logger.info(f"    → {response.query_level}, {elapsed:.0f}ms, faith={result['faithfulness']:.2f}")

            except Exception as e:
                logger.error(f"    → Failed: {e}")
                details.append({"question": tc.question, "error": str(e)})

        summary = aggregate_results(details)
        all_results[config_name] = {"summary": summary, "details": details}

        # 每个配置跑完后增量保存
        save_incremental(args.output, all_results)
        logger.info(f"  [Saved] {args.output}")

        logger.info(f"\n  {config_name} Summary:")
        logger.info(f"    Classification Accuracy: {summary['classification_accuracy']:.2%}")
        logger.info(f"    Hit Rate@5: {summary['hit_rate']:.2%}")
        logger.info(f"    MRR: {summary['mrr']:.2%}")
        logger.info(f"    Faithfulness: {summary['faithfulness']:.2%}")
        logger.info(f"    Answer Relevance: {summary['answer_relevance']:.2%}")
        logger.info(f"    Context Recall: {summary['context_recall']:.2%}")
        logger.info(f"    Context Precision: {summary['context_precision']:.2%}")

    # 打印汇总表格
    print(f"\n{'='*100}")
    print(f"ABLATION RESULTS ({len(test_cases)} cases)")
    print(f"{'='*100}")
    print(f"{'Config':<20} {'Cls.Acc':>8} {'Hit@5':>8} {'MRR':>8} {'Faith':>8} {'Rel':>8} {'Recall':>8} {'Prec':>8}")
    print(f"{'-'*100}")
    for config_name in list(ABLATION_CONFIGS.keys()):
        if config_name in all_results and "summary" in all_results[config_name]:
            s = all_results[config_name]["summary"]
            print(
                f"{config_name:<20} "
                f"{s['classification_accuracy']:>7.2%} "
                f"{s['hit_rate']:>7.2%} "
                f"{s['mrr']:>7.2%} "
                f"{s['faithfulness']:>7.2%} "
                f"{s['answer_relevance']:>7.2%} "
                f"{s['context_recall']:>7.2%} "
                f"{s['context_precision']:>7.2%}"
            )
    print(f"{'='*100}")

    # 打印按级别细分表格
    print(f"\n{'='*100}")
    print("PER-LEVEL FAITHFULNESS BREAKDOWN")
    print(f"{'='*100}")
    for config_name in list(ABLATION_CONFIGS.keys()):
        if config_name in all_results and "summary" in all_results[config_name]:
            pl = all_results[config_name]["summary"].get("per_level", {})
            row = f"{config_name:<20}"
            for lv in ["L1", "L2", "L3", "L4"]:
                if lv in pl:
                    row += f"  {lv}: {pl[lv]['faithfulness']:.2%} ({pl[lv]['count']})"
                else:
                    row += f"  {lv}: --"
            print(row)
    print(f"{'='*100}")
    print(f"Results saved to {args.output}")


if __name__ == "__main__":
    asyncio.run(main())
