"""GradusRAG 端到端评估脚本。

一键完成：生成测试集 → 运行 Pipeline → 计算指标 → 输出报告。

用法：
    # 使用已有测试集评估
    python scripts/run_evaluation.py

    # 自动生成测试集并评估
    python scripts/run_evaluation.py --auto-generate --count 50

    # 指定测试集文件
    python scripts/run_evaluation.py --test-set my_test_set.json

    # 只评估特定级别
    python scripts/run_evaluation.py --level L3

    # 保存增量结果（支持中断后续跑）
    python scripts/run_evaluation.py --save-progress results/eval_progress.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.settings import load_settings
from src.core.types import EvalTestCase
from src.libs.llm_service import LLMService
from src.evaluation.evaluator import Evaluator
from src.evaluation.test_set import TestSetManager
from src.generation.pipeline import RAGPipeline
from src.query_classifier.classifier import create_classifier
from src.generation.document_grader import DocumentGrader
from src.generation.query_rewriter import QueryRewriter
from src.generation.response_generator import ResponseGenerator
from src.libs.embedding_service import EmbeddingService
from src.retrieval.milvus_store import MilvusStore
from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.sparse_retriever import SparseRetriever
from src.retrieval.hybrid_search import HybridSearch
from src.retrieval.reranker import Reranker
from src.core.trace import TraceCollector

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def build_pipeline(settings) -> RAGPipeline:
    """构建完整的 RAG Pipeline（与 app.py 一致）。"""
    # 基础服务
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

    # 检索器
    dense_retriever = DenseRetriever(embedding_service, milvus_store)
    sparse_retriever = SparseRetriever(embedding_service, milvus_store)
    reranker = Reranker(llm_service=llm_service)

    # 图检索器（可选）
    graph_retriever = None
    if settings.graph.enabled:
        from src.ingestion.graph_builder.graph_store import GraphStore
        from src.retrieval.graph_retriever import GraphRetriever
        graph_store = GraphStore(persist_path="data/knowledge_graph.json")
        graph_retriever = GraphRetriever(graph_store, embedding_service)

    hybrid_search = HybridSearch(
        settings=settings,
        dense_retriever=dense_retriever,
        sparse_retriever=sparse_retriever,
        graph_retriever=graph_retriever,
        reranker=reranker,
    )

    # 生成器
    classifier = create_classifier(settings.query_classifier)
    grader = DocumentGrader(llm_service)
    rewriter = QueryRewriter(llm_service)
    generator = ResponseGenerator(llm_service, hybrid_search)
    trace_collector = TraceCollector()

    pipeline = RAGPipeline(
        settings=settings,
        classifier=classifier,
        hybrid_search=hybrid_search,
        grader=grader,
        rewriter=rewriter,
        generator=generator,
        trace_collector=trace_collector,
    )

    return pipeline


def print_report(result, level_filter=None):
    """打印评估报告。"""
    print("\n" + "=" * 60)
    print("  GradusRAG 评估报告")
    print("=" * 60)

    if level_filter:
        print(f"  过滤级别: {level_filter}")

    print(f"  测试用例数: {result.total_cases}")
    print()

    # 分类准确率
    print(f"  分类准确率 (Classification Accuracy):  {result.classification_accuracy:.1%}")

    # 检索指标
    if result.hit_rate > 0:
        print(f"  Hit Rate@5:                          {result.hit_rate:.1%}")
    else:
        print(f"  Hit Rate@5:                          N/A (无 ground_truth_chunks)")

    if result.mrr > 0:
        print(f"  MRR:                                 {result.mrr:.1%}")
    else:
        print(f"  MRR:                                 N/A (无 ground_truth_chunks)")

    # 生成指标
    print(f"  忠实度 (Faithfulness):                 {result.faithfulness:.1%}")
    print(f"  相关性 (Answer Relevance):             {result.answer_relevance:.1%}")
    print(f"  上下文召回 (Context Recall):           {result.context_recall:.1%}")
    print(f"  上下文精确 (Context Precision):        {result.context_precision:.1%}")

    # 按级别统计
    if result.details:
        print("\n  --- 按级别统计 ---")
        level_stats = defaultdict(lambda: {"correct": 0, "total": 0, "faith": [], "relev": []})
        for d in result.details:
            if "error" in d:
                continue
            level = d.get("expected_level", "L1")
            level_stats[level]["total"] += 1
            if d.get("level_correct"):
                level_stats[level]["correct"] += 1
            if d.get("faithfulness") is not None:
                level_stats[level]["faith"].append(d["faithfulness"])
            if d.get("answer_relevance") is not None:
                level_stats[level]["relev"].append(d["answer_relevance"])

        for level in ["L1", "L2", "L3", "L4"]:
            stats = level_stats[level]
            if stats["total"] == 0:
                continue
            acc = stats["correct"] / stats["total"]
            faith = sum(stats["faith"]) / len(stats["faith"]) if stats["faith"] else 0
            relev = sum(stats["relev"]) / len(stats["relev"]) if stats["relev"] else 0
            print(f"  {level}: 分类准确率={acc:.1%}, 忠实度={faith:.1%}, 相关性={relev:.1%} ({stats['total']}条)")

    print("=" * 60)


async def main():
    parser = argparse.ArgumentParser(description="GradusRAG 端到端评估")
    parser.add_argument("--test-set", type=str, default="seed_test_set.json", help="测试集文件名")
    parser.add_argument("--auto-generate", action="store_true", help="自动生成测试集")
    parser.add_argument("--count", type=int, default=50, help="自动生成数量")
    parser.add_argument("--level", type=str, help="只评估特定级别 (L1/L2/L3/L4)")
    parser.add_argument("--max-cases", type=int, default=0, help="最大评估用例数（0=全部）")
    parser.add_argument("--save-progress", type=str, help="增量保存路径")
    parser.add_argument("--output", type=str, help="结果输出路径")
    args = parser.parse_args()

    settings = load_settings()

    # 自动生成测试集
    if args.auto_generate:
        from scripts.generate_test_set import TestSetGenerator
        from src.retrieval.milvus_store import MilvusStore

        llm_service = LLMService.from_settings(settings)
        generator = TestSetGenerator(llm_service)
        milvus = MilvusStore(
            host=settings.vector_store.host,
            port=settings.vector_store.port,
            collection=settings.vector_store.collection,
        )
        test_cases = await generator.generate_from_milvus(milvus, count=args.count)
        if test_cases:
            generator.save(test_cases, args.test_set)
        else:
            print("自动生成失败，请检查 Milvus 中是否有文档")
            return

    # 构建 Pipeline
    logger.info("Building RAG Pipeline...")
    pipeline = build_pipeline(settings)

    # 构建评估器
    evaluator = Evaluator(pipeline=pipeline, llm_service=LLMService.from_settings(settings))

    # 按级别过滤
    if args.level:
        # 先加载再过滤
        test_cases = evaluator._test_set_mgr.load(args.test_set)
        filtered = evaluator._test_set_mgr.filter_by_level(test_cases, args.level)
        filtered_file = f"_temp_{args.level}.json"
        evaluator._test_set_mgr.save(filtered_file, filtered)
        test_set_file = filtered_file
    else:
        test_set_file = args.test_set

    # 运行评估
    logger.info(f"Running evaluation on {test_set_file}...")
    result = evaluator.evaluate(
        test_set_file=test_set_file,
        max_cases=args.max_cases,
        save_path=args.save_progress or "",
    )

    # 打印报告
    print_report(result, level_filter=args.level)

    # 保存结果
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(result.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.info(f"Results saved to {output_path}")

    # 清理临时文件
    if args.level:
        temp_path = evaluator._test_set_mgr._data_dir / f"_temp_{args.level}.json"
        if temp_path.exists():
            temp_path.unlink()


if __name__ == "__main__":
    asyncio.run(main())
