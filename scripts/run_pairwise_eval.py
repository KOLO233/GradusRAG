"""Pairwise 评估脚本（对标 LightRAG 论文 Table 6）。

对每道题分别用 GradusRAG 和 Baseline 生成回答，
然后用 LLM-as-Judge 比较两个回答的质量。

评估指标（与 LightRAG 一致）：
- Comprehensiveness（全面性）
- Diversity（多样性）
- Empowerment（赋能度）
- Overall（综合）

用法：
    python scripts/run_pairwise_eval.py
    python scripts/run_pairwise_eval.py --resume
    python scripts/run_pairwise_eval.py --max-cases 50  # 先跑 50 题测试
"""

import argparse
import json
import logging
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ============================================================
# Judge Prompt（与 LightRAG 完全一致）
# ============================================================

JUDGE_SYSTEM_PROMPT = """You are an expert tasked with evaluating two answers to the same question based on three criteria: **Comprehensiveness**, **Diversity**, and **Empowerment**."""

JUDGE_PROMPT = """You will evaluate two answers to the same question based on three criteria: **Comprehensiveness**, **Diversity**, and **Empowerment**.

- **Comprehensiveness**: How much detail does the answer provide to cover all aspects and details of the question?
- **Diversity**: How varied and rich is the answer in providing different perspectives and insights on the question?
- **Empowerment**: How well does the answer help the reader understand and make informed judgments about the topic?

For each criterion, choose the better answer (either Answer 1 or Answer 2) and explain why. Then, select an overall winner based on these three categories.

Here is the question:
{question}

Here are the two answers:

**Answer 1:**
{answer1}

**Answer 2:**
{answer2}

Evaluate both answers using the three criteria listed above and provide detailed explanations for each criterion.

Output your evaluation in the following JSON format:

{{
    "Comprehensiveness": {{
        "Winner": "[Answer 1 or Answer 2]",
        "Explanation": "[Provide explanation here]"
    }},
    "Diversity": {{
        "Winner": "[Answer 1 or Answer 2]",
        "Explanation": "[Provide explanation here]"
    }},
    "Empowerment": {{
        "Winner": "[Answer 1 or Answer 2]",
        "Explanation": "[Provide explanation here]"
    }},
    "Overall": {{
        "Winner": "[Answer 1 or Answer 2]",
        "Explanation": "[Summarize why this answer is the overall winner]"
    }}
}}"""

# Baseline Prompt（统一生成，无 L1-L4 分级）
BASELINE_PROMPT = """你是一个专业的领域问答助手。请根据提供的参考资料回答用户问题。

参考资料：
{context}

用户问题：{question}

请提供详细、结构化的回答。包含相关事实、解释和例子。
如果参考资料不足以完整回答，说明你能确定的部分和不确定的部分。
引用时标注来源编号 [1][2]。"""


def parse_judge_result(response: str) -> Optional[Dict]:
    """解析 LLM 裁判的评判结果。"""
    # 去掉 <think> 标签
    cleaned = re.sub(r'<think>[\s\S]*?</think>', '', response)

    # 提取 JSON
    match = re.search(r'\{[\s\S]*\}', cleaned)
    if not match:
        return None

    try:
        data = json.loads(match.group())
        return data
    except json.JSONDecodeError:
        # 尝试修复
        try:
            fixed = re.sub(r',\s*}', '}', match.group())
            fixed = re.sub(r',\s*]', ']', fixed)
            return json.loads(fixed)
        except (json.JSONDecodeError, ValueError):
            return None


def count_wins(results: List[Dict], answer1_name: str, answer2_name: str) -> Dict:
    """统计胜/负/平。"""
    wins = {"answer1_wins": 0, "answer2_wins": 0, "ties": 0}
    criteria = ["Comprehensiveness", "Diversity", "Empowerment", "Overall"]

    for r in results:
        if r.get("judge_result") is None:
            wins["ties"] += 1
            continue

        overall = r["judge_result"].get("Overall", {})
        winner = overall.get("Winner", "")

        if "1" in winner and "2" not in winner:
            wins["answer1_wins"] += 1
        elif "2" in winner and "1" not in winner:
            wins["answer2_wins"] += 1
        else:
            wins["ties"] += 1

    return wins


async def run_evaluation(
    llm,
    questions: List[Dict],
    rag_pipeline=None,
    max_cases: int = 0,
    save_path: str = "",
) -> Dict:
    """运行 Pairwise 评估。

    Args:
        llm: LLM 服务
        questions: 测试题目列表
        rag_pipeline: GradusRAG Pipeline 实例
        max_cases: 最大评估题数（0=全部）
        save_path: 增量保存路径
    """
    if max_cases > 0:
        questions = questions[:max_cases]

    results = []
    completed_questions = set()

    # 断点续传
    if save_path and Path(save_path).exists():
        try:
            existing = json.loads(Path(save_path).read_text(encoding="utf-8"))
            results = existing.get("results", [])
            completed_questions = {r["question"] for r in results}
            logger.info(f"Resumed: {len(results)} already completed")
        except Exception:
            pass

    for i, q_data in enumerate(questions):
        question = q_data.get("question", "")
        domain = q_data.get("domain", "unknown")

        if question in completed_questions:
            continue

        logger.info(f"[{i+1}/{len(questions)}] {domain}: {question[:60]}...")

        try:
            # 1. GradusRAG 回答
            t0 = time.monotonic()
            if rag_pipeline:
                rag_response = await rag_pipeline.run(question)
                answer1 = rag_response.answer
            else:
                # 无 pipeline 时用简单 LLM 回答
                answer1 = await llm.ainvoke(f"Answer this question comprehensively: {question}")
            rag_time = (time.monotonic() - t0) * 1000

            # 2. Baseline 回答（LightRAG 风格：有检索 + 统一 Prompt，无 L1-L4 分级）
            t0 = time.monotonic()
            if rag_pipeline and rag_pipeline._search:
                # 用 GradusRAG 的检索系统获取上下文
                from src.core.types import ProcessedQuery
                from src.libs.text_utils import extract_keywords
                keywords = extract_keywords(question, min_length=2)
                baseline_query = ProcessedQuery(
                    original_query=question,
                    classified_level="L1",  # 基线不分类，统一用 L1
                    keywords=keywords,
                )
                search_result = await rag_pipeline._search.search(query=baseline_query)
                baseline_context = rag_pipeline._format_context(search_result.results)
                baseline_answer = await llm.ainvoke(
                    BASELINE_PROMPT.format(context=baseline_context[:4000], question=question)
                )
            else:
                baseline_answer = await llm.ainvoke(
                    BASELINE_PROMPT.format(context="", question=question)
                )
            baseline_time = (time.monotonic() - t0) * 1000

            # 3. LLM Judge（随机交换顺序避免位置偏差）
            import random
            if random.random() < 0.5:
                judge_answer1, judge_answer2 = answer1, baseline_answer
                answer1_is_rag = True
            else:
                judge_answer1, judge_answer2 = baseline_answer, answer1
                answer1_is_rag = False

            judge_prompt = JUDGE_PROMPT.format(
                question=question,
                answer1=judge_answer1[:2000],
                answer2=judge_answer2[:2000],
            )
            judge_response = await llm.ainvoke(judge_prompt, system_prompt=JUDGE_SYSTEM_PROMPT)
            judge_result = parse_judge_result(judge_response)

            # 还原 Winner 标记（考虑位置交换）
            if judge_result and not answer1_is_rag:
                # 交换 Winner 标记
                for key in ["Comprehensiveness", "Diversity", "Empowerment", "Overall"]:
                    if key in judge_result:
                        w = judge_result[key].get("Winner", "")
                        if "1" in w and "2" not in w:
                            judge_result[key]["Winner"] = "Answer 2"
                        elif "2" in w and "1" not in w:
                            judge_result[key]["Winner"] = "Answer 1"

            result = {
                "question": question,
                "domain": domain,
                "rag_answer": answer1[:500],
                "baseline_answer": baseline_answer[:500],
                "judge_result": judge_result,
                "rag_time_ms": rag_time,
                "baseline_time_ms": baseline_time,
            }
            results.append(result)

            # 输出单题结果
            if judge_result:
                overall_winner = judge_result.get("Overall", {}).get("Winner", "?")
                logger.info(f"  Winner: {overall_winner}")
            else:
                logger.warning(f"  Judge parse failed")

        except Exception as e:
            logger.error(f"  Failed: {e}")
            results.append({
                "question": question,
                "domain": domain,
                "error": str(e),
            })

        # 增量保存
        if save_path:
            _save_partial(save_path, results)

        time.sleep(0.5)

    # 汇总
    summary = _compute_summary(results)
    return {"results": results, "summary": summary}


def _save_partial(path: str, results: list):
    """增量保存。"""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(
        json.dumps({"results": results}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _compute_summary(results: List[Dict]) -> Dict:
    """计算汇总统计。"""
    summary = {"total": len(results), "by_domain": {}, "overall": {}}

    # 按领域统计
    by_domain = {}
    for r in results:
        d = r.get("domain", "unknown")
        if d not in by_domain:
            by_domain[d] = []
        by_domain[d].append(r)

    for domain, domain_results in by_domain.items():
        wins = count_wins(domain_results, "GradusRAG", "Baseline")
        total = len(domain_results)
        summary["by_domain"][domain] = {
            "total": total,
            "gradusrag_wins": wins["answer1_wins"],
            "baseline_wins": wins["answer2_wins"],
            "ties": wins["ties"],
            "gradusrag_win_rate": f"{wins['answer1_wins']/total*100:.1f}%" if total else "0%",
        }

    # 总体统计
    overall_wins = count_wins(results, "GradusRAG", "Baseline")
    total = len(results)
    summary["overall"] = {
        "total": total,
        "gradusrag_wins": overall_wins["answer1_wins"],
        "baseline_wins": overall_wins["answer2_wins"],
        "ties": overall_wins["ties"],
        "gradusrag_win_rate": f"{overall_wins['answer1_wins']/total*100:.1f}%" if total else "0%",
    }

    return summary


def create_rag_pipeline():
    """创建完整的 GradusRAG Pipeline。"""
    from src.core.settings import load_settings
    from src.core.trace import TraceCollector
    from src.libs.llm_service import LLMService
    from src.libs.embedding_service import EmbeddingService
    from src.retrieval.milvus_store import MilvusStore
    from src.retrieval.dense_retriever import DenseRetriever
    from src.retrieval.sparse_retriever import SparseRetriever
    from src.retrieval.hybrid_search import HybridSearch
    from src.retrieval.reranker import Reranker
    from src.retrieval.graph_retriever import GraphRetriever
    from src.ingestion.graph_builder.graph_store import GraphStore
    from src.query_classifier.classifier import create_classifier
    from src.generation.pipeline import RAGPipeline
    from src.generation.document_grader import DocumentGrader
    from src.generation.query_rewriter import QueryRewriter
    from src.generation.response_generator import ResponseGenerator

    settings = load_settings()
    llm_service = LLMService.from_settings(settings)
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

    dense_retriever = DenseRetriever(embedding_service, milvus_store)
    sparse_retriever = SparseRetriever(embedding_service, milvus_store)

    graph_store = GraphStore(persist_path="data/knowledge_graph.json")
    graph_retriever = GraphRetriever(graph_store, embedding_service) if settings.graph.enabled else None

    reranker = Reranker(llm_service=llm_service)

    hybrid_search = HybridSearch(
        settings=settings,
        dense_retriever=dense_retriever,
        sparse_retriever=sparse_retriever,
        graph_retriever=graph_retriever,
        reranker=reranker,
    )

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

    logger.info("RAG Pipeline initialized")
    return pipeline


async def main():
    parser = argparse.ArgumentParser(description="Pairwise 评估（对标 LightRAG Table 6）")
    parser.add_argument("--input", "-i", default="data/test_sets/formal_test_set.json")
    parser.add_argument("--output", "-o", default="results/pairwise_cn_eval.json")
    parser.add_argument("--max-cases", type=int, default=0, help="最大评估题数（0=全部）")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    from src.core.settings import load_settings
    from src.libs.llm_service import LLMService

    settings = load_settings()
    llm = LLMService.from_settings(settings)

    # 加载题目
    questions = json.loads(Path(args.input).read_text(encoding="utf-8"))
    logger.info(f"Loaded {len(questions)} questions")

    # 初始化 GradusRAG Pipeline
    logger.info("Initializing RAG Pipeline...")
    rag_pipeline = create_rag_pipeline()

    # 清空旧结果（如果不用 --resume）
    if not args.resume and Path(args.output).exists():
        Path(args.output).unlink()

    # 运行评估
    result = await run_evaluation(
        llm=llm,
        questions=questions,
        rag_pipeline=rag_pipeline,
        max_cases=args.max_cases,
        save_path=args.output,
    )

    # 保存
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # 打印汇总
    summary = result["summary"]
    print(f"\n{'='*60}")
    print(f"Pairwise Evaluation Results (GradusRAG vs Baseline)")
    print(f"{'='*60}")
    print(f"{'Domain':<15} {'Total':>6} {'GradusRAG':>10} {'Baseline':>10} {'Ties':>6} {'Win%':>8}")
    print(f"{'-'*60}")
    for domain, stats in summary.get("by_domain", {}).items():
        print(f"{domain:<15} {stats['total']:>6} {stats['gradusrag_wins']:>10} {stats['baseline_wins']:>10} {stats['ties']:>6} {stats['gradusrag_win_rate']:>8}")
    overall = summary.get("overall", {})
    print(f"{'-'*60}")
    print(f"{'Overall':<15} {overall.get('total', 0):>6} {overall.get('gradusrag_wins', 0):>10} {overall.get('baseline_wins', 0):>10} {overall.get('ties', 0):>6} {overall.get('gradusrag_win_rate', '0%'):>8}")
    print(f"{'='*60}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
