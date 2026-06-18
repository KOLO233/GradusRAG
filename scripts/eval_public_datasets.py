"""公开数据集下载与评估脚本。

支持数据集：
- HotpotQA（多跳推理，L2/L3）— 域外拒绝测试
- SQuAD 2.0（单段落阅读理解，L1）— 域外拒绝测试
- MuSiQue（多步推理，L3/L4）— 域外拒绝测试
- PubMedQA（医学问答）— 域内泛化测试

流程：
1. 从 HuggingFace 下载数据集
2. 抽取指定数量的样本
3. 自动分类为 L1-L4
4. 转换为 GradusRAG 测试集格式
5. 运行评估

用法：
    # 下载并转换数据集
    python scripts/eval_public_datasets.py --dataset hotpotqa --sample 50
    python scripts/eval_public_datasets.py --dataset squad --sample 50
    python scripts/eval_public_datasets.py --dataset pubmedqa --sample 50

    # 运行评估
    python scripts/eval_public_datasets.py --evaluate --dataset hotpotqa
    python scripts/eval_public_datasets.py --evaluate --dataset all
"""

import argparse
import json
import logging
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def download_hotpotqa(sample_size: int = 100) -> list:
    """下载 HotpotQA 数据集。"""
    from datasets import load_dataset
    logger.info("Downloading HotpotQA...")
    ds = load_dataset("hotpot_qa", "distractor", split="validation", trust_remote_code=True)

    samples = []
    for i, item in enumerate(ds):
        if i >= sample_size:
            break
        question = item["question"]
        answer = item["answer"]
        # HotpotQA 的 type 字段: "bridge" (两跳) 或 "comparison" (比较)
        qtype = item.get("type", "bridge")

        # 自动分类
        if qtype == "comparison":
            level = "L2"
            category = "比较"
        else:
            level = "L3"
            category = "因果推理"

        # 构造上下文
        context_parts = []
        for ctx in item.get("context", []):
            if isinstance(ctx, dict):
                title = ctx.get("title", "")
                sentences = ctx.get("sentences", [])
                context_parts.append(f"{title}: {' '.join(sentences)}")
            elif isinstance(ctx, (list, tuple)) and len(ctx) >= 2:
                context_parts.append(f"{ctx[0]}: {' '.join(ctx[1]) if isinstance(ctx[1], list) else ctx[1]}")

        samples.append({
            "question": question,
            "expected_answer": answer,
            "expected_level": level,
            "ground_truth_chunks": [],
            "category": category,
            "source": "hotpotqa",
            "context": "\n\n".join(context_parts)[:3000],
        })

    logger.info(f"Downloaded {len(samples)} HotpotQA samples")
    return samples


def download_squad(sample_size: int = 100) -> list:
    """下载 SQuAD 2.0 数据集。"""
    from datasets import load_dataset
    logger.info("Downloading SQuAD 2.0...")
    ds = load_dataset("rajpurkar/squad_v2", split="validation")

    samples = []
    count = 0
    for item in ds:
        if count >= sample_size:
            break
        # SQuAD 2.0 有些问题没有答案（answerable=False）
        if not item.get("answers", {}).get("text"):
            continue

        question = item["question"]
        answer = item["answers"]["text"][0] if item["answers"]["text"] else ""
        context = item["context"]
        title = item.get("title", "")

        # SQuAD 都是单段落事实问题 → L1
        samples.append({
            "question": question,
            "expected_answer": answer,
            "expected_level": "L1",
            "ground_truth_chunks": [],
            "category": "事实",
            "source": "squad",
            "context": f"{title}: {context}"[:3000],
        })
        count += 1

    logger.info(f"Downloaded {len(samples)} SQuAD samples")
    return samples


def download_musique(sample_size: int = 50) -> list:
    """下载 MuSiQue 数据集。"""
    from datasets import load_dataset
    logger.info("Downloading MuSiQue...")
    ds = load_dataset("bdsaglp/musique", "answerable", split="validation", trust_remote_code=True)

    samples = []
    for i, item in enumerate(ds):
        if i >= sample_size:
            break
        question = item["question"]
        answer = item["answer"]

        # MuSiQue 问题通常需要 2-4 步推理
        # 根据支撑段落数量判断级别
        paragraphs = item.get("paragraphs", [])
        supporting = [p for p in paragraphs if p.get("is_supporting")]

        if len(supporting) <= 2:
            level = "L3"
            category = "因果推理"
        else:
            level = "L4"
            category = "假设推理"

        context_parts = []
        for p in paragraphs:
            title = p.get("title", "")
            text = p.get("paragraph_text", "")
            context_parts.append(f"{title}: {text}")

        samples.append({
            "question": question,
            "expected_answer": answer,
            "expected_level": level,
            "ground_truth_chunks": [],
            "category": category,
            "source": "musique",
            "context": "\n\n".join(context_parts)[:3000],
        })

    logger.info(f"Downloaded {len(samples)} MuSiQue samples")
    return samples


def download_pubmedqa(sample_size: int = 50) -> list:
    """下载 PubMedQA 数据集（医学问答，域内泛化测试）。"""
    from datasets import load_dataset
    logger.info("Downloading PubMedQA...")
    try:
        ds = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train", trust_remote_code=True)
    except Exception:
        # 备选下载方式
        ds = load_dataset("pubmed_qa", "pqa_labeled", split="train", trust_remote_code=True)

    samples = []
    for i, item in enumerate(ds):
        if i >= sample_size:
            break
        question = item.get("question", "")
        # PubMedQA 的答案在 long_answer 或 final_decision 中
        answer = item.get("long_answer", item.get("final_decision", ""))
        if not answer:
            continue

        # PubMedQA 的 context
        context_parts = item.get("context", {})
        if isinstance(context_parts, dict):
            contexts = context_parts.get("contexts", [])
            context = "\n\n".join(contexts) if isinstance(contexts, list) else str(contexts)
        elif isinstance(context_parts, list):
            context = "\n\n".join(str(c) for c in context_parts)
        else:
            context = str(context_parts)

        # PubMedQA 问题通常是因果/推理类型
        samples.append({
            "question": question,
            "expected_answer": answer[:500],
            "expected_level": "L3",
            "ground_truth_chunks": [],
            "category": "医学推理",
            "source": "pubmedqa",
            "context": context[:3000],
        })

    logger.info(f"Downloaded {len(samples)} PubMedQA samples")
    return samples


def save_test_set(samples: list, dataset_name: str, output_dir: str = "data/test_sets"):
    """保存为 GradusRAG 测试集格式。"""
    output_path = Path(output_dir) / f"{dataset_name}_test_set.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(samples, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(f"Saved {len(samples)} cases to {output_path}")

    # 统计
    level_counts = {}
    for s in samples:
        lv = s["expected_level"]
        level_counts[lv] = level_counts.get(lv, 0) + 1
    for lv in sorted(level_counts):
        logger.info(f"  {lv}: {level_counts[lv]}")
    return str(output_path)


def run_evaluation(test_set_path: str, max_cases: int = 0, output_path: str = ""):
    """运行评估。"""
    from src.core.settings import load_settings
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
    from src.evaluation.evaluator import Evaluator
    from src.evaluation.test_set import TestSetManager

    settings = load_settings()

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

    from src.retrieval.graph_retriever import GraphRetriever
    from src.ingestion.graph_builder.graph_store import GraphStore

    dense = DenseRetriever(embedding_service, milvus_store)
    sparse = SparseRetriever(embedding_service, milvus_store)
    graph_store = GraphStore(persist_path="data/knowledge_graph.json")
    graph_retriever = GraphRetriever(graph_store) if settings.graph.enabled else None
    reranker = Reranker(llm_service=llm_service)

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

    evaluator = Evaluator(pipeline=pipeline, llm_service=llm_service)
    test_set_name = Path(test_set_path).name

    result = evaluator.evaluate(test_set_file=test_set_name, max_cases=max_cases)

    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info(f"Results saved to {out}")

    return result


def main():
    parser = argparse.ArgumentParser(description="公开数据集下载与评估")
    parser.add_argument("--dataset", choices=["hotpotqa", "squad", "musique", "pubmedqa", "all"], default="hotpotqa")
    parser.add_argument("--sample", type=int, default=100, help="抽取样本数")
    parser.add_argument("--evaluate", action="store_true", help="运行评估")
    parser.add_argument("--max-cases", type=int, default=0, help="最大评估数（0=全部）")
    parser.add_argument("--output", "-o", default="", help="评估结果输出路径")
    args = parser.parse_args()

    datasets_to_process = ["hotpotqa", "squad", "pubmedqa"] if args.dataset == "all" else [args.dataset]

    if not args.evaluate:
        # 下载并转换
        for ds_name in datasets_to_process:
            if ds_name == "hotpotqa":
                samples = download_hotpotqa(args.sample)
            elif ds_name == "squad":
                samples = download_squad(args.sample)
            elif ds_name == "musique":
                samples = download_musique(min(args.sample, 50))
            elif ds_name == "pubmedqa":
                samples = download_pubmedqa(args.sample)
            else:
                continue
            save_test_set(samples, ds_name)
    else:
        # 运行评估
        for ds_name in datasets_to_process:
            test_set_path = f"data/test_sets/{ds_name}_test_set.json"
            if not Path(test_set_path).exists():
                logger.error(f"Test set not found: {test_set_path}. Run without --evaluate first.")
                continue

            output = args.output or f"results/{ds_name}_eval.json"
            logger.info(f"\n{'='*60}")
            logger.info(f"Evaluating: {ds_name}")
            logger.info(f"{'='*60}")

            result = run_evaluation(test_set_path, args.max_cases, output)

            print(f"\n{'='*60}")
            print(f"{ds_name.upper()} Results:")
            print(f"  Cases: {result.total_cases}")
            print(f"  Classification Accuracy: {result.classification_accuracy:.2%}")
            print(f"  Hit Rate@5: {result.hit_rate:.2%}")
            print(f"  MRR: {result.mrr:.2%}")
            print(f"  Faithfulness: {result.faithfulness:.2%}")
            print(f"  Answer Relevance: {result.answer_relevance:.2%}")
            print(f"  Context Recall: {result.context_recall:.2%}")
            print(f"  Context Precision: {result.context_precision:.2%}")
            print(f"{'='*60}")


if __name__ == "__main__":
    main()
