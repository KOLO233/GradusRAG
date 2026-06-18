"""用实际文档内容重新生成种子测试集的参考答案。

解决参考答案与文档内容不匹配的问题。

用法：
    python scripts/regenerate_answers.py
    python scripts/regenerate_answers.py --input seed_test_set.json --output seed_test_set_v2.json
"""

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.settings import load_settings
from src.core.types import ProcessedQuery
from src.libs.llm_service import LLMService
from src.libs.embedding_service import EmbeddingService
from src.retrieval.milvus_store import MilvusStore
from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.sparse_retriever import SparseRetriever
from src.retrieval.hybrid_search import HybridSearch
from src.retrieval.reranker import Reranker
from src.libs.text_utils import extract_keywords


REGENERATE_PROMPT = """你是一个领域专家。请根据以下检索到的参考资料，回答用户问题。

参考资料：
{context}

问题：{question}

要求：
1. 严格基于参考资料回答，不要使用参考资料以外的知识
2. 回答要完整、准确
3. 如果参考资料中有相关内容，尽量详细引用

请直接给出回答："""


async def regenerate(input_file: str, output_file: str):
    settings = load_settings()

    # 构建检索器
    embedding_service = EmbeddingService(
        model_name=settings.embedding.model,
        device=settings.embedding.device,
        dimensions=settings.embedding.dimensions,
        api_key=settings.embedding.api_key,
        api_base_url=settings.embedding.api_base_url,
    )
    milvus = MilvusStore(
        host=settings.vector_store.host,
        port=settings.vector_store.port,
        collection=settings.vector_store.collection,
    )
    llm_service = LLMService.from_settings(settings)
    dense = DenseRetriever(embedding_service, milvus)
    sparse = SparseRetriever(embedding_service, milvus)
    reranker = Reranker(llm_service=llm_service)

    hybrid = HybridSearch(
        settings=settings,
        dense_retriever=dense,
        sparse_retriever=sparse,
        reranker=reranker,
    )

    # 加载测试集
    test_set_path = Path(__file__).resolve().parent.parent / "data" / "test_sets" / input_file
    test_cases = json.loads(test_set_path.read_text(encoding="utf-8"))

    print(f"加载 {len(test_cases)} 条测试用例，开始重新生成参考答案...\n")

    for i, tc in enumerate(test_cases):
        question = tc["question"]
        level = tc["expected_level"]
        old_answer = tc["expected_answer"]

        # 检索相关文档
        keywords = extract_keywords(question)
        query = ProcessedQuery(
            original_query=question,
            classified_level=level,
            keywords=keywords,
        )
        result = await hybrid.search(query, top_k=5)

        if not result.results:
            print(f"[{i+1}] {question[:50]}... → 无检索结果，跳过")
            continue

        # 拼接上下文
        context_parts = []
        for j, r in enumerate(result.results, 1):
            fname = r.metadata.get("filename", "Unknown")
            context_parts.append(f"[{j}] {fname}:\n{r.text}")
        context = "\n\n---\n\n".join(context_parts)

        # 用 LLM 生成参考答案
        prompt = REGENERATE_PROMPT.format(context=context[:4000], question=question)
        try:
            new_answer = await llm_service.ainvoke(prompt)
            tc["expected_answer"] = new_answer.strip()
            print(f"[{i+1}] {question[:50]}... → 已重新生成 ({len(new_answer)} 字)")
        except Exception as e:
            print(f"[{i+1}] {question[:50]}... → 失败: {e}")

    # 保存
    output_path = test_set_path.parent / output_file
    output_path.write_text(
        json.dumps(test_cases, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\n已保存到: {output_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="seed_test_set.json")
    parser.add_argument("--output", default="seed_test_set.json")  # 直接覆盖原文件
    args = parser.parse_args()
    asyncio.run(regenerate(args.input, args.output))
