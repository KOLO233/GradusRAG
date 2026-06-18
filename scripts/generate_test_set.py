"""测试集自动生成器。

从已有文档自动生成 L1-L4 四级测试用例，无需人工标注。

流程：
1. 加载已入库的文档分块
2. 用 LLM 对每个分块生成 4 个级别的问题
3. 用 LLM 生成参考答案（基于原文）
4. 保存为标准测试集 JSON

用法：
    python scripts/generate_test_set.py                     # 从 Milvus 生成
    python scripts/generate_test_set.py --docs ./data/docs  # 从文档目录生成
    python scripts/generate_test_set.py --count 50          # 生成 50 条
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import random
import sys
from pathlib import Path

# 添加项目根目录到 sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.settings import load_settings
from src.libs.llm_service import LLMService
from src.ingestion.pipeline import IngestionPipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ===========================================================================
# Prompt 模板
# ===========================================================================

GENERATE_QUESTIONS_PROMPT = """你是一个专业的测试题出题专家。请根据以下参考资料，分别为 L1-L4 四个级别各生成 1 个问题。

参考资料：
{context}

级别定义：
- L1（显性事实）：答案直接出现在参考资料中的简单事实查询。如："什么是X？""列举X的类型"
- L2（隐性事实）：需要跨段落推理的比较/对比类查询。如："比较A和B的区别"
- L3（可解释原理）：需要因果推理的查询。如："为什么X会导致Y？""解释X的原理"
- L4（隐藏原理）：需要假设推理的查询。如："如果改变X会怎样？""假设X成立，推断Y"

要求：
1. 每个问题必须能从参考资料中找到答案依据
2. 问题要自然、真实，像真正的用户会问的
3. 涵盖参考资料中的不同知识点

请以 JSON 格式输出：
```json
{{
  "L1": "问题文本",
  "L2": "问题文本",
  "L3": "问题文本",
  "L4": "问题文本"
}}
```"""

GENERATE_ANSWER_PROMPT = """你是一个领域专家。请根据以下参考资料，回答用户问题。

参考资料：
{context}

问题：{question}

要求：
1. 严格基于参考资料回答，不要编造
2. 回答要完整、准确
3. 如果参考资料不足以完整回答，说明哪些部分可以回答、哪些不能

请直接给出回答："""


class TestSetGenerator:
    """测试集自动生成器。

    Example:
        >>> generator = TestSetGenerator(llm_service)
        >>> test_cases = await generator.generate_from_milvus(milvus_store, count=50)
        >>> generator.save(test_cases, "auto_test_set.json")
    """

    def __init__(self, llm_service: LLMService):
        self._llm = llm_service

    async def generate_from_chunks(
        self,
        chunks: list,
        count: int = 50,
        category: str = "auto",
    ) -> list:
        """从文档分块生成测试用例。

        Args:
            chunks: 分块列表，每项需有 .text 和 .id 属性
            count: 目标生成数量
            category: 测试类别标签

        Returns:
            测试用例字典列表
        """
        test_cases = []
        per_level = count // 4  # 每个级别目标数量
        level_counts = {"L1": 0, "L2": 0, "L3": 0, "L4": 0}
        level_targets = {"L1": per_level, "L2": per_level, "L3": per_level, "L4": per_level}
        # 多余的分配给 L3/L4（它们更有价值）
        remainder = count - per_level * 4
        for level in ["L3", "L4", "L2", "L1"]:
            if remainder > 0:
                level_targets[level] += 1
                remainder -= 1

        # 随机采样分块（多取一些，因为不是每个都能成功生成）
        sampled = random.sample(chunks, min(count * 2, len(chunks)))

        for i, chunk in enumerate(sampled):
            # 检查是否所有级别都已达到目标
            if all(level_counts[lv] >= level_targets[lv] for lv in level_targets):
                break

            logger.info(f"[{i+1}/{len(sampled)}] Generating questions from chunk {chunk.id[:20]}...")

            try:
                # Step 1: 生成 4 个级别的问题
                prompt = GENERATE_QUESTIONS_PROMPT.format(context=chunk.text[:3000])
                response = await self._llm.ainvoke(prompt)

                # 解析 JSON
                import re
                json_match = re.search(r'\{[\s\S]*\}', response)
                if not json_match:
                    logger.warning(f"  Failed to parse questions, skipping")
                    continue

                questions = json.loads(json_match.group())

                # Step 2: 为每个问题生成参考答案（只生成还缺的级别）
                for level, question in questions.items():
                    if level not in level_targets:
                        continue
                    if level_counts[level] >= level_targets[level]:
                        continue  # 该级别已满，跳过
                    if not question or len(question) < 5:
                        continue

                    answer_prompt = GENERATE_ANSWER_PROMPT.format(
                        context=chunk.text[:3000],
                        question=question,
                    )
                    answer = await self._llm.ainvoke(answer_prompt)

                    test_cases.append({
                        "question": question.strip(),
                        "expected_answer": answer.strip(),
                        "expected_level": level,
                        "ground_truth_chunks": [chunk.id],
                        "category": category,
                        "source_chunk_id": chunk.id,
                    })
                    level_counts[level] += 1

                    logger.info(f"  [{level}] {question[:50]}... ({level_counts[level]}/{level_targets[level]})")

            except Exception as e:
                logger.error(f"  Failed: {e}")
                continue

        # 打印分布统计
        logger.info(f"Generated {len(test_cases)} test cases: {dict(level_counts)}")
        return test_cases

    async def generate_from_milvus(
        self,
        milvus_store,
        count: int = 50,
        category: str = "auto",
    ) -> list:
        """从 Milvus 中的已有文档生成测试用例。"""
        client = milvus_store._get_client()
        collection = milvus_store._collection

        # 查询所有文档
        all_docs = client.query(
            collection,
            output_fields=["chunk_id", "text"],
            limit=min(count * 3, 16384),  # 多取一些，因为不是每个都能成功生成
        )

        if not all_docs:
            logger.error("No documents found in Milvus")
            return []

        # 转为简单对象
        class ChunkProxy:
            def __init__(self, d):
                self.id = d["chunk_id"]
                self.text = d["text"]
                self.metadata = {"source_path": "milvus"}

        chunks = [ChunkProxy(d) for d in all_docs if d.get("text")]
        logger.info(f"Found {len(chunks)} chunks in Milvus")

        return await self.generate_from_chunks(chunks, count=count, category=category)

    async def generate_from_directory(
        self,
        dir_path: str,
        settings,
        count: int = 50,
        category: str = "auto",
    ) -> list:
        """从文档目录生成测试用例（先摄取再生成）。"""
        from src.core.types import Chunk

        pipeline = IngestionPipeline(settings)
        chunks, _ = pipeline.ingest_directory(dir_path)

        if not chunks:
            logger.error(f"No chunks generated from {dir_path}")
            return []

        logger.info(f"Generated {len(chunks)} chunks from {dir_path}")
        return await self.generate_from_chunks(chunks, count=count, category=category)

    @staticmethod
    def save(test_cases: list, filename: str, data_dir: str = None):
        """保存测试集到 JSON 文件。"""
        if data_dir is None:
            data_dir = Path(__file__).resolve().parent.parent / "data" / "test_sets"
        else:
            data_dir = Path(data_dir)

        data_dir.mkdir(parents=True, exist_ok=True)
        path = data_dir / filename

        path.write_text(
            json.dumps(test_cases, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.info(f"Saved {len(test_cases)} test cases to {path}")


async def main():
    parser = argparse.ArgumentParser(description="GradusRAG 测试集自动生成器")
    parser.add_argument("--count", type=int, default=50, help="生成数量（默认 50）")
    parser.add_argument("--docs", type=str, help="文档目录路径（从文档生成）")
    parser.add_argument("--output", type=str, default="auto_test_set.json", help="输出文件名")
    parser.add_argument("--category", type=str, default="auto", help="测试类别标签")
    args = parser.parse_args()

    settings = load_settings()
    llm_service = LLMService.from_settings(settings)
    generator = TestSetGenerator(llm_service)

    if args.docs:
        # 从文档目录生成
        test_cases = await generator.generate_from_directory(
            args.docs, settings, count=args.count, category=args.category
        )
    else:
        # 从 Milvus 生成
        from src.retrieval.milvus_store import MilvusStore
        milvus = MilvusStore(
            host=settings.vector_store.host,
            port=settings.vector_store.port,
            collection=settings.vector_store.collection,
        )
        test_cases = await generator.generate_from_milvus(
            milvus, count=args.count, category=args.category
        )

    if test_cases:
        generator.save(test_cases, args.output)
        # 打印统计
        from collections import Counter
        levels = Counter(tc["expected_level"] for tc in test_cases)
        print(f"\n生成完成：{len(test_cases)} 条测试用例")
        print(f"级别分布：{dict(levels)}")
        print(f"保存至：data/test_sets/{args.output}")
    else:
        print("未生成任何测试用例，请检查文档是否已入库")


if __name__ == "__main__":
    asyncio.run(main())
