"""从 Milvus 真实 chunk 生成测试集（反向生成法）。

核心思路：不是从文档生成题目再找 chunk，而是从 Milvus 里的真实 chunk 生成题目。
这样 ground truth chunk ID 一定是正确的，因为题目就是根据这个 chunk 内容出的。

流程：
1. 从 Milvus 加载所有 chunk（含 chunk_id, text, filename）
2. 按文件名分组，识别领域
3. 对每批 chunk 用 LLM 生成 L1-L4 题目
4. 题目的 ground_truth_chunks 直接设为来源 chunk
5. 质量过滤

用法：
    python scripts/build_testset_v3.py --target 120
    python scripts/build_testset_v3.py --target 80 --skip-quality
"""

import argparse
import json
import logging
import re
import sys
import time
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# 领域映射（按文件名前缀匹配）
DOMAIN_PREFIXES = {
    "AI/ML": ["01_", "02_", "03_", "04_", "05_", "06_", "07_", "08_", "09_", "10_"],
    "医学": ["11_", "12_", "13_", "14_", "15_", "16_", "17_", "18_", "19_", "20_"],
    "教育": ["21_", "22_", "23_", "24_", "25_", "26_", "27_", "28_", "29_", "30_"],
    "法律": ["31_", "32_", "33_", "34_", "35_", "36_", "37_", "38_", "39_", "40_"],
    "金融": ["41_", "42_", "43_", "44_", "45_", "46_", "47_", "48_", "49_", "50_"],
    "跨域": ["51_", "52_", "53_", "54_", "55_", "56_", "57_", "58_", "59_", "60_"],
}


def get_domain(filename: str) -> str:
    """根据文件名前缀判断领域。"""
    stem = Path(filename).stem
    for domain, prefixes in DOMAIN_PREFIXES.items():
        for prefix in prefixes:
            if stem.startswith(prefix):
                return domain
    return "其他"


def load_chunks_from_milvus() -> List[Dict]:
    """从 Milvus 加载所有 chunk。"""
    from src.core.settings import load_settings
    from src.retrieval.milvus_store import MilvusStore

    settings = load_settings()
    store = MilvusStore(
        host=settings.vector_store.host,
        port=settings.vector_store.port,
        collection=settings.vector_store.collection,
    )

    client = store._get_client()
    # 查询所有记录（用 chunk_id != "" 代替空 filter，兼容所有 Milvus 版本）
    results = client.query(
        collection_name=settings.vector_store.collection,
        filter='chunk_id != ""',
        output_fields=["chunk_id", "text", "filename"],
        limit=16384,
    )

    logger.info(f"Loaded {len(results)} chunks from Milvus")
    return results


def group_chunks_by_file(chunks: List[Dict]) -> Dict[str, List[Dict]]:
    """按文件名分组。"""
    by_file = {}
    for c in chunks:
        fname = c.get("filename", "unknown")
        if fname not in by_file:
            by_file[fname] = []
        by_file[fname].append(c)
    return by_file


# 从 chunk 生成题目的 prompt
QUESTION_FROM_CHUNK_PROMPT = """你是一个教育测试题设计专家。请根据以下文档片段，生成 {count} 个测试题。

文档片段内容：
---
{chunk_text}
---

文档来源：{filename}

要求：
- 只生成 L1 和 L2 级别的问题
- L1（显性事实，{l1_count} 个）：答案直接出现在上面的文档片段中。示例："什么是X？" "X有哪些分类？"
- L2（隐性事实，{l2_count} 个）：需要对比该片段中不同概念。示例："X和Y有什么区别？" "X分别适用于什么场景？"

关键要求：
- 问题必须能仅用上面的文档片段来回答
- 不要引用"文档中提到"之类的话
- L2 题必须是真正的比较/对比，不能是换个说法的 L1

请严格按 JSON 数组格式输出：
```json
[
  {{"question": "问题文本", "expected_level": "L1", "category": "定义", "expected_answer": "基于文档片段的参考答案(50-100字)"}},
  ...
]
```"""


# 从文档章节生成 L3/L4 题目的 prompt（需要更大上下文）
QUESTION_FROM_SECTION_PROMPT = """你是一个教育测试题设计专家。请根据以下文档章节内容，生成 {count} 个需要深层推理的测试题。

文档章节内容：
---
{section_text}
---

文档来源：{filename}

要求：
- 只生成 L3 和 L4 级别的问题
- L3（可解释原理，{l3_count} 个）：需要因果推理，解释"为什么"或"如何"。示例："为什么会出现X？" "X的工作原理是什么？"
  L3 题目的答案必须能从文档中找到因果链或机理描述。
- L4（隐藏原理，{l4_count} 个）：需要假设推理，基于文档知识进行推断。示例："如果改变X会怎样？" "假设X不存在，对Y有什么影响？"
  L4 题目需要基于文档中的原理进行假设推理，不能凭空编造。

关键要求：
- L3 题的答案必须在文档中有明确的因果描述或原理说明作为依据
- L4 题必须基于文档中的具体知识进行假设推理，不能问文档完全没涉及的内容
- 问题表述清晰自然，像真实用户会问的问题

请严格按 JSON 数组格式输出：
```json
[
  {{"question": "问题文本", "expected_level": "L3", "category": "因果推理", "expected_answer": "基于文档内容的推理答案(100-200字)"}},
  ...
]
```"""


def parse_json_response(response: str) -> list:
    """从 LLM 响应中提取 JSON 数组。"""
    # 去掉 Qwen3 思考模式的 <think>...</think> 标签内容
    cleaned = re.sub(r'<think>[\s\S]*?</think>', '', response)
    cleaned = re.sub(r'```(?:json)?\s*', '', cleaned)
    cleaned = cleaned.strip()

    json_match = re.search(r'\[[\s\S]*\]', cleaned)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass
    return []


async def generate_questions_from_chunks(
    llm,
    chunks: List[Dict],
    count: int,
    filename: str,
) -> List[Dict]:
    """从一批 chunk 生成题目。"""
    per_level = max(count // 4, 1)
    questions = []

    # 将 chunk 分组（每 3-5 个 chunk 一批生成题目，增加上下文丰富度）
    batch_size = 3
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        combined_text = "\n\n".join([c.get("text", "")[:1500] for c in batch])
        chunk_ids = [c.get("chunk_id", "") for c in batch]

        if len(combined_text.strip()) < 100:
            continue

        questions_this_batch = max(count * batch_size // len(chunks), 2)
        per_level_batch = max(questions_this_batch // 4, 1)

        prompt = QUESTION_FROM_CHUNK_PROMPT.format(
            count=questions_this_batch,
            per_level=per_level_batch,
            chunk_text=combined_text[:3000],
            filename=filename,
            l1_count=max(questions_this_batch // 2, 1),
            l2_count=max(questions_this_batch // 2, 1),
        )

        try:
            response = await llm.ainvoke(prompt)
            batch_questions = parse_json_response(response)

            for q in batch_questions:
                q["ground_truth_chunks"] = chunk_ids
                # 直接存储 chunk 文本，审核和评估时不再依赖 Milvus 查询
                q["ground_truth_texts"] = [c.get("text", "")[:800] for c in batch]
                q["_source_file"] = filename
                q["_source_domain"] = get_domain(filename)

            questions.extend(batch_questions)
        except Exception as e:
            logger.error(f"  Generation failed for batch {i}: {e}")

        time.sleep(0.3)

        if len(questions) >= count * 1.5:
            break

    return questions


def balance_levels(questions: List[Dict], target_per_level: int) -> List[Dict]:
    """按级别平衡数量。"""
    from difflib import SequenceMatcher

    by_level = {"L1": [], "L2": [], "L3": [], "L4": []}
    for q in questions:
        lv = q.get("expected_level", "L1")
        if lv in by_level:
            by_level[lv].append(q)

    balanced = []
    for lv in ["L1", "L2", "L3", "L4"]:
        pool = by_level[lv]
        # 去重
        unique = []
        for q in pool:
            is_dup = False
            for u in unique:
                if SequenceMatcher(None, q["question"], u["question"]).ratio() > 0.6:
                    is_dup = True
                    break
            if not is_dup:
                unique.append(q)
        balanced.extend(unique[:target_per_level])

    return balanced


def balance_domains(questions: List[Dict], target: int = 300) -> List[Dict]:
    """按领域平衡数量，确保每个领域都有代表性。"""
    from difflib import SequenceMatcher

    all_domains = ["AI/ML", "医学", "教育", "法律", "金融", "跨域"]
    by_domain = {d: [] for d in all_domains}
    for q in questions:
        dm = q.get("_source_domain", "AI/ML")
        if dm in by_domain:
            by_domain[dm].append(q)

    # 计算每个领域的目标数量（按比例分配，但保证每个领域至少 20 题）
    total_available = sum(len(v) for v in by_domain.values())
    min_per_domain = 20
    balanced = []

    for dm in all_domains:
        pool = by_domain[dm]
        if not pool:
            logger.warning(f"  领域 [{dm}] 无题目，跳过")
            continue

        # 按比例分配，但至少 min_per_domain
        proportion = len(pool) / total_available if total_available > 0 else 1 / len(all_domains)
        domain_target = max(min_per_domain, int(target * proportion))

        # 按级别均匀采样
        by_level = {"L1": [], "L2": [], "L3": [], "L4": []}
        for q in pool:
            lv = q.get("expected_level", "L1")
            if lv in by_level:
                by_level[lv].append(q)

        per_level = max(domain_target // 4, 1)
        domain_questions = []
        for lv in ["L1", "L2", "L3", "L4"]:
            # 去重
            unique = []
            for q in by_level[lv]:
                is_dup = False
                for u in unique:
                    if SequenceMatcher(None, q["question"], u["question"]).ratio() > 0.6:
                        is_dup = True
                        break
                if not is_dup:
                    unique.append(q)
            domain_questions.extend(unique[:per_level])

        balanced.extend(domain_questions)
        logger.info(f"  领域 [{dm}]: {len(domain_questions)} 题（目标 {domain_target}）")

    return balanced


def balance_domains_and_levels(questions: List[Dict], target: int = 600) -> List[Dict]:
    """先按领域均分，再在每个领域内按级别平衡。保证 6 个领域 × 4 个级别都有题目。"""
    from difflib import SequenceMatcher

    all_domains = ["AI/ML", "医学", "教育", "法律", "金融", "跨域"]

    # Step 1: 按领域分组
    by_domain = {d: [] for d in all_domains}
    for q in questions:
        dm = q.get("_source_domain", "AI/ML")
        if dm in by_domain:
            by_domain[dm].append(q)

    # Step 2: 每个领域内按级别分组 + 去重
    domain_level_pools = {}
    for dm in all_domains:
        domain_level_pools[dm] = {"L1": [], "L2": [], "L3": [], "L4": []}
        for q in by_domain[dm]:
            lv = q.get("expected_level", "L1")
            if lv in domain_level_pools[dm]:
                domain_level_pools[dm][lv].append(q)

    # 去重
    for dm in all_domains:
        for lv in ["L1", "L2", "L3", "L4"]:
            pool = domain_level_pools[dm][lv]
            unique = []
            for q in pool:
                is_dup = False
                for u in unique:
                    if SequenceMatcher(None, q["question"], u["question"]).ratio() > 0.6:
                        is_dup = True
                        break
                if not is_dup:
                    unique.append(q)
            domain_level_pools[dm][lv] = unique

    # Step 3: 统计每个领域可用的题目数
    domain_available = {dm: sum(len(v) for v in pools.values()) for dm, pools in domain_level_pools.items()}
    total_available = sum(domain_available.values())

    # Step 4: 按比例分配每个领域的目标数，保底 30 题
    min_per_domain = 30
    domain_targets = {}
    for dm in all_domains:
        if domain_available[dm] == 0:
            domain_targets[dm] = 0
            continue
        proportion = domain_available[dm] / total_available if total_available > 0 else 1 / len(all_domains)
        domain_targets[dm] = max(min_per_domain, int(target * proportion))

    # 归一化使总数不超过 target
    total_targets = sum(domain_targets.values())
    if total_targets > target:
        scale = target / total_targets
        domain_targets = {dm: max(min_per_domain, int(t * scale)) for dm, t in domain_targets.items()}

    # Step 5: 每个领域内按级别均分采样
    balanced = []
    for dm in all_domains:
        dm_target = domain_targets[dm]
        if dm_target == 0 or domain_available[dm] == 0:
            logger.warning(f"  领域 [{dm}]: 无可用题目，跳过")
            continue

        per_level = max(dm_target // 4, 1)
        domain_questions = []
        for lv in ["L1", "L2", "L3", "L4"]:
            pool = domain_level_pools[dm][lv]
            domain_questions.extend(pool[:per_level])

        balanced.extend(domain_questions)
        logger.info(f"  领域 [{dm}]: {len(domain_questions)} 题（可用 {domain_available[dm]}，目标 {dm_target}）")

    return balanced


async def quality_check(llm, questions: List[Dict]) -> List[Dict]:
    """快速质量检查：只检查级别标注是否正确。"""
    passed = []
    for i, q in enumerate(questions):
        prompt = f"""判断以下问题的复杂度级别是否正确。

问题：{q['question']}
标注级别：{q.get('expected_level', 'L1')}

级别标准：
L1 = 直接从文档中找答案（什么是、定义、列举）
L2 = 需要比较/对比信息（区别、比较、分别）
L3 = 需要因果推理（为什么、原因、机制）
L4 = 需要假设推理（如果、假设、会怎样）

只输出 JSON：{{"correct": true/false}}"""

        try:
            response = await llm.ainvoke(prompt)
            result = parse_json_response(response)
            if result and isinstance(result, list) and len(result) > 0:
                if result[0].get("correct", True):
                    passed.append(q)
            elif result and isinstance(result, dict):
                if result.get("correct", True):
                    passed.append(q)
            else:
                passed.append(q)  # 解析失败时保留
        except Exception:
            passed.append(q)

        if (i + 1) % 20 == 0:
            logger.info(f"  Quality check: {i+1}/{len(questions)} ({len(passed)} passed)")

    logger.info(f"Quality check: {len(passed)}/{len(questions)} passed")
    return passed


def _save_progress(progress_path: Path, questions: list, completed_files: set):
    """保存生成进度到文件（支持断点续传）。"""
    data = {
        "questions": questions,
        "completed_files": list(completed_files),
        "total": len(questions),
    }
    progress_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info(f"Progress saved: {len(questions)} questions, {len(completed_files)} files done")


async def main():
    parser = argparse.ArgumentParser(description="从 Milvus chunk 反向生成测试集")
    parser.add_argument("--target", "-n", type=int, default=400, help="目标用例数（默认 400）")
    parser.add_argument("--output", "-o", default="data/test_sets/formal_test_set.json", help="输出路径")
    parser.add_argument("--skip-quality", action="store_true", help="跳过质量检查")
    parser.add_argument("--resume", action="store_true", help="从上次中断处继续")
    args = parser.parse_args()

    from src.core.settings import load_settings
    from src.libs.llm_service import LLMService

    settings = load_settings()
    llm = LLMService.from_settings(settings)

    # 断点续传：检查进度文件
    progress_path = Path(args.output).with_suffix(".progress.json")
    completed_files = set()
    all_questions = []

    if args.resume and progress_path.exists():
        try:
            progress_data = json.loads(progress_path.read_text(encoding="utf-8"))
            all_questions = progress_data.get("questions", [])
            completed_files = set(progress_data.get("completed_files", []))
            logger.info(f"Resumed: {len(all_questions)} questions from {len(completed_files)} files already done")
        except Exception as e:
            logger.warning(f"Failed to load progress: {e}, starting fresh")

    # Step 1: 从 Milvus 加载所有 chunk
    logger.info("Loading chunks from Milvus...")
    all_chunks = load_chunks_from_milvus()
    if not all_chunks:
        logger.error("No chunks found in Milvus. Run ingest first.")
        return

    # Step 2: 按文件分组
    by_file = group_chunks_by_file(all_chunks)
    for fname, chunks in by_file.items():
        domain = get_domain(fname)
        logger.info(f"  {fname} ({domain}): {len(chunks)} chunks")

    target_l12 = args.target // 2  # L1/L2 占一半
    target_l34 = args.target - target_l12  # L3/L4 占一半

    # Step 3: L1/L2 题目（从 chunk 生成）
    logger.info(f"\n--- Generating L1/L2 questions (target: {target_l12}) ---")
    per_file_l12 = max(target_l12 // max(len(by_file), 1), 4)
    for fname, chunks in by_file.items():
        if fname in completed_files:
            logger.info(f"Skipping {fname} (already done)")
            continue
        logger.info(f"Generating L1/L2 from {fname} ({len(chunks)} chunks)...")
        questions = await generate_questions_from_chunks(llm, chunks, per_file_l12, fname)
        all_questions.extend(questions)
        completed_files.add(fname)
        logger.info(f"  Generated {len(questions)} L1/L2 questions")

        # 保存进度
        _save_progress(progress_path, all_questions, completed_files)

    # Step 4: L3/L4 题目（从文档章节生成，上下文更大）
    logger.info(f"\n--- Generating L3/L4 questions (target: {target_l34}) ---")

    # 初始化检索系统（用于 L3/L4 的 ground truth 标注）
    from src.libs.embedding_service import EmbeddingService
    from src.retrieval.milvus_store import MilvusStore
    from src.retrieval.dense_retriever import DenseRetriever

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

    per_file_l34 = max(target_l34 // max(len(by_file), 1), 3)
    for fname, file_chunks in by_file.items():
        # L3/L4 用 "fname:L34" 标记，避免和 L1/L2 的 completed_files 冲突
        l34_key = f"{fname}:L34"
        if l34_key in completed_files:
            logger.info(f"Skipping L3/L4 for {fname} (already done)")
            continue

        l3_count = max(per_file_l34 // 2, 1)
        l4_count = max(per_file_l34 // 2, 1)

        # L3/L4 用更大的 batch（5 个 chunk 一批），提供更丰富的上下文
        l34_batch_size = 5
        domain = get_domain(fname)
        logger.info(f"Generating L3/L4 from {fname} [{domain}] ({len(file_chunks)} chunks)...")
        for i in range(0, len(file_chunks), l34_batch_size):
            batch = file_chunks[i:i + l34_batch_size]
            combined_text = "\n\n".join([c.get("text", "")[:1200] for c in batch])

            if len(combined_text.strip()) < 200:
                continue

            questions_this_batch = max((l3_count + l4_count) * l34_batch_size // max(len(file_chunks), 1), 2)

            prompt = QUESTION_FROM_SECTION_PROMPT.format(
                count=questions_this_batch,
                l3_count=max(questions_this_batch // 2, 1),
                         l4_count=max(questions_this_batch // 2, 1),
                section_text=combined_text[:4000],
                filename=fname,
            )
            try:
                response = await llm.ainvoke(prompt)
                batch_qs = parse_json_response(response)

                for q in batch_qs:
                    question_text = q.get("question", "")
                    try:
                        retrieval_results = await dense_retriever.retrieve(
                            query=question_text, top_k=5
                        )
                        gt_chunk_ids = [r.chunk_id for r in retrieval_results[:5]]
                        gt_texts = [r.text[:800] for r in retrieval_results[:5]]
                    except Exception:
                        gt_chunk_ids = [c.get("chunk_id", "") for c in batch[:3]]
                        gt_texts = [c.get("text", "")[:800] for c in batch[:3]]

                    q["ground_truth_chunks"] = gt_chunk_ids
                    q["ground_truth_texts"] = gt_texts
                    q["_source_file"] = fname
                    q["_source_domain"] = get_domain(fname)

                all_questions.extend(batch_qs)
                logger.info(f"    Generated {len(batch_qs)} L3/L4 questions")
            except Exception as e:
                logger.error(f"  L3/L4 generation failed for {fname}: {e}")

            time.sleep(0.3)

        _save_progress(progress_path, all_questions, completed_files)

    logger.info(f"\nTotal generated: {len(all_questions)}")

    # Step 5: 先按领域平衡，再在每个领域内按级别平衡
    all_questions = balance_domains_and_levels(all_questions, target=args.target)
    logger.info(f"After balancing: {len(all_questions)}")

    # Step 6: 质量检查
    if not args.skip_quality:
        logger.info("Running quality check...")
        checked = await quality_check(llm, all_questions)
        if len(checked) >= len(all_questions) * 0.6:
            all_questions = checked
        else:
            logger.warning(f"Quality check rejected too many ({len(checked)}/{len(all_questions)}), keeping all")

    # Step 7: 保存
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(all_questions, ensure_ascii=False, indent=2), encoding="utf-8")

    # 统计
    level_counts = {}
    domain_counts = {}
    for q in all_questions:
        lv = q.get("expected_level", "?")
        level_counts[lv] = level_counts.get(lv, 0) + 1
        dm = q.get("_source_domain", "?")
        domain_counts[dm] = domain_counts.get(dm, 0) + 1

    logger.info(f"\n{'='*50}")
    logger.info(f"Test set saved to {output_path}")
    logger.info(f"Total: {len(all_questions)} cases")
    for lv in ["L1", "L2", "L3", "L4"]:
        logger.info(f"  {lv}: {level_counts.get(lv, 0)} cases")
    for dm, cnt in sorted(domain_counts.items()):
        logger.info(f"  Domain [{dm}]: {cnt} cases")
    logger.info(f"{'='*50}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
