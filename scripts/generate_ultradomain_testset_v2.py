"""UltraDomain 测试集生成（简化版，更稳健）。

每个领域独立生成 125 题（不依赖复杂的格式解析）。
直接让 LLM 输出 JSON 数组，比解析嵌套列表可靠得多。

用法：
    python scripts/generate_ultradomain_testset_v2.py
    python scripts/generate_ultradomain_testset_v2.py --resume
"""

import argparse
import json
import logging
import re
import sys
import time
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DOMAINS = ["agriculture", "cs", "legal", "mix"]


def load_domain_chunks(domain: str) -> List[dict]:
    """从 Milvus 加载一个领域的所有 chunks。"""
    from src.core.settings import load_settings
    from src.retrieval.milvus_store import MilvusStore

    settings = load_settings()
    store = MilvusStore(
        host=settings.vector_store.host,
        port=settings.vector_store.port,
        collection=settings.vector_store.collection,
    )
    client = store._get_client()

    results = client.query(
        collection_name=settings.vector_store.collection,
        filter=f'filename == "{domain}.md"',
        output_fields=["chunk_id", "text", "filename"],
        limit=16384,
    )
    logger.info(f"  Loaded {len(results)} chunks for {domain}")
    return results


def build_description(chunks: list, max_chars: int = 6000) -> str:
    """从 chunks 构建数据集描述（取样代表性内容）。"""
    if not chunks:
        return ""

    # 均匀采样 50 个 chunk（避免太多导致截断太严重）
    step = max(1, len(chunks) // 50)
    sampled = chunks[::step][:50]

    parts = []
    for c in sampled:
        text = c.get("text", "")
        if len(text) > 200:
            # 取前 150 字作为摘要
            parts.append(text[:150].replace("\n", " "))

    description = "\n".join(parts)
    return description[:max_chars]


PROMPT = """You are given a collection of text passages from a knowledge base about {domain}.

Here are representative passages from this knowledge base:

{description}

Your task: Generate exactly 25 diverse questions that require deep understanding across multiple passages in this knowledge base. These questions should NOT be answerable from a single passage alone.

Requirements:
- Questions should span different complexity levels:
  - 6 factual questions (What is...? Define...? List...)
  - 7 comparative questions (Compare... How does X differ from Y?)
  - 6 causal/mechanism questions (Why does...? How does... work? Explain the mechanism of...)
  - 6 hypothetical questions (What would happen if...? Suppose... How would... change?)
- Questions must be specific to the content shown above
- Questions should require synthesizing information from multiple passages
- Output ONLY a JSON array of question strings, nothing else

Output format (strictly follow):
["question 1", "question 2", ..., "question 25"]"""


def repair_json(text: str) -> str:
    """修复常见的 JSON 格式问题。"""
    text = re.sub(r',\s*]', ']', text)
    text = re.sub(r',\s*}', '}', text)
    return text


def extract_questions_from_text(text: str) -> List[str]:
    """从文本中提取问题字符串，不依赖 JSON 解析。

    处理 Qwen 输出的各种格式问题（未转义引号、LaTeX 公式等）。
    """
    questions = []

    # 策略1：尝试标准 JSON 解析
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return [q for q in parsed if isinstance(q, str) and len(q) > 10]
    except (json.JSONDecodeError, ValueError):
        pass

    # 策略2：尝试修复后 JSON 解析
    try:
        repaired = repair_json(text)
        parsed = json.loads(repaired)
        if isinstance(parsed, list):
            return [q for q in parsed if isinstance(q, str) and len(q) > 10]
    except (json.JSONDecodeError, ValueError):
        pass

    # 策略3：正则逐行提取引号内的字符串
    # 匹配 "..." 模式（支持多行）
    pattern = r'"([^"]{15,})"'
    matches = re.findall(pattern, text, re.DOTALL)
    for m in matches:
        q = m.strip().replace('\n', ' ')
        if len(q) > 15 and not q.startswith('{') and not q.startswith('['):
            questions.append(q)

    # 策略4：按编号提取（1. xxx 2. xxx）
    if len(questions) < 5:
        for line in text.split('\n'):
            match = re.match(r'^\s*\d+[\.\)]\s+"?(.+?)"?\s*$', line)
            if match:
                q = match.group(1).strip().strip('"').strip(',')
                if len(q) > 15:
                    questions.append(q)

    return questions


async def generate_for_domain(llm, domain: str, chunks: list) -> List[dict]:
    """为一个领域生成 25 题。"""
    description = build_description(chunks)
    logger.info(f"  Description: {len(description)} chars")

    prompt = PROMPT.format(domain=domain, description=description)

    for attempt in range(3):  # 最多重试 3 次
        try:
            response = await llm.ainvoke(prompt)
        except Exception as e:
            logger.error(f"  LLM call failed (attempt {attempt+1}): {e}")
            time.sleep(2)
            continue

        # 去掉 <think> 标签
        cleaned = re.sub(r'<think>[\s\S]*?</think>', '', response)
        # 去掉 markdown 代码块
        cleaned = re.sub(r'```(?:json)?\s*', '', cleaned).strip()

        # 提取问题（多策略，不依赖 JSON 解析）
        raw_questions = extract_questions_from_text(cleaned)

        if not raw_questions:
            logger.warning(f"  No questions extracted (attempt {attempt+1})")
            debug_path = Path(f"data/test_sets/debug_{domain}_attempt{attempt}.txt")
            debug_path.parent.mkdir(parents=True, exist_ok=True)
            debug_path.write_text(response[:3000], encoding="utf-8")
            time.sleep(2)
            continue

        # 构建题目列表
        questions = []
        for q in raw_questions:
            if isinstance(q, str) and len(q.strip()) > 10:
                questions.append({
                    "question": q.strip(),
                    "domain": domain,
                    "source": "ultradomain",
                })

        logger.info(f"  Parsed {len(questions)} questions")
        return questions

    logger.error(f"  All 3 attempts failed for {domain}")
    return []


async def main():
    parser = argparse.ArgumentParser(description="UltraDomain 测试集生成 v2")
    parser.add_argument("--output", "-o", default="data/test_sets/ultradomain_testset.json")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    from src.core.settings import load_settings
    from src.libs.llm_service import LLMService

    settings = load_settings()
    llm = LLMService.from_settings(settings)

    # 断点续传
    progress_path = Path(args.output).with_suffix(".progress.json")
    all_questions = []
    completed = set()

    if args.resume and progress_path.exists():
        try:
            data = json.loads(progress_path.read_text(encoding="utf-8"))
            all_questions = data.get("questions", [])
            completed = set(data.get("completed_domains", []))
            logger.info(f"Resumed: {len(all_questions)} questions from {completed}")
        except Exception:
            pass

    # 逐领域生成
    for domain in DOMAINS:
        if domain in completed:
            logger.info(f"Skipping {domain} (done)")
            continue

        chunks = load_domain_chunks(domain)
        if not chunks:
            logger.warning(f"No chunks for {domain}")
            continue

        # 生成 5 轮 × 25 题 = 125 题
        domain_questions = []
        for round_num in range(5):
            logger.info(f"[{domain}] Round {round_num+1}/5...")
            qs = await generate_for_domain(llm, domain, chunks)
            domain_questions.extend(qs)
            time.sleep(1)

        # 去重
        seen = set()
        unique = []
        for q in domain_questions:
            if q["question"] not in seen:
                seen.add(q["question"])
                unique.append(q)

        logger.info(f"[{domain}] Total: {len(unique)} unique questions")
        all_questions.extend(unique)
        completed.add(domain)

        # 保存进度
        progress_data = {"questions": all_questions, "completed_domains": list(completed)}
        progress_path.write_text(json.dumps(progress_data, ensure_ascii=False, indent=2), encoding="utf-8")

    # 保存最终结果
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(all_questions, ensure_ascii=False, indent=2), encoding="utf-8")

    # 统计
    domain_counts = {}
    for q in all_questions:
        d = q.get("domain", "?")
        domain_counts[d] = domain_counts.get(d, 0) + 1

    logger.info(f"\n{'='*50}")
    logger.info(f"Total: {len(all_questions)} questions")
    for d in DOMAINS:
        logger.info(f"  {d}: {domain_counts.get(d, 0)}")
    logger.info(f"{'='*50}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
