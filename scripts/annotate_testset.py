"""测试集人工审核工具。

将测试集生成可审核的 Markdown 文档：
- 每道题显示：题目、预期级别、参考答案
- 从 Milvus 查询 ground truth chunk 的实际内容
- 标注需要人工确认的检查项
- 标记无效 chunk（文档已删除/无法找到）

用法：
    python scripts/annotate_testset.py -i data/test_sets/golden_test_set.json
    python scripts/annotate_testset.py -i data/test_sets/formal_test_set.json
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def build_chunk_cache() -> Dict[str, str]:
    """从 Milvus 加载所有 chunk 内容，构建 chunk_id → text 的缓存。"""
    try:
        from src.core.settings import load_settings
        from src.libs.embedding_service import EmbeddingService
        from src.retrieval.milvus_store import MilvusStore

        settings = load_settings()
        milvus = MilvusStore(
            host=settings.vector_store.host,
            port=settings.vector_store.port,
            collection=settings.vector_store.collection,
        )

        cache = {}
        # 查询 Milvus 中所有 chunk
        try:
            results = milvus.query_with_text(limit=10000)
            for r in results:
                cid = r.get("chunk_id", "")
                text = r.get("text", "")
                if cid and text:
                    cache[cid] = text[:400]
            logger.info(f"Loaded {len(cache)} chunks from Milvus")
        except Exception as e:
            logger.warning(f"Could not query Milvus directly: {e}")
            # 降级：尝试从文件加载
            pass

        return cache
    except Exception as e:
        logger.warning(f"Could not connect to Milvus: {e}")
        return {}


def build_chunk_cache_from_docs(doc_dir: str = "data/documents") -> Dict[str, str]:
    """降级方案：从本地文档按段落切分构建 chunk 缓存。"""
    cache = {}
    doc_path = Path(doc_dir)

    for f in sorted(doc_path.glob("*.md")):
        if f.name.startswith("."):
            continue
        content = f.read_text(encoding="utf-8")
        # 按二级标题切分段落
        sections = content.split("\n## ")
        for i, section in enumerate(sections):
            text = section.strip()[:400]
            if text:
                # 生成可能的 chunk_id 格式
                cache[f"{f.name}::p{i}"] = text
                cache[f"{f.name}::p{i}::l3::0"] = text

    logger.info(f"Built fallback cache with {len(cache)} entries from local docs")
    return cache


def get_chunk_content(chunk_id: str, chunk_cache: Dict[str, str], tc: dict = None, idx: int = 0) -> tuple:
    """获取 chunk 内容，返回 (content, is_valid)。

    优先从测试用例自带的 ground_truth_texts 字段读取（v3 方案）。
    其次从 Milvus 缓存读取（v2 方案）。
    """
    # v3 方案：测试集里直接存了 chunk 文本
    if tc and "ground_truth_texts" in tc:
        texts = tc["ground_truth_texts"]
        if idx < len(texts) and texts[idx]:
            return texts[idx], True

    # v2 方案：从 Milvus 缓存查找
    if chunk_id in chunk_cache:
        return chunk_cache[chunk_id], True

    # 模糊匹配
    for key, text in chunk_cache.items():
        if chunk_id in key or key in chunk_id:
            return text, True

    return "", False


def generate_audit_markdown(test_cases: list, output_path: str, chunk_cache: Dict[str, str]):
    """生成可审核的 Markdown 文档。"""
    lines = []
    lines.append("# 测试集人工审核文档")
    lines.append("")
    lines.append("## 审核说明")
    lines.append("")
    lines.append("请逐题检查以下项目，用 ✅ 或 ❌ 标记：")
    lines.append("")
    lines.append("| 检查项 | 说明 |")
    lines.append("|--------|------|")
    lines.append("| 1. 题目清晰 | 问题表述无歧义，不会产生多种理解 |")
    lines.append("| 2. 级别正确 | L1=定义/列举, L2=比较/对比, L3=因果推理, L4=假设推理 |")
    lines.append("| 3. 有据可答 | 问题能在文档中找到明确答案，不是凭空编的 |")
    lines.append("| 4. 答案正确 | 参考答案与文档内容一致 |")
    lines.append("| 5. chunk 命中 | ground_truth_chunks 中确实包含回答该问题所需的段落 |")
    lines.append("")
    lines.append("**级别标注标准：**")
    lines.append("- **L1 显性事实**：答案直接出现在某一段落中，不需要跨段落或推理")
    lines.append("- **L2 隐性事实**：需要比较、对比文档中多处信息")
    lines.append("- **L3 可解释原理**：需要因果推理、解释机理")
    lines.append("- **L4 隐藏原理**：需要假设推理、预测、推断")
    lines.append("")
    lines.append("---")
    lines.append("")

    # 按级别分组
    by_level = {"L1": [], "L2": [], "L3": [], "L4": []}
    for i, tc in enumerate(test_cases):
        lv = tc.get("expected_level", "L1")
        if lv in by_level:
            by_level[lv].append((i, tc))

    invalid_count = 0

    for level in ["L1", "L2", "L3", "L4"]:
        items = by_level[level]
        lines.append(f"## {level} 题目（共 {len(items)} 题）")
        lines.append("")

        for idx, (orig_i, tc) in enumerate(items, 1):
            q = tc.get("question", "")
            answer = tc.get("expected_answer", "（无参考答案）")
            chunks = tc.get("ground_truth_chunks", [])
            domain = tc.get("_source_domain", tc.get("domain", "?"))
            category = tc.get("category", "")

            lines.append(f"### {level}-{idx}")
            lines.append("")
            lines.append(f"**领域：** {domain}  **类别：** {category}")
            lines.append("")
            safe_q = q.replace("#", "\\#").replace("|", "\\|").replace("\n", " ").strip()
            lines.append(f"**题目：** {safe_q}")
            lines.append("")
            safe_answer = answer[:300].replace("#", "\\#").replace("|", "\\|").replace("\n", " ").strip()
            lines.append(f"**参考答案：** {safe_answer}")
            lines.append("")
            lines.append(f"**Ground Truth Chunks ({len(chunks)} 个)：**")
            lines.append("")

            valid_chunks = 0
            for ci, chunk_id in enumerate(chunks):
                content, is_valid = get_chunk_content(chunk_id, chunk_cache, tc, ci)
                if is_valid:
                    valid_chunks += 1
                    lines.append(f"{ci+1}. `{chunk_id}`")
                    if content:
                        # 转义 markdown 特殊字符，避免破坏审核文档格式
                        safe_content = content[:300].replace("#", "\\#").replace("|", "\\|").replace("*", "\\*").replace("\n", " ").replace("---", "—").strip()
                        lines.append(f"   > {safe_content}")
                else:
                    invalid_count += 1
                    lines.append(f"{ci+1}. `{chunk_id}` ⚠️ **[无效]**")
                lines.append("")

            if valid_chunks == 0 and chunks:
                lines.append("> ⚠️ **警告：所有 ground truth chunks 均无效，需要重新标注**")
                lines.append("")

            lines.append("**审核：**")
            lines.append("")
            lines.append("| 检查项 | 结果 | 修改建议 |")
            lines.append("|--------|------|---------|")
            lines.append("| 1. 题目清晰 | ✅/❌ | |")
            lines.append("| 2. 级别正确 | ✅/❌ | 正确级别应为: |")
            lines.append("| 3. 有据可答 | ✅/❌ | |")
            lines.append("| 4. 答案正确 | ✅/❌ | |")
            lines.append("| 5. chunk 命中 | ✅/❌ | |")
            lines.append("")
            lines.append("**处理决定：** 保留 / 修改级别 / 修改题目 / 删除")
            lines.append("")
            lines.append("---")
            lines.append("")

    # 汇总表
    lines.append("## 汇总统计")
    lines.append("")
    lines.append("| 级别 | 题数 | 审核通过 | 修改级别 | 删除 |")
    lines.append("|------|------|---------|---------|------|")
    for level in ["L1", "L2", "L3", "L4"]:
        lines.append(f"| {level} | {len(by_level[level])} | | | |")
    lines.append(f"| **总计** | **{len(test_cases)}** | | | |")
    lines.append("")
    if invalid_count > 0:
        lines.append(f"> ⚠️ **共发现 {invalid_count} 个无效 chunk 引用，建议运行 `build_testset.py --chunk-only` 重新标注**")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text("\n".join(lines), encoding="utf-8")
    logger.info(f"Audit document saved to {output_path}")
    logger.info(f"Total: {len(test_cases)} questions, {invalid_count} invalid chunk references")


def main():
    parser = argparse.ArgumentParser(description="测试集人工审核")
    parser.add_argument("--input", "-i", default="data/test_sets/formal_test_set.json", help="输入测试集")
    parser.add_argument("--output", "-o", default="", help="审核文档输出路径（默认同目录 audit_review.md）")
    args = parser.parse_args()

    test_cases = json.loads(Path(args.input).read_text(encoding="utf-8"))
    logger.info(f"Loaded {len(test_cases)} test cases")

    # 先尝试从 Milvus 加载 chunk 内容，失败则用本地文档
    chunk_cache = build_chunk_cache()
    if not chunk_cache:
        logger.info("Milvus not available, using local documents as fallback")
        chunk_cache = build_chunk_cache_from_docs()

    output_path = args.output or str(Path(args.input).parent / "audit_review.md")
    generate_audit_markdown(test_cases, output_path, chunk_cache)


if __name__ == "__main__":
    main()
