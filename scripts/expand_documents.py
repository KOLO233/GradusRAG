#!/usr/bin/env python3
"""
GradusRAG 知识库文档批量扩写脚本

功能：调用 LLM API 将知识库文档从当前长度自动扩写到 15000+ 字符
用法：python scripts/expand_documents.py [--dry-run] [--doc 01] [--resume]

依赖：已配置 config/settings.yaml 中的 LLM API

流程：
1. 读取 config/settings.yaml 获取 API 配置
2. 遍历 data/documents/ 下所有 .md 文件
3. 对每篇文档调用 LLM 进行内容扩写（分段调用，每段约 5000 字）
4. 保存扩写后的文档（原文件备份为 .bak）
5. 记录进度到 results/expand_progress.json，支持断点续跑
"""

import os
import sys
import json
import time
import yaml
import argparse
import logging
from pathlib import Path
from datetime import datetime

# ============================================================
# 路径配置
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "expand_settings.yaml"  # 默认用扩写专用配置
DOCS_DIR = PROJECT_ROOT / "data" / "documents"
PROGRESS_FILE = PROJECT_ROOT / "results" / "expand_progress.json"
BACKUP_DIR = PROJECT_ROOT / "data" / "documents_backup"
LOG_FILE = PROJECT_ROOT / "results" / "expand_log.txt"

TARGET_CHARS = 15000  # 目标字数
CHUNK_TARGET = 15000   # 每次 LLM 调用生成的字数（提高到 8000 减少调用次数）

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(str(LOG_FILE), encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)


# ============================================================
# LLM 调用
# ============================================================
class LLMCaller:
    """调用 OpenAI 兼容 API（支持 DashScope / DeepSeek / OpenAI 等）。"""

    def __init__(self, config: dict):
        llm_cfg = config.get("llm", {})
        self.provider = llm_cfg.get("provider", "openai")
        self.model = llm_cfg.get("model", "gpt-4o")
        self.api_key = llm_cfg.get("api_key", "")
        self.base_url = llm_cfg.get("base_url", "")
        self.temperature = llm_cfg.get("temperature", 0.7)
        self.max_tokens = llm_cfg.get("max_tokens", 4096)

        if not self.api_key or self.api_key == "YOUR_API_KEY_HERE":
            raise ValueError(
                "请在 config/settings.yaml 中配置有效的 api_key"
            )

        self._init_client()
        logger.info(f"LLM 初始化完成: {self.provider} / {self.model}")

    def _init_client(self):
        """初始化 OpenAI 兼容客户端。"""
        try:
            from openai import OpenAI
        except ImportError:
            logger.error("请安装 openai 库: pip install openai --break-system-packages")
            sys.exit(1)

        kwargs = {"api_key": self.api_key}
        if self.base_url:
            kwargs["base_url"] = self.base_url
        elif self.provider == "deepseek":
            kwargs["base_url"] = "https://api.deepseek.com/v1"
        elif "dashscope" in self.api_key.lower() or "dashscope" in str(self.base_url):
            # DashScope OpenAI 兼容模式
            kwargs["base_url"] = "https://dashscope.aliyuncs.com/compatible-mode/v1"

        self.client = OpenAI(**kwargs)

    def call(self, system_prompt: str, user_prompt: str, max_tokens: int = 0) -> str:
        """调用 LLM 获取回复。如果输出被截断（hit max_tokens），自动续写一次。"""
        mt = max_tokens or self.max_tokens
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=self.temperature,
                max_tokens=mt,
            )
            content = response.choices[0].message.content.strip()
            finish_reason = response.choices[0].finish_reason

            # 检测截断：如果 finish_reason 是 length，说明被 max_tokens 截断了
            if finish_reason == "length":
                logger.warning(f"输出被 max_tokens({mt}) 截断，自动续写...")
                try:
                    # 用续写 prompt 补全
                    cont_response = self.client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt},
                            {"role": "assistant", "content": content},
                            {"role": "user", "content": "请继续上面的内容，从断开处接着写，不要重复已有内容。"},
                        ],
                        temperature=self.temperature,
                        max_tokens=mt,
                    )
                    cont = cont_response.choices[0].message.content.strip()
                    # 去重：如果续写内容开头与已有内容末尾重叠，去掉重复部分
                    content = content + "\n\n" + cont
                    logger.info(f"续写完成，总长度 {len(content)} 字")
                except Exception as e:
                    logger.warning(f"续写失败，使用已有内容: {e}")

            return content
        except Exception as e:
            logger.error(f"LLM 调用失败: {e}")
            raise


# ============================================================
# 扩写 Prompt
# ============================================================
SYSTEM_PROMPT = """你是一位资深的学科专家和教材编写者。你的任务是将一篇简短的知识概要扩写为一篇内容丰富、结构完整、适合领域问答测试的专业文档。

扩写要求：
1. **内容深度**：每个知识点都要展开详细解释，包含原理、机制、案例、数据
2. **结构完整**：至少包含 5-8 个章节，每章节有 2-4 个子节
3. **语言风格**：类似专业教材或百科全书，准确、严谨、通俗易懂
4. **包含元素**：
   - 核心概念的定义和解释
   - 原理和机制的详细描述
   - 实际应用案例和具体数据
   - 历史发展和重要里程碑
   - 与相关领域的联系
   - 当前发展趋势和前沿动态
5. **格式**：使用 Markdown 格式，用 ## 标记章节，### 标记子节
6. **目标字数**：{target} 字左右
7. **语言**：中文
8. **不要**使用"本文将介绍"等过渡性语句，直接进入内容
"""


def make_expand_prompt(topic: str, existing_content: str, target_chars: int, part: int = 1, total_parts: int = 1) -> str:
    """生成扩写 prompt。"""
    if total_parts == 1:
        return f"""请将以下关于"{topic}"的知识概要扩写为一篇完整的专业文档（目标 {target_chars} 字）。

现有内容（仅供参考结构，需要大幅扩写）：
---
{existing_content}
---

请直接输出扩写后的完整文档（Markdown 格式）。确保内容丰富、有深度、包含具体案例和数据。"""
    else:
        # 分段扩写
        if part == 1:
            return f"""请将以下关于"{topic}"的知识概要扩写为一篇完整的专业文档。
本次请撰写文档的**前半部分**（约 {target_chars} 字），包含概述和前几个核心章节。

现有内容（参考结构）：
---
{existing_content}
---

请直接输出 Markdown 格式的前半部分内容。最后一节用 "## " 开头但不要写完，留作续写。"""
        else:
            return f"""这是关于"{topic}"的文档扩写（续）。请撰写文档的**后半部分**（约 {target_chars} 字），包含剩余章节和总结。

前半部分最后一节为："{existing_content[-200:]}"

请从上一节的续写开始，直接输出后半部分的 Markdown 内容。"""


# ============================================================
# 进度管理
# ============================================================
def load_progress() -> dict:
    """加载进度文件。"""
    if PROGRESS_FILE.exists():
        try:
            return json.loads(PROGRESS_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"completed": {}, "started_at": datetime.now().isoformat()}


def save_progress(progress: dict):
    """保存进度文件。"""
    PROGRESS_FILE.parent.mkdir(parents=True, exist_ok=True)
    progress["updated_at"] = datetime.now().isoformat()
    PROGRESS_FILE.write_text(
        json.dumps(progress, ensure_ascii=False, indent=2), encoding="utf-8"
    )


# ============================================================
# 主扩写逻辑
# ============================================================
def expand_document(
    llm: LLMCaller,
    doc_path: Path,
    target_chars: int = TARGET_CHARS,
    dry_run: bool = False,
) -> dict:
    """扩写单篇文档。

    Returns:
        {"file": str, "before": int, "after": int, "parts": int, "status": str}
    """
    fname = doc_path.name
    original = doc_path.read_text(encoding="utf-8", errors="replace")
    current_chars = len(original)

    if current_chars >= target_chars:
        logger.info(f"[{fname}] 已达标 ({current_chars}字)，跳过")
        return {"file": fname, "before": current_chars, "after": current_chars, "parts": 0, "status": "skip"}

    # 提取主题（从文件名或第一个标题）
    topic = fname.split("_", 1)[1].replace(".md", "") if "_" in fname else fname.replace(".md", "")
    lines = original.strip().split("\n")
    for line in lines:
        if line.startswith("# "):
            topic = line[2:].strip()
            break

    needed = target_chars - current_chars
    parts_needed = max(1, (needed + CHUNK_TARGET - 1) // CHUNK_TARGET)
    parts_needed = min(parts_needed, 4)  # 最多分 4 段

    logger.info(f"[{fname}] 当前 {current_chars}字, 目标 {target_chars}字, 需 {parts_needed} 段扩写")

    if dry_run:
        return {"file": fname, "before": current_chars, "after": current_chars, "parts": parts_needed, "status": "dry_run"}

    # 分段调用 LLM 扩写
    expanded_parts = []
    for part_idx in range(parts_needed):
        logger.info(f"[{fname}] 第 {part_idx+1}/{parts_needed} 段扩写中...")

        if part_idx == 0:
            context = original
        else:
            # 后续段落提供前一段的末尾作为上下文
            prev_text = "\n".join(expanded_parts)
            context = prev_text[-500:] if len(prev_text) > 500 else prev_text

        prompt = make_expand_prompt(topic, context, CHUNK_TARGET, part_idx + 1, parts_needed)

        for attempt in range(3):
            try:
                result = llm.call(SYSTEM_PROMPT.format(target=CHUNK_TARGET), prompt)
                expanded_parts.append(result)
                logger.info(f"[{fname}] 第 {part_idx+1} 段完成 ({len(result)}字)")
                break
            except Exception as e:
                logger.warning(f"[{fname}] 第 {part_idx+1} 段第 {attempt+1} 次尝试失败: {e}")
                if attempt < 2:
                    time.sleep(5 * (attempt + 1))
                else:
                    logger.error(f"[{fname}] 第 {part_idx+1} 段扩写失败，跳过该段")

        # 避免 API 限流
        if part_idx < parts_needed - 1:
            time.sleep(1)

    if not expanded_parts:
        return {"file": fname, "before": current_chars, "after": current_chars, "parts": 0, "status": "failed"}

    # 合并：保留原文结构，将扩写内容替换或追加
    if parts_needed == 1:
        # 单段扩写：直接替换
        expanded = expanded_parts[0]
    else:
        # 多段扩写：拼接
        expanded = "\n\n".join(expanded_parts)

    # 确保以换行结尾
    if not expanded.endswith("\n"):
        expanded += "\n"

    # 备份原文件
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    backup_path = BACKUP_DIR / fname
    if not backup_path.exists():
        backup_path.write_text(original, encoding="utf-8")

    # 保存扩写结果
    doc_path.write_text(expanded, encoding="utf-8")
    after_chars = len(expanded)

    logger.info(f"[{fname}] 扩写完成: {current_chars}字 → {after_chars}字")
    return {"file": fname, "before": current_chars, "after": after_chars, "parts": parts_needed, "status": "done"}


def main():
    parser = argparse.ArgumentParser(description="GradusRAG 知识库文档批量扩写")
    parser.add_argument("--dry-run", action="store_true", help="只分析不实际扩写")
    parser.add_argument("--doc", type=str, default="", help="只扩写指定文档（如 '01' 或 '01,02,03'）")
    parser.add_argument("--resume", action="store_true", help="从上次中断处继续")
    parser.add_argument("--target", type=int, default=TARGET_CHARS, help=f"目标字数（默认 {TARGET_CHARS}）")
    parser.add_argument("--domain", type=str, default="", help="只扩写指定领域（AI, 医学, 教育, 法律, 金融, 跨域）")
    parser.add_argument("--config", type=str, default="", help="自定义配置文件路径（默认 config/expand_settings.yaml）")
    args = parser.parse_args()

    # 加载配置
    config_path = Path(args.config) if args.config else CONFIG_PATH
    if not config_path.exists():
        logger.error(f"配置文件不存在: {config_path}")
        logger.error("请先复制 settings.yaml.example 为 settings.yaml 并填入 API 配置")
        sys.exit(1)

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # 初始化 LLM
    try:
        llm = LLMCaller(config)
    except Exception as e:
        logger.error(f"LLM 初始化失败: {e}")
        sys.exit(1)

    # 获取文档列表
    docs = sorted(DOCS_DIR.glob("*.md"))
    if not docs:
        logger.error(f"未找到文档: {DOCS_DIR}")
        sys.exit(1)

    # 领域过滤
    domain_map = {
        "AI": range(1, 11), "医学": range(11, 21), "教育": range(21, 31),
        "法律": range(31, 41), "金融": range(41, 51), "跨域": range(51, 61),
    }
    if args.domain:
        if args.domain in domain_map:
            nums = [f"{i:02d}" for i in domain_map[args.domain]]
            docs = [d for d in docs if d.name[:2] in nums]
            logger.info(f"仅扩写 {args.domain} 领域: {len(docs)} 篇")
        else:
            logger.error(f"未知领域: {args.domain}，可选: {', '.join(domain_map.keys())}")
            sys.exit(1)

    # 指定文档过滤
    if args.doc:
        doc_nums = [n.strip().zfill(2) for n in args.doc.split(",")]
        docs = [d for d in docs if d.name[:2] in doc_nums]
        logger.info(f"仅扩写指定文档: {[d.name for d in docs]}")

    # 加载进度
    progress = load_progress() if args.resume else {"completed": {}, "started_at": datetime.now().isoformat()}
    completed = progress.get("completed", {})

    # 执行扩写
    results = []
    total = len(docs)
    skipped = 0
    success = 0
    failed = 0

    logger.info(f"=" * 60)
    logger.info(f"开始批量扩写: {total} 篇文档, 目标 {args.target} 字/篇")
    logger.info(f"Dry Run: {args.dry_run}")
    logger.info(f"=" * 60)

    for i, doc_path in enumerate(docs):
        fname = doc_path.name

        # 断点续跑：跳过已完成的
        if args.resume and fname in completed and completed[fname].get("status") == "done":
            logger.info(f"[{i+1}/{total}] {fname} 已完成，跳过")
            skipped += 1
            continue

        logger.info(f"\n[{i+1}/{total}] 处理: {fname}")

        try:
            result = expand_document(llm, doc_path, args.target, args.dry_run)
            results.append(result)

            if result["status"] == "done":
                success += 1
                completed[fname] = result
                progress["completed"] = completed
                save_progress(progress)
            elif result["status"] == "skip":
                skipped += 1
            else:
                failed += 1

        except Exception as e:
            logger.error(f"[{fname}] 异常: {e}")
            failed += 1
            results.append({"file": fname, "status": "error", "error": str(e)})

            # 每篇之间间隔
            if i < total - 1 and not args.dry_run:
                time.sleep(1)

    # 汇总
    logger.info(f"\n{'=' * 60}")
    logger.info(f"扩写完成!")
    logger.info(f"  成功: {success}")
    logger.info(f"  跳过: {skipped}")
    logger.info(f"  失败: {failed}")
    logger.info(f"  进度文件: {PROGRESS_FILE}")
    logger.info(f"  原文备份: {BACKUP_DIR}")
    logger.info(f"{'=' * 60}")

    # 保存最终结果
    summary_path = PROJECT_ROOT / "results" / "expand_summary.json"
    summary = {
        "total": total,
        "success": success,
        "skipped": skipped,
        "failed": failed,
        "target_chars": args.target,
        "results": results,
        "timestamp": datetime.now().isoformat(),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(f"汇总报告: {summary_path}")


if __name__ == "__main__":
    main()
