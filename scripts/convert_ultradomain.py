"""将 UltraDomain JSONL 数据集转为 GradusRAG 可摄取的 Markdown 文件。

用法：
    python scripts/convert_ultradomain.py                         # 转换 LightRAG 用的 4 个领域
    python scripts/convert_ultradomain.py --domains agriculture cs legal mix
    python scripts/convert_ultradomain.py --all                   # 转换全部 20 个领域
"""

import argparse
import json
from pathlib import Path


# LightRAG 论文使用的 4 个领域
LIGHTRAG_DOMAINS = ["agriculture", "cs", "legal", "mix"]


def convert_jsonl_to_md(jsonl_path: Path, output_dir: Path, max_sections: int = 100):
    """将一个 JSONL 文件转为一个 Markdown 文件。"""
    domain_name = jsonl_path.stem  # 如 "agriculture"

    # 读取所有 context（限制数量，与 LightRAG 对齐）
    contexts = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if len(contexts) >= max_sections:
                break
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                ctx = obj.get("context", "")
                if ctx and len(ctx.strip()) > 50:  # 过滤太短的段落
                    contexts.append(ctx.strip())
            except json.JSONDecodeError:
                continue

    if not contexts:
        print(f"  {domain_name}: no valid contexts found")
        return 0

    # 拼接为 Markdown
    md_lines = [f"# {domain_name.title()} Domain Knowledge Base\n"]
    for i, ctx in enumerate(contexts, 1):
        md_lines.append(f"## Section {i}\n")
        md_lines.append(ctx)
        md_lines.append("\n")

    # 保存
    output_path = output_dir / f"{domain_name}.md"
    output_path.write_text("\n".join(md_lines), encoding="utf-8")

    total_chars = sum(len(c) for c in contexts)
    print(f"  {domain_name}: {len(contexts)} sections, {total_chars:,} chars → {output_path.name}")
    return len(contexts)


def main():
    parser = argparse.ArgumentParser(description="Convert UltraDomain JSONL to GradusRAG Markdown")
    parser.add_argument("--input", "-i", default="data/ultradomain", help="UltraDomain 数据集目录")
    parser.add_argument("--output", "-o", default="data/ultradomain_md", help="输出目录")
    parser.add_argument("--domains", "-d", nargs="+", help="指定领域列表")
    parser.add_argument("--all", action="store_true", help="转换全部领域")
    parser.add_argument("--max-sections", type=int, default=100, help="每个领域最多取多少个 section（默认 100，与 LightRAG 对齐）")
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 确定要转换的领域
    if args.all:
        domains = [f.stem for f in sorted(input_dir.glob("*.jsonl"))]
    elif args.domains:
        domains = args.domains
    else:
        domains = LIGHTRAG_DOMAINS

    print(f"Converting {len(domains)} domains from {input_dir} to {output_dir}\n")

    total_sections = 0
    for domain in domains:
        jsonl_path = input_dir / f"{domain}.jsonl"
        if not jsonl_path.exists():
            print(f"  {domain}: file not found, skipping")
            continue
        n = convert_jsonl_to_md(jsonl_path, output_dir, max_sections=args.max_sections)
        total_sections += n

    print(f"\nDone: {len(domains)} domains, {total_sections} total sections")
    print(f"Output: {output_dir}")
    print(f"\nNext step: python scripts/ingest.py --input {output_dir}")


if __name__ == "__main__":
    main()
