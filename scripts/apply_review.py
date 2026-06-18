"""解析审核文档，将修改应用到测试集。

修复版：用编号精确匹配 + 模糊匹配双重校验，避免匹配错误。

用法：
    python scripts/apply_review.py data/test_sets/audit_review.md
"""

import json
import re
import sys
from pathlib import Path
from difflib import SequenceMatcher


def parse_audit_decisions(md_path: str) -> list:
    """从审核 Markdown 中提取每题的决定。"""
    content = Path(md_path).read_text(encoding="utf-8")
    decisions = []

    # 按 ### 标题分割，提取 label 和 body
    sections = re.split(r'^### (L\d+-\d+)', content, flags=re.MULTILINE)

    for i in range(1, len(sections), 2):
        if i + 1 >= len(sections):
            break
        label = sections[i].strip()  # 如 "L1-3"
        body = sections[i + 1]

        # 提取题目文本
        question_match = re.search(r'\*\*题目：\*\*\s*(.+)', body)
        question = question_match.group(1).strip() if question_match else ""
        # 清理转义字符
        question = question.replace("\\#", "#").replace("\\|", "|").replace("\\*", "*")

        # 提取处理决定
        decision_lines = re.findall(r'\*\*处理决定：\*\*\s*(.+)', body)
        decision = "保留"
        for line in decision_lines:
            line = line.strip()
            if "保留 / 修改级别" in line:  # 跳过模板
                continue
            if line and line != "保留":
                decision = line

        # 提取级别修正
        level_fix = None
        level_matches = re.findall(r'正确级别应为:\s*(L\d)', body)
        if level_matches:
            level_fix = level_matches[-1]

        # 解析动作
        if "删除" in decision:
            action = "delete"
        elif "修改级别" in decision:
            action = "relevel"
        elif "修改题目" in decision:
            action = "edit"
        else:
            action = "keep"

        decisions.append({
            "label": label,          # 如 "L1-3"
            "level": label.split("-")[0],  # 如 "L1"
            "index": int(label.split("-")[1]),  # 如 3
            "question": question,
            "action": action,
            "new_level": level_fix,
        })

    return decisions


def match_decisions_to_testcases(test_cases: list, decisions: list) -> list:
    """将审核决定匹配到测试用例。

    策略：先按编号精确匹配（L1-3 = 第3个L1题），再用文本相似度验证。
    """
    # 按级别分组，保持顺序
    by_level = {"L1": [], "L2": [], "L3": [], "L4": []}
    for i, tc in enumerate(test_cases):
        lv = tc.get("expected_level", "L1")
        if lv in by_level:
            by_level[lv].append((i, tc))

    # 构建编号 → 测试用例的映射
    matched = [None] * len(test_cases)
    match_report = {"exact": 0, "fuzzy": 0, "unmatched": 0, "errors": []}

    for d in decisions:
        level = d["level"]
        index = d["index"]
        label = d["label"]

        # 方式1：按编号精确匹配
        if level in by_level and 1 <= index <= len(by_level[level]):
            tc_idx, tc = by_level[level][index - 1]
            # 验证文本相似度（防止编号错位）
            if d["question"]:
                similarity = SequenceMatcher(
                    None, d["question"][:50], tc.get("question", "")[:50]
                ).ratio()
                if similarity < 0.5:
                    match_report["errors"].append(
                        f"  {label}: 编号匹配但文本不相似 ({similarity:.0%})，跳过"
                    )
                    match_report["unmatched"] += 1
                    continue
            matched[tc_idx] = d
            match_report["exact"] += 1
        else:
            # 方式2：降级到模糊匹配
            best_idx = -1
            best_score = 0
            for i, tc in enumerate(test_cases):
                if matched[i] is not None:
                    continue  # 已被匹配，跳过
                score = SequenceMatcher(
                    None, d["question"][:50], tc.get("question", "")[:50]
                ).ratio()
                if score > best_score:
                    best_score = score
                    best_idx = i

            if best_idx >= 0 and best_score > 0.6:
                matched[best_idx] = d
                match_report["fuzzy"] += 1
                match_report["errors"].append(
                    f"  {label}: 编号未命中，模糊匹配到第 {best_idx+1} 题 ({best_score:.0%})"
                )
            else:
                match_report["unmatched"] += 1
                match_report["errors"].append(
                    f"  {label}: 无法匹配任何题目"
                )

    return matched, match_report


def apply_decisions(test_cases: list, matched: list) -> tuple:
    """应用审核决定，返回 (新测试集, 统计)。"""
    stats = {"kept": 0, "deleted": 0, "releveled": 0, "edited": 0}
    new_cases = []

    for i, tc in enumerate(test_cases):
        d = matched[i]
        if d is None:
            # 未被审核的题目，默认保留
            new_cases.append(tc)
            stats["kept"] += 1
            continue

        if d["action"] == "delete":
            stats["deleted"] += 1
            continue
        elif d["action"] == "relevel" and d.get("new_level"):
            old_level = tc.get("expected_level", "?")
            tc["expected_level"] = d["new_level"]
            stats["releveled"] += 1
        elif d["action"] == "edit":
            stats["edited"] += 1
            new_cases.append(tc)
            continue
        else:
            stats["kept"] += 1

        new_cases.append(tc)

    return new_cases, stats


def main():
    audit_path = sys.argv[1] if len(sys.argv) > 1 else "data/test_sets/audit_review.md"
    test_path = sys.argv[2] if len(sys.argv) > 2 else "data/test_sets/formal_test_set.json"
    output_path = sys.argv[3] if len(sys.argv) > 3 else "data/test_sets/formal_test_set.json"

    # 加载
    print(f"读取审核文档: {audit_path}")
    decisions = parse_audit_decisions(audit_path)
    print(f"解析到 {len(decisions)} 条审核决定")

    test_cases = json.loads(Path(test_path).read_text(encoding="utf-8"))
    print(f"读取测试集: {len(test_cases)} 题")

    # 匹配
    matched, report = match_decisions_to_testcases(test_cases, decisions)

    print(f"\n匹配结果:")
    print(f"  编号精确匹配: {report['exact']}")
    print(f"  模糊匹配: {report['fuzzy']}")
    print(f"  未匹配: {report['unmatched']}")

    if report["errors"]:
        print(f"\n匹配详情:")
        for err in report["errors"]:
            print(err)

    # 应用修改
    new_cases, stats = apply_decisions(test_cases, matched)

    # 数量校验
    expected_count = len(test_cases) - stats["deleted"]
    if len(new_cases) != expected_count:
        print(f"\n⚠️ 数量异常！预期 {expected_count}，实际 {len(new_cases)}")
    else:
        print(f"\n✅ 数量校验通过: {len(test_cases)} → {len(new_cases)}")

    # 保存
    Path(output_path).write_text(
        json.dumps(new_cases, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # 汇总
    level_counts = {}
    domain_counts = {}
    for tc in new_cases:
        lv = tc.get("expected_level", "?")
        level_counts[lv] = level_counts.get(lv, 0) + 1
        dm = tc.get("_source_domain", "?")
        domain_counts[dm] = domain_counts.get(dm, 0) + 1

    print(f"\n{'='*50}")
    print(f"审核结果汇总")
    print(f"{'='*50}")
    print(f"原始题数: {len(test_cases)}")
    print(f"保留: {stats['kept']}")
    print(f"修改级别: {stats['releveled']}")
    print(f"删除: {stats['deleted']}")
    print(f"最终题数: {len(new_cases)}")
    print(f"\n级别分布:")
    for lv in ["L1", "L2", "L3", "L4"]:
        print(f"  {lv}: {level_counts.get(lv, 0)}")
    print(f"\n领域分布:")
    for dm, cnt in sorted(domain_counts.items()):
        print(f"  {dm}: {cnt}")
    print(f"{'='*50}")
    print(f"已保存到: {output_path}")


if __name__ == "__main__":
    main()
