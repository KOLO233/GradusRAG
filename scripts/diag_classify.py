"""快速诊断：UltraDomain 查询的 L1-L4 分类分布。"""
import json, sys, asyncio
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.settings import load_settings
from src.query_classifier.classifier import create_classifier

async def main():
    settings = load_settings()
    classifier = create_classifier(settings.query_classifier)

    questions = json.loads(Path("data/test_sets/ultradomain_testset.json").read_text(encoding="utf-8"))

    level_counts = {"L1": 0, "L2": 0, "L3": 0, "L4": 0}
    for i, q in enumerate(questions[:50]):
        result = await classifier.classify(q["question"])
        level_counts[result.level] += 1
        if i < 10:
            print(f"  [{result.level}] {result.confidence:.0%} | {q['question'][:70]}")

    print(f"\nDistribution (first 50): {level_counts}")

asyncio.run(main())
