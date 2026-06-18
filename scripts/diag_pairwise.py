"""快速诊断：对比 GradusRAG Pipeline 和直接 LLM 调用的回答。"""
import json, sys, asyncio
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

async def main():
    from src.core.settings import load_settings
    from src.libs.llm_service import LLMService
    from scripts.run_pairwise_eval import create_rag_pipeline

    settings = load_settings()
    llm = LLMService.from_settings(settings)
    pipeline = create_rag_pipeline()

    question = "机器学习的典型流程包含哪些主要步骤？"

    # 1. GradusRAG Pipeline
    print("=== GradusRAG Pipeline ===")
    response = await pipeline.run(question)
    print(f"Level: {response.query_level}")
    print(f"Answer ({len(response.answer)} chars): {response.answer[:500]}")
    print(f"Citations: {len(response.citations)}")

    # 2. 直接 LLM 调用（用同样的 Prompt）
    print("\n=== Direct LLM (same prompt) ===")
    from src.generation.response_generator import _load_prompt
    prompt_template = _load_prompt("generate_l1.txt")
    # 获取 pipeline 用的 context
    context = "\n".join([c.text_snippet for c in response.citations[:5]])
    prompt = prompt_template.format(context=context[:3000], question=question)
    direct_answer = await llm.ainvoke(prompt)
    print(f"Answer ({len(direct_answer)} chars): {direct_answer[:500]}")

    # 3. Baseline Prompt
    print("\n=== Baseline Prompt ===")
    baseline_prompt = f"""你是一个专业的领域问答助手。请根据提供的参考资料回答用户问题。

参考资料：
{context[:3000]}

用户问题：{question}

请提供详细、结构化的回答。包含相关事实、解释和例子。
如果参考资料不足以完整回答，说明你能确定的部分和不确定的部分。
引用时标注来源编号 [1][2]。"""
    baseline_answer = await llm.ainvoke(baseline_prompt)
    print(f"Answer ({len(baseline_answer)} chars): {baseline_answer[:500]}")

asyncio.run(main())
