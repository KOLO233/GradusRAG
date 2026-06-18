import json
data = json.loads(open("results/lightrag_comparison.json", encoding="utf-8").read())
for name in ["LightRAG_Baseline", "GradusRAG"]:
    if name in data and "summary" in data[name]:
        s = data[name]["summary"]
        print(f'{name}: Hit={s["hit_rate"]:.2%} MRR={s["mrr"]:.2%} Faith={s["faithfulness"]:.2%} Rel={s["answer_relevance"]:.2%}')
if "pairwise" in data:
    pw = data["pairwise"]
    total = pw.get("GradusRAG_wins", 0) + pw.get("LightRAG_wins", 0) + pw.get("Tie", 0)
    print(f'Pairwise: GradusRAG={pw.get("GradusRAG_wins",0)} LightRAG={pw.get("LightRAG_wins",0)} Tie={pw.get("Tie",0)} (total={total})')
else:
    print("Pairwise: not completed")
