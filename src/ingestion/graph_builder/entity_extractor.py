"""实体和关系抽取器。

使用 LLM 从文本中抽取实体和关系，构建知识图谱。
Prompt 设计参考 LightRAG 的 entity_extraction_system_prompt。
"""

from __future__ import annotations

import json
import logging
import re
from typing import Dict, List, Tuple

from src.core.types import Entity, Relation

logger = logging.getLogger(__name__)

EXTRACTION_PROMPT = """You are a knowledge graph expert. Extract entities and relations from the text.

---Instructions---

1. **Entity Extraction Rules**:
   - Identify clear, meaningful entities (concepts, technologies, methods, diseases, people, organizations, etc.)
   - Assign each entity a type: {entity_types}
   - Entity names must be specific (avoid pronouns like "it", "this method")
   - Use the same name for the same entity throughout
   - Entity descriptions must be based on the text, concise (10-30 words)

2. **Relation Extraction Rules**:
   - Extract direct relations between identified entities only
   - Use concise verb phrases for relation types (e.g., belongs_to, causes, contains, uses, treats, compares, depends_on, applies_to)
   - Relation descriptions must be based on the text

3. **Quantity Requirements**:
   - Extract at least 3 entities and 2 relations per text
   - Prioritize domain-specific concepts over generic terms

4. **Output Format**:
   - Output ONLY a valid JSON object, nothing else
   - Do not use markdown code blocks
   - Do not output explanations or reasoning

---Example---

Input text: "Convolutional neural networks (CNNs) extract local features from images through convolutional layers. Pooling layers reduce the dimensionality of feature maps. ResNet introduced residual connections to solve the vanishing gradient problem in deep networks."

Output:
{{"entities": [{{"name": "Convolutional Neural Network", "type": "technology", "description": "Deep learning architecture that extracts local features from images using convolutional layers"}}, {{"name": "Convolutional Layer", "type": "concept", "description": "Neural network layer that extracts local features through convolution operations"}}, {{"name": "Pooling Layer", "type": "concept", "description": "Layer that reduces the dimensionality of feature maps"}}, {{"name": "ResNet", "type": "technology", "description": "Deep CNN architecture that introduced residual connections"}}, {{"name": "Residual Connection", "type": "concept", "description": "Skip connection technique that solves vanishing gradient in deep networks"}}], "relations": [{{"source": "Convolutional Neural Network", "target": "Convolutional Layer", "type": "contains", "description": "CNNs are composed of convolutional layers among other components"}}, {{"source": "ResNet", "target": "Residual Connection", "type": "uses", "description": "ResNet introduced residual connections to solve deep network degradation"}}, {{"source": "Residual Connection", "target": "Convolutional Neural Network", "type": "solves", "description": "Residual connections solve the vanishing gradient problem in deep CNNs"}}]}}

---Text---
{text}

---Output---
"""


class EntityExtractor:
    """LLM 实体/关系抽取器。

    Example:
        >>> extractor = EntityExtractor(llm_service, entity_types=["疾病", "药物", "症状"])
        >>> entities, relations = await extractor.extract("阿司匹林可以缓解头痛...")
    """

    def __init__(self, llm_service=None, entity_types: List[str] = None):
        self._llm = llm_service
        self._entity_types = entity_types or [
            "concept", "technology", "method", "person", "organization", "tool", "theory"
        ]

    async def extract(self, text: str) -> Tuple[List[Entity], List[Relation]]:
        """从文本中抽取实体和关系。

        Args:
            text: 输入文本

        Returns:
            (entities, relations) 二元组
        """
        if not text.strip():
            return [], []

        if self._llm is None:
            logger.warning("No LLM configured, using rule-based extraction")
            return self._rule_based_extract(text)

        prompt = EXTRACTION_PROMPT.format(
            entity_types=", ".join(self._entity_types),
            text=text[:3000],  # 截断避免超 token
        )

        try:
            response = await self._llm.ainvoke(prompt)
            return self._parse_response(response)
        except Exception as e:
            logger.error(f"LLM extraction failed: {e}")
            return self._rule_based_extract(text)

    def _parse_response(self, response: str) -> Tuple[List[Entity], List[Relation]]:
        """解析 LLM 的 JSON 响应。"""
        # 去掉 Qwen3 思考模式的 <think>...</think> 标签内容
        cleaned = re.sub(r'<think>[\s\S]*?</think>', '', response)
        # 去掉 markdown 代码块标记
        cleaned = re.sub(r'```(?:json)?\s*', '', cleaned)
        cleaned = cleaned.strip()

        # 提取 JSON 块（从第一个 { 到最后一个 }）
        json_match = re.search(r'\{[\s\S]*\}', cleaned)
        if not json_match:
            logger.warning("No JSON found in LLM response")
            return [], []

        try:
            data = json.loads(json_match.group())
        except json.JSONDecodeError:
            logger.warning("Failed to parse JSON from LLM response")
            return [], []

        entities = []
        for item in data.get("entities", []):
            entities.append(Entity(
                name=item.get("name", ""),
                entity_type=item.get("type", "概念"),
                description=item.get("description", ""),
            ))

        relations = []
        for item in data.get("relations", []):
            relations.append(Relation(
                source=item.get("source", ""),
                target=item.get("target", ""),
                relation_type=item.get("type", "related"),
                description=item.get("description", ""),
            ))

        return entities, relations

    def _rule_based_extract(self, text: str) -> Tuple[List[Entity], List[Relation]]:
        """规则基线抽取（无 LLM 时的降级方案）。

        使用 jieba 词性标注识别名词短语作为实体，
        比纯正则模式匹配覆盖更广（医疗、法律、教育领域通用）。
        """
        entities = []
        seen = set()

        # 策略 1：jieba 词性标注提取名词短语
        try:
            import jieba.posseg as pseg
            # 词性标注：n=名词, nr=人名, ns=地名, nt=机构, nz=其他专名,
            #           vn=动名词, an=名形词
            noun_pos = {"n", "nr", "ns", "nt", "nz", "vn", "an"}

            for word, pos in pseg.cut(text):
                word = word.strip()
                if len(word) < 2 or word in seen:
                    continue

                # 根据词性映射实体类型
                if pos in ("nr",):
                    etype = "person"
                elif pos in ("ns",):
                    etype = "location"
                elif pos in ("nt",):
                    etype = "organization"
                elif pos in ("n", "vn", "an", "nz"):
                    tech_suffixes = ("learning", "network", "algorithm", "model", "function",
                                     "mechanism", "method", "system", "framework", "theory",
                                     "protocol", "technology", "approach", "technique")
                    if any(word.endswith(s) for s in tech_suffixes):
                        etype = "technology"
                    else:
                        etype = "concept"
                else:
                    continue

                seen.add(word)
                entities.append(Entity(
                    name=word,
                    entity_type=etype,
                    description="",
                ))

        except ImportError:
            # jieba 不可用时，回退到正则匹配
            patterns = [
                (r'([A-Z][a-zA-Z]+(?:\s[A-Z][a-zA-Z]+)*)', "technology"),
            ]
            for pattern, etype in patterns:
                for match in re.finditer(pattern, text):
                    name = match.group(1).strip()
                    if name not in seen and len(name) >= 2:
                        seen.add(name)
                        entities.append(Entity(name=name, entity_type=etype))

        # Strategy 2: English term supplement
        for match in re.finditer(r'([A-Z][a-zA-Z]{2,}(?:\s[A-Z][a-zA-Z]+)*)', text):
            name = match.group(1).strip()
            if name not in seen and len(name) >= 3:
                seen.add(name)
                entities.append(Entity(name=name, entity_type="technology"))

        # 关系抽取：基于共现和句法模式
        relations = []
        for i in range(len(entities)):
            for j in range(i + 1, len(entities)):
                e1, e2 = entities[i], entities[j]
                if e1.name in text and e2.name in text:
                    idx1 = text.index(e1.name)
                    idx2 = text.index(e2.name)
                    dist = abs(idx1 - idx2)

                    if dist < 50:
                        between = text[min(idx1, idx2) + len(e1.name if idx1 < idx2 else e2.name):
                                       max(idx1, idx2)]
                        rel_type = "related_to"
                        if any(kw in between for kw in ("belongs to", "is a", "is a type of", "is an")):
                            rel_type = "belongs_to"
                        elif any(kw in between for kw in ("causes", "leads to", "results in", "produces")):
                            rel_type = "causes"
                        elif any(kw in between for kw in ("uses", "utilizes", "employs", "based on")):
                            rel_type = "uses"
                        elif any(kw in between for kw in ("contains", "includes", "comprises", "consists of")):
                            rel_type = "contains"

                        relations.append(Relation(
                            source=e1.name,
                            target=e2.name,
                            relation_type=rel_type,
                            description=f"{e1.name} {rel_type} {e2.name}",
                        ))
                    elif dist < 200:
                        # 中距离共现
                        relations.append(Relation(
                            source=e1.name,
                            target=e2.name,
                            relation_type="co_occurs",
                            description=f"{e1.name} and {e2.name} appear in the same context",
                        ))

        return entities, relations
