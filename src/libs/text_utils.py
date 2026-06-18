"""文本处理工具。

提供中英文关键词提取、停用词过滤等公共功能。
被 Pipeline、GraphRetriever、SparseRetriever 等模块共享使用。
"""

from __future__ import annotations

import re
from typing import List

# 英文停用词（覆盖常见疑问词和虚词）
_EN_STOP = frozenset({
    "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would",
    "shall", "should", "may", "might", "must", "can", "could",
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to",
    "for", "of", "with", "by", "from", "as", "into", "about",
    "what", "which", "who", "when", "where", "how", "why",
    "this", "that", "these", "those", "it", "its", "they", "them",
    "if", "then", "else", "not", "no", "all", "each", "every",
    "both", "few", "more", "most", "other", "some", "such",
    "than", "too", "very", "just", "also", "different",
})

# 中文停用词（覆盖常见虚词和疑问词）
_CN_STOP = frozenset({
    "的", "了", "是", "在", "和", "有", "为", "这", "那", "个",
    "什么", "怎么", "为什么", "如何", "可以", "会", "将", "被",
    "如果", "出现", "使用", "比较", "哪些", "与", "对", "中",
    "上", "下", "请", "一", "不", "也", "就", "都", "而", "及",
    "等", "到", "从", "把", "让", "向", "给", "又", "或", "但",
})


def extract_keywords(query: str, min_length: int = 2) -> List[str]:
    """从查询中提取关键词（中英文双语支持）。

    优先使用 jieba 分词处理中文，回退到字符级切分。
    英文按空格分词并过滤停用词。

    Args:
        query: 查询文本
        min_length: 最小关键词长度（中文为字符数，英文为单词长度）

    Returns:
        去重后的关键词列表
    """
    keywords = []

    # 英文关键词：提取 3+ 字母的单词，过滤停用词
    en_words = re.findall(r'[a-zA-Z]{3,}', query)
    for w in en_words:
        if w.lower() not in _EN_STOP:
            keywords.append(w)

    # 中文关键词：jieba 分词
    try:
        import jieba
        for w in jieba.cut(query):
            w = w.strip()
            if len(w) >= min_length and w not in _CN_STOP:
                keywords.append(w)
    except ImportError:
        # jieba 不可用时，按字符级切分（效果差但不报错）
        cn_chars = re.findall(r'[一-鿿]{2,}', query)
        for w in cn_chars:
            if w not in _CN_STOP:
                keywords.append(w)

    return list(dict.fromkeys(keywords))  # 去重并保持顺序
