"""嵌入服务。

提供文本向量化能力，支持：
- 密集向量（Dense）：使用 BGE-M3 本地模型
- 稀疏向量（Sparse）：手动 BM25 实现，统计持久化

设计参考 SuperMew 的 EmbeddingService，但接口更简洁。
"""

from __future__ import annotations

import json
import logging
import math
import os
# 设置 HuggingFace 镜像（国内加速）
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
# 离线模式：模型下载完成后取消注释下一行以加速启动
# os.environ.setdefault("HF_HUB_OFFLINE", "1")
import re
import threading
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

_DEFAULT_STATE_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "bm25_state.json"


class EmbeddingService:
    """文本向量化服务。

    提供密集向量（BGE-M3）和稀疏向量（BM25）两种向量化方式。

    Example:
        >>> service = EmbeddingService(model_name="BAAI/bge-m3")
        >>> dense = service.embed_dense(["什么是机器学习？"])
        >>> print(len(dense[0]))  # 1024
        >>> sparse = service.embed_sparse(["什么是机器学习？"])
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        device: str = "cpu",
        dimensions: int = 1024,
        state_path: Optional[str | Path] = None,
        api_key: Optional[str] = None,
        api_base_url: Optional[str] = None,
    ):
        self._model_name = model_name
        self._device = device
        self._dimensions = dimensions
        self._embedder = None  # 懒加载
        self._api_key = api_key
        self._api_base_url = api_base_url
        self._use_api = api_key and api_base_url  # 有 API 配置时用 API

        # BM25 参数
        self._state_path = Path(state_path) if state_path else _DEFAULT_STATE_PATH
        self._index_path = self._state_path.parent / "bm25_inverted_index.json"
        self._lock = threading.Lock()
        self.k1 = 1.5
        self.b = 0.75
        self._vocab: Dict[str, int] = {}
        self._vocab_counter = 0
        self._doc_freq: Counter = Counter()
        self._total_docs = 0
        self._sum_token_len = 0
        self._avg_doc_len = 1.0
        # 倒排索引: token → [(chunk_id, tf, doc_len), ...]
        self._inverted_index: Dict[str, List[tuple]] = {}
        # 文档元数据缓存: chunk_id → {text, filename, source_path, ...}
        self._doc_meta: Dict[str, Dict] = {}
        self._load_bm25_state()
        self._load_inverted_index()

    # =========================================================================
    # 密集向量 (Dense)
    # =========================================================================

    def _embed_via_api(self, texts: List[str]) -> List[List[float]]:
        """通过 OpenAI 兼容 API 生成密集向量。"""
        import httpx

        url = f"{self._api_base_url.rstrip('/')}/embeddings"
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self._model_name,
            "input": texts,
        }

        resp = httpx.post(url, json=payload, headers=headers, timeout=60)
        resp.raise_for_status()
        data = resp.json()

        # 按 index 排序保证顺序正确
        sorted_results = sorted(data["data"], key=lambda x: x["index"])
        return [item["embedding"] for item in sorted_results]

    def _get_embedder(self):
        """懒加载嵌入模型。"""
        if self._embedder is None:
            if self._model_name == "lightweight":
                # 轻量后端：jieba 分词 + TF-IDF，不依赖 PyTorch
                self._embedder = self._create_lightweight_embedder()
                logger.info("Using lightweight embedder (jieba + TF-IDF)")
            else:
                try:
                    from langchain_huggingface import HuggingFaceEmbeddings
                    self._embedder = HuggingFaceEmbeddings(
                        model_name=self._model_name,
                        model_kwargs={"device": self._device},
                        encode_kwargs={"normalize_embeddings": True},
                    )
                    logger.info(f"Loaded embedding model: {self._model_name} on {self._device}")
                except Exception as e:
                    logger.error(f"Failed to load model {self._model_name}: {e}")
                    logger.info("Falling back to lightweight embedder")
                    self._embedder = self._create_lightweight_embedder()
        return self._embedder

    @staticmethod
    def _create_lightweight_embedder():
        """创建轻量嵌入器（jieba + TF-IDF），不依赖 PyTorch。"""
        import math
        import numpy as np

        try:
            import jieba
        except ImportError:
            raise ImportError("jieba is required for lightweight embedder. Install: pip install jieba")

        class LightweightEmbedder:
            """jieba 分词 + TF-IDF 向量化。不依赖 PyTorch，瞬间可用。"""

            def __init__(self):
                self._vocab = {}       # token -> index
                self._idf = {}         # token -> idf score
                self._doc_count = 0
                self._fitted = False

            def _tokenize(self, text: str) -> list:
                return [w for w in jieba.cut(text) if len(w.strip()) > 0]

            def _fit(self, texts: list):
                """基于语料库构建 IDF。"""
                if self._fitted:
                    return
                df = Counter()
                self._doc_count = len(texts)
                for text in texts:
                    tokens = set(self._tokenize(text))
                    for t in tokens:
                        df[t] += 1
                # 构建词表
                self._vocab = {t: i for i, t in enumerate(df.keys())}
                # 计算 IDF
                for t, freq in df.items():
                    self._idf[t] = math.log((self._doc_count + 1) / (freq + 1)) + 1
                self._fitted = True

            def _text_to_vec(self, text: str, dim: int = 512) -> list:
                """将文本转成 TF-IDF 向量。"""
                tokens = self._tokenize(text)
                if not tokens:
                    return [0.0] * dim

                tf = Counter(tokens)
                vec = {}
                for t, count in tf.items():
                    if t in self._idf:
                        idx = self._vocab.get(t, hash(t) % dim)
                        vec[idx % dim] = vec.get(idx % dim, 0) + count * self._idf[t]

                # 转为固定维度向量并归一化
                result = [0.0] * dim
                for idx, val in vec.items():
                    result[idx] = val

                # L2 归一化
                norm = math.sqrt(sum(v * v for v in result)) or 1.0
                result = [v / norm for v in result]
                return result

            def embed_documents(self, texts: list) -> list:
                self._fit(texts)
                return [self._text_to_vec(t) for t in texts]

            def embed_query(self, query: str) -> list:
                if not self._fitted:
                    self._fit([query])
                return self._text_to_vec(query)

        return LightweightEmbedder()

    def embed_dense(self, texts: List[str]) -> List[List[float]]:
        """生成密集向量。"""
        if not texts:
            return []
        if self._use_api:
            return self._embed_via_api(texts)
        embedder = self._get_embedder()
        return embedder.embed_documents(texts)

    def embed_dense_query(self, query: str) -> List[float]:
        """为查询生成密集向量。"""
        if self._use_api:
            return self._embed_via_api([query])[0]
        embedder = self._get_embedder()
        return embedder.embed_query(query)

    @property
    def dense_dim(self) -> int:
        return self._dimensions

    # =========================================================================
    # 稀疏向量 (Sparse / BM25)
    # =========================================================================

    @staticmethod
    def tokenize(text: str) -> List[str]:
        """中英文混合分词。

        中文使用 jieba 分词（词级别），英文按空格分词。
        比字符级分词精度高得多——"监督学习"不会被拆成"监督"+"学习"。
        """
        tokens = []
        # 英文单词
        english_words = re.findall(r'[a-zA-Z]+(?:\.\w+)*', text)
        tokens.extend(english_words)
        # 中文：使用 jieba 分词
        try:
            import jieba
            chinese_tokens = [w for w in jieba.cut(text) if len(w.strip()) >= 2]
            tokens.extend(chinese_tokens)
        except ImportError:
            # jieba 不可用时回退到字符级
            chinese_chars = re.findall(r'[一-鿿]', text)
            tokens.extend(chinese_chars)
        # 数字
        numbers = re.findall(r'\d+', text)
        tokens.extend(numbers)
        return tokens

    def _load_bm25_state(self) -> None:
        """从文件加载 BM25 统计状态。"""
        if not self._state_path.is_file():
            return
        try:
            raw = json.loads(self._state_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return
        if raw.get("version") != 1:
            return
        self._vocab = {str(k): int(v) for k, v in raw.get("vocab", {}).items()}
        self._doc_freq = Counter({str(k): int(v) for k, v in raw.get("doc_freq", {}).items()})
        self._total_docs = int(raw.get("total_docs", 0))
        self._sum_token_len = int(raw.get("sum_token_len", 0))
        if self._vocab:
            self._vocab_counter = max(self._vocab.values()) + 1
        self._recompute_avg_len()

    def _persist_bm25_state(self) -> None:
        """持久化 BM25 统计到文件。"""
        self._state_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "total_docs": self._total_docs,
            "sum_token_len": self._sum_token_len,
            "vocab": self._vocab,
            "doc_freq": dict(self._doc_freq),
        }
        tmp = self._state_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        tmp.replace(self._state_path)

    def _recompute_avg_len(self) -> None:
        self._avg_doc_len = (
            self._sum_token_len / self._total_docs if self._total_docs > 0 else 1.0
        )

    def bm25_increment_add(self, texts: List[str]) -> None:
        """增量添加文档到 BM25 统计。"""
        if not texts:
            return
        with self._lock:
            for text in texts:
                tokens = self.tokenize(text)
                doc_len = len(tokens)
                self._total_docs += 1
                self._sum_token_len += doc_len
                seen = set()
                for token in tokens:
                    if token not in self._vocab:
                        self._vocab[token] = self._vocab_counter
                        self._vocab_counter += 1
                    if token not in seen:
                        self._doc_freq[token] += 1
                        seen.add(token)
            self._recompute_avg_len()
            self._persist_bm25_state()

    def bm25_increment_remove(self, texts: List[str]) -> None:
        """增量从 BM25 统计中移除文档。"""
        if not texts:
            return
        with self._lock:
            for text in texts:
                tokens = self.tokenize(text)
                doc_len = len(tokens)
                self._total_docs = max(0, self._total_docs - 1)
                self._sum_token_len = max(0, self._sum_token_len - doc_len)
                seen = set()
                for token in tokens:
                    if token not in seen:
                        self._doc_freq[token] = max(0, self._doc_freq.get(token, 0) - 1)
                        seen.add(token)
            self._recompute_avg_len()
            self._persist_bm25_state()

    def compute_bm25_score(self, query_tokens: List[str], doc_tokens: List[str]) -> float:
        """计算单个文档的 BM25 分数。"""
        doc_len = len(doc_tokens)
        tf_map: Dict[str, int] = Counter(doc_tokens)
        score = 0.0
        for qt in query_tokens:
            tf = tf_map.get(qt, 0)
            df = self._doc_freq.get(qt, 0)
            if df == 0:
                continue
            idf = math.log(
                (self._total_docs - df + 0.5) / (df + 0.5) + 1.0
            )
            tf_norm = (tf * (self.k1 + 1)) / (
                tf + self.k1 * (1 - self.b + self.b * doc_len / self._avg_doc_len)
            )
            score += idf * tf_norm
        return score

    def embed_sparse(self, texts: List[str]) -> List[Dict[str, float]]:
        """为文本生成 BM25 稀疏向量。

        Returns:
            每个文本的稀疏向量，格式 {token_index: bm25_score}
        """
        results = []
        for text in texts:
            tokens = self.tokenize(text)
            query_tokens = list(set(tokens))
            scores: Dict[str, float] = {}

            for qt in query_tokens:
                tf = tokens.count(qt)
                df = self._doc_freq.get(qt, 0)
                if df == 0 or self._total_docs == 0:
                    continue
                idf = math.log(
                    (self._total_docs - df + 0.5) / (df + 0.5) + 1.0
                )
                tf_norm = (tf * (self.k1 + 1)) / (
                    tf + self.k1 * (1 - self.b + self.b * len(tokens) / self._avg_doc_len)
                )
                idx = self._vocab.get(qt)
                if idx is not None:
                    scores[str(idx)] = idf * tf_norm

            results.append(scores)
        return results

    def embed_all(self, texts: List[str]) -> Tuple[List[List[float]], List[Dict[str, float]]]:
        """同时生成密集和稀疏向量。"""
        dense = self.embed_dense(texts)
        sparse = self.embed_sparse(texts)
        return dense, sparse

    # =========================================================================
    # 倒排索引 (Inverted Index for fast BM25 retrieval)
    # =========================================================================

    def _load_inverted_index(self) -> None:
        """从文件加载倒排索引。"""
        if not self._index_path.is_file():
            return
        try:
            raw = json.loads(self._index_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return

        self._inverted_index = {}
        for token, postings in raw.get("index", {}).items():
            self._inverted_index[token] = [
                (p[0], p[1], p[2]) for p in postings
            ]
        self._doc_meta = raw.get("doc_meta", {})
        logger.info(
            f"Loaded inverted index: {len(self._inverted_index)} tokens, "
            f"{len(self._doc_meta)} docs"
        )

    def _persist_inverted_index(self) -> None:
        """持久化倒排索引到文件。"""
        self._index_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "index": {
                token: [list(p) for p in postings]
                for token, postings in self._inverted_index.items()
            },
            "doc_meta": self._doc_meta,
        }
        tmp = self._index_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        tmp.replace(self._index_path)

    def bm25_index_add(
        self,
        chunk_ids: List[str],
        texts: List[str],
        metas: Optional[List[Dict]] = None,
    ) -> None:
        """将文档添加到倒排索引。

        在 bm25_increment_add 之后调用，将文档的详细信息写入倒排索引。
        这样检索时可以直接从索引获取结果，无需从 Milvus 全量拉取。

        Args:
            chunk_ids: 每个文档的 chunk_id
            texts: 每个文档的文本
            metas: 每个文档的元数据（filename, source_path, page 等）
        """
        if not chunk_ids:
            return

        with self._lock:
            for i, (cid, text) in enumerate(zip(chunk_ids, texts)):
                tokens = self.tokenize(text)
                doc_len = len(tokens)

                # 计算每个 token 在该文档中的 tf
                tf_map: Dict[str, int] = {}
                for t in tokens:
                    tf_map[t] = tf_map.get(t, 0) + 1

                # 写入倒排索引
                for token, tf in tf_map.items():
                    if token not in self._inverted_index:
                        self._inverted_index[token] = []
                    # 检查是否已存在（幂等）
                    existing = [p for p in self._inverted_index[token] if p[0] == cid]
                    if not existing:
                        self._inverted_index[token].append((cid, tf, doc_len))

                # 保存文档元数据
                if metas and i < len(metas):
                    meta = metas[i].copy()
                    meta["text"] = text[:2000]  # 保存部分文本用于检索结果
                    self._doc_meta[cid] = meta
                elif cid not in self._doc_meta:
                    self._doc_meta[cid] = {"text": text[:2000]}

            self._persist_inverted_index()

    def bm25_index_remove(self, chunk_ids: List[str]) -> None:
        """从倒排索引中移除文档。"""
        with self._lock:
            remove_set = set(chunk_ids)
            for token in list(self._inverted_index.keys()):
                self._inverted_index[token] = [
                    p for p in self._inverted_index[token]
                    if p[0] not in remove_set
                ]
                if not self._inverted_index[token]:
                    del self._inverted_index[token]
            for cid in chunk_ids:
                self._doc_meta.pop(cid, None)
            self._persist_inverted_index()

    def bm25_search(
        self,
        query_text: str,
        top_k: int = 10,
    ) -> List[Dict]:
        """使用倒排索引进行 BM25 检索（O(K) 复杂度）。

        只遍历查询词命中的文档，不扫描全量数据。

        Args:
            query_text: 查询文本
            top_k: 返回的最大结果数

        Returns:
            检索结果列表，每项包含 chunk_id, score, text, 及元数据
        """
        query_tokens = self.tokenize(query_text)
        if not query_tokens or not self._inverted_index:
            return []

        # 收集所有命中文档的 BM25 分数
        doc_scores: Dict[str, float] = {}

        for qt in query_tokens:
            postings = self._inverted_index.get(qt, [])
            df = len(postings)  # 该 token 出现在多少文档中
            if df == 0:
                continue

            idf = math.log(
                (self._total_docs - df + 0.5) / (df + 0.5) + 1.0
            )

            for chunk_id, tf, doc_len in postings:
                tf_norm = (tf * (self.k1 + 1)) / (
                    tf + self.k1 * (1 - self.b + self.b * doc_len / self._avg_doc_len)
                )
                score = idf * tf_norm
                doc_scores[chunk_id] = doc_scores.get(chunk_id, 0.0) + score

        # 排序并取 top_k
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

        results = []
        for chunk_id, score in sorted_docs:
            meta = self._doc_meta.get(chunk_id, {})
            results.append({
                "chunk_id": chunk_id,
                "score": score,
                "text": meta.get("text", ""),
                "filename": meta.get("filename", ""),
                "source_path": meta.get("source_path", ""),
                "page": meta.get("page", 0),
                "chunk_index": meta.get("chunk_index", 0),
                "chunk_level": meta.get("chunk_level", 3),
                "parent_chunk_id": meta.get("parent_chunk_id", ""),
                "root_chunk_id": meta.get("root_chunk_id", ""),
            })

        return results

    def bm25_index_stats(self) -> Dict:
        """获取倒排索引统计信息。"""
        total_postings = sum(len(p) for p in self._inverted_index.values())
        return {
            "tokens": len(self._inverted_index),
            "documents": len(self._doc_meta),
            "total_postings": total_postings,
        }
