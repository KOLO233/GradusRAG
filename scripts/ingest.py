"""文档摄取脚本。

用法：
    python scripts/ingest.py --input data/documents/
    python scripts/ingest.py --input data/documents/教材.pdf
"""

import argparse
import logging
import sys
from pathlib import Path

# 确保项目根目录在 sys.path 中
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.settings import load_settings
from src.ingestion.pipeline import IngestionPipeline
from src.libs.embedding_service import EmbeddingService
from src.retrieval.milvus_store import MilvusStore


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="GradusRAG 文档摄取")
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="输入文件或目录路径",
    )
    parser.add_argument(
        "--chunk-only",
        action="store_true",
        help="仅分块，不写入 Milvus",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="批量写入大小（默认 50）",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: {input_path} does not exist")
        sys.exit(1)

    settings = load_settings()
    print(f"Loaded settings: LLM={settings.llm.provider}/{settings.llm.model}")

    if args.chunk_only:
        # 纯分块模式
        pipeline = IngestionPipeline(settings)

        if input_path.is_file():
            chunks, parents = pipeline.ingest_file(input_path)
        else:
            chunks, parents = pipeline.ingest_directory(input_path)

        print(f"\n{'='*50}")
        print(f"Chunking complete:")
        print(f"  Leaf chunks (L3): {len(chunks)}")
        print(f"  Parent blocks:    {len(parents)}")
        if chunks:
            print(f"  Sample chunk ID:  {chunks[0].id}")
            print(f"  Sample text:      {chunks[0].text[:100]}...")
    else:
        # 完整入库模式
        print(f"Embedding model: {settings.embedding.model}")
        print(f"Milvus: {settings.vector_store.host}:{settings.vector_store.port}")
        print()

        embedding = EmbeddingService(
            model_name=settings.embedding.model,
            device=settings.embedding.device,
            dimensions=settings.embedding.dimensions,
            api_key=settings.embedding.api_key,
            api_base_url=settings.embedding.api_base_url,
        )
        store = MilvusStore(
            host=settings.vector_store.host,
            port=settings.vector_store.port,
            collection=settings.vector_store.collection,
        )
        store.init_collection(dense_dim=settings.embedding.dimensions)

        pipeline = IngestionPipeline(
            settings,
            embedding_service=embedding,
            milvus_store=store,
        )

        if input_path.is_file():
            total_chunks, total_parents = pipeline.ingest_file_to_milvus(
                input_path, batch_size=args.batch_size
            )
        else:
            total_chunks, total_parents = pipeline.ingest_directory_to_milvus(
                input_path, batch_size=args.batch_size
            )

        print(f"\n{'='*50}")
        print(f"Ingestion complete:")
        print(f"  Total chunks written: {total_chunks}")
        print(f"  Parent blocks:        {total_parents}")


if __name__ == "__main__":
    main()
