"""知识图谱构建脚本。

用法：
    python scripts/build_graph.py --input data/documents/
    python scripts/build_graph.py --input data/documents/AI与机器学习基础教程.md
    python scripts/build_graph.py --stats   # 查看图统计
"""

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.settings import load_settings
from src.core.types import Chunk
from src.ingestion.loaders.markdown_loader import MarkdownLoader
from src.ingestion.loaders.pdf_loader import PDFLoader
from src.ingestion.graph_builder.graph_store import GraphStore
from src.ingestion.graph_builder.graph_builder import GraphBuilder
from src.libs.llm_service import LLMService


async def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="GradusRAG 知识图谱构建")
    parser.add_argument("--input", "-i", help="输入文件或目录")
    parser.add_argument("--stats", action="store_true", help="显示图统计信息")
    parser.add_argument("--query", "-q", help="查询图谱")
    args = parser.parse_args()

    settings = load_settings()
    graph_store = GraphStore(persist_path="data/knowledge_graph.json")

    if args.stats:
        stats = graph_store.stats()
        print(f"实体数量: {stats['entity_count']}")
        print(f"关系数量: {stats['relation_count']}")
        return

    if args.query:
        results = graph_store.search_entities(args.query)
        if results:
            print(f"找到 {len(results)} 个匹配实体:")
            for r in results:
                print(f"  - {r['name']} [{r.get('entity_type', '')}]: {r.get('description', '')[:80]}")

                neighbors = graph_store.get_neighbors(r["name"], hops=2)
                for rel in neighbors.get("relations", []):
                    print(f"    → {rel['source']} --[{rel.get('relation_type', '')}]--> {rel['target']}")
        else:
            print(f"未找到匹配 '{args.query}' 的实体")
        return

    if not args.input:
        parser.print_help()
        return

    input_path = Path(args.input)
    loaders = {".pdf": PDFLoader(), ".md": MarkdownLoader(), ".markdown": MarkdownLoader()}

    # 加载文档（用 L1 根块构建图谱，减少 API 调用次数）
    chunks = []
    files = [input_path] if input_path.is_file() else sorted(input_path.rglob("*"))
    for f in files:
        loader = loaders.get(f.suffix.lower())
        if loader:
            docs = loader.load(f)
            from src.ingestion.chunking.document_chunker import DocumentChunker
            from src.ingestion.chunking.sliding_window import SlidingWindowChunker
            chunker = SlidingWindowChunker(settings.ingestion)
            doc_chunker = DocumentChunker(settings.ingestion)
            for doc in docs:
                # 只取 L1 根块用于图谱构建
                nodes = chunker.chunk_text(
                    text=doc.text,
                    filename=doc.metadata.get("filename", ""),
                    page=doc.metadata.get("page", 0),
                    doc_id=doc.id,
                )
                l1_nodes = [n for n in nodes if n.chunk_level == 1]
                for node in l1_nodes:
                    chunk = Chunk(
                        id=node.id,
                        text=node.text,
                        metadata={**doc.metadata, "chunk_level": 1, "parent_chunk_id": "", "root_chunk_id": node.id},
                        source_ref=doc.id,
                    )
                    chunks.append(chunk)

    print(f"加载了 {len(chunks)} 个分块")

    # 构建图谱
    llm_service = LLMService.from_settings(settings)
    builder = GraphBuilder(settings.graph, llm_service, graph_store)
    stats = await builder.build_from_chunks(chunks)

    print(f"\n{'='*50}")
    print(f"图谱构建完成:")
    print(f"  处理分块: {stats['processed_chunks']}")
    print(f"  实体数量: {stats['total_entities']}")
    print(f"  关系数量: {stats['total_relations']}")


if __name__ == "__main__":
    asyncio.run(main())
