"""清理 Milvus 向量库。

清空所有数据，重新开始。用于文档更新后清理旧 chunk。

用法：
    python scripts/clear_milvus.py                    # 清空默认 collection
    python scripts/clear_milvus.py --yes              # 跳过确认
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main():
    parser = argparse.ArgumentParser(description="清空 Milvus 向量库")
    parser.add_argument("--yes", "-y", action="store_true", help="跳过确认")
    args = parser.parse_args()

    from src.core.settings import load_settings
    settings = load_settings()

    host = settings.vector_store.host
    port = settings.vector_store.port
    collection = settings.vector_store.collection

    print(f"Milvus: {host}:{port}, collection: {collection}")

    if not args.yes:
        confirm = input(f"确认清空 collection '{collection}' 的所有数据？(y/N): ")
        if confirm.lower() != "y":
            print("已取消")
            return

    from pymilvus import connections, Collection, utility

    connections.connect(host=host, port=port)

    if not utility.has_collection(collection):
        print(f"Collection '{collection}' 不存在，无需清理")
        return

    col = Collection(collection)
    count_before = col.num_entities
    col.drop()
    print(f"已删除 collection '{collection}'（原有 {count_before} 条记录）")
    print("下次入库时会自动重建")


if __name__ == "__main__":
    main()
