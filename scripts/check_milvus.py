from pymilvus import MilvusClient
client = MilvusClient(uri="http://127.0.0.1:19530")
print(f"Collections: {client.list_collections()}")
stats = client.get_collection_stats("gradusrag")
print(f"Row count: {stats['row_count']}")
