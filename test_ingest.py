from src.ingest import load_config, run_ingestion

config = load_config()
chunks = run_ingestion(config)

print(f"\n Total chunks created: {len(chunks)}")
print(f"\n Sample chunk:")
print(f"  ID     : {chunks[0]['chunk_id']}")
print(f"  Source : {chunks[0]['source']}")
print(f"  Text   : {chunks[0]['text'][:200]}...")