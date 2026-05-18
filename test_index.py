from src.ingest import load_config
from src.retriever import build_index, load_index, retrieve
from src.embedder import load_embedding_model

config = load_config()

# Build the FAISS index
print("Building FAISS index...")
build_index(config)

# Load it back
print("\nLoading index...")
index, chunks = load_index(config)
model = load_embedding_model(config)

# Test retrieval
query = "What are the main components of RAG?"
print(f"\nQuery: {query}")
results = retrieve(query, index, chunks, model, config)

print(f"\n Top {len(results)} retrieved chunks:\n")
for i, r in enumerate(results):
    print(f"[{i+1}] Score: {r['score']} | Source: {r['source']}")
    print(f"     {r['text'][:150]}...")
    print()