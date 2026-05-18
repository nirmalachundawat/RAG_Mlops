from src.ingest import load_config
from src.retriever import load_index, retrieve
from src.embedder import load_embedding_model
from src.generator import load_llm_client, generate_answer

config = load_config()

# Load everything
print("Loading index and model...")
index, chunks = load_index(config)
model         = load_embedding_model(config)
client        = load_llm_client()

# Test query
query = "What are the three stages of RAG development?"

print(f"\nQuery: {query}")
print("-" * 60)

# Retrieve relevant chunks
retrieved = retrieve(query, index, chunks, model, config)

# Generate answer
result = generate_answer(query, retrieved, client, config)

print(f"\n Answer:\n{result['answer']}")
print(f"\n Metrics:")
print(f"   Model      : {result['model']}")
print(f"   Latency    : {result['latency_sec']}s")
print(f"   Tokens in  : {result['input_tokens']}")
print(f"   Tokens out : {result['output_tokens']}")
print(f"   Sources    : {result['sources']}")