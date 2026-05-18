import os
import time
from typing import List, Dict

import mlflow
import yaml
from dotenv import load_dotenv
from groq import Groq
from loguru import logger

load_dotenv()


# ── Load config ───────────────────────────────────────────────
def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# ── Load Groq client ──────────────────────────────────────────
def load_llm_client() -> Groq:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not found in .env file")
    logger.info("Groq client initialized")
    return Groq(api_key=api_key)


# ── Build prompt from retrieved chunks ────────────────────────
def build_prompt(query: str, retrieved_chunks: List[Dict]) -> str:
    context_parts = []
    for i, chunk in enumerate(retrieved_chunks):
        context_parts.append(
            f"[Source {i+1}: {chunk['source']} | chunk {chunk['chunk_index']}]\n"
            f"{chunk['text']}"
        )
    context = "\n\n---\n\n".join(context_parts)

    prompt = f"""You are an expert assistant that answers questions based strictly on the provided context.

CONTEXT:
{context}

QUESTION:
{query}

INSTRUCTIONS:
- Answer based only on the context above
- If the context does not contain enough information, say "I don't have enough information in the provided documents to answer this"
- Be concise, clear, and cite which source your answer comes from
- Do not make up information

ANSWER:"""
    return prompt


# ── Call Groq API ─────────────────────────────────────────────
def generate_answer(
    query: str,
    retrieved_chunks: List[Dict],
    client: Groq,
    config: dict,
    log_to_mlflow: bool = False,
) -> Dict:
    model       = config["llm"]["model"]
    temperature = config["llm"]["temperature"]
    max_tokens  = config["llm"]["max_tokens"]

    prompt = build_prompt(query, retrieved_chunks)

    logger.info(f"Calling Groq API — model: {model}")
    start = time.time()

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant that answers questions based on provided document context."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )

    latency      = round(time.time() - start, 3)
    answer       = response.choices[0].message.content
    input_tokens = response.usage.prompt_tokens
    out_tokens   = response.usage.completion_tokens
    total_tokens = response.usage.total_tokens

    logger.info(
        f"Generated answer in {latency}s | "
        f"tokens: {input_tokens} in / {out_tokens} out"
    )

    result = {
        "query":           query,
        "answer":          answer,
        "model":           model,
        "latency_sec":     latency,
        "input_tokens":    input_tokens,
        "output_tokens":   out_tokens,
        "total_tokens":    total_tokens,
        "num_chunks_used": len(retrieved_chunks),
        "sources": list(set(c["source"] for c in retrieved_chunks)),
    }

    # ── Log to MLflow if inside a run ─────────────────────────
    if log_to_mlflow:
        mlflow.log_metric("llm_latency_sec",   latency)
        mlflow.log_metric("input_tokens",      input_tokens)
        mlflow.log_metric("output_tokens",     out_tokens)
        mlflow.log_metric("total_tokens",      total_tokens)
        mlflow.log_param("llm_model",          model)
        mlflow.log_param("temperature",        temperature)

    return result