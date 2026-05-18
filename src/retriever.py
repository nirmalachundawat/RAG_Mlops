import os
import json
import time
from pathlib import Path
from typing import List, Dict, Tuple

import faiss
import mlflow
import numpy as np
import yaml
from loguru import logger

from src.embedder import load_config, load_embedding_model, embed_chunks
from src.ingest import load_chunks


# ── Save FAISS index + metadata ───────────────────────────────
def save_index(
    index: faiss.Index,
    chunks: List[Dict],
    config: dict,
):
    index_dir  = Path(config["vector_store"]["index_dir"])
    index_name = config["vector_store"]["index_name"]
    index_dir.mkdir(parents=True, exist_ok=True)

    # Save FAISS binary
    index_path = index_dir / f"{index_name}.faiss"
    faiss.write_index(index, str(index_path))
    logger.info(f"FAISS index saved → {index_path}")

    # Save chunk metadata (so we can map index ID → original text)
    meta_path = index_dir / f"{index_name}_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)
    logger.info(f"Metadata saved → {meta_path}")

    return str(index_path), str(meta_path)


# ── Load FAISS index + metadata ───────────────────────────────
def load_index(config: dict) -> Tuple[faiss.Index, List[Dict]]:
    index_dir  = Path(config["vector_store"]["index_dir"])
    index_name = config["vector_store"]["index_name"]

    index_path = index_dir / f"{index_name}.faiss"
    meta_path  = index_dir / f"{index_name}_metadata.json"

    if not index_path.exists():
        raise FileNotFoundError(f"No FAISS index at {index_path}. Build it first.")

    index = faiss.read_index(str(index_path))
    with open(meta_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    logger.info(f"Loaded FAISS index with {index.ntotal} vectors")
    return index, chunks


# ── Build FAISS index from chunks ─────────────────────────────
def build_index(config: dict) -> faiss.Index:
    mlflow.set_tracking_uri(config["mlops"]["tracking_uri"])
    mlflow.set_experiment(config["mlops"]["experiment_name"])

    with mlflow.start_run(run_name="index_build"):

        # -- Log config params
        mlflow.log_param("embedding_model",   config["embedding"]["model_name"])
        mlflow.log_param("embedding_version", config["embedding"]["version"])
        mlflow.log_param("index_name",        config["vector_store"]["index_name"])
        mlflow.log_param("top_k",             config["vector_store"]["top_k"])

        # -- Load chunks
        chunks = load_chunks(config)
        logger.info(f"Loaded {len(chunks)} chunks for indexing")
        mlflow.log_metric("num_chunks", len(chunks))

        # -- Load model & embed
        model = load_embedding_model(config)
        dim   = model.get_sentence_embedding_dimension()

        start = time.time()
        embeddings = embed_chunks(chunks, model)
        embed_time = round(time.time() - start, 2)

        mlflow.log_metric("embedding_dim",      dim)
        mlflow.log_metric("embedding_time_sec", embed_time)
        logger.info(f"Embeddings shape: {embeddings.shape} in {embed_time}s")

        # -- Build FAISS flat index (exact search, great for <100k chunks)
        index = faiss.IndexFlatL2(dim)
        index = faiss.IndexIDMap(index)
        ids   = np.arange(len(chunks)).astype("int64")
        index.add_with_ids(embeddings, ids)

        mlflow.log_metric("index_total_vectors", index.ntotal)
        logger.info(f"FAISS index built — {index.ntotal} vectors")

        # -- Save index & metadata
        index_path, meta_path = save_index(index, chunks, config)
        mlflow.log_artifact(index_path)
        mlflow.log_artifact(meta_path)
        mlflow.log_metric("build_time_sec", round(time.time() - start, 2))

    return index


# ── Retrieve top-k chunks for a query ─────────────────────────
def retrieve(
    query: str,
    index: faiss.Index,
    chunks: List[Dict],
    model,
    config: dict,
) -> List[Dict]:
    top_k = config["vector_store"]["top_k"]

    # Embed the query
    query_vec = model.encode([query]).astype("float32")

    # Search FAISS
    distances, ids = index.search(query_vec, top_k)

    results = []
    for dist, idx in zip(distances[0], ids[0]):
        if idx == -1:
            continue
        chunk = chunks[idx].copy()
        chunk["score"] = round(float(dist), 4)
        results.append(chunk)

    logger.info(f"Retrieved {len(results)} chunks for query: '{query[:60]}...'")
    return results