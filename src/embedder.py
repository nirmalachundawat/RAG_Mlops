import os
import time
import json
from pathlib import Path
from typing import List, Dict

import mlflow
import yaml
import numpy as np
from loguru import logger
from tqdm import tqdm
from sentence_transformers import SentenceTransformer


# ── Load config ───────────────────────────────────────────────
def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path, "r") as f:
        import yaml
        return yaml.safe_load(f)


# ── Load embedding model ──────────────────────────────────────
def load_embedding_model(config: dict) -> SentenceTransformer:
    model_name = config["embedding"]["model_name"]
    logger.info(f"Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)
    logger.info(f"Model loaded — dimension: {model.get_sentence_embedding_dimension()}")
    return model


# ── Embed chunks in batches ───────────────────────────────────
def embed_chunks(
    chunks: List[Dict],
    model: SentenceTransformer,
    batch_size: int = 64,
) -> np.ndarray:
    texts = [c["text"] for c in chunks]
    logger.info(f"Embedding {len(texts)} chunks in batches of {batch_size}...")

    all_embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding"):
        batch = texts[i : i + batch_size]
        embeddings = model.encode(batch, show_progress_bar=False)
        all_embeddings.append(embeddings)

    return np.vstack(all_embeddings).astype("float32")