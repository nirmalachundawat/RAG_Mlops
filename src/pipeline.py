import json
import time
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional

import mlflow
import yaml
from loguru import logger

from src.embedder import load_embedding_model, embed_chunks
from src.retriever import build_index, load_index, retrieve
from src.generator import load_llm_client, generate_answer
from src.ingest import run_ingestion, load_chunks, load_config


# ── Query logger ──────────────────────────────────────────────
class QueryLogger:
    def __init__(self, log_dir: str):
        self.log_path = Path(log_dir) / "query_log.jsonl"
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, result: Dict):
        record = {**result, "timestamp": datetime.now().isoformat()}
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")


# ── RAG Pipeline class ────────────────────────────────────────
class RAGPipeline:
    def __init__(self, config_path: str = "configs/config.yaml"):
        self.config       = load_config(config_path)
        self.query_logger = QueryLogger(self.config["logging"]["log_dir"])
        self.index        = None
        self.chunks       = None
        self.embed_model  = None
        self.llm_client   = None

    # ── Load all components ───────────────────────────────────
    def load(self):
        logger.info("Loading RAG pipeline components...")
        self.embed_model = load_embedding_model(self.config)
        self.index, self.chunks = load_index(self.config)
        self.llm_client  = load_llm_client()
        logger.info(" RAG pipeline ready")
        return self

    # ── Build index from scratch ──────────────────────────────
    def build(self, ingest: bool = True):
        logger.info("Building RAG pipeline from scratch...")
        if ingest:
            run_ingestion(self.config)
        build_index(self.config)
        self.load()
        return self

    # ── Run a single query ────────────────────────────────────
    def query(
        self,
        question: str,
        log_to_mlflow: bool = False,
    ) -> Dict:
        if not all([self.index, self.chunks, self.embed_model, self.llm_client]):
            raise RuntimeError("Pipeline not loaded. Call .load() first.")

        start = time.time()

        # Step 1 — Retrieve
        retrieved = retrieve(
            question,
            self.index,
            self.chunks,
            self.embed_model,
            self.config,
        )

        # Step 2 — Generate
        result = generate_answer(
            question,
            retrieved,
            self.llm_client,
            self.config,
            log_to_mlflow=log_to_mlflow,
        )

        result["total_pipeline_sec"] = round(time.time() - start, 3)
        result["retrieved_chunks"]   = [
            {"chunk_id": c["chunk_id"], "score": c["score"], "text": c["text"][:200]}
            for c in retrieved
        ]

        # Step 3 — Log query
        self.query_logger.log(result)

        return result

    # ── Run batch queries with MLflow tracking ────────────────
    def run_batch(self, questions: List[str]) -> List[Dict]:
        mlflow.set_tracking_uri(self.config["mlops"]["tracking_uri"])
        mlflow.set_experiment(self.config["mlops"]["experiment_name"])

        results = []
        with mlflow.start_run(run_name="batch_query"):
            mlflow.log_param("num_questions",    len(questions))
            mlflow.log_param("llm_model",        self.config["llm"]["model"])
            mlflow.log_param("embedding_model",  self.config["embedding"]["model_name"])
            mlflow.log_param("top_k",            self.config["vector_store"]["top_k"])

            total_tokens  = 0
            total_latency = 0

            for i, question in enumerate(questions):
                logger.info(f"Query {i+1}/{len(questions)}: {question[:60]}...")
                result = self.query(question, log_to_mlflow=False)
                results.append(result)

                total_tokens  += result["total_tokens"]
                total_latency += result["latency_sec"]

            # Log aggregate metrics
            mlflow.log_metric("total_tokens_used",   total_tokens)
            mlflow.log_metric("avg_latency_sec",     round(total_latency / len(questions), 3))
            mlflow.log_metric("avg_tokens_per_query",round(total_tokens  / len(questions), 1))
            mlflow.log_metric("total_queries",       len(questions))

            # Save batch results
            output_path = Path(self.config["logging"]["log_dir"]) / "batch_results.json"
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            mlflow.log_artifact(str(output_path))

        logger.info(f"Batch complete — {len(results)} queries processed")
        return results

    # ── Pretty print a result ─────────────────────────────────
    @staticmethod
    def display(result: Dict):
        print("\n" + "="*60)
        print(f" Query   : {result['query']}")
        print("="*60)
        print(f"\n Answer:\n{result['answer']}")
        print(f"\n Metrics:")
        print(f"   Model          : {result['model']}")
        print(f"   LLM latency    : {result['latency_sec']}s")
        print(f"   Pipeline time  : {result['total_pipeline_sec']}s")
        print(f"   Tokens in/out  : {result['input_tokens']} / {result['output_tokens']}")
        print(f"   Sources        : {result['sources']}")
        print("\n Retrieved chunks:")
        for i, c in enumerate(result["retrieved_chunks"]):
            print(f"   [{i+1}] score={c['score']} | {c['text'][:100]}...")
        print("="*60)