import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import mlflow
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from pydantic import BaseModel

from src.pipeline import RAGPipeline
from src.ingest import load_config

load_dotenv()

# ── FastAPI app ───────────────────────────────────────────────
app = FastAPI(
    title="RAG MLOps API",
    description="Document Q&A system powered by RAG + Groq + FAISS",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Global pipeline instance ──────────────────────────────────
pipeline: Optional[RAGPipeline] = None
config   = load_config()
start_time = datetime.now()


# ── Request / Response models ─────────────────────────────────
class QueryRequest(BaseModel):
    question: str
    top_k:    Optional[int] = None

class QueryResponse(BaseModel):
    question:           str
    answer:             str
    sources:            List[str]
    model:              str
    latency_sec:        float
    total_pipeline_sec: float
    input_tokens:       int
    output_tokens:      int
    retrieved_chunks:   List[dict]
    timestamp:          str

class BatchQueryRequest(BaseModel):
    questions: List[str]

class HealthResponse(BaseModel):
    status:        str
    pipeline_ready: bool
    uptime_seconds: float
    model:         str
    embedding:     str
    index_vectors: int
    timestamp:     str


# ── Startup: load pipeline ────────────────────────────────────
@app.on_event("startup")
async def startup_event():
    global pipeline
    logger.info("Starting RAG API — loading pipeline...")
    try:
        pipeline = RAGPipeline()
        pipeline.load()
        logger.info("✅ Pipeline loaded and ready")
    except Exception as e:
        logger.error(f"Failed to load pipeline: {e}")


# ── GET / — welcome ───────────────────────────────────────────
@app.get("/")
def root():
    return {
        "message": "RAG MLOps API is running 🚀",
        "docs":    "http://localhost:8000/docs",
        "health":  "http://localhost:8000/health",
    }


# ── GET /health — system status ───────────────────────────────
@app.get("/health", response_model=HealthResponse)
def health_check():
    uptime = round((datetime.now() - start_time).total_seconds(), 1)
    ready  = pipeline is not None and pipeline.index is not None

    return HealthResponse(
        status         = "healthy" if ready else "loading",
        pipeline_ready = ready,
        uptime_seconds = uptime,
        model          = config["llm"]["model"],
        embedding      = config["embedding"]["model_name"],
        index_vectors  = pipeline.index.ntotal if ready else 0,
        timestamp      = datetime.now().isoformat(),
    )


# ── POST /query — single question ────────────────────────────
@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    if not pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not ready yet")

    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    # Override top_k if provided
    if request.top_k:
        pipeline.config["vector_store"]["top_k"] = request.top_k

    try:
        result = pipeline.query(request.question)
        return QueryResponse(
            question           = result["query"],
            answer             = result["answer"],
            sources            = result["sources"],
            model              = result["model"],
            latency_sec        = result["latency_sec"],
            total_pipeline_sec = result["total_pipeline_sec"],
            input_tokens       = result["input_tokens"],
            output_tokens      = result["output_tokens"],
            retrieved_chunks   = result["retrieved_chunks"],
            timestamp          = datetime.now().isoformat(),
        )
    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ── POST /batch — multiple questions ─────────────────────────
@app.post("/batch")
def batch_query(request: BatchQueryRequest):
    if not pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not ready yet")

    if not request.questions:
        raise HTTPException(status_code=400, detail="Questions list cannot be empty")

    if len(request.questions) > 20:
        raise HTTPException(status_code=400, detail="Max 20 questions per batch")

    try:
        results = pipeline.run_batch(request.questions)
        return {
            "total_questions": len(results),
            "results": [
                {
                    "question":    r["query"],
                    "answer":      r["answer"],
                    "sources":     r["sources"],
                    "latency_sec": r["latency_sec"],
                    "tokens":      r["total_tokens"],
                }
                for r in results
            ],
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        logger.error(f"Batch query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ── GET /metrics — query log summary ─────────────────────────
@app.get("/metrics")
def get_metrics():
    log_path = Path(config["logging"]["log_dir"]) / "query_log.jsonl"

    if not log_path.exists():
        return {"message": "No queries logged yet", "total_queries": 0}

    queries = []
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                queries.append(json.loads(line))
            except:
                continue

    if not queries:
        return {"message": "No queries logged yet", "total_queries": 0}

    avg_latency = round(
        sum(q["latency_sec"] for q in queries) / len(queries), 3
    )
    avg_tokens = round(
        sum(q["total_tokens"] for q in queries) / len(queries), 1
    )
    total_tokens = sum(q["total_tokens"] for q in queries)

    return {
        "total_queries":    len(queries),
        "avg_latency_sec":  avg_latency,
        "avg_tokens":       avg_tokens,
        "total_tokens_used": total_tokens,
        "last_query":       queries[-1]["query"] if queries else None,
        "last_query_time":  queries[-1]["timestamp"] if queries else None,
    }


# ── GET /config — current system config ──────────────────────
@app.get("/config")
def get_config():
    return {
        "embedding_model":   config["embedding"]["model_name"],
        "embedding_version": config["embedding"]["version"],
        "llm_model":         config["llm"]["model"],
        "temperature":       config["llm"]["temperature"],
        "max_tokens":        config["llm"]["max_tokens"],
        "chunk_size":        config["data"]["chunk_size"],
        "chunk_overlap":     config["data"]["chunk_overlap"],
        "top_k":             config["vector_store"]["top_k"],
        "index_name":        config["vector_store"]["index_name"],
    }


# ── Run server ────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info",
    )