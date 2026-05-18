# RAG MLOps — Production-Grade Document Q&A System

A fully production-ready **Retrieval-Augmented Generation (RAG)** system designed with MLOps best practices. Ingest any PDF documents, build a versioned semantic search index, generate accurate grounded answers via Groq's LLaMA 3.3, evaluate quality with a custom LLM-as-judge pipeline, and serve everything through a REST API — with every experiment tracked in MLflow.

---

## Table of Contents

- [Overview](#overview)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [API Reference](#api-reference)
- [Evaluation](#evaluation)
- [MLflow Tracking](#mlflow-tracking)

---

## Overview

Most RAG tutorials show you how to build a chatbot. This project shows you how to **operate** one — with versioning, evaluation, monitoring, and reproducibility built in from the start.

The system answers questions grounded strictly in your uploaded documents, cites which source and chunk it drew from, tracks every run in MLflow, and scores itself automatically using a 4-metric LLM-as-judge evaluation framework.

**Baseline evaluation results on the RAG survey paper:**

| Metric | Score | Description |
|---|---|---|
| Faithfulness | 0.90 | No hallucination — answers grounded in context |
| Answer Relevancy | 0.80 | Answers directly address the question |
| Context Precision | 0.50 | Proportion of retrieved chunks that are relevant |
| Context Recall | 0.60 | Coverage of information needed to answer |
| **Overall** | **0.70** | **Production-ready baseline** |

---

## Tech Stack

| Layer | Tool | Purpose |
|---|---|---|
| LLM | Groq — LLaMA 3.3 70B Versatile | Answer generation |
| Embeddings | sentence-transformers — all-MiniLM-L6-v2 | Semantic encoding |
| Vector Store | FAISS (CPU) | Similarity search |
| PDF Parsing | PyMuPDF | Native text extraction |
| OCR | Tesseract 5 | Scanned PDF fallback |
| Experiment Tracking | MLflow 2.15 | Run logging & comparison |
| Orchestration | LangChain | Text splitting, pipeline wiring |
| API | FastAPI + Uvicorn | REST serving |
| Logging | Loguru | Structured file + console logs |
| Config | PyYAML + python-dotenv | Settings management |

---

---

## Getting Started

### Prerequisites

- Python 3.11+
- [Tesseract OCR](https://github.com/UB-Mannheim/tesseract/wiki) installed on your system
- A free [Groq API key](https://console.groq.com)

### 1. Clone the repository

```bash
git clone https://github.com/nirmalachundawat/RAG_Mlops.git
cd RAG_Mlops
```

### 2. Create and activate a virtual environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Install Tesseract OCR

**Windows:** Download and run the installer from [UB Mannheim](https://github.com/UB-Mannheim/tesseract/wiki). Keep the default install path (`C:\Program Files\Tesseract-OCR\`).

**macOS:**
```bash
brew install tesseract
```

**Linux:**
```bash
sudo apt install tesseract-ocr
```

### 5. Configure environment variables

```bash
cp .env.example .env
```

Edit `.env` and add your keys:

```env
GROQ_API_KEY=your_groq_api_key_here
EMBEDDING_MODEL=all-MiniLM-L6-v2
LLM_MODEL=llama-3.3-70b-versatile
```

### 6. Add your documents

Drop any PDF files into the `data/raw/` directory.

### 7. Run the ingestion pipeline

```bash
python test_ingest.py
```

### 8. Build the FAISS index

```bash
python test_index.py
```

### 9. Start the API

```bash
python api.py
```

Visit **http://localhost:8000/docs** for the interactive Swagger UI.

---

## Evaluation

Run the LLM-as-judge evaluation pipeline:

```bash
python evaluation/evaluate.py
```

The pipeline scores 4 metrics per question using a second LLM call as the judge:

- **Faithfulness** — Is every claim in the answer supported by the retrieved context? Catches hallucinations.
- **Answer Relevancy** — Does the answer actually address what was asked?
- **Context Precision** — What proportion of the retrieved chunks were actually useful?
- **Context Recall** — Did the retriever surface all the information needed?

All scores are logged to MLflow under the `ragas_evaluation` run for tracking over time.

---

## MLflow Tracking

Start the MLflow UI:

```bash
mlflow ui --backend-store-uri experiments/mlruns
```

Open **http://localhost:5000** to see all tracked runs:

| Run Name | What's Tracked |
|---|---|
| `ingestion` | num_docs, total_chunks, avg_chunk_length, chunk_size, overlap |
| `index_build` | embedding_model, dimension, build_time_sec, num_vectors |
| `batch_query` | avg_latency_sec, total_tokens, avg_tokens_per_query |
| `custom_evaluation` | faithfulness, answer_relevancy, context_precision, context_recall |

---

<p align="center">Built with ❤️ using RAG, Groq, FAISS, MLflow, and FastAPI</p>
