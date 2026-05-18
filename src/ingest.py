import os
import json
import hashlib
from pathlib import Path
from datetime import datetime
from typing import List, Dict
import fitz  
import pytesseract
from PIL import Image
import io
import shutil
import mlflow
import yaml
from loguru import logger
from tqdm import tqdm
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter


# ── Load config ──────────────────────────────────────────────
def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# ── Logging setup ─────────────────────────────────────────────
def setup_logger(log_dir: str):
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    log_file = Path(log_dir) / f"ingest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logger.add(log_file, rotation="10 MB", level="INFO")
    logger.info("Logger initialized")


# ── Document hash (for change detection) ─────────────────────
def get_file_hash(filepath: str) -> str:
    """MD5 hash of a file — used to detect if a doc changed."""
    with open(filepath, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


# ── PDF text extraction (with OCR fallback) ───────────────────
def extract_text_from_pdf(filepath: str) -> str:
    tesseract_path = shutil.which("tesseract") or r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    pytesseract.pytesseract.tesseract_cmd = tesseract_path
    logger.info(f"  Using Tesseract at: {tesseract_path}")

    doc = fitz.open(filepath)
    full_text = ""

    for page_num, page in enumerate(doc):
        # Try native text extraction first
        text = page.get_text().strip()

        if text and len(text) > 50:
            # Page has real text
            full_text += text + "\n"
            logger.info(f"  Page {page_num + 1}: native text ({len(text)} chars)")
        else:
            # Page is an image — use OCR
            logger.info(f"  Page {page_num + 1}: no text found, running OCR...")
            pix = page.get_pixmap(dpi=300)
            img = Image.open(io.BytesIO(pix.tobytes("png")))
            ocr_text = pytesseract.image_to_string(img, lang="eng").strip()
            if ocr_text:
                full_text += ocr_text + "\n"
                logger.info(f"  Page {page_num + 1}: OCR extracted ({len(ocr_text)} chars)")
            else:
                logger.warning(f"  Page {page_num + 1}: OCR returned nothing")

    doc.close()
    return full_text.strip()


# ── Chunk a single document ───────────────────────────────────
def chunk_document(
    text: str,
    filename: str,
    file_hash: str,
    chunk_size: int,
    chunk_overlap: int,
) -> List[Dict]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_text(text)

    records = []
    for i, chunk in enumerate(chunks):
        records.append({
            "chunk_id":   f"{file_hash[:8]}_{i:04d}",
            "source":     filename,
            "file_hash":  file_hash,
            "chunk_index": i,
            "total_chunks": len(chunks),
            "text":       chunk,
            "ingested_at": datetime.now().isoformat(),
        })

    logger.info(f"{filename} → {len(chunks)} chunks")
    return records


# ── Process all PDFs in raw_dir ───────────────────────────────
def run_ingestion(config: dict) -> List[Dict]:
    raw_dir       = config["data"]["raw_dir"]
    processed_dir = config["data"]["processed_dir"]
    chunk_size    = config["data"]["chunk_size"]
    chunk_overlap = config["data"]["chunk_overlap"]
    log_dir       = config["logging"]["log_dir"]

    setup_logger(log_dir)
    Path(processed_dir).mkdir(parents=True, exist_ok=True)

    pdf_files = list(Path(raw_dir).glob("*.pdf"))
    if not pdf_files:
        logger.warning(f"No PDFs found in {raw_dir}")
        return []

    logger.info(f"Found {len(pdf_files)} PDF(s) in {raw_dir}")

    all_chunks = []

    # ── MLflow run ────────────────────────────────────────────
    mlflow.set_tracking_uri(config["mlops"]["tracking_uri"])
    mlflow.set_experiment(config["mlops"]["experiment_name"])

    with mlflow.start_run(run_name="ingestion"):
        mlflow.log_param("chunk_size",    chunk_size)
        mlflow.log_param("chunk_overlap", chunk_overlap)
        mlflow.log_param("num_documents", len(pdf_files))
        mlflow.log_param("embedding_model_version",
                         config["embedding"]["version"])

        for pdf_path in tqdm(pdf_files, desc="Ingesting PDFs"):
            filename  = pdf_path.name
            file_hash = get_file_hash(str(pdf_path))

            logger.info(f"Processing: {filename} (hash: {file_hash[:8]})")

            try:
                text   = extract_text_from_pdf(str(pdf_path))
                chunks = chunk_document(
                    text, filename, file_hash,
                    chunk_size, chunk_overlap
                )
                all_chunks.extend(chunks)

            except Exception as e:
                logger.error(f"Failed to process {filename}: {e}")
                mlflow.log_param(f"error_{filename}", str(e))

        # ── Save processed chunks as JSON ─────────────────────
        output_path = Path(processed_dir) / "chunks.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(all_chunks, f, indent=2, ensure_ascii=False)

        # ── Log summary metrics to MLflow ─────────────────────
        mlflow.log_metric("total_chunks",    len(all_chunks))
        mlflow.log_metric("total_documents", len(pdf_files))
        avg_len = (
            sum(len(c["text"]) for c in all_chunks) / len(all_chunks)
            if all_chunks else 0
        )
        mlflow.log_metric("avg_chunk_length", round(avg_len, 2))
        mlflow.log_artifact(str(output_path))

        logger.info(
            f"Ingestion complete — {len(all_chunks)} chunks from "
            f"{len(pdf_files)} docs saved to {output_path}"
        )

    return all_chunks


# ── Load already-processed chunks ────────────────────────────
def load_chunks(config: dict) -> List[Dict]:
    chunks_path = Path(config["data"]["processed_dir"]) / "chunks.json"
    if not chunks_path.exists():
        raise FileNotFoundError(
            f"No chunks found at {chunks_path}. Run ingestion first."
        )
    with open(chunks_path, "r", encoding="utf-8") as f:
        return json.load(f)