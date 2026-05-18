import json
import os
import sys
import time
from pathlib import Path
from datetime import datetime
from typing import List, Dict

sys.path.append(str(Path(__file__).parent.parent))

import mlflow
from dotenv import load_dotenv
from groq import Groq
from loguru import logger

from src.pipeline import RAGPipeline
from src.ingest import load_config

load_dotenv()


# ── Load eval dataset ─────────────────────────────────────────
def load_eval_dataset(path: str = "evaluation/eval_dataset.json") -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ── LLM-as-judge scorer ───────────────────────────────────────
def score_with_llm(client: Groq, prompt: str) -> float:
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=10,
    )
    text = response.choices[0].message.content.strip()
    try:
        score = float(text.split()[0])
        return max(0.0, min(1.0, score))
    except:
        return 0.5


# ── Metric 1: Faithfulness ────────────────────────────────────
def score_faithfulness(
    client: Groq,
    answer: str,
    contexts: List[str],
) -> float:
    context_text = "\n\n".join(contexts)
    prompt = f"""You are an evaluation judge. Score whether the ANSWER is fully supported by the CONTEXT.

CONTEXT:
{context_text}

ANSWER:
{answer}

Rules:
- Score 1.0 if every claim in the answer is supported by the context
- Score 0.5 if the answer is partially supported
- Score 0.0 if the answer contains information not in the context (hallucination)

Respond with ONLY a single number between 0.0 and 1.0. Nothing else."""
    return score_with_llm(client, prompt)


# ── Metric 2: Answer Relevancy ────────────────────────────────
def score_answer_relevancy(
    client: Groq,
    question: str,
    answer: str,
) -> float:
    prompt = f"""You are an evaluation judge. Score whether the ANSWER directly addresses the QUESTION.

QUESTION:
{question}

ANSWER:
{answer}

Rules:
- Score 1.0 if the answer directly and completely addresses the question
- Score 0.5 if the answer partially addresses the question
- Score 0.0 if the answer is off-topic or does not address the question

Respond with ONLY a single number between 0.0 and 1.0. Nothing else."""
    return score_with_llm(client, prompt)


# ── Metric 3: Context Precision ───────────────────────────────
def score_context_precision(
    client: Groq,
    question: str,
    contexts: List[str],
) -> float:
    scores = []
    for ctx in contexts:
        prompt = f"""You are an evaluation judge. Score whether this CONTEXT chunk is relevant to the QUESTION.

QUESTION:
{question}

CONTEXT CHUNK:
{ctx}

Rules:
- Score 1.0 if the chunk is highly relevant to answering the question
- Score 0.5 if the chunk is somewhat relevant
- Score 0.0 if the chunk is not relevant

Respond with ONLY a single number between 0.0 and 1.0. Nothing else."""
        scores.append(score_with_llm(client, prompt))
    return round(sum(scores) / len(scores), 4) if scores else 0.0


# ── Metric 4: Context Recall ──────────────────────────────────
def score_context_recall(
    client: Groq,
    ground_truth: str,
    contexts: List[str],
) -> float:
    context_text = "\n\n".join(contexts)
    prompt = f"""You are an evaluation judge. Score whether the CONTEXT contains enough information to produce the GROUND TRUTH answer.

CONTEXT:
{context_text}

GROUND TRUTH ANSWER:
{ground_truth}

Rules:
- Score 1.0 if the context contains all information needed to produce the ground truth
- Score 0.5 if the context contains some but not all needed information
- Score 0.0 if the context is missing critical information

Respond with ONLY a single number between 0.0 and 1.0. Nothing else."""
    return score_with_llm(client, prompt)


# ── Run full evaluation ───────────────────────────────────────
def run_evaluation(config_path: str = "configs/config.yaml"):
    config = load_config(config_path)

    mlflow.set_tracking_uri(config["mlops"]["tracking_uri"])
    mlflow.set_experiment(config["mlops"]["experiment_name"])

    with mlflow.start_run(run_name="custom_evaluation"):

        mlflow.log_param("eval_dataset",    "evaluation/eval_dataset.json")
        mlflow.log_param("embedding_model", config["embedding"]["model_name"])
        mlflow.log_param("llm_model",       config["llm"]["model"])
        mlflow.log_param("judge_model",     "llama-3.3-70b-versatile")
        mlflow.log_param("top_k",           config["vector_store"]["top_k"])
        mlflow.log_param("eval_timestamp",  datetime.now().isoformat())

        # -- Load pipeline and Groq judge
        logger.info("Loading RAG pipeline...")
        pipeline = RAGPipeline(config_path)
        pipeline.load()
        judge = Groq(api_key=os.getenv("GROQ_API_KEY"))

        # -- Load eval data
        eval_data = load_eval_dataset()
        logger.info(f"Loaded {len(eval_data)} evaluation questions")
        mlflow.log_metric("num_eval_questions", len(eval_data))

        # -- Score each question
        all_scores = []
        results    = []

        print("\n" + "="*60)
        print("🔍 Running evaluation on each question...")
        print("="*60)

        for i, item in enumerate(eval_data):
            question     = item["question"]
            ground_truth = item["ground_truth"]

            logger.info(f"Evaluating {i+1}/{len(eval_data)}: {question[:50]}...")

            # Run pipeline
            result   = pipeline.query(question)
            answer   = result["answer"]
            contexts = [c["text"] for c in result["retrieved_chunks"]]

            # Score all 4 metrics
            faithfulness      = score_faithfulness(judge, answer, contexts)
            answer_relevancy  = score_answer_relevancy(judge, question, answer)
            context_precision = score_context_precision(judge, question, contexts)
            context_recall    = score_context_recall(judge, ground_truth, contexts)
            overall           = round(
                (faithfulness + answer_relevancy + context_precision + context_recall) / 4, 4
            )

            scores = {
                "question":          question,
                "answer":            answer,
                "ground_truth":      ground_truth,
                "faithfulness":      faithfulness,
                "answer_relevancy":  answer_relevancy,
                "context_precision": context_precision,
                "context_recall":    context_recall,
                "overall":           overall,
            }
            all_scores.append(scores)
            results.append(scores)

            print(f"\n[{i+1}] {question}")
            print(f"     Faithfulness      : {faithfulness:.2f}")
            print(f"     Answer Relevancy  : {answer_relevancy:.2f}")
            print(f"     Context Precision : {context_precision:.2f}")
            print(f"     Context Recall    : {context_recall:.2f}")
            print(f"     Overall           : {overall:.2f}")

            # Small delay to avoid Groq rate limits
            time.sleep(0.5)

        # -- Aggregate scores
        avg = {
            "faithfulness":      round(sum(s["faithfulness"]      for s in all_scores) / len(all_scores), 4),
            "answer_relevancy":  round(sum(s["answer_relevancy"]  for s in all_scores) / len(all_scores), 4),
            "context_precision": round(sum(s["context_precision"] for s in all_scores) / len(all_scores), 4),
            "context_recall":    round(sum(s["context_recall"]    for s in all_scores) / len(all_scores), 4),
        }
        avg["overall_score"] = round(sum(avg.values()) / len(avg), 4)

        # -- Log to MLflow
        for metric, score in avg.items():
            mlflow.log_metric(metric, score)

        # -- Save results
        output_path = Path("evaluation") / f"eval_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        mlflow.log_artifact(str(output_path))

        # -- Final report
        print("\n" + "="*60)
        print("📊 FINAL EVALUATION REPORT")
        print("="*60)
        print(f"  Faithfulness      : {avg['faithfulness']:.4f}  (hallucination check)")
        print(f"  Answer Relevancy  : {avg['answer_relevancy']:.4f}  (on-topic answers)")
        print(f"  Context Precision : {avg['context_precision']:.4f}  (retrieval quality)")
        print(f"  Context Recall    : {avg['context_recall']:.4f}  (coverage)")
        print("-"*60)
        print(f"  Overall Score     : {avg['overall_score']:.4f}")
        print("="*60)
        print(f"\n✅ Results saved to {output_path}")
        print("✅ All scores logged to MLflow")

        return avg


if __name__ == "__main__":
    run_evaluation()