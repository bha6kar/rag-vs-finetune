"""Shared config for the RAG-vs-fine-tuning knowledge experiment.

The experiment: fine-tune a model on financial-document Q/A, then answer the
held-out questions CLOSED-BOOK (no retrieved context) and compare against the
same questions answered WITH RAG (the bake-off's tuned gpt-5.4-mini pipeline).
Same questions, same judge, so the only variable is fine-tune-into-weights
versus retrieve-into-context.
"""
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
# Standalone repo: the slice of bake-off data this experiment needs (questions,
# tuned scored runs, chunk stores) is vendored under ./bakeoff/.
BAKEOFF = HERE / "bakeoff"
WORK = HERE / "work"

# Local (free, Apple Silicon) base. 7B-4bit runs comfortably on a 48 GB M4 Pro
# and is a far better grounded extractor than 3B (3B abstains on long context).
LOCAL_BASE = os.environ.get("LOCAL_BASE", "mlx-community/Qwen2.5-7B-Instruct-4bit")
LOCAL_ADAPTER = str(WORK / "mlx_adapter")

# Hosted (paid) base: the SAME model the RAG side used, for an exact comparison.
OPENAI_BASE = os.environ.get("OPENAI_BASE", "gpt-5.4-mini")
JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "gpt-5.4-mini")

# RAG condition: the bake-off's tuned pipeline (gpt-5.4-mini + emb-large), pulled
# per-question from its already-scored runs so the comparison is exact.
RAG_CONFIG = "tuned"

SEED = 13

# Closed-book: the model must answer from its own parameters, no context.
CLOSED_BOOK_SYSTEM = (
    "You are a financial-document QA assistant. Answer from your own knowledge. "
    "Be concise. When the answer is a number, include the unit and the period. "
    "If you do not know the answer, reply exactly: I don't know."
)
