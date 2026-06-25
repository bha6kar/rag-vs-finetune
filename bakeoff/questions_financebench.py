"""FinanceBench question loader (https://arxiv.org/abs/2311.11944, PatronusAI).

Maps the FinanceBench schema to our internal question schema:
  {id, bucket, question, ground_truth}

Buckets are inferred from FinanceBench's `question_reasoning` field.
"""
from __future__ import annotations

import json
from pathlib import Path

DATA = Path(__file__).resolve().parent / "data" / "financebench"
QUESTIONS_FILE = DATA / "questions.jsonl"


REASONING_TO_BUCKET = {
    "Information extraction": "lookup",
    "Numerical reasoning": "numeric_table",
    "Logical reasoning": "multi_section",
    "Information extraction (and computation)": "numeric_table",
}


def load_for_doc(doc_name: str) -> list[dict]:
    """Load all FinanceBench questions for a given doc_name (e.g. 'AMD_2022_10K')."""
    out = []
    for line in QUESTIONS_FILE.open():
        q = json.loads(line)
        if q["doc_name"] != doc_name:
            continue
        bucket = REASONING_TO_BUCKET.get(q.get("question_reasoning", ""), "multi_section")
        out.append({
            "id": q["financebench_id"].replace("financebench_id_", "FB"),
            "bucket": bucket,
            "question": q["question"],
            "ground_truth": q["answer"],
        })
    if not out:
        raise ValueError(f"no FinanceBench questions for {doc_name}")
    return out


# Pre-built bindings used by the runner scripts. Add more as needed.
DOCS = {
    "amd":    "AMD_2022_10K",
    "boeing": "BOEING_2022_10K",
}


def for_alias(alias: str) -> list[dict]:
    if alias not in DOCS:
        raise KeyError(f"unknown FinanceBench alias '{alias}', try {list(DOCS)}")
    return load_for_doc(DOCS[alias])
