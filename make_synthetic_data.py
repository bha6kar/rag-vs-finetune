"""Give knowledge fine-tuning its best fair shot.

The first fine-tune trained on only 37 question/answer pairs and overfit. This
builds a real corpus: it asks the judge model to write self-contained Q/A pairs
from the SAME document chunks the RAG pipeline retrieves from, so fine-tuning and
RAG both get the document content. Fine-tuning bakes it into weights; RAG
retrieves it. Fair fight.

Robust by design: writes each pair to work/mlx/train.jsonl as it goes (so a
stall or rate-limit never loses work and `wc -l` shows live progress), bounded
retries + timeout so the Azure dev resource can't hang the run, flushed logging.
The held-out test 17 (splits.json) is never touched.
"""
from __future__ import annotations

import json
import sys
import time

from openai import AzureOpenAI

import azure_client as az
from config import BAKEOFF, CLOSED_BOOK_SYSTEM, WORK

DOCS = ["apple", "berkshire", "amd", "boeing"]
MAX_CHUNKS_PER_DOC = 45
QA_PER_CHUNK = 3
MIN_CHARS = 350
MLX = WORK / "mlx"
RAW = MLX / "train.jsonl"
VALID = MLX / "valid.jsonl"

GEN_PROMPT = """You are creating training data from an excerpt of a financial filing ({doc}).

Write {n} self-contained question/answer pairs answerable ONLY from the excerpt.
Prefer specific facts: figures with units and periods, named items, concrete
disclosures. Each question must stand alone (name the company and period). Each
answer must be a short factual statement grounded in the excerpt.

Return ONLY JSON: {{"pairs": [{{"q": "...", "a": "..."}}, ...]}}

Excerpt:
{chunk}"""


def log(m):
    print(m, flush=True)


def chunks_for(doc):
    path = BAKEOFF / "results" / f"baseline_chunks_{doc}_tuned.jsonl"
    out = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    return [c for c in out if len(c.get("text", "")) >= MIN_CHARS][:MAX_CHUNKS_PER_DOC]


def chat_row(q, a):
    return {"messages": [
        {"role": "system", "content": CLOSED_BOOK_SYSTEM},
        {"role": "user", "content": q},
        {"role": "assistant", "content": a},
    ]}


def main():
    # Tight client: short timeout, 1 retry, so a rate-limit surfaces fast.
    client = AzureOpenAI(azure_endpoint=az.AZURE_ENDPOINT, api_key=az.AZURE_KEY,
                         api_version=az.API_VERSION, timeout=30.0, max_retries=1)
    dep = az.require_deployment(az.CHAT_DEPLOYMENT)

    MLX.mkdir(parents=True, exist_ok=True)
    RAW.write_text("")              # fresh start
    total = 0
    for doc in DOCS:
        chunks = chunks_for(doc)
        made = 0
        for i, c in enumerate(chunks):
            try:
                r = client.chat.completions.create(
                    model=dep,
                    messages=[{"role": "user", "content": GEN_PROMPT.format(
                        doc=doc, n=QA_PER_CHUNK, chunk=c["text"][:4000])}],
                    response_format={"type": "json_object"},
                    max_completion_tokens=az.MAX_COMPLETION_TOKENS,
                )
                pairs = json.loads(r.choices[0].message.content).get("pairs", [])
            except Exception as e:
                log(f"  [{doc}] chunk {i}: {type(e).__name__} (skip)")
                time.sleep(1.0)
                continue
            with RAW.open("a") as f:
                for p in pairs:
                    if p.get("q") and p.get("a"):
                        f.write(json.dumps(chat_row(p["q"].strip(), p["a"].strip())) + "\n")
                        made += 1
                        total += 1
            if (i + 1) % 10 == 0:
                log(f"  [{doc}] {i+1}/{len(chunks)} chunks, {total} pairs total")
        log(f"{doc}: +{made} pairs")

    # Carve a small valid split from the tail.
    lines = RAW.read_text().splitlines()
    n_valid = max(5, round(len(lines) * 0.05))
    VALID.write_text("\n".join(lines[:n_valid]))
    RAW.write_text("\n".join(lines[n_valid:]))
    log(f"\nDONE: {len(lines)-n_valid} train / {n_valid} valid synthetic Q/A. test split untouched.")


if __name__ == "__main__":
    sys.exit(main())
