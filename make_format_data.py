"""Training data for the behavioural (format) fine-tune.

For each sampled document chunk, ask gpt-5.4-mini to produce a question and a
GOLD schema object answerable from that chunk. The training example is
(format instruction + question + chunk as context) -> the schema JSON. This
teaches the model to always emit the rigid schema while reading from context,
exactly the inference setup. Writes work/fmt/{train,valid}.jsonl incrementally.
"""
from __future__ import annotations

import json
import time

from openai import AzureOpenAI

import azure_client as az
from config import BAKEOFF, WORK
from schema import ALLOWED_CONFIDENCE, FORMAT_INSTRUCTION, REQUIRED_KEYS

DOCS = ["apple", "berkshire", "amd", "boeing"]
MAX_CHUNKS_PER_DOC = 60
MIN_CHARS = 350
FMT = WORK / "fmt"
TRAIN = FMT / "train.jsonl"
VALID = FMT / "valid.jsonl"

GEN_PROMPT = """From this excerpt of a financial filing ({doc}), write ONE question
answerable only from it and a gold answer as a strict JSON object with exactly
these keys: answer (string), figure (string or null), unit (string or null),
period (string or null), section (string or null), confidence (high|medium|low).
Fill figure/unit/period when the answer is a number; else null.

Return ONLY JSON: {{"question": "...", "gold": {{"answer": "...", "figure": ..., "unit": ..., "period": ..., "section": ..., "confidence": "..."}}}}

Excerpt:
{chunk}"""


def chunks_for(doc):
    path = BAKEOFF / "results" / f"baseline_chunks_{doc}_tuned.jsonl"
    out = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    return [c for c in out if len(c.get("text", "")) >= MIN_CHARS][:MAX_CHUNKS_PER_DOC]


def valid_gold(g):
    return (isinstance(g, dict)
            and all(k in g for k in REQUIRED_KEYS)
            and not [k for k in g if k not in REQUIRED_KEYS]
            and g.get("confidence") in ALLOWED_CONFIDENCE)


def row(question, chunk, gold):
    user = f"{FORMAT_INSTRUCTION}\n\nContext: {chunk}\n\nQuestion: {question}"
    return {"messages": [
        {"role": "user", "content": user},
        {"role": "assistant", "content": json.dumps(gold, ensure_ascii=False)},
    ]}


def main():
    client = AzureOpenAI(azure_endpoint=az.AZURE_ENDPOINT, api_key=az.AZURE_KEY,
                         api_version=az.API_VERSION, timeout=30.0, max_retries=1)
    dep = az.require_deployment(az.CHAT_DEPLOYMENT)
    FMT.mkdir(parents=True, exist_ok=True)
    TRAIN.write_text("")
    total = 0
    for doc in DOCS:
        chunks = chunks_for(doc)
        for i, c in enumerate(chunks):
            try:
                r = client.chat.completions.create(
                    model=dep,
                    messages=[{"role": "user", "content": GEN_PROMPT.format(
                        doc=doc, chunk=c["text"][:4000])}],
                    response_format={"type": "json_object"},
                    max_completion_tokens=az.MAX_COMPLETION_TOKENS)
                obj = json.loads(r.choices[0].message.content)
                q, gold = obj.get("question"), obj.get("gold")
            except Exception as e:
                print(f"  [{doc}] chunk {i}: {type(e).__name__} (skip)", flush=True)
                time.sleep(1.0)
                continue
            if q and valid_gold(gold):
                with TRAIN.open("a") as f:
                    f.write(json.dumps(row(q.strip(), c["text"][:4000], gold)) + "\n")
                total += 1
            if (i + 1) % 15 == 0:
                print(f"  [{doc}] {i+1}/{len(chunks)} chunks, {total} examples", flush=True)
        print(f"{doc}: total {total}", flush=True)

    lines = TRAIN.read_text().splitlines()
    n_valid = max(5, round(len(lines) * 0.05))
    VALID.write_text("\n".join(lines[:n_valid]))
    TRAIN.write_text("\n".join(lines[n_valid:]))
    print(f"\nDONE: {len(lines)-n_valid} train / {n_valid} valid format examples.", flush=True)


if __name__ == "__main__":
    main()
