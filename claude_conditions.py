"""Add Claude conditions to the comparison, via OpenRouter (no Anthropic key
present; OpenRouter proxies Claude). Claude CANNOT be fine-tuned (Anthropic
offers no fine-tuning), so only two conditions exist:

  claude_base   Claude closed-book, no context
  claude_rag    Claude + the SAME retrieved context gpt-5.4-mini's RAG used

Generation is Claude via OpenRouter; judging stays on Azure gpt-5.4-mini, same
judge as every other condition. Tier-matched to gpt-5.4-mini with a small Claude
(claude-haiku-4.5); override with CLAUDE_MODEL.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from openai import OpenAI

import eval_all as E
from config import BAKEOFF, CLOSED_BOOK_SYSTEM, RAG_CONFIG, WORK

CLAUDE_MODEL = os.environ.get("CLAUDE_MODEL", "anthropic/claude-haiku-4.5")
RESULTS = WORK / "results.json"

# The bake-off's exact answer prompt, so claude_rag mirrors the RAG pipeline.
RAG_PROMPT = """You are a careful financial-document QA assistant.
Answer the question using ONLY the provided context. If the answer is not
present in the context, reply exactly: Not in document.

Be concise. When the answer is a number, include the unit and the period.
When citing, mention the section name.

Question: {question}
Context: {context}
Answer:"""


def openrouter() -> OpenAI:
    key = (Path.home() / ".secrets" / "openrouter_api_key").read_text().strip()
    return OpenAI(base_url="https://openrouter.ai/api/v1", api_key=key)


def _retrieved_map(doc: str) -> dict:
    path = BAKEOFF / "results" / f"runs_{doc}_{RAG_CONFIG}_scored.jsonl"
    out = {}
    for line in path.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            out[r["id"]] = r["retrieved"]
    return out


def _chunks(doc: str) -> dict:
    # Normalize ids to str: the chunk store and the runs' retrieved indices can
    # be int or str across files, so key everything by str to match reliably.
    path = BAKEOFF / "results" / f"baseline_chunks_{doc}_{RAG_CONFIG}.jsonl"
    return {str(c["id"]): c["text"]
            for c in (json.loads(l) for l in path.read_text().splitlines() if l.strip())}


def rag_context(q: dict) -> str:
    """Reconstruct the exact context the tuned gpt-5.4-mini RAG pipeline used."""
    ids = _retrieved_map(q["doc"]).get(q["id"], [])
    chunks = _chunks(q["doc"])
    return "\n\n".join(chunks[str(i)] for i in ids if str(i) in chunks)


def gen_closedbook(client):
    def g(q):
        r = client.chat.completions.create(model=CLAUDE_MODEL, max_tokens=600,
            messages=[{"role": "system", "content": CLOSED_BOOK_SYSTEM},
                      {"role": "user", "content": q["question"]}])
        return (r.choices[0].message.content or "").strip()
    return g


def gen_rag(client):
    def g(q):
        ctx = rag_context(q)
        r = client.chat.completions.create(model=CLAUDE_MODEL, max_tokens=600,
            messages=[{"role": "user", "content": RAG_PROMPT.format(
                question=q["question"], context=ctx)}])
        return (r.choices[0].message.content or "").strip()
    return g


def main():
    qs = E.test_questions()
    print(f"Claude model: {CLAUDE_MODEL}  |  test set: {len(qs)} questions")
    judge_client = E.az.client()       # Azure judge, same as everything else
    orc = openrouter()

    store = json.loads(RESULTS.read_text()) if RESULTS.exists() else {}
    for cond, gen in [("claude_base", gen_closedbook(orc)), ("claude_rag", gen_rag(orc))]:
        print(f"\n=== {cond} ===")
        rows = E.run_generated(cond, gen, qs, judge_client)
        store[cond] = {"rows": rows, **E.aggregate(rows)}
        RESULTS.write_text(json.dumps(store, indent=2))
        a = store[cond]
        print(f"  -> {cond}: correct {a['correct']}/{a['n']} ({a['acc']:.0%}), faithful {a['faith']:.0%}")
    print("\ndone; run: python make_report.py")


if __name__ == "__main__":
    main()
