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

# RAG prompt and context reconstruction are shared with eval_all.py.
RAG_PROMPT = E.RAG_PROMPT
rag_context = E.rag_context


def openrouter() -> OpenAI:
    key = (Path.home() / ".secrets" / "openrouter_api_key").read_text().strip()
    return OpenAI(base_url="https://openrouter.ai/api/v1", api_key=key)


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
