"""Score the held-out finance questions across conditions and write a report.

Conditions:
  rag           the bake-off's tuned gpt-5.4-mini + emb-large, pulled per-question
                from its already-scored runs (the WITH-RAG number)
  base_local    local base model, closed-book (no context)
  ft_local      local LoRA fine-tune, closed-book
  base_openai   hosted base gpt-5.4-mini, closed-book
  ft_openai     hosted fine-tuned gpt-5.4-mini, closed-book

Same questions, same judge (gpt-5.4-mini, correct + faithful) for every
generated condition. Results accumulate into work/results.json so the OpenAI
conditions can be added after the async fine-tune finishes, then the report and
chart are rebuilt from whatever is present.

Usage:
  python eval_all.py --conditions rag,base_local,ft_local
  python eval_all.py --conditions base_openai,ft_openai --ft-model ft:gpt-5.4-mini:...
"""
from __future__ import annotations

import argparse
import json
import sys

import azure_client as az
from config import (BAKEOFF, CLOSED_BOOK_SYSTEM, LOCAL_ADAPTER, LOCAL_BASE,
                    RAG_CONFIG, WORK)
from load_questions import load_all

RESULTS = WORK / "results.json"

JUDGE_PROMPT = """You are evaluating a QA system's answer against a known ground truth.

Question: {question}

Ground truth: {ground_truth}

System answer: {answer}

Score the system answer on two binary criteria:
- "correct": 1 if it agrees with ground truth on the substantive facts. Minor
  wording differences or extra detail are OK. Wrong number, missing key fact,
  fabricated claim, or "I don't know" / "Not in document" when ground truth has
  the answer = 0.
- "faithful": 1 if every claim in the answer is plausibly grounded (no obvious
  hallucinations). 0 if it invents numbers or contradicts ground truth.

Return ONLY JSON: {{"correct": 0 or 1, "faithful": 0 or 1, "reason": "<one sentence>"}}"""


def test_questions():
    ids = set(json.loads((WORK / "splits.json").read_text())["test"])
    return [q for q in load_all() if q["id"] in ids]


def judge(client, q, answer):
    r = client.chat.completions.create(
        model=az.require_deployment(az.JUDGE_DEPLOYMENT),
        messages=[{"role": "user", "content": JUDGE_PROMPT.format(
            question=q["question"], ground_truth=q["ground_truth"], answer=answer)}],
        response_format={"type": "json_object"},
        max_completion_tokens=az.MAX_COMPLETION_TOKENS,
    )
    return json.loads(r.choices[0].message.content)


# ---- generators -------------------------------------------------------------

def gen_azure(client, deployment):
    """Closed-book generation through an Azure deployment (base or fine-tuned)."""
    dep = az.require_deployment(deployment)

    def g(q):
        r = client.chat.completions.create(
            model=dep,
            messages=[{"role": "system", "content": CLOSED_BOOK_SYSTEM},
                      {"role": "user", "content": q["question"]}],
            max_completion_tokens=az.MAX_COMPLETION_TOKENS,
        )
        return (r.choices[0].message.content or "").strip()
    return g


def gen_mlx(adapter):
    from mlx_lm import generate, load
    model, tok = load(LOCAL_BASE, adapter_path=adapter)

    def g(q):
        prompt = tok.apply_chat_template(
            [{"role": "system", "content": CLOSED_BOOK_SYSTEM},
             {"role": "user", "content": q["question"]}],
            tokenize=False, add_generation_prompt=True)
        try:
            return generate(model, tok, prompt=prompt, max_tokens=200, verbose=False).strip()
        except TypeError:
            return generate(model, tok, prompt, max_tokens=200).strip()
    return g


# ---- conditions -------------------------------------------------------------

def run_generated(label, gen, qs, client):
    rows = []
    for i, q in enumerate(qs, 1):
        ans = gen(q)
        v = judge(client, q, ans)
        rows.append({"id": q["id"], "doc": q["doc"], "bucket": q["bucket"],
                     "answer": ans, "correct": v["correct"], "faithful": v["faithful"]})
        print(f"  [{label}] {i}/{len(qs)} {q['id']}: correct={v['correct']} faithful={v['faithful']}")
    return rows


def run_rag(qs):
    """Pull per-question correctness from the bake-off's tuned scored runs."""
    sys.path.insert(0, str(BAKEOFF))
    verdict = {}
    for doc in {q["doc"] for q in qs}:
        path = BAKEOFF / "results" / f"runs_{doc}_{RAG_CONFIG}_scored.jsonl"
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            verdict[r["id"]] = r["judge"]
    rows = []
    for q in qs:
        j = verdict.get(q["id"])
        if j is None:
            print(f"  [rag] WARNING no scored run for {q['id']}")
            continue
        rows.append({"id": q["id"], "doc": q["doc"], "bucket": q["bucket"],
                     "answer": "(RAG pipeline answer)",
                     "correct": int(j["correct"]), "faithful": int(j["faithful"])})
    return rows


def aggregate(rows):
    n = len(rows)
    return {"n": n,
            "correct": sum(r["correct"] for r in rows),
            "faithful": sum(r["faithful"] for r in rows),
            "acc": sum(r["correct"] for r in rows) / n if n else 0,
            "faith": sum(r["faithful"] for r in rows) / n if n else 0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conditions", required=True)
    ap.add_argument("--ft-model", default=None, help="hosted fine-tuned model id")
    args = ap.parse_args()
    want = args.conditions.split(",")

    qs = test_questions()
    print(f"test set: {len(qs)} held-out questions")
    client = az.client()

    store = json.loads(RESULTS.read_text()) if RESULTS.exists() else {}

    for cond in want:
        print(f"\n=== {cond} ===")
        if cond == "rag":
            rows = run_rag(qs)
        elif cond == "base_local":
            rows = run_generated(cond, gen_mlx(None), qs, client)
        elif cond == "ft_local":
            rows = run_generated(cond, gen_mlx(LOCAL_ADAPTER), qs, client)
        elif cond == "base_azure":
            rows = run_generated(cond, gen_azure(client, az.CHAT_DEPLOYMENT), qs, client)
        elif cond == "ft_azure":
            if not args.ft_model:
                sys.exit("ft_azure needs --ft-model (the fine-tuned deployment name)")
            rows = run_generated(cond, gen_azure(client, args.ft_model), qs, client)
        else:
            sys.exit(f"unknown condition {cond}")
        store[cond] = {"rows": rows, **aggregate(rows)}
        RESULTS.write_text(json.dumps(store, indent=2))
        a = store[cond]
        print(f"  -> {cond}: correct {a['correct']}/{a['n']} ({a['acc']:.0%}), "
              f"faithful {a['faith']:.0%}")

    print(f"\nsaved {RESULTS}")
    print("now run: python make_report.py")


if __name__ == "__main__":
    main()
