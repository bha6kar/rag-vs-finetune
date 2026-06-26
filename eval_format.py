"""Behavioural experiment: does fine-tuning beat prompting at strict output
format, with RAG supplying the facts?

Both conditions answer the held-out 17 WITH the same retrieved context and must
emit the schema. We measure schema adherence (strict) and answer correctness
(same judge as everything else).

  fmt_base   Qwen2.5-7B + RAG context + a strong format prompt (prompt-only)
  fmt_ft     Qwen2.5-7B fine-tuned for the schema + RAG context

Writes work/format_results.json and images/format_results.png.
"""
from __future__ import annotations

import json
from pathlib import Path

import eval_all as E
import schema as S
from config import WORK

ADAPTER = str(WORK / "fmt_adapter")
OUT = WORK / "format_results.json"


def gen_format(adapter):
    from mlx_lm import generate, load
    from config import LOCAL_BASE
    model, tok = load(LOCAL_BASE, adapter_path=adapter)

    def g(q):
        user = f"{S.FORMAT_INSTRUCTION}\n\nContext: {E.rag_context(q)}\n\nQuestion: {q['question']}"
        prompt = tok.apply_chat_template([{"role": "user", "content": user}],
                                         tokenize=False, add_generation_prompt=True)
        try:
            return generate(model, tok, prompt=prompt, max_tokens=400, verbose=False).strip()
        except TypeError:
            return generate(model, tok, prompt, max_tokens=400).strip()
    return g


def run(label, gen, qs, client):
    adh = cor = 0
    rows = []
    for i, q in enumerate(qs, 1):
        raw = gen(q)
        ok, reason = S.validate(raw)
        v = E.judge(client, q, S.to_answer_text(raw))
        adh += ok
        cor += v["correct"]
        rows.append({"id": q["id"], "adherent": int(ok), "reason": reason,
                     "correct": v["correct"], "raw": raw[:300]})
        print(f"  [{label}] {i}/{len(qs)} {q['id']}: adherent={int(ok)} correct={v['correct']}", flush=True)
    n = len(qs)
    return {"n": n, "adherence": adh / n, "correct": cor / n,
            "adh_count": adh, "cor_count": cor, "rows": rows}


def chart(store):
    import matplotlib.pyplot as plt
    labels = ["Prompt-only\n(Qwen-7B + RAG)", "Fine-tuned\n(Qwen-7B + RAG)"]
    keys = ["fmt_base", "fmt_ft"]
    adh = [store[k]["adherence"] * 100 for k in keys]
    cor = [store[k]["correct"] * 100 for k in keys]
    x = range(len(keys))
    fig, ax = plt.subplots(figsize=(9, 5.5))
    w = 0.36
    b1 = ax.bar([i - w/2 for i in x], adh, w, color="#2563eb", label="Schema adherence %")
    b2 = ax.bar([i + w/2 for i in x], cor, w, color="#10b981", label="Answer correctness %")
    for b in list(b1) + list(b2):
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 1.5,
                f"{b.get_height():.0f}%", ha="center", fontsize=11, fontweight="bold")
    ax.set_xticks(list(x)); ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylim(0, 112); ax.set_ylabel("Percent of held-out questions", color="#6b7280")
    ax.set_title("Strict JSON format: a good prompt already aces it; fine-tuning does not help",
                 loc="left", fontsize=12.5, fontweight="bold", pad=12)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color("#e5e7eb"); ax.spines["bottom"].set_color("#e5e7eb")
    ax.tick_params(length=0); ax.grid(axis="y", color="#e5e7eb", lw=0.8); ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=10)
    fig.patch.set_facecolor("#fafbfc"); ax.set_facecolor("#fafbfc")
    fig.tight_layout()
    out = Path(__file__).resolve().parent / "images" / "format_results.png"
    fig.savefig(out, bbox_inches="tight", facecolor="#fafbfc", dpi=200)
    print("wrote", out, flush=True)


def main():
    qs = E.test_questions()
    client = E.az.client()
    store = json.loads(OUT.read_text()) if OUT.exists() else {}
    print("=== fmt_base (prompt-only) ===", flush=True)
    store["fmt_base"] = run("fmt_base", gen_format(None), qs, client)
    OUT.write_text(json.dumps(store, indent=2))
    print("=== fmt_ft (fine-tuned) ===", flush=True)
    store["fmt_ft"] = run("fmt_ft", gen_format(ADAPTER), qs, client)
    OUT.write_text(json.dumps(store, indent=2))
    for k in ["fmt_base", "fmt_ft"]:
        s = store[k]
        print(f"{k}: adherence {s['adh_count']}/{s['n']} ({s['adherence']:.0%}), "
              f"correct {s['cor_count']}/{s['n']} ({s['correct']:.0%})", flush=True)
    chart(store)


if __name__ == "__main__":
    main()
