"""Render the experiment results (work/results.json) as a markdown table and a
chart, in the bake-off's visual style. Only renders conditions that are present,
so it works before and after the Azure fine-tune lands.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

from config import WORK

RESULTS = WORK / "results.json"
OUT_MD = WORK / "report.md"
OUT_CHART = Path(__file__).resolve().parent / "images" / "experiment_results.png"
OUT_CHART.parent.mkdir(parents=True, exist_ok=True)

PAGE_BG = "#fafbfc"
INK = "#111827"
MUTED = "#6b7280"
GRID = "#e5e7eb"

LABELS = {
    "rag":         "gpt-5.4-mini + RAG (retrieval)",
    "claude_rag":  "Claude Haiku 4.5 + RAG (retrieval)",
    "local_rag":   "Qwen2.5-7B + RAG (retrieval)",
    "base_azure":  "gpt-5.4-mini, closed-book (no fine-tune)",
    "claude_base": "Claude Haiku 4.5, closed-book (no fine-tune)",
    "ft_azure":    "gpt-5.4-mini, fine-tuned, closed-book",
    "base_local":  "Qwen2.5-7B, closed-book (no fine-tune)",
    "ft_local":    "Qwen2.5-7B, fine-tuned, closed-book",
}
ORDER = ["rag", "claude_rag", "local_rag", "base_azure", "claude_base", "ft_azure", "base_local", "ft_local"]
COLORS = {
    "rag": "#065f46", "claude_rag": "#047857", "local_rag": "#10b981",
    "base_azure": "#fca5a5", "claude_base": "#f0abfc", "ft_azure": "#2563eb",
    "base_local": "#fdba74", "ft_local": "#1d4ed8",
}


def main() -> None:
    store = json.loads(RESULTS.read_text())
    present = [c for c in ORDER if c in store]

    # Markdown table
    lines = ["# Experiment: RAG vs fine-tuning on the same finance questions", "",
             f"Held-out test set: {store[present[0]]['n']} questions. "
             "Judge: gpt-5.4-mini (correct + faithful), same as the bake-off.", "",
             "| Condition | Correct | Faithful |", "|---|---|---|"]
    for c in present:
        a = store[c]
        lines.append(f"| {LABELS[c]} | {a['correct']}/{a['n']} ({a['acc']:.0%}) "
                     f"| {a['faithful']}/{a['n']} ({a['faith']:.0%}) |")
    OUT_MD.write_text("\n".join(lines) + "\n")
    print("wrote", OUT_MD)

    # Chart: grouped bars, correctness + faithfulness. Two fixed colours so the
    # legend matches every bar (one colour = correct, one = faithful).
    CORRECT_C = "#0f766e"   # teal-700
    FAITHFUL_C = "#5eead4"  # teal-300
    fig, ax = plt.subplots(figsize=(11, 0.9 * len(present) + 2))
    ys = range(len(present))
    accs = [store[c]["acc"] * 100 for c in present]
    faiths = [store[c]["faith"] * 100 for c in present]
    h = 0.38
    ax.barh([y + h / 2 for y in ys], accs, height=h, color=CORRECT_C, label="Correct %")
    ax.barh([y - h / 2 for y in ys], faiths, height=h, color=FAITHFUL_C, label="Faithful %")
    for y, c in zip(ys, present):
        ax.text(store[c]["acc"] * 100 + 1, y + h / 2,
                f"{store[c]['acc']:.0%}", va="center", fontsize=10, color=INK, fontweight="bold")
        ax.text(store[c]["faith"] * 100 + 1, y - h / 2,
                f"{store[c]['faith']:.0%}", va="center", fontsize=9, color=MUTED)
    ax.set_yticks(list(ys))
    ax.set_yticklabels([LABELS[c] for c in present], fontsize=10, color=INK)
    ax.set_xlim(0, 108)
    ax.set_xlabel("Percent of held-out questions", color=MUTED)
    ax.set_title("Same questions, same judge: retrieve into context vs fine-tune into weights",
                 loc="left", color=INK, fontsize=14, fontweight="bold", pad=12)
    ax.invert_yaxis()
    for s in ["top", "right", "left"]:
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(length=0)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    fig.patch.set_facecolor(PAGE_BG)
    ax.set_facecolor(PAGE_BG)
    fig.tight_layout()
    fig.savefig(OUT_CHART, bbox_inches="tight", facecolor=PAGE_BG, dpi=200)
    print("wrote", OUT_CHART)


if __name__ == "__main__":
    main()
