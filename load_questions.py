"""Load all 54 bake-off finance questions uniformly: {id, doc, bucket, question,
ground_truth}. Single source so train, eval, and RAG-lookup all agree."""
from __future__ import annotations

from bakeoff.questions import QUESTIONS as APPLE
from bakeoff.questions_brk import QUESTIONS as BERKSHIRE
from bakeoff.questions_financebench import load_for_doc


def load_all() -> list[dict]:
    out = []
    for q in APPLE:
        out.append({**q, "doc": "apple"})
    for q in BERKSHIRE:
        out.append({**q, "doc": "berkshire"})
    for q in load_for_doc("AMD_2022_10K"):
        out.append({**q, "doc": "amd"})
    for q in load_for_doc("BOEING_2022_10K"):
        out.append({**q, "doc": "boeing"})
    return out


if __name__ == "__main__":
    qs = load_all()
    print(f"{len(qs)} questions")
