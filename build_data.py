"""Split the 54 finance questions into train/test (stratified by document) and
write closed-book fine-tuning files for both the local MLX path and the hosted
OpenAI path.

Closed-book training target: (system, question) -> ground-truth answer, with NO
context. We are deliberately trying to push document knowledge into the weights,
which is exactly the thing the literature says is hard. The held-out test split
is what we score.
"""
from __future__ import annotations

import json
import random
from collections import defaultdict

from config import CLOSED_BOOK_SYSTEM, SEED, WORK
from load_questions import load_all

MLX = WORK / "mlx"
OPENAI = WORK / "openai"
for d in (MLX, OPENAI):
    d.mkdir(parents=True, exist_ok=True)


def chat_row(q: dict) -> dict:
    return {"messages": [
        {"role": "system", "content": CLOSED_BOOK_SYSTEM},
        {"role": "user", "content": q["question"]},
        {"role": "assistant", "content": q["ground_truth"]},
    ]}


def main() -> None:
    qs = load_all()
    rng = random.Random(SEED)

    # Stratify by doc so the test set covers every document.
    by_doc = defaultdict(list)
    for q in qs:
        by_doc[q["doc"]].append(q)
    train, test = [], []
    for doc, items in by_doc.items():
        items = items[:]
        rng.shuffle(items)
        k = max(1, round(len(items) * 0.33))   # ~1/3 held out per doc
        test += items[:k]
        train += items[k:]
    rng.shuffle(train)
    rng.shuffle(test)

    splits = {"train": [q["id"] for q in train], "test": [q["id"] for q in test]}
    (WORK / "splits.json").write_text(json.dumps(splits, indent=2))

    # MLX: train.jsonl + a small valid.jsonl carved from train for val loss.
    n_valid = max(2, round(len(train) * 0.1))
    mlx_valid, mlx_train = train[:n_valid], train[n_valid:]
    (MLX / "train.jsonl").write_text("\n".join(json.dumps(chat_row(q)) for q in mlx_train))
    (MLX / "valid.jsonl").write_text("\n".join(json.dumps(chat_row(q)) for q in mlx_valid))
    (MLX / "test.jsonl").write_text("\n".join(json.dumps(chat_row(q)) for q in test))

    # OpenAI hosted FT: one chat-format file (it makes its own val split).
    (OPENAI / "train.jsonl").write_text("\n".join(json.dumps(chat_row(q)) for q in train))

    print(f"{len(qs)} questions -> {len(train)} train / {len(test)} test")
    print(f"  MLX:    {len(mlx_train)} train / {len(mlx_valid)} valid / {len(test)} test")
    print(f"  OpenAI: {len(train)} train rows -> {OPENAI / 'train.jsonl'}")
    print(f"  splits -> {WORK / 'splits.json'}")
    from collections import Counter
    print("  test by doc:", dict(Counter(q["doc"] for q in test)))


if __name__ == "__main__":
    main()
