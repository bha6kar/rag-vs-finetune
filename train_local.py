"""Local LoRA fine-tune on Apple Silicon via MLX. Offline: no API, no Azure.

Trains a small adapter on the closed-book finance Q/A train split so we can then
test whether the knowledge made it into the weights (eval_all.py).
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from config import LOCAL_ADAPTER, LOCAL_BASE, WORK

DATA = WORK / "mlx"


def main() -> None:
    if not (DATA / "train.jsonl").exists():
        raise SystemExit("no work/mlx/train.jsonl; run build_data.py first")
    Path(LOCAL_ADAPTER).mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "-m", "mlx_lm", "lora",
        "--model", LOCAL_BASE,
        "--train",
        "--data", str(DATA),
        "--iters", "200",
        "--batch-size", "4",
        "--num-layers", "8",
        "--fine-tune-type", "lora",
        "--adapter-path", LOCAL_ADAPTER,
    ]
    print("running:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    print("adapter saved to", LOCAL_ADAPTER, flush=True)


if __name__ == "__main__":
    main()
