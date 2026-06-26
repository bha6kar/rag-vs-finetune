"""LoRA fine-tune for the behavioural (format) task, via MLX on Apple Silicon.

Trains the open model to always emit the strict schema. Format is low-entropy,
so this needs fewer iterations than the knowledge fine-tune.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from config import LOCAL_BASE, WORK

DATA = WORK / "fmt"
ADAPTER = str(WORK / "fmt_adapter")


def main():
    if not (DATA / "train.jsonl").exists():
        raise SystemExit("no work/fmt/train.jsonl; run make_format_data.py first")
    Path(ADAPTER).mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "-m", "mlx_lm", "lora",
        "--model", LOCAL_BASE, "--train", "--data", str(DATA),
        "--iters", "250", "--batch-size", "4", "--num-layers", "12",
        "--learning-rate", "1e-5", "--fine-tune-type", "lora",
        "--adapter-path", ADAPTER,
    ]
    print("running:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    print("format adapter saved to", ADAPTER, flush=True)


if __name__ == "__main__":
    main()
