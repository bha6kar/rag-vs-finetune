set -e
cd /Users/onyx/crasher/rag-vs-finetune
while pgrep -f make_format_data >/dev/null; do sleep 12; done
echo "[fmt] corpus: $(wc -l < work/fmt/train.jsonl) train"
echo "[fmt] training format adapter..."
uv run python train_format.py
echo "[fmt] evaluating prompt-only vs fine-tuned..."
uv run python eval_format.py
echo "FMT_PIPELINE_DONE"
