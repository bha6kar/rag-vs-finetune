set -e
cd /Users/onyx/crasher/rag-vs-finetune
echo "[pipeline] waiting for generation..." 
while pgrep -f make_synthetic_data >/dev/null; do sleep 15; done
echo "[pipeline] corpus: $(wc -l < work/mlx/train.jsonl) train / $(wc -l < work/mlx/valid.jsonl) valid"
echo "[pipeline] retraining 7B..."
uv run python train_local.py
echo "[pipeline] re-scoring ft_local..."
uv run python eval_all.py --conditions ft_local
echo "[pipeline] rebuilding report..."
uv run python make_report.py
echo "PIPELINE_DONE"
