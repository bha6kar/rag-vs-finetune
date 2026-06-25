# Posts

The write-ups behind this repo.

- [RAG or fine-tuning?](rag-or-finetuning/rag-vs-finetuning.md) — I measured both on the same finance documents. RAG lifts every model; fine-tuning the documents into the weights, even with a 513-pair corpus, still loses (24% vs 65% on the same model) and collapses faithfulness.
- [How to fine-tune, once you have admitted RAG is doing most of the work](how-to-finetune/how-to-finetune.md) — a fine-tuning workflow that starts by trying not to, with the measured head-to-head as the closing argument.

The retrieval bake-off these build on: [bha6kar/rag-bake-off](https://github.com/bha6kar/rag-bake-off).
