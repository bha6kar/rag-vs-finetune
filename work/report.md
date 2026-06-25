# Experiment: RAG vs fine-tuning on the same finance questions

Held-out test set: 17 questions. Judge: gpt-5.4-mini (correct + faithful), same as the bake-off.

| Condition | Correct | Faithful |
|---|---|---|
| gpt-5.4-mini + RAG (retrieval) | 16/17 (94%) | 17/17 (100%) |
| Claude Haiku 4.5 + RAG (retrieval) | 15/17 (88%) | 13/17 (76%) |
| gpt-5.4-mini, closed-book (no fine-tune) | 7/17 (41%) | 12/17 (71%) |
| Claude Haiku 4.5, closed-book (no fine-tune) | 2/17 (12%) | 14/17 (82%) |
| Qwen2.5-3B, closed-book (no fine-tune) | 1/17 (6%) | 16/17 (94%) |
| Qwen2.5-3B, fine-tuned, closed-book | 0/17 (0%) | 3/17 (18%) |
