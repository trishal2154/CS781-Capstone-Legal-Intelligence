# Phase 6 — Legal Judgment Summarization

The task is to generate a summary of a long legal judgment. My main approach was two-stage hierarchical summarization: first summarize chunks, then combine those intermediate summaries into the final summary.

## Approach

For chunk-stage training, reference-summary sentences are assigned to source chunks using TF–IDF similarity. The chunk model learns from these pairs. It then generates intermediate summaries from the source text alone; validation and test references are not used to build generation inputs.

The final-stage model starts from a fresh adapted Phase 1 checkpoint and learns to summarize the generated intermediates. Prompt tokens are masked out of the training loss.

The Model 1 LoRMAPlus configuration uses:

| Setting | Value |
| --- | --- |
| Context length | 4,096 tokens |
| Source chunk length / overlap | 768 / 64 tokens |
| Chunk training target budget | 256 tokens |
| Intermediate generation limit | 96 new tokens per chunk |
| Final generation limit | 2,048 new tokens |
| Beam count / length penalty | 4 / 0.8 |
| No-repeat n-gram size | 3 |

This decoder pipeline has no fixed 16-chunk cap, although the final model still has a limited input budget. LoRMAPlus adapts the query, key, value and output projections. Other runs use LoRA or Houlsby adapters.

## Notebooks

| Notebook | Experiment |
| --- | --- |
| `model1-with-phase6-lora.ipynb` | Decoder-only LoRA |
| `model1-with-phase6-lorma.ipynb` | Decoder-only LoRMAPlus |
| `model1-with-phase6-lora-kd.ipynb` | Decoder-only LoRA distillation |
| `model2-with-phase6-lora.ipynb` | Encoder–decoder LoRA |
| `model3-with-phase6-houlsby-adapters.ipynb` | Hybrid Houlsby adapters |
| `model4-with-phase6-lora.ipynb` | Efficient LoRA |

## Running this phase

1. Load the Phase 1 checkpoint and matching tokenizer. Set the data and output paths and provide Hugging Face access for the gated dataset.
2. Remove empty pairs and duplicates, create the fixed document-level split, and save its IDs.
3. Build chunk training pairs and train the chunk-stage model.
4. Load its selected checkpoint and generate intermediate rows for training, validation and test documents.
5. Train the final-stage model on the generated training intermediates.
6. Load `final_best.pt`, generate validation summaries and compute ROUGE and BERTScore.
7. Generate test summaries and export the submission.

The notebooks use PyTorch, NumPy, scikit-learn, Hugging Face Hub, `rouge-score` and `bert-score`. The logged split contains 6,301 training and 700 held-out examples; a fixed 100-example validation subset limits generation cost.

## Training and implementation details

The Model 1 LoRMAPlus run uses rank 16, alpha 32 and adapter dropout 0.05. The recorded setup uses batch size 4 with four accumulation steps, AdamW, label smoothing 0.05 and gradient clipping at 1.0. The chunk-stage learning rate is `1e-4`, and the final-stage learning rate is `5e-5`, with two and five planned epochs respectively.

Stage checkpoints are selected using development loss. Generation-based evaluation is run separately. The other backbone notebooks can use different context and output budgets; their configuration cells define those limits.

## Saved files and continuation

Keep `split.json`, the stage checkpoints, intermediate-row files and partial predictions. After a timeout, load the selected stage checkpoint and the required intermediate rows before generation. Training does not need to be repeated just because prediction generation stopped.

A partial prediction file supports continuation only when the generation loop reads it and skips completed IDs. The final input still has a fixed context budget, so long intermediate collections are compressed before the final stage. Generation with multiple beams and a long output limit can take substantial time.
