# Phase 7 — Prior Case Retrieval

Given a legal query case, this phase returns seven relevant prior cases from the supplied candidate pool. I combined text retrieval with a neural retriever and then tested a query–candidate reranker.

## Approach

1. **Text retrieval:** TF–IDF ranks the candidate cases.
2. **Neural retrieval:** a Phase 1 backbone encodes the query and candidate separately. The selected pipeline averages up to eight chunk vectors and projects them to a normalized 256-dimensional representation.
3. **First-stage fusion:** weighted reciprocal rank fusion combines the two top-100 lists.
4. **Reranking:** a cross-encoder scores up to 50 shortlisted candidates using query–candidate passage pairs.
5. **Final ranking:** a second fusion combines the retrieval and reranker rankings and returns seven case IDs.

Training uses relevant cases, hard negatives from text retrieval and random negatives. Checkpoints and fusion weights are selected on validation queries. Distillation trains smaller retrieval and reranking models from their teachers.

## Notebooks

| Notebook | Experiment |
| --- | --- |
| `model1-with-phase7-lora.ipynb` | Decoder-only LoRA and KD |
| `model1-with-phase7-lorma.ipynb` | Decoder-only LoRMAPlus and KD |
| `model1-with-phase7_pt.ipynb` | Decoder-only partial fine-tuning |
| `model2-with-phase7-lora.ipynb` | Encoder–decoder LoRA |
| `model3-with-phase7-lora.ipynb` | Hybrid LoRA |
| `model4-with-phase7-lora.ipynb` | Efficient LoRA |

## Running this phase

1. Attach the matching Phase 1 checkpoint and tokenizer and set the notebook paths.
2. Load each split's queries, relevance labels where available, and candidate cases. Keep candidate IDs tied to their own split.
3. Build the TF–IDF index and prepare neural-retrieval training pairs.
4. Train the retriever, load its best checkpoint, encode candidates and evaluate retrieval on validation queries.
5. Select the first fusion weight and prepare the reranking shortlist.
6. Train the reranker, load its selected checkpoint and choose the final fusion weight on validation data.
7. For KD, run the student sections after the teacher checkpoints are available.
8. Apply the selected settings to test queries and export seven ranked case IDs per query.

The notebooks use PyTorch, NumPy, scikit-learn and Hugging Face Hub. The recorded data has 827 training queries, 118 validation queries and 237 test queries. The corresponding candidate pools contain 4,320, 1,023 and 1,727 cases.

## Training and implementation details

The Model 1 LoRMAPlus run uses rank 8, a learning rate of `1e-4`, batches of eight queries for retrieval, and eight passage pairs for reranking. The planned schedule is five retrieval epochs followed by three reranking epochs. The settings for other variants remain in their notebook configuration cells.

The retriever and reranker are separate models. Query and candidate embeddings are normalized for retrieval similarity. The reranker reads both texts together, so it is applied to a shortlist rather than the full candidate pool. Fusion weights are selected on validation queries and kept fixed for test prediction.

For KD, create the student with the same adapter class and configuration expected by its saved weights. A LoRA checkpoint cannot be loaded into a LoRMAPlus model just by changing the method name.

## Saved files and prediction

The teacher pipeline saves `best_retriever.pt`, `best_reranker.pt` and `settings.json`. The KD pipeline saves `best_student_retriever.pt`, `best_student_reranker.pt` and `fusion_weights.json`. Restore the matching models and fusion settings before rerunning prediction.

Candidate embeddings depend on the checkpoint, tokenizer and chunk selection. Regenerate them when any of these changes. Final predictions contain seven unique candidate IDs per query, drawn from the correct split's pool.
