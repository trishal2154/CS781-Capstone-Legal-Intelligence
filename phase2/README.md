# Phase 2 — Legal Named Entity Recognition

The task is to identify legal entities and return their character spans in the original document. I adapted the Phase 1 models with a token-classification head and compared fine-tuning, adapters and knowledge distillation.

## Labels and approach

The 12 entity types are `APP`, `RESP`, `A.COUNSEL`, `R.COUNSEL`, `JUDGE`, `WIT`, `AUTH`, `COURT`, `STAT`, `PREC`, `DATE` and `CASENO`. Tokens receive BIO labels, which are converted back to typed character spans.

The important preprocessing step is keeping the tokenizer offsets aligned with the original text. The gold-label round-trip check reconstructs spans from token labels before training. It checks the label conversion independently of model predictions.

Long documents are split into overlapping windows. The hybrid uses a 512-token window with context on both sides. Other saved variants use 1,024-token windows with left context. Weighted token loss handles label imbalance, and constrained BIO decoding prevents invalid label transitions.

## Notebooks

| Notebook | Backbone |
| --- | --- |
| `phase2_with_model1.ipynb` | Decoder-only |
| `phase2_with_model2.ipynb` | Encoder–decoder |
| `phase2_with_model3.ipynb` | Hybrid |
| `phase2_with_model4.ipynb` | Efficient; also includes distillation |

The efficient student keeps six of the teacher's twelve blocks. Its copied adapters are merged, and the student is trained with both gold labels and teacher outputs.

## Running this phase

1. Attach the matching Phase 1 checkpoint and `tokenizer.json`.
2. Load the official text and gold-label files. The recorded split combines the two labelled folds and creates 63 training and 7 validation documents.
3. Run offset alignment and the gold round-trip check before training.
4. Create windows, label mappings and loaders, then train the chosen model.
5. Load the checkpoint selected by validation span F1. For KD, load the teacher before creating and training the student.
6. Merge window predictions, decode BIO labels and reconstruct character spans for the 35 test documents.
7. Export `prediction.json` and `submission.zip` using the notebook's submission cells.

Use the tokenizer and alignment functions from the same notebook. The notebooks use PyTorch, NumPy, pandas and Hugging Face Hub; their Kaggle paths need updating for another environment.

## Training and implementation details

Full fine-tuning updates the backbone and classifier. Adapter runs freeze the original projections and train the adapters and classifier. The notebook specifies the projection names for its backbone; fused `qkv` and separate `q_proj` / `k_proj` / `v_proj` layers are not interchangeable.

Padding and context-only positions must be excluded from the supervised loss where the windowing scheme requires it. The same label mapping, offset handling and window-merging logic are used for validation and test prediction. BIO constraints allow an inside tag only after a compatible entity tag.

The efficient KD configuration uses eight windows per batch, learning rate `5e-5`, and up to 30 epochs. The teacher is frozen, and the loss combines gold labels with teacher targets on labelled positions. These settings describe that student run, not every notebook.

## Saved files

The teacher workflow saves `best_f1.pt`. The efficient KD section saves `student_best_f1.pt`. Load the relevant checkpoint before running its prediction cells. Export character offsets against the original text, without changing whitespace after encoding.
