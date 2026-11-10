# Phase 5 — Legal Statute Identification

This phase predicts the statutes associated with a judgment. A case can have several correct labels, so I used multi-label classification with independent sigmoid outputs.

## Approach

The baseline uses up to 16 chunks of 512 tokens. A Phase 1 backbone encodes the chunks, a BiGRU processes them in order, and attention pools their representations. The Model 1 partial fine-tuning baseline updates the final two decoder blocks and the task head.

I compared two attention heads:

- **Shared attention:** one document vector is used for all statute labels.
- **Statute-specific attention:** each statute learns its own weighting over the retained chunks.

I also trained a half-depth student using gold labels and cached teacher scores. A shared probability threshold is selected from 0.01 to 0.99 on validation Macro-F1 and saved with the checkpoint and statute mapping.

## Notebooks

| Notebook | Experiment |
| --- | --- |
| `model1-with-phase5.ipynb` | Decoder-only shared attention |
| `model1-with-phase5-statute-attention.ipynb` | Statute-specific attention and KD |
| `model1-with-phase5-lorma.ipynb` | Decoder-only LoRMAPlus |
| `model2-with-phase5.ipynb` | Encoder–decoder variant |
| `model3-with-phase5.ipynb` | Hybrid variant |
| `model4-with-phase5.ipynb` | Efficient variant |

## Running this phase

1. Attach the matching Phase 1 model and tokenizer and update the notebook paths.
2. Load the official train, development and test data. Build the statute mapping from the training setup and use it consistently across splits.
3. Tokenize judgments, select chunks and build multi-hot targets.
4. Train the shared-attention or statute-attention model and select its checkpoint and threshold on validation data.
5. For KD, load the teacher and cache its scores by case ID before training the student.
6. Reload the chosen checkpoint, its threshold and its label mapping.
7. Generate the predicted statute lists and export `prediction.json` and `submission.zip`.

The notebooks use PyTorch, NumPy, scikit-learn and Hugging Face Hub. Some saved notebooks contain continuation cells or commented training cells from earlier runs. Use either the fresh-training path or the continuation path for the experiment being reproduced.

## Training and implementation details

The model uses multi-hot targets and a binary cross-entropy objective. A softmax over statute labels would force the labels to compete, so the output uses independent sigmoid probabilities instead. Padding chunks are masked before attention pooling.

The statute-attention KD setup uses eight documents per batch, 32 chunks per backbone pass, a BiGRU hidden size of 256, and head dropout 0.2. Its learning rates are `2e-5` for the backbone and `1e-4` for the head. Teacher scores are cached by case ID, and the student combines supervised targets with independent teacher probabilities.

The saved statute-ID mapping determines the output order. A checkpoint, threshold and label mapping from different runs should not be mixed. If the teacher, tokenizer or chunk selection changes, regenerate the teacher cache.

## Saved files and continuation

The teacher uses `best_f1.pt`; the KD section uses `teacher_logits.pt`, `best_student.pt` and `latest_student.pt`. Use the latest student checkpoint for continuation and the selected student checkpoint for prediction. Restore the saved threshold before converting probabilities into statute lists.
