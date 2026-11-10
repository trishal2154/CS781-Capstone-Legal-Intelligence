# Phase 3 — Rhetorical Role Classification

This phase assigns a rhetorical role to each sentence in a legal judgment. I first trained sentence classifiers and then added document context with a bidirectional GRU.

## Approach

The sentence model starts from a Phase 1 checkpoint. It encodes the target sentence and produces a sentence vector. Pooling differs across variants: the Model 1 LoRA run uses target-token pooling, while other variants also use the final EOS representation.

For the document model, sentence vectors are cached in their original order. A BiGRU reads the full sentence sequence and predicts one role per sentence. This lets the classifier use both earlier and later sentences. I also tested relative sentence position, IA3 and distillation.

## Notebooks

| Notebook | Experiment |
| --- | --- |
| `phase3_model1-with-lora-bigru-kd.ipynb` | Decoder-only LoRA, document BiGRU and KD |
| `phase3-model1-with-ia3.ipynb` | Decoder-only IA3 variant |
| `phase3_model2.ipynb` | Encoder–decoder |
| `phase3_model3.ipynb` | Hybrid |
| `phase3_model4.ipynb` | Efficient |

## Running this phase

1. Attach the Phase 1 model and matching tokenizer, then set the notebook's data and output paths.
2. Load the CL and IT data and `label_vocab.json`. Combine the domains within their official splits.
3. Build sentence examples and train the sentence model. Keep sentences from the same judgment in their document split.
4. Load the best sentence checkpoint and cache the sentence vectors.
5. Train the document BiGRU on the ordered vectors and select its checkpoint using validation Macro-F1.
6. Run the optional KD section only after the teacher is available.
7. Generate test labels in document and sentence order, then export `prediction.json` and `submission.zip`.

The notebooks use PyTorch, NumPy, scikit-learn and Hugging Face Hub. Saved paths may refer to earlier Kaggle runs. For the relative-position variant, training, validation and test loaders must all provide the same position feature.

## Training and implementation details

The Model 1 LoRA configuration uses rank 8 and alpha 16. The recorded sentence-model batch size is 16, and the document BiGRU uses batches of four judgments with hidden size 512 and dropout 0.2. The corresponding learning rates are `1e-4` for sentence-model adaptation and `1e-3` for the recurrent head.

The document model uses cached sentence vectors, so changing the sentence checkpoint or pooling method requires regenerating the cache. Padding must be excluded when processing variable-length documents. Relative sentence position is an optional input and needs to be supplied consistently by every loader when enabled.

The KD section trains the recurrent head from teacher outputs and gold labels. In this implementation, both heads use hidden size 512; distillation alone does not imply a smaller head.

## Saved files

The sentence model saves `best_f1.pt`, the document model saves `best_bigru.pt`, and the KD section saves `best_student.pt`. Keep their output directories separate. Test predictions must preserve the original document IDs and sentence order.
