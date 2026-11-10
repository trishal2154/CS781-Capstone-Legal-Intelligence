# Phase 4 — Court Judgment Prediction

This is a binary classification task: predict the supplied case outcome label from a judgment. I used chunk representations so the model could read more than a single short prefix.

## Approach

Each judgment contributes up to 16 chunks of 512 tokens in the selected pipeline. When a judgment is longer, evenly spaced chunks are retained. The backbone encodes each chunk, a BiGRU reads the chunk sequence, and masked attention creates a document vector for the two-class classifier.

I compared partial fine-tuning, LoRA, LoRMAPlus and knowledge distillation. The class-1 decision threshold is selected on validation Macro-F1 and reused for test predictions.

## Notebooks

| Notebook | Experiment |
| --- | --- |
| `model1-with-phase4-partial_fine_tuning_kd.ipynb` | Decoder-only partial fine-tuning and KD |
| `model1-phase4-lora.ipynb` | Decoder-only LoRA |
| `model2-with-phase4_with_lormaplus.ipynb` | Encoder–decoder LoRMAPlus |
| `model3-with-phase4_with_lora.ipynb` | Hybrid LoRA |
| `model4-with-phase4_with_lora.ipynb` | Efficient LoRA |

## Running this phase

1. Load the matching Phase 1 checkpoint and tokenizer.
2. Read the official training and development cases. Remove identical duplicate IDs and reject conflicting records.
3. Tokenize the judgments, select chunks and create the document loaders.
4. Train the selected model and save the checkpoint with the best validation Macro-F1.
5. For KD, load the trained teacher, prepare its scores and train the student.
6. Collect validation probabilities, select a threshold and save it with the selected model's settings.
7. Load the selected model and threshold, predict the 1,517 test cases, and export the submission.

The main dependencies are PyTorch, NumPy, scikit-learn and Hugging Face Hub. Update all mounted checkpoint paths, including teacher paths in the KD section. Keep the threshold from the same model whose predictions are being generated.

## Training and implementation details

The partial fine-tuning pipeline updates selected final backbone blocks and the task head. Adapter variants freeze the base weights. In the encoder–decoder LoRMAPlus configuration, the encoder is frozen and adapters are added to the last two decoder blocks.

One recorded Model 1 partial/KD setup uses four documents per batch, 32 chunks per backbone pass, a BiGRU hidden size of 256, and learning rates of `1e-4` for the backbone and `5e-4` for the head. The KD setup uses temperature 2 and distillation weight 0.5. Other notebook configurations differ.

`MAX_CHUNKS` controls document coverage, `DOC_BATCH_SIZE` controls cases per batch, and `CHUNK_BATCH_SIZE` controls chunks per backbone pass. Increasing chunk batch size does not increase the amount of text selected from a case. Activation checkpointing trades additional computation for lower activation memory.

## Saved files and prediction

Keep the selected checkpoint and validation-selected threshold together. The KD workflow uses `best_student.pt` and writes the selected threshold to `threshold.json`. Apply the threshold to the class-1 probability, then export one integer label per case ID. For training continuation, restore optimizer and scaler state when present in the resume checkpoint.
