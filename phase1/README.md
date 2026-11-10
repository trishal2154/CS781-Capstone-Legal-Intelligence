# Phase 1 — Legal Language Modelling

This phase builds the base models used in the remaining tasks. I trained four Transformer models from scratch on legal text and compared their generated continuations. The submission requires a 128-word continuation for each test prompt.

## Models and notebooks

| Notebook | Model |
| --- | --- |
| `phase1_model1.ipynb` | Decoder-only Transformer |
| `phase1_model2.ipynb` | Encoder–decoder Transformer |
| `phase1_model3.ipynb` | Hybrid GPT–BERT model |
| `phase1_model4.ipynb` | Efficient sliding-window Transformer |

Model 1 uses causal multi-head attention and tied token embeddings. Model 2 has six encoder and six decoder layers and predicts a continuation from a legal prefix. Model 3 alternates causal and bidirectional masked next-token objectives. Model 4 uses a 256-token attention window, eight query heads, two key/value heads, RoPE, RMSNorm and SwiGLU.

## Data preparation

I used the official Acts, High Courts, Supreme Court and Tribunal sources. Large JSON files are streamed with `ijson`. Exact duplicates are removed, and documents are split before tokenization. Each model uses its matching custom byte-level BPE tokenizer; the encoded text is stored in memory-mapped arrays.

The source ratios, in the order listed above, are 1:3:2:2 for Models 1 and 3, 1:2:3:2 for Model 2, and 1:2:1:1 for Model 4. Models 1–3 use a vocabulary of 20,000, while Model 4 uses 10,000.

## Running the notebooks

1. Open the notebook for the required model in Kaggle with a GPU enabled.
2. Set the data and output paths. Download the official data using an account with dataset access.
3. Run preprocessing, tokenizer training and token-array creation.
4. Run the model definition and training cells. Use the resume section only when continuing a saved run.
5. Save the model and its matching tokenizer, then load the selected checkpoint for generation.
6. Generate the continuations and export `prediction.json` inside `submission.zip`.

The notebooks use PyTorch, NumPy, `ijson` and Hugging Face Hub. Kaggle-specific paths and secrets need changing when running elsewhere. Some later cells load previously saved checkpoints from mounted Kaggle datasets, so those paths must point to the current run's files.

## Training setup

| Model | Layers | Hidden size | Attention heads | Training block length |
| --- | --- | ---: | --- | ---: |
| Decoder-only | 12 | 560 | 8 query / 8 KV | 1,024 |
| Encoder–decoder | 6 encoder + 6 decoder | 512 | 8 query / 8 KV | 1,024 |
| Hybrid | 12 | 512 | 8 query / 8 KV | 512 |
| Efficient | 12 | 512 | 8 query / 2 KV | 1,024 |

The efficient model's 256-token attention window is smaller than its training block. Model 3 uses a masking schedule from 30% to 15%. The training cells define the learning-rate schedule, batch size, gradient accumulation and stopping budget for each model.

## Saved files and continuation

Keep `model.pt` and `tokenizer.json` together for downstream tasks. A tokenizer from another run can assign different IDs to the same text. `latest.pt` is used to continue training when the optimizer and training step are saved; the selected checkpoint is used for evaluation.

The prepared corpus budget and the number of tokens processed are separate settings. Gradient accumulation and repeated passes over token windows affect the training counter. Generation also needs a word-count check because a token limit does not enforce the required 128 words.
