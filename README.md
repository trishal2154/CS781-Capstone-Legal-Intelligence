# CS781 Capstone — Legal Intelligence

This repository contains my CS781 capstone work on legal language modelling and downstream legal NLP tasks. I trained four base models on legal text and adapted them for entity recognition, rhetorical roles, judgment prediction, statute identification, summarization and prior case retrieval.

**Name:** Trishal Reddy Bokka  
**Roll number:** 261110070  
**Institute:** IIT Kanpur  
**Email:** trishalr26@iitk.ac.in

## Project phases

| Phase | Task | Main implementation |
| --- | --- | --- |
| [Phase 1](phase1/README.md) | Legal language modelling | Four Transformers trained from scratch with custom BPE |
| [Phase 2](phase2/README.md) | Named entity recognition | Token classification, offset alignment and BIO decoding |
| [Phase 3](phase3/README.md) | Rhetorical role classification | Sentence representations and document-level BiGRU |
| [Phase 4](phase4/README.md) | Court judgment prediction | Chunk encoder, BiGRU and attention pooling |
| [Phase 5](phase5/README.md) | Statute identification | Multi-label classification with statute-specific attention |
| [Phase 6](phase6/README.md) | Judgment summarization | Chunk summaries followed by final-summary generation |
| [Phase 7](phase7/README.md) | Prior case retrieval | TF–IDF, neural retrieval, reranking and rank fusion |

Each phase README lists its notebooks, model variants, implementation details and run order. Multiple experiment stages can be in one notebook, including a baseline, adapter variant, distillation section or continuation cells.

## Base models

- **Model 1 — Decoder-only:** causal self-attention and a next-token prediction objective.
- **Model 2 — Encoder–decoder:** a bidirectional encoder reads a prefix; a causal decoder generates the continuation using cross-attention.
- **Model 3 — Hybrid GPT–BERT:** one backbone alternates causal and bidirectional masked next-token objectives.
- **Model 4 — Efficient Transformer:** sliding-window attention with grouped-query attention, RoPE, RMSNorm and SwiGLU.

The downstream phases load the matching Phase 1 weights and tokenizer. They are separate task models; Phase 3 does not require Phase 2 predictions, for example. Fine-tuning variants include full and partial fine-tuning, LoRA, LoRMAPlus, IA3 and Houlsby adapters where implemented. Several notebooks also contain teacher–student distillation.

## Environment and data

I ran the experiments in Kaggle notebooks with GPU acceleration, using Tesla T4 GPUs. Some runs use two devices. The notebooks include their imports and setup cells; the main packages used across the project are:

- PyTorch and NumPy for model training.
- pandas and scikit-learn for data handling and task utilities.
- `ijson` for streaming large pretraining files.
- `huggingface_hub` for downloading the official data.
- Matplotlib for plots.
- `rouge-score` and `bert-score` for summarization evaluation.

The official datasets come from `Exploration-Lab/CS-781-Capstone` on Hugging Face. Dataset access must be granted to the account used for downloading. The Kaggle notebooks use secrets for authentication; keep access tokens out of committed files.

## Running an experiment

1. Choose a phase and model notebook from the corresponding README.
2. Enable a GPU and run the notebook's package setup and import cells.
3. Set the data, checkpoint and output paths. Paths under `/kaggle/input/` refer to attached datasets; `/kaggle/working/` is used for generated files.
4. For Phases 2–7, attach the correct Phase 1 `model.pt` and `tokenizer.json`. Run Phase 1 first if those files are not available.
5. Run the data preparation, model definition and training sections for the chosen experiment.
6. Load the validation-selected checkpoint before inference. Restore its label mapping, threshold or fusion settings when the task uses them.
7. Run the submission cells to produce `prediction.json` and `submission.zip`.

The saved notebooks include cells from fresh runs and resumed runs. Some later cells load mounted checkpoints from a previous session. Update those paths and choose the appropriate training or continuation section instead of running every alternative stage together.

For a local Jupyter environment, replace Kaggle-specific paths and `kaggle_secrets` authentication with local paths and a supported Hugging Face login method. The notebook code is the entry point; the project does not use a single training command for all phases.

## Experiment settings and saved outputs

The configuration cells define learning rates, batch sizes, chunk budgets, adapter settings and epoch counts. The phase READMEs describe selected setups, while the notebook settings identify the exact variant being run. Keep separate output directories for separate experiments so checkpoints and predictions are not overwritten.

For each submitted variant, keep the notebook or configuration used for that run alongside its prediction JSON or submission ZIP. A different threshold, adapter type, checkpoint or decoding configuration can produce a separate submission even when the notebook is shared. Model checkpoints can remain in external storage, but their location and matching tokenizer are needed to reproduce inference.

Use a latest checkpoint with optimizer state to continue training. Use the validation-selected checkpoint for final prediction. Distillation requires the teacher checkpoint first, and cached teacher outputs must match the same examples and preprocessing used by the student.

## Author

Maintained by Trishal Reddy Bokka for the CS781 capstone project.
