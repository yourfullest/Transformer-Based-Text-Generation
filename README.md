# Transformer-Based Text Generation

A small, self-contained PyTorch project that implements a basic Transformer
encoder-decoder architecture from scratch for text generation.

The project includes:

- Regex-based text tokenization and vocabulary building
- Sliding-window source/target pair creation for continuation prediction
- Hand-written multi-head attention, encoder layers, and decoder layers
- Training, checkpointing, generation, and preliminary evaluation scripts
- A tiny public-domain sample corpus for quick smoke tests

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Train

```bash
python train.py --epochs 20 --batch-size 16
```

By default, training uses `data/sample_corpus.txt` and writes a checkpoint to
`runs/tiny_transformer/checkpoint.pt`.

For a larger local text file:

```bash
python train.py --data-path /path/to/corpus.txt --epochs 30
```

## Generate Text

```bash
python generate.py --prompt "to be or not to be" --max-new-tokens 40
```

## Evaluate

```bash
python evaluate.py
```

The evaluation reports validation loss, perplexity, and token-level accuracy.

## Project Layout

```text
.
├── data/sample_corpus.txt
├── evaluate.py
├── generate.py
├── requirements.txt
├── src/
│   ├── __init__.py
│   ├── data.py
│   └── model.py
└── train.py
```

## Notes

This is intentionally compact and educational. It is suitable for demonstrating
the full pipeline from preprocessing through model training and preliminary
evaluation on a small public text corpus.
