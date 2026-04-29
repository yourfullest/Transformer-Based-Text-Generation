from __future__ import annotations

import argparse
import math
from functools import partial
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.data import TextPairDataset, Vocab, collate_batch, read_text, tokenize, train_val_split
from src.model import TransformerSeq2Seq
from train import choose_device, run_epoch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained Transformer checkpoint.")
    parser.add_argument("--checkpoint", default="runs/tiny_transformer/checkpoint.pt")
    parser.add_argument("--data-path", default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}. Run train.py first.")

    device = choose_device()
    checkpoint = torch.load(checkpoint_path, map_location=device)
    vocab = Vocab.from_tokens(checkpoint["vocab_tokens"])
    train_args = checkpoint["train_args"]
    data_path = args.data_path or train_args["data_path"]

    tokens = tokenize(read_text(data_path))
    _, val_tokens = train_val_split(tokens, val_fraction=train_args["val_fraction"])
    val_ids = vocab.encode(val_tokens)
    val_dataset = TextPairDataset(
        val_ids,
        train_args["context_len"],
        train_args["target_len"],
        vocab.bos_id,
        vocab.eos_id,
        stride=train_args["stride"],
    )
    loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=partial(collate_batch, pad_id=vocab.pad_id),
    )

    model = TransformerSeq2Seq(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    criterion = nn.CrossEntropyLoss(ignore_index=vocab.pad_id)
    val_loss, val_acc = run_epoch(model, loader, criterion, device)

    print(f"validation_loss={val_loss:.4f}")
    print(f"validation_perplexity={math.exp(min(val_loss, 20)):.2f}")
    print(f"token_accuracy={val_acc:.3f}")


if __name__ == "__main__":
    main()
