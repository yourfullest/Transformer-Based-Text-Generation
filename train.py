from __future__ import annotations

import argparse
import math
from functools import partial
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data import TextPairDataset, Vocab, collate_batch, read_text, tokenize, train_val_split
from src.model import TransformerSeq2Seq


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a from-scratch Transformer text generator.")
    parser.add_argument("--data-path", default="data/sample_corpus.txt")
    parser.add_argument("--output-dir", default="runs/tiny_transformer")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--context-len", type=int, default=8)
    parser.add_argument("--target-len", type=int, default=8)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--d-ff", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--min-freq", type=int, default=1)
    parser.add_argument("--max-vocab-size", type=int, default=8000)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def choose_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def run_epoch(
    model: TransformerSeq2Seq,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, float]:
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    total_tokens = 0
    correct = 0

    progress = tqdm(loader, leave=False, desc="train" if is_train else "valid")
    for src, decoder_in, labels in progress:
        src = src.to(device)
        decoder_in = decoder_in.to(device)
        labels = labels.to(device)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        logits = model(src, decoder_in)
        loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))

        if is_train:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        with torch.no_grad():
            valid_tokens = labels != model.pad_id
            predictions = logits.argmax(dim=-1)
            correct += ((predictions == labels) & valid_tokens).sum().item()
            token_count = valid_tokens.sum().item()
            total_tokens += token_count
            total_loss += loss.item() * token_count
            progress.set_postfix(loss=loss.item())

    avg_loss = total_loss / max(1, total_tokens)
    accuracy = correct / max(1, total_tokens)
    return avg_loss, accuracy


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    text = read_text(args.data_path)
    tokens = tokenize(text)
    if len(tokens) < args.context_len + args.target_len + 2:
        raise ValueError("Corpus is too small for the selected context and target lengths.")

    train_tokens, val_tokens = train_val_split(tokens, val_fraction=args.val_fraction)
    vocab = Vocab.build(train_tokens, min_freq=args.min_freq, max_size=args.max_vocab_size)
    train_ids = vocab.encode(train_tokens)
    val_ids = vocab.encode(val_tokens)

    train_dataset = TextPairDataset(
        train_ids, args.context_len, args.target_len, vocab.bos_id, vocab.eos_id, stride=args.stride
    )
    val_dataset = TextPairDataset(
        val_ids, args.context_len, args.target_len, vocab.bos_id, vocab.eos_id, stride=args.stride
    )
    if len(train_dataset) == 0 or len(val_dataset) == 0:
        raise ValueError("Not enough train/validation examples. Lower lengths or val_fraction.")

    collate = partial(collate_batch, pad_id=vocab.pad_id)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    device = choose_device()
    model_config = {
        "vocab_size": len(vocab),
        "pad_id": vocab.pad_id,
        "d_model": args.d_model,
        "num_heads": args.num_heads,
        "num_layers": args.num_layers,
        "d_ff": args.d_ff,
        "max_len": args.context_len + args.target_len + 8,
        "dropout": args.dropout,
    }
    model = TransformerSeq2Seq(**model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss(ignore_index=vocab.pad_id)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")

    print(f"tokens={len(tokens)} vocab={len(vocab)} train_examples={len(train_dataset)} val_examples={len(val_dataset)}")
    print(f"device={device}")

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, device, optimizer)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, device)
        perplexity = math.exp(min(val_loss, 20))

        print(
            f"epoch {epoch:03d} | train_loss={train_loss:.4f} train_acc={train_acc:.3f} "
            f"| val_loss={val_loss:.4f} val_acc={val_acc:.3f} val_ppl={perplexity:.2f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "model_config": model_config,
                    "vocab_tokens": vocab.id_to_token,
                    "train_args": vars(args),
                    "best_val_loss": best_val_loss,
                },
                output_dir / "checkpoint.pt",
            )

    print(f"best checkpoint saved to {output_dir / 'checkpoint.pt'}")


if __name__ == "__main__":
    main()
