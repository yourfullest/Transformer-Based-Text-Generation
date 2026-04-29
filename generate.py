from __future__ import annotations

import argparse
from pathlib import Path

import torch

from src.data import Vocab, tokenize
from src.model import TransformerSeq2Seq


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate text with a trained Transformer checkpoint.")
    parser.add_argument("--checkpoint", default="runs/tiny_transformer/checkpoint.pt")
    parser.add_argument("--prompt", default="to be or not to be")
    parser.add_argument("--max-new-tokens", type=int, default=40)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=20)
    return parser.parse_args()


def choose_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def sample_next(logits: torch.Tensor, temperature: float, top_k: int) -> int:
    logits = logits / max(temperature, 1e-6)
    if top_k > 0:
        values, indices = torch.topk(logits, k=min(top_k, logits.size(-1)))
        probs = torch.softmax(values, dim=-1)
        return indices[torch.multinomial(probs, num_samples=1)].item()
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).item()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}. Run train.py first.")

    device = choose_device()
    checkpoint = torch.load(checkpoint_path, map_location=device)
    vocab = Vocab.from_tokens(checkpoint["vocab_tokens"])
    model = TransformerSeq2Seq(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    train_args = checkpoint.get("train_args", {})
    context_len = train_args.get("context_len", 16)
    prompt_tokens = tokenize(args.prompt)
    source_tokens = prompt_tokens[-context_len:]
    source_ids = vocab.encode(source_tokens)
    if len(source_ids) < context_len:
        source_ids = [vocab.pad_id] * (context_len - len(source_ids)) + source_ids

    src = torch.tensor([source_ids], dtype=torch.long, device=device)
    memory, src_mask = model.encode(src)
    generated = [vocab.bos_id]

    for _ in range(args.max_new_tokens):
        decoder_in = torch.tensor([generated], dtype=torch.long, device=device)
        decoded = model.decode(decoder_in, memory, src_mask)
        logits = model.output(decoded[:, -1, :]).squeeze(0)
        next_id = sample_next(logits, args.temperature, args.top_k)
        if next_id == vocab.eos_id:
            break
        generated.append(next_id)

    output_ids = vocab.encode(prompt_tokens) + generated[1:]
    print(vocab.decode(output_ids))


if __name__ == "__main__":
    main()
