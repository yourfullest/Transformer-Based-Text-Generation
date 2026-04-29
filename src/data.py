from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import torch
from torch.utils.data import Dataset


TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", re.UNICODE)
SPECIAL_TOKENS = ["<pad>", "<unk>", "<bos>", "<eos>"]


def read_text(path: str | Path) -> str:
    return Path(path).read_text(encoding="utf-8")


def tokenize(text: str, lowercase: bool = True) -> List[str]:
    if lowercase:
        text = text.lower()
    return TOKEN_PATTERN.findall(text)


@dataclass
class Vocab:
    token_to_id: dict
    id_to_token: list

    @classmethod
    def build(cls, tokens: Iterable[str], min_freq: int = 1, max_size: int | None = None) -> "Vocab":
        counts: dict[str, int] = {}
        for token in tokens:
            counts[token] = counts.get(token, 0) + 1

        sorted_tokens = sorted(
            (token for token, count in counts.items() if count >= min_freq),
            key=lambda item: (-counts[item], item),
        )
        if max_size is not None:
            sorted_tokens = sorted_tokens[: max(0, max_size - len(SPECIAL_TOKENS))]

        id_to_token = SPECIAL_TOKENS + [token for token in sorted_tokens if token not in SPECIAL_TOKENS]
        token_to_id = {token: idx for idx, token in enumerate(id_to_token)}
        return cls(token_to_id=token_to_id, id_to_token=id_to_token)

    @classmethod
    def from_tokens(cls, id_to_token: Sequence[str]) -> "Vocab":
        return cls(token_to_id={token: idx for idx, token in enumerate(id_to_token)}, id_to_token=list(id_to_token))

    @property
    def pad_id(self) -> int:
        return self.token_to_id["<pad>"]

    @property
    def unk_id(self) -> int:
        return self.token_to_id["<unk>"]

    @property
    def bos_id(self) -> int:
        return self.token_to_id["<bos>"]

    @property
    def eos_id(self) -> int:
        return self.token_to_id["<eos>"]

    def __len__(self) -> int:
        return len(self.id_to_token)

    def encode(self, tokens: Sequence[str]) -> List[int]:
        return [self.token_to_id.get(token, self.unk_id) for token in tokens]

    def decode(self, ids: Sequence[int], skip_special: bool = True) -> str:
        tokens = []
        for idx in ids:
            token = self.id_to_token[int(idx)]
            if skip_special and token in SPECIAL_TOKENS:
                continue
            tokens.append(token)
        return detokenize(tokens)

    def to_json(self) -> str:
        return json.dumps(self.id_to_token, ensure_ascii=False)

    @classmethod
    def from_json(cls, value: str) -> "Vocab":
        return cls.from_tokens(json.loads(value))


def detokenize(tokens: Sequence[str]) -> str:
    text = " ".join(tokens)
    text = re.sub(r"\s+([,.;:!?%])", r"\1", text)
    text = re.sub(r"([({\[])\s+", r"\1", text)
    text = re.sub(r"\s+([)}\]])", r"\1", text)
    text = text.replace(" ' ", "'")
    return text.strip()


def train_val_split(tokens: Sequence[str], val_fraction: float = 0.1) -> Tuple[List[str], List[str]]:
    if not 0.0 < val_fraction < 0.5:
        raise ValueError("val_fraction must be between 0.0 and 0.5")
    split_idx = max(1, int(len(tokens) * (1.0 - val_fraction)))
    return list(tokens[:split_idx]), list(tokens[split_idx:])


class TextPairDataset(Dataset):
    """Creates encoder source chunks and decoder continuation targets."""

    def __init__(
        self,
        token_ids: Sequence[int],
        context_len: int,
        target_len: int,
        bos_id: int,
        eos_id: int,
        stride: int = 1,
    ) -> None:
        if context_len < 1 or target_len < 1:
            raise ValueError("context_len and target_len must be positive")
        if stride < 1:
            raise ValueError("stride must be positive")

        self.token_ids = list(token_ids)
        self.context_len = context_len
        self.target_len = target_len
        self.bos_id = bos_id
        self.eos_id = eos_id
        self.starts = list(range(0, max(0, len(self.token_ids) - context_len - target_len), stride))

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        start = self.starts[index]
        src = self.token_ids[start : start + self.context_len]
        target = self.token_ids[start + self.context_len : start + self.context_len + self.target_len]
        decoder_in = [self.bos_id] + target
        labels = target + [self.eos_id]
        return (
            torch.tensor(src, dtype=torch.long),
            torch.tensor(decoder_in, dtype=torch.long),
            torch.tensor(labels, dtype=torch.long),
        )


def collate_batch(batch: Sequence[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]], pad_id: int):
    src, decoder_in, labels = zip(*batch)
    return (
        torch.nn.utils.rnn.pad_sequence(src, batch_first=True, padding_value=pad_id),
        torch.nn.utils.rnn.pad_sequence(decoder_in, batch_first=True, padding_value=pad_id),
        torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=pad_id),
    )
