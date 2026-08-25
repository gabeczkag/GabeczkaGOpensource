from pathlib import Path

import torch
from torch.utils.data import Dataset


class ByteCodeDataset(Dataset):
    """Temporary byte dataset; replace it with a trained tokenizer before pretraining."""

    def __init__(self, data_dir: str | Path, context_length: int) -> None:
        self.context_length = context_length
        paths = sorted(Path(data_dir).rglob("*"))
        text = "\n".join(path.read_text(encoding="utf-8", errors="ignore") for path in paths if path.is_file())
        if not text:
            raise ValueError(f"No readable training files found in {data_dir}")
        self.tokens = torch.tensor(list(text.encode("utf-8")), dtype=torch.long)
        if len(self.tokens) <= context_length:
            raise ValueError("Training data must be longer than context_length")

    def __len__(self) -> int:
        return len(self.tokens) - self.context_length

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {"input_ids": self.tokens[index : index + self.context_length]}
