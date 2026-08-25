import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from .tokenizer import WordTokenizer


def load_json_records(data_dir: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted(Path(data_dir).glob("*.json")):
        if path.name == "downloadlink.json":
            continue
        with path.open(encoding="utf-8") as file:
            payload = json.load(file)
        if isinstance(payload, dict):
            payload = payload.get("records", [payload])
        if not isinstance(payload, list) or not all(isinstance(item, dict) for item in payload):
            raise ValueError(f"Expected a JSON object or object list in {path}")
        records.extend(payload)
    if not records:
        raise ValueError(f"No JSON records found in {data_dir}")
    return records


def format_record(record: dict[str, Any]) -> str:
    language = record.get("language", "text")
    instruction = record.get("instruction", "")
    code = record.get("code", "")
    tests = record.get("tests", "")
    word = record.get("word", "")
    explanation = record.get("explanation", "")
    examples = record.get("examples", "")
    return f"<language>{language}</language>\n<word>{word}</word>\n<explanation>{explanation}</explanation>\n<examples>{examples}</examples>\n<instruction>{instruction}</instruction>\n<code>\n{code}\n</code>\n<tests>\n{tests}\n</tests>\n"


class JsonCodeDataset(Dataset):
    """Byte-level fallback dataset for structured JSON code records."""

    def __init__(self, records: list[dict[str, Any]], context_length: int, tokenizer: WordTokenizer) -> None:
        self.context_length = context_length
        text = "\n".join(format_record(record) for record in records)
        if not text:
            raise ValueError("JSON records produced no training text")
        self.tokens = torch.tensor(tokenizer.encode(text), dtype=torch.long)
        if len(self.tokens) <= context_length:
            raise ValueError("Training data must be longer than context_length")

    def __len__(self) -> int:
        return len(self.tokens) - self.context_length

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {"input_ids": self.tokens[index : index + self.context_length]}


def build_datasets(data_dir: str | Path, context_length: int, test_ratio: float, vocab_size: int) -> tuple[JsonCodeDataset, JsonCodeDataset, WordTokenizer]:
    if not 0.0 < test_ratio < 1.0:
        raise ValueError("test_ratio must be between 0 and 1")
    records = load_json_records(data_dir)
    split_index = max(1, int(len(records) * (1.0 - test_ratio)))
    split_index = min(split_index, len(records) - 1)
    tokenizer = WordTokenizer.train((format_record(record) for record in records[:split_index]), vocab_size)
    return (
        JsonCodeDataset(records[:split_index], context_length, tokenizer),
        JsonCodeDataset(records[split_index:], context_length, tokenizer),
        tokenizer,
    )
