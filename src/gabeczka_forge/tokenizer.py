import json
import re
from collections import Counter
from pathlib import Path
from typing import Iterable


_TOKEN_PATTERN = re.compile(r"\s+|[A-Za-z_][A-Za-z_0-9]*|[^\sA-Za-z_0-9]", re.UNICODE)


class WordTokenizer:
    """Small trainable tokenizer that preserves whitespace and source punctuation."""

    special_tokens = ("<pad>", "<unk>", "<eos>")

    def __init__(self, vocabulary: list[str]) -> None:
        self.id_to_token = vocabulary
        self.token_to_id = {token: index for index, token in enumerate(vocabulary)}
        self.unknown_id = self.token_to_id["<unk>"]
        self.eos_id = self.token_to_id["<eos>"]

    @classmethod
    def train(cls, texts: Iterable[str], vocab_size: int = 32768) -> "WordTokenizer":
        counts = Counter(token for text in texts for token in _TOKEN_PATTERN.findall(text))
        vocabulary = list(cls.special_tokens)
        vocabulary.extend(token for token, _ in counts.most_common(max(0, vocab_size - len(vocabulary))))
        return cls(vocabulary)

    @property
    def vocab_size(self) -> int:
        return len(self.id_to_token)

    def encode(self, text: str, add_eos: bool = True) -> list[int]:
        tokens = [self.token_to_id.get(token, self.unknown_id) for token in _TOKEN_PATTERN.findall(text)]
        if add_eos:
            tokens.append(self.eos_id)
        return tokens

    def decode(self, token_ids: Iterable[int]) -> str:
        return "".join(self.id_to_token[token_id] for token_id in token_ids if 0 <= token_id < self.vocab_size and token_id not in (0, 1, 2))

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps({"vocabulary": self.id_to_token}, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "WordTokenizer":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(payload["vocabulary"])
