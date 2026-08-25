import math

import torch
from torch import nn
from torch.nn import functional as F

from .config import ModelConfig


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        return self.weight * hidden_states * torch.rsqrt(variance + self.eps)


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, max_position: int, theta: float) -> None:
        super().__init__()
        inverse_frequency = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
        positions = torch.arange(max_position).float()
        frequencies = torch.outer(positions, inverse_frequency)
        self.register_buffer("cos", frequencies.cos(), persistent=False)
        self.register_buffer("sin", frequencies.sin(), persistent=False)

    def forward(self, query: torch.Tensor, key: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        length = query.shape[-2]
        cos = self.cos[:length].to(query.dtype)[None, None, :, :]
        sin = self.sin[:length].to(query.dtype)[None, None, :, :]
        query = self._rotate(query, cos, sin)
        key = self._rotate(key, cos, sin)
        return query, key

    @staticmethod
    def _rotate(states: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        first, second = states[..., ::2], states[..., 1::2]
        rotated = torch.stack((-second, first), dim=-1).flatten(-2)
        frequencies = torch.stack((cos, cos), dim=-1).flatten(-1)
        sine = torch.stack((sin, sin), dim=-1).flatten(-1)
        return states * frequencies + rotated * sine


class Attention(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        if config.num_attention_heads % config.num_key_value_heads != 0:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads")
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.query = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.key = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.value = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.output = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.rotary = RotaryEmbedding(self.head_dim, config.context_length, config.rope_theta)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch, length, _ = hidden_states.shape
        query = self.query(hidden_states).view(batch, length, self.num_heads, self.head_dim).transpose(1, 2)
        key = self.key(hidden_states).view(batch, length, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value = self.value(hidden_states).view(batch, length, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        query, key = self.rotary(query, key)
        repeats = self.num_heads // self.num_key_value_heads
        key = key.repeat_interleave(repeats, dim=1)
        value = value.repeat_interleave(repeats, dim=1)
        attended = F.scaled_dot_product_attention(query, key, value, is_causal=True)
        attended = attended.transpose(1, 2).contiguous().view(batch, length, -1)
        return self.output(attended)


class DecoderBlock(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.input_norm = RMSNorm(config.hidden_size)
        self.attention = Attention(config)
        self.post_attention_norm = RMSNorm(config.hidden_size)
        self.gate = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = hidden_states + self.attention(self.input_norm(hidden_states))
        feed_forward_input = self.post_attention_norm(hidden_states)
        feed_forward = F.silu(self.gate(feed_forward_input)) * self.up(feed_forward_input)
        return hidden_states + self.down(feed_forward)


class GabeczkaForgeModel(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(DecoderBlock(config) for _ in range(config.num_layers))
        self.output_norm = RMSNorm(config.hidden_size)
        self.output = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.output.weight = self.embedding.weight

    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        hidden_states = self.embedding(input_ids)
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        logits = self.output(self.output_norm(hidden_states))
        result = {"logits": logits}
        if labels is not None:
            result["loss"] = F.cross_entropy(logits[:, :-1].reshape(-1, logits.size(-1)), labels[:, 1:].reshape(-1))
        return result

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())
