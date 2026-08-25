from dataclasses import dataclass, replace


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int = 128256
    context_length: int = 8192
    hidden_size: int = 8192
    intermediate_size: int = 18432
    num_layers: int = 40
    num_attention_heads: int = 64
    num_key_value_heads: int = 8
    rope_theta: float = 500000.0

    def with_context_length(self, context_length: int) -> "ModelConfig":
        return replace(self, context_length=context_length)


@dataclass(frozen=True)
class TrainConfig:
    batch_size: int = 1
    gradient_accumulation_steps: int = 8
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    max_steps: int = 1000
    warmup_steps: int = 100
    log_every: int = 10
    save_every: int = 500
    output_dir: str = "checkpoints"
