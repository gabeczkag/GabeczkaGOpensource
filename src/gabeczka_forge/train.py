import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .config import ModelConfig, TrainConfig
from .data import ByteCodeDataset
from .model import GabeczkaForgeModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Gabeczka Forge from scratch")
    parser.add_argument("--data", default="data")
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--context-length", type=int, default=256)
    parser.add_argument("--tiny", action="store_true", help="Use a CPU-friendly smoke-test model")
    parser.add_argument("--output", default="checkpoints")
    return parser.parse_args()


def save_checkpoint(model: GabeczkaForgeModel, optimizer: torch.optim.Optimizer, step: int, output_dir: str) -> None:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    torch.save({"step": step, "model": model.state_dict(), "optimizer": optimizer.state_dict()}, Path(output_dir) / f"step-{step}.pt")


def main() -> None:
    args = parse_args()
    model_config = ModelConfig(context_length=args.context_length)
    if args.tiny:
        model_config = ModelConfig(vocab_size=256, context_length=args.context_length, hidden_size=128, intermediate_size=256, num_layers=2, num_attention_heads=4, num_key_value_heads=2)
    train_config = TrainConfig(max_steps=args.steps, output_dir=args.output)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GabeczkaForgeModel(model_config).to(device)
    dataset = ByteCodeDataset(args.data, model_config.context_length)
    loader = DataLoader(dataset, batch_size=train_config.batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_config.learning_rate, weight_decay=train_config.weight_decay)
    model.train()
    for step, batch in enumerate(loader, start=1):
        if step > train_config.max_steps:
            break
        input_ids = batch["input_ids"].to(device)
        loss = model(input_ids, labels=input_ids)["loss"]
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if step % train_config.log_every == 0 or step == 1:
            print(f"step={step} loss={loss.item():.4f} device={device}")
        if step % train_config.save_every == 0:
            save_checkpoint(model, optimizer, step, train_config.output_dir)
    save_checkpoint(model, optimizer, step, train_config.output_dir)


if __name__ == "__main__":
    main()
