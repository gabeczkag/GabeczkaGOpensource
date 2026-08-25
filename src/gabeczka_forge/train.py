import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .config import ModelConfig, TrainConfig
from .data import build_datasets
from .model import GabeczkaForgeModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Gabeczka Forge from scratch")
    parser.add_argument("--data", default="data")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--context-length", type=int, default=256)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--tiny", action="store_true", help="Use a CPU-friendly smoke-test model")
    parser.add_argument("--output", default="checkpoints")
    return parser.parse_args()


def save_checkpoint(model: GabeczkaForgeModel, optimizer: torch.optim.Optimizer, step: int, output_dir: str) -> None:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    torch.save({"step": step, "model": model.state_dict(), "optimizer": optimizer.state_dict()}, Path(output_dir) / f"step-{step}.pt")


@torch.no_grad()
def evaluate(model: GabeczkaForgeModel, loader: DataLoader, device: torch.device) -> tuple[float, int]:
    model.eval()
    losses = []
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        losses.append(model(input_ids, labels=input_ids)["loss"].item())
    model.train()
    return sum(losses) / len(losses), len(losses)


def main() -> None:
    args = parse_args()
    model_config = ModelConfig(context_length=args.context_length)
    if args.tiny:
        model_config = ModelConfig(vocab_size=256, context_length=args.context_length, hidden_size=128, intermediate_size=256, num_layers=2, num_attention_heads=4, num_key_value_heads=2)
    train_config = TrainConfig(output_dir=args.output)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GabeczkaForgeModel(model_config).to(device)
    train_dataset, test_dataset = build_datasets(args.data, model_config.context_length, args.test_ratio)
    loader = DataLoader(train_dataset, batch_size=train_config.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=train_config.batch_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_config.learning_rate, weight_decay=train_config.weight_decay)
    model.train()
    step = 0
    for epoch in range(1, args.epochs + 1):
        for batch in loader:
            step += 1
            input_ids = batch["input_ids"].to(device)
            loss = model(input_ids, labels=input_ids)["loss"]
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            if step % train_config.log_every == 0 or step == 1:
                print(f"epoch={epoch} step={step} loss={loss.item():.4f} device={device}")
            if step % train_config.save_every == 0:
                save_checkpoint(model, optimizer, step, train_config.output_dir)
        test_loss, tests_total = evaluate(model, test_loader, device)
        print(f"epoch={epoch} loss test {test_loss:.4f}/{tests_total}")
    save_checkpoint(model, optimizer, step, train_config.output_dir)


if __name__ == "__main__":
    main()
