import argparse
from pathlib import Path

import torch

from gabeczka_forge.config import ModelConfig
from gabeczka_forge.model import GabeczkaForgeModel


def find_checkpoint(path: str) -> Path:
    requested = Path(path)
    if requested.is_file():
        return requested
    candidates = sorted(requested.glob("step-*.pt"), key=lambda item: int(item.stem.split("-")[-1]))
    if not candidates:
        raise FileNotFoundError(f"No checkpoint found at {requested}")
    return candidates[-1]


def load_model(checkpoint_path: str, tiny: bool, context_length: int, device: torch.device) -> GabeczkaForgeModel:
    config = ModelConfig(context_length=context_length)
    if tiny:
        config = ModelConfig(vocab_size=256, context_length=context_length, hidden_size=128, intermediate_size=256, num_layers=2, num_attention_heads=4, num_key_value_heads=2)
    checkpoint = torch.load(find_checkpoint(checkpoint_path), map_location=device, weights_only=False)
    model = GabeczkaForgeModel(config).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model


def encode(text: str) -> list[int]:
    return list(text.encode("utf-8"))


def decode(tokens: list[int]) -> str:
    return bytes(token for token in tokens if 0 <= token < 256).decode("utf-8", errors="ignore")


@torch.no_grad()
def generate(model: GabeczkaForgeModel, prompt: str, max_new_tokens: int, temperature: float, device: torch.device) -> str:
    if temperature <= 0:
        raise ValueError("temperature must be greater than 0")
    tokens = encode(prompt)
    if not tokens:
        raise ValueError("prompt cannot be empty")
    generated = torch.tensor([tokens], dtype=torch.long, device=device)
    for _ in range(max_new_tokens):
        input_ids = generated[:, -model.config.context_length :]
        logits = model(input_ids)["logits"][:, -1, :] / temperature
        next_token = torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1)
        generated = torch.cat((generated, next_token), dim=1)
    return decode(generated[0].tolist())


def main() -> None:
    parser = argparse.ArgumentParser(description="Chat with a Gabeczka Forge checkpoint")
    parser.add_argument("--checkpoint", default="checkpoints", help="Checkpoint file or directory")
    parser.add_argument("--context-length", type=int, default=256)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--tiny", action="store_true", help="Load a CPU-friendly checkpoint")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.checkpoint, args.tiny, args.context_length, device)
    print(f"Gabeczka Forge chat ({device}). Type 'exit' to quit.")
    while True:
        try:
            prompt = input("You: ")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if prompt.strip().lower() == "exit":
            break
        try:
            response = generate(model, prompt, args.max_new_tokens, args.temperature, device)
            print(f"Model: {response}")
        except ValueError as error:
            print(f"Error: {error}")


if __name__ == "__main__":
    main()
