import argparse
from pathlib import Path

import torch

from gabeczka_forge.config import ModelConfig
from gabeczka_forge.data import format_record, load_json_records
from gabeczka_forge.model import GabeczkaForgeModel
from gabeczka_forge.tokenizer import WordTokenizer


def find_checkpoint(path: str) -> Path:
    requested = Path(path)
    if requested.is_file():
        return requested
    candidates = sorted(requested.glob("step-*.pt"), key=lambda item: int(item.stem.split("-")[-1]))
    if not candidates:
        raise FileNotFoundError(f"No checkpoint found at {requested}")
    return candidates[-1]


def load_model(checkpoint_path: str, tiny: bool, context_length: int, data_dir: str, device: torch.device) -> tuple[GabeczkaForgeModel, WordTokenizer]:
    checkpoint_file = find_checkpoint(checkpoint_path)
    checkpoint = torch.load(checkpoint_file, map_location=device, weights_only=False)
    checkpoint_vocab_size = checkpoint["model"]["embedding.weight"].shape[0]
    tokenizer_file = checkpoint_file.parent / "tokenizer.json"
    tokenizer = WordTokenizer.load(tokenizer_file) if tokenizer_file.exists() else None
    if tokenizer is None or tokenizer.vocab_size != checkpoint_vocab_size:
        records = load_json_records(data_dir)
        vocabulary_size = checkpoint_vocab_size
        tokenizer = WordTokenizer.train((format_record(record) for record in records), vocabulary_size)
        if tokenizer.vocab_size < checkpoint_vocab_size:
            tokenizer = WordTokenizer(tokenizer.id_to_token + [f"<extra-{index}>" for index in range(checkpoint_vocab_size - tokenizer.vocab_size)])
        tokenizer.save(tokenizer_file)
        print(f"Warning: tokenizer.json was missing; rebuilt from {data_dir} and saved it.")
    config = ModelConfig(vocab_size=tokenizer.vocab_size, context_length=context_length)
    if tiny:
        config = ModelConfig(vocab_size=tokenizer.vocab_size, context_length=context_length, hidden_size=128, intermediate_size=256, num_layers=2, num_attention_heads=4, num_key_value_heads=2)
    model = GabeczkaForgeModel(config).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, tokenizer


@torch.no_grad()
def generate(model: GabeczkaForgeModel, tokenizer: WordTokenizer, prompt: str, max_new_tokens: int, min_new_tokens: int, temperature: float, device: torch.device) -> str:
    if temperature < 0:
        raise ValueError("temperature cannot be negative")
    tokens = tokenizer.encode(prompt, add_eos=False)
    if not tokens:
        raise ValueError("prompt cannot be empty")
    generated = torch.tensor([tokens], dtype=torch.long, device=device)
    prompt_length = generated.shape[1]
    for _ in range(max_new_tokens):
        input_ids = generated[:, -model.config.context_length :]
        logits = model(input_ids)["logits"][:, -1, :]
        if temperature == 0:
            next_token = logits.argmax(dim=-1, keepdim=True)
        else:
            next_token = torch.multinomial(torch.softmax(logits / temperature, dim=-1), num_samples=1)
        generated = torch.cat((generated, next_token), dim=1)
        if next_token.item() == tokenizer.eos_id and generated.shape[1] - prompt_length >= min_new_tokens:
            break
    return tokenizer.decode(generated[0, prompt_length:].tolist())


def main() -> None:
    parser = argparse.ArgumentParser(description="Chat with a Gabeczka Forge checkpoint")
    parser.add_argument("--checkpoint", default="checkpoints", help="Checkpoint file or directory")
    parser.add_argument("--data", default="data", help="JSON data used to rebuild a missing tokenizer")
    parser.add_argument("--context-length", type=int, default=256)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--min-new-tokens", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.7, help="0 uses greedy decoding")
    parser.add_argument("--tiny", action="store_true", help="Load a CPU-friendly checkpoint")
    args = parser.parse_args()
    if args.min_new_tokens < 0 or args.max_new_tokens < args.min_new_tokens:
        parser.error("--min-new-tokens must be non-negative and no greater than --max-new-tokens")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer = load_model(args.checkpoint, args.tiny, args.context_length, args.data, device)
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
            response = generate(model, tokenizer, prompt, args.max_new_tokens, args.min_new_tokens, args.temperature, device)
            print(f"Model: {response}")
        except ValueError as error:
            print(f"Error: {error}")


if __name__ == "__main__":
    main()
