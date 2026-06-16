import argparse
from pathlib import Path

import torch
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from config import MambaLMConfig, TrainConfig
from models.model import MambaLanguageModel
from train import get_hf_token, resolve_device


def load_model_from_checkpoint(checkpoint_path: Path, device: torch.device) -> tuple[MambaLanguageModel, str]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if "config" not in checkpoint or "model" not in checkpoint:
        raise ValueError("Checkpoint must contain 'config' and 'model'")

    config = MambaLMConfig.from_dict(checkpoint["config"])
    model = MambaLanguageModel(config).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    tokenizer_name = checkpoint.get("tokenizer_name")
    if tokenizer_name is None:
        tokenizer_name = TrainConfig.from_yaml("config/train.yaml").tokenizer.name
    return model, tokenizer_name


def sample_next_token(logits: torch.Tensor, temperature: float, top_k: int | None) -> torch.Tensor:
    if temperature <= 0:
        raise ValueError("temperature must be greater than 0")

    logits = logits / temperature
    if top_k is not None and top_k > 0 and top_k < logits.shape[-1]:
        values, _ = torch.topk(logits, top_k, dim=-1)
        min_values = values[:, -1].unsqueeze(-1)
        logits = torch.where(logits < min_values, torch.full_like(logits, float("-inf")), logits)

    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


@torch.no_grad()
def generate(
    model: MambaLanguageModel,
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_k: int | None,
    device: torch.device,
) -> str:
    encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    input_ids = encoded["input_ids"].to(device)
    if input_ids.numel() == 0:
        eos_token_id = tokenizer.eos_token_id
        if eos_token_id is None:
            raise ValueError("Prompt is empty and tokenizer does not define eos_token_id")
        input_ids = torch.tensor([[eos_token_id]], device=device)

    for _ in range(max_new_tokens):
        logits = model.infer(input_ids)
        next_token = sample_next_token(logits[:, -1, :], temperature=temperature, top_k=top_k)
        input_ids = torch.cat([input_ids, next_token], dim=1)
        if tokenizer.eos_token_id is not None and next_token.item() == tokenizer.eos_token_id:
            break

    return tokenizer.decode(input_ids[0], skip_special_tokens=True)


def parse_args() -> argparse.Namespace:
    default_train_config = TrainConfig.from_yaml("config/train.yaml")
    parser = argparse.ArgumentParser(description="Generate text from a trained Mamba checkpoint.")
    parser.add_argument("--config", type=Path, default=Path("config/train.yaml"))
    parser.add_argument("--checkpoint", type=Path, default=Path(default_train_config.output.checkpoint_path))
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_config = TrainConfig.from_yaml(args.config)
    device = resolve_device(args.device if args.device is not None else train_config.runtime.device)
    hf_token = get_hf_token()

    model, tokenizer_name = load_model_from_checkpoint(args.checkpoint, device)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, token=hf_token)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    text = generate(
        model=model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        device=device,
    )
    print(text)


if __name__ == "__main__":
    main()
