import argparse
import math
import random
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from config import MambaLMConfig, TrainConfig
from dataset import build_dataloaders
from models.model import MambaLanguageModel


@torch.no_grad()
def evaluate(
    model: MambaLanguageModel,
    eval_loader: DataLoader[dict[str, Tensor]] | None,
    device: torch.device,
    max_batches: int,
) -> float | None:
    if eval_loader is None:
        return None

    model.eval()
    losses: list[float] = []
    for batch_index, batch in enumerate(eval_loader):
        if batch_index >= max_batches:
            break
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        _, loss = model(input_ids, labels=labels)
        if loss is not None:
            losses.append(loss.item())
    model.train()
    if not losses:
        return None
    return sum(losses) / len(losses)


def save_checkpoint(
    path: Path,
    model: MambaLanguageModel,
    optimizer: torch.optim.Optimizer,
    config: MambaLMConfig,
    tokenizer_name: str,
    global_step: int,
    train_loss: float,
    eval_loss: float | None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": config.to_dict(),
            "tokenizer_name": tokenizer_name,
            "global_step": global_step,
            "train_loss": train_loss,
            "eval_loss": eval_loss,
        },
        path,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pre-train a small Mamba language model.")
    parser.add_argument("--config", type=Path, default=Path("config/train.yaml"))
    parser.add_argument("--dataset-name", default=None)
    parser.add_argument("--dataset-config", default=None)
    parser.add_argument("--text-column", default=None)
    parser.add_argument("--tokenizer-name", default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=None)

    parser.add_argument("--block-size", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-eval-samples", type=int, default=None)
    parser.add_argument("--eval-interval", type=int, default=None)
    parser.add_argument("--eval-batches", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--grad-clip", type=float, default=None)

    parser.add_argument("--d-model", type=int, default=None)
    parser.add_argument("--d-state", type=int, default=None)
    parser.add_argument("--d-conv", type=int, default=None)
    parser.add_argument("--dt-rank", type=int, default=None)
    parser.add_argument("--num-layers", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--expand", type=int, default=None)
    tying_group = parser.add_mutually_exclusive_group()
    tying_group.add_argument("--tie-embeddings", action="store_true", default=None)
    tying_group.add_argument("--no-tie-embeddings", action="store_true", default=None)
    return parser.parse_args()


def load_train_config(args: argparse.Namespace) -> TrainConfig:
    config = TrainConfig.from_yaml(args.config)
    overrides: dict[str, Any] = {
        "dataset.name": args.dataset_name,
        "dataset.config_name": args.dataset_config,
        "dataset.text_column": args.text_column,
        "dataset.max_train_samples": args.max_train_samples,
        "dataset.max_eval_samples": args.max_eval_samples,
        "tokenizer.name": args.tokenizer_name,
        "output.checkpoint_path": str(args.output) if args.output is not None else None,
        "runtime.device": args.device,
        "runtime.seed": args.seed,
        "model.block_size": args.block_size,
        "model.d_model": args.d_model,
        "model.d_state": args.d_state,
        "model.d_conv": args.d_conv,
        "model.dt_rank": args.dt_rank,
        "model.num_layers": args.num_layers,
        "model.dropout": args.dropout,
        "model.expand": args.expand,
        "training.batch_size": args.batch_size,
        "training.max_steps": args.max_steps,
        "training.learning_rate": args.learning_rate,
        "training.weight_decay": args.weight_decay,
        "training.grad_clip": args.grad_clip,
        "training.eval_interval": args.eval_interval,
        "training.eval_batches": args.eval_batches,
    }
    if args.tie_embeddings:
        overrides["model.tie_embeddings"] = True
    if args.no_tie_embeddings:
        overrides["model.tie_embeddings"] = False
    return config.with_overrides(overrides)


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def main() -> None:
    args = parse_args()
    train_config = load_train_config(args)
    torch.manual_seed(train_config.runtime.seed)
    random.seed(train_config.runtime.seed)

    tokenizer = AutoTokenizer.from_pretrained(train_config.tokenizer.name)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    config = train_config.to_mamba_lm_config(vocab_size=len(tokenizer))

    device = resolve_device(train_config.runtime.device)
    model = MambaLanguageModel(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_config.training.learning_rate,
        weight_decay=train_config.training.weight_decay,
    )
    train_loader, eval_loader = build_dataloaders(train_config, tokenizer)

    global_step = 0
    last_loss = math.nan
    while global_step < train_config.training.max_steps:
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            _, loss = model(input_ids, labels=labels)
            if loss is None:
                raise RuntimeError("Training loss was not computed")

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if train_config.training.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.training.grad_clip)
            optimizer.step()

            global_step += 1
            last_loss = loss.item()
            if (
                global_step == 1
                or global_step % train_config.training.eval_interval == 0
                or global_step == train_config.training.max_steps
            ):
                eval_loss = evaluate(model, eval_loader, device, train_config.training.eval_batches)
                eval_text = "n/a" if eval_loss is None else f"{eval_loss:.4f}"
                print(f"step={global_step:04d} train_loss={last_loss:.4f} eval_loss={eval_text}")
                save_checkpoint(
                    Path(train_config.output.checkpoint_path),
                    model,
                    optimizer,
                    config,
                    train_config.tokenizer.name,
                    global_step,
                    last_loss,
                    eval_loss,
                )

            if global_step >= train_config.training.max_steps:
                break


if __name__ == "__main__":
    main()
