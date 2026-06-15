import argparse
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

from config import TrainConfig  # noqa: E402


def default_output_path(checkpoint_path: Path) -> Path:
    return checkpoint_path.with_name(f"{checkpoint_path.stem}_loss.png")


def load_loss_history(checkpoint_path: Path) -> list[dict[str, float | int | None]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    history = checkpoint.get("history")
    if history:
        return [_normalize_history_entry(entry) for entry in history]

    if "global_step" not in checkpoint or "train_loss" not in checkpoint:
        raise ValueError("Checkpoint must contain either 'history' or legacy loss fields")
    return [
        {
            "step": int(checkpoint["global_step"]),
            "train_loss": float(checkpoint["train_loss"]),
            "eval_loss": _optional_float(checkpoint.get("eval_loss")),
        }
    ]


def plot_losses(history: list[dict[str, float | int | None]], output_path: Path) -> Path:
    if not history:
        raise ValueError("Loss history is empty")

    steps = [int(entry["step"]) for entry in history]
    train_losses = [float(entry["train_loss"]) for entry in history]
    eval_points = [(int(entry["step"]), float(entry["eval_loss"])) for entry in history if entry.get("eval_loss") is not None]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(steps, train_losses, marker="o", linewidth=1.5, label="train")
    if eval_points:
        eval_steps, eval_losses = zip(*eval_points, strict=True)
        ax.plot(eval_steps, eval_losses, marker="o", linewidth=1.5, label="eval")

    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    ax.set_title("Training and evaluation loss")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def parse_args() -> argparse.Namespace:
    default_checkpoint = Path(TrainConfig.from_yaml("config/train.yaml").output.checkpoint_path)
    parser = argparse.ArgumentParser(description="Plot train/eval losses saved in a checkpoint.")
    parser.add_argument("--checkpoint", type=Path, default=default_checkpoint)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = args.output if args.output is not None else default_output_path(args.checkpoint)
    history = load_loss_history(args.checkpoint)
    path = plot_losses(history, output_path)
    print(f"saved plot: {path}")


def _normalize_history_entry(entry: dict[str, Any]) -> dict[str, float | int | None]:
    if "step" not in entry or "train_loss" not in entry:
        raise ValueError("Each history entry must contain 'step' and 'train_loss'")
    return {
        "step": int(entry["step"]),
        "train_loss": float(entry["train_loss"]),
        "eval_loss": _optional_float(entry.get("eval_loss")),
    }


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


if __name__ == "__main__":
    main()
