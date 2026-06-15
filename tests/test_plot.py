from pathlib import Path

import torch

from utils.plot import default_output_path, load_loss_history, plot_losses


def test_default_output_path_uses_checkpoint_directory() -> None:
    checkpoint_path = Path("checkpoints/mamba_lm.pt")

    assert default_output_path(checkpoint_path) == Path("checkpoints/mamba_lm_loss.png")


def test_plot_losses_writes_png(tmp_path: Path) -> None:
    output_path = tmp_path / "loss.png"
    history = [
        {"step": 1, "train_loss": 2.0, "eval_loss": 2.5},
        {"step": 2, "train_loss": 1.5, "eval_loss": 2.0},
    ]

    path = plot_losses(history, output_path)

    assert path == output_path
    assert output_path.exists()
    assert output_path.read_bytes().startswith(b"\x89PNG")


def test_load_loss_history_reads_history_checkpoint(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "checkpoint.pt"
    torch.save(
        {
            "history": [
                {"step": 1, "train_loss": 2.0, "eval_loss": 2.5},
                {"step": 2, "train_loss": 1.5, "eval_loss": None},
            ]
        },
        checkpoint_path,
    )

    history = load_loss_history(checkpoint_path)

    assert history == [
        {"step": 1, "train_loss": 2.0, "eval_loss": 2.5},
        {"step": 2, "train_loss": 1.5, "eval_loss": None},
    ]


def test_load_loss_history_supports_legacy_checkpoint(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "checkpoint.pt"
    torch.save(
        {
            "global_step": 3,
            "train_loss": 1.25,
            "eval_loss": 1.75,
        },
        checkpoint_path,
    )

    assert load_loss_history(checkpoint_path) == [{"step": 3, "train_loss": 1.25, "eval_loss": 1.75}]
