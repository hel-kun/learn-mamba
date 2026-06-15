from argparse import Namespace
from pathlib import Path

import pytest

from config import TrainConfig
from train import load_train_config, resolve_device


def test_train_config_loads_nested_yaml(tmp_path: Path) -> None:
    config_path = tmp_path / "train.yaml"
    config_path.write_text(
        """
dataset:
  name: tiny-dataset
  text_column: text
tokenizer:
  name: tiny-tokenizer
model:
  d_model: 16
  d_state: 4
  num_layers: 2
  block_size: 8
training:
  batch_size: 3
  max_steps: 7
runtime:
  device: cpu
  seed: 123
output:
  checkpoint_path: checkpoints/tiny.pt
""",
    )

    config = TrainConfig.from_yaml(config_path)

    assert config.dataset.name == "tiny-dataset"
    assert config.dataset.text_column == "text"
    assert config.tokenizer.name == "tiny-tokenizer"
    assert config.model.d_model == 16
    assert config.model.d_state == 4
    assert config.model.num_layers == 2
    assert config.model.block_size == 8
    assert config.training.batch_size == 3
    assert config.training.max_steps == 7
    assert config.runtime.device == "cpu"
    assert config.runtime.seed == 123
    assert config.output.checkpoint_path == "checkpoints/tiny.pt"


def test_train_config_rejects_unknown_keys(tmp_path: Path) -> None:
    config_path = tmp_path / "train.yaml"
    config_path.write_text(
        """
dataset:
  name: tiny-dataset
unknown: true
""",
    )

    with pytest.raises(ValueError, match="Unknown config key"):
        TrainConfig.from_yaml(config_path)


def test_train_config_builds_mamba_lm_config() -> None:
    config = TrainConfig.default().with_overrides(
        {
            "model.d_model": 16,
            "model.d_state": 4,
            "model.block_size": 8,
            "model.tie_embeddings": False,
        }
    )

    model_config = config.to_mamba_lm_config(vocab_size=99)

    assert model_config.vocab_size == 99
    assert model_config.d_model == 16
    assert model_config.d_state == 4
    assert model_config.block_size == 8
    assert model_config.tie_embeddings is False


def test_cli_overrides_only_specified_values(tmp_path: Path) -> None:
    config_path = tmp_path / "train.yaml"
    config_path.write_text(
        """
dataset:
  name: yaml-dataset
tokenizer:
  name: yaml-tokenizer
training:
  batch_size: 2
model:
  d_model: 16
""",
    )
    args = Namespace(
        config=config_path,
        dataset_name="cli-dataset",
        dataset_config=None,
        text_column=None,
        tokenizer_name=None,
        output=None,
        device="cpu",
        seed=None,
        block_size=4,
        batch_size=None,
        max_steps=5,
        max_train_samples=None,
        max_eval_samples=None,
        eval_interval=None,
        eval_batches=None,
        learning_rate=None,
        weight_decay=None,
        grad_clip=None,
        d_model=None,
        d_state=None,
        d_conv=None,
        dt_rank=None,
        num_layers=None,
        dropout=None,
        expand=None,
        tie_embeddings=None,
        no_tie_embeddings=True,
    )

    config = load_train_config(args)

    assert config.dataset.name == "cli-dataset"
    assert config.tokenizer.name == "yaml-tokenizer"
    assert config.training.batch_size == 2
    assert config.training.max_steps == 5
    assert config.model.d_model == 16
    assert config.model.block_size == 4
    assert config.model.tie_embeddings is False
    assert config.runtime.device == "cpu"


def test_resolve_device_accepts_auto_and_explicit_cpu() -> None:
    assert resolve_device("auto").type in {"cpu", "cuda"}
    assert resolve_device("cpu").type == "cpu"
