from pathlib import Path

import pytest
import torch

from config import MambaLMConfig
from infer import load_model_from_checkpoint, sample_next_token
from models.model import MambaLanguageModel


def test_load_model_from_checkpoint_restores_model_and_tokenizer_name(tmp_path: Path, small_config: MambaLMConfig) -> None:
    model = MambaLanguageModel(small_config)
    checkpoint_path = tmp_path / "checkpoint.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "config": small_config.to_dict(),
            "tokenizer_name": "tiny-tokenizer",
        },
        checkpoint_path,
    )

    loaded_model, tokenizer_name = load_model_from_checkpoint(checkpoint_path, torch.device("cpu"))

    assert isinstance(loaded_model, MambaLanguageModel)
    assert tokenizer_name == "tiny-tokenizer"
    assert loaded_model.config.to_dict() == small_config.to_dict()


def test_sample_next_token_returns_valid_token_id() -> None:
    torch.manual_seed(0)
    logits = torch.tensor([[0.0, 1.0, 2.0, 3.0]])

    token = sample_next_token(logits, temperature=1.0, top_k=2)

    assert token.shape == (1, 1)
    assert token.item() in {2, 3}


def test_sample_next_token_rejects_non_positive_temperature() -> None:
    with pytest.raises(ValueError, match="temperature"):
        sample_next_token(torch.randn(1, 4), temperature=0.0, top_k=None)


def test_load_model_from_checkpoint_requires_model_and_config(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "checkpoint.pt"
    torch.save({"config": MambaLMConfig(vocab_size=8).to_dict()}, checkpoint_path)

    with pytest.raises(ValueError, match="config.*model"):
        load_model_from_checkpoint(checkpoint_path, torch.device("cpu"))
