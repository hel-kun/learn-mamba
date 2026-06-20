from pathlib import Path

import torch
from datasets import Dataset, DatasetDict
from torch.utils.data import DataLoader

import dataset as dataset_module
from config import MambaLMConfig, TrainConfig
from dataset import TokenBlockDataset, build_dataloaders, collate_batch
from models.embedding import TokenEmbedding
from models.lm_head import LMHead
from models.model import MambaLanguageModel
from tests.conftest import TinyDataset, TinyTokenizer
from train import evaluate, load_checkpoint, save_checkpoint


def test_token_embedding_shape() -> None:
    embedding = TokenEmbedding(vocab_size=32, d_model=16, dropout=0.0)
    input_ids = torch.randint(0, 32, (2, 5))

    output = embedding(input_ids)

    assert output.shape == (2, 5, 16)


def test_lm_head_shape() -> None:
    lm_head = LMHead(d_model=16, vocab_size=32)
    hidden_states = torch.randn(2, 5, 16)

    logits = lm_head(hidden_states)

    assert logits.shape == (2, 5, 32)


def test_mamba_language_model_forward_and_backward(small_config: MambaLMConfig) -> None:
    config = small_config
    model = MambaLanguageModel(config)
    input_ids = torch.randint(0, config.vocab_size, (2, config.block_size))
    labels = torch.randint(0, config.vocab_size, (2, config.block_size))

    logits, loss = model(input_ids, labels=labels)

    assert logits.shape == (2, config.block_size, config.vocab_size)
    assert loss is not None
    assert torch.isfinite(loss)
    loss.backward()


def test_mamba_language_model_infer_matches_forward(small_config: MambaLMConfig) -> None:
    torch.manual_seed(0)
    model = MambaLanguageModel(small_config)
    model.eval()
    input_ids = torch.randint(0, small_config.vocab_size, (2, small_config.block_size))

    with torch.no_grad():
        forward_logits, _ = model(input_ids)
        infer_logits = model.infer(input_ids)

    torch.testing.assert_close(infer_logits, forward_logits, rtol=1e-5, atol=1e-6)


def test_mamba_language_model_without_labels_returns_no_loss(small_config: MambaLMConfig) -> None:
    model = MambaLanguageModel(small_config)
    input_ids = torch.randint(0, small_config.vocab_size, (2, small_config.block_size))

    logits, loss = model(input_ids)

    assert logits.shape == (2, small_config.block_size, small_config.vocab_size)
    assert loss is None


def test_mamba_language_model_ties_embedding_weights_by_default(small_config: MambaLMConfig) -> None:
    model = MambaLanguageModel(small_config)

    assert model.lm_head.weight is model.embedding.weight


def test_mamba_language_model_can_disable_embedding_weight_tying() -> None:
    config = MambaLMConfig(vocab_size=32, d_model=16, d_state=4, tie_embeddings=False)
    model = MambaLanguageModel(config)

    assert model.lm_head.weight is not model.embedding.weight


def test_token_block_dataset_yields_next_token_labels() -> None:
    dataset = TokenBlockDataset(
        TinyDataset([{"story": "abcdef"}]),  # type: ignore[arg-type]
        TinyTokenizer(),  # type: ignore[arg-type]
        text_column="story",
        block_size=3,
        max_samples=None,
        shuffle=False,
        seed=0,
    )

    batch = next(iter(dataset))

    assert batch["input_ids"].tolist() == [2, 3, 4]
    assert batch["labels"].tolist() == [3, 4, 5]


def test_collate_batch_stacks_inputs_and_labels() -> None:
    examples = [
        {"input_ids": torch.tensor([1, 2, 3]), "labels": torch.tensor([2, 3, 4])},
        {"input_ids": torch.tensor([5, 6, 7]), "labels": torch.tensor([6, 7, 8])},
    ]

    batch = collate_batch(examples)

    assert batch["input_ids"].shape == (2, 3)
    assert batch["labels"].shape == (2, 3)
    assert batch["input_ids"].tolist() == [[1, 2, 3], [5, 6, 7]]


def test_evaluate_returns_none_without_eval_loader(small_config: MambaLMConfig) -> None:
    model = MambaLanguageModel(small_config)

    loss = evaluate(model, None, torch.device("cpu"), max_batches=1)

    assert loss is None


def test_evaluate_returns_finite_loss_for_eval_loader(small_config: MambaLMConfig) -> None:
    model = MambaLanguageModel(small_config)
    examples = [
        {
            "input_ids": torch.randint(0, small_config.vocab_size, (small_config.block_size,)),
            "labels": torch.randint(0, small_config.vocab_size, (small_config.block_size,)),
        }
    ]
    eval_loader = DataLoader(examples, batch_size=1, collate_fn=collate_batch)

    loss = evaluate(model, eval_loader, torch.device("cpu"), max_batches=1)

    assert loss is not None
    assert torch.isfinite(torch.tensor(loss))


def test_save_checkpoint_writes_training_state(tmp_path: Path, small_config: MambaLMConfig) -> None:
    model = MambaLanguageModel(small_config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    path = tmp_path / "checkpoint.pt"

    save_checkpoint(
        path,
        model,
        optimizer,
        small_config,
        tokenizer_name="tiny-tokenizer",
        global_step=3,
        train_loss=1.5,
        eval_loss=2.5,
        history=[{"step": 3, "train_loss": 1.5, "eval_loss": 2.5}],
    )

    checkpoint = torch.load(path, weights_only=False)
    assert set(checkpoint) == {
        "model",
        "optimizer",
        "config",
        "tokenizer_name",
        "global_step",
        "train_loss",
        "eval_loss",
        "history",
    }
    assert checkpoint["config"] == small_config.to_dict()
    assert checkpoint["tokenizer_name"] == "tiny-tokenizer"
    assert checkpoint["global_step"] == 3
    assert checkpoint["train_loss"] == 1.5
    assert checkpoint["eval_loss"] == 2.5
    assert checkpoint["history"] == [{"step": 3, "train_loss": 1.5, "eval_loss": 2.5}]


def test_load_checkpoint_restores_training_state(tmp_path: Path, small_config: MambaLMConfig) -> None:
    torch.manual_seed(0)
    model = MambaLanguageModel(small_config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.2)
    input_ids = torch.randint(0, small_config.vocab_size, (2, small_config.block_size))
    labels = torch.randint(0, small_config.vocab_size, (2, small_config.block_size))
    _, loss = model(input_ids, labels=labels)
    assert loss is not None
    loss.backward()
    optimizer.step()
    path = tmp_path / "checkpoint.pt"
    history = [{"step": 7, "train_loss": 1.25, "eval_loss": 1.75}]

    save_checkpoint(
        path,
        model,
        optimizer,
        small_config,
        tokenizer_name="tiny-tokenizer",
        global_step=7,
        train_loss=1.25,
        eval_loss=1.75,
        history=history,
    )

    state = load_checkpoint(path, torch.device("cpu"), TrainConfig.default())

    assert state.config == small_config
    assert state.tokenizer_name == "tiny-tokenizer"
    assert state.global_step == 7
    assert state.train_loss == 1.25
    assert state.eval_loss == 1.75
    assert state.history == history
    for key, value in model.state_dict().items():
        torch.testing.assert_close(state.model.state_dict()[key], value)
    assert state.optimizer.state_dict()["state"]


def test_load_checkpoint_defaults_missing_optional_state(tmp_path: Path, small_config: MambaLMConfig) -> None:
    model = MambaLanguageModel(small_config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    path = tmp_path / "legacy_checkpoint.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": small_config.to_dict(),
            "tokenizer_name": "tiny-tokenizer",
            "global_step": 1,
            "train_loss": 2.0,
        },
        path,
    )

    state = load_checkpoint(path, torch.device("cpu"), TrainConfig.default())

    assert state.eval_loss is None
    assert state.history == []


def test_build_dataloaders_passes_hf_token(monkeypatch) -> None:
    captured: dict[str, str | None] = {}

    def fake_load_dataset(path: str, name: str | None = None, token: str | None = None) -> DatasetDict:
        del path, name
        captured["token"] = token
        return DatasetDict(
            {
                "train": Dataset.from_dict({"story": ["abcdef"]}),
                "test": Dataset.from_dict({"story": ["abcdef"]}),
            }
        )

    monkeypatch.setattr(dataset_module, "load_dataset", fake_load_dataset)

    train_loader, eval_loader = build_dataloaders(TrainConfig.default(), TinyTokenizer(), token="hf_test")

    assert captured["token"] == "hf_test"
    assert train_loader is not None
    assert eval_loader is not None
