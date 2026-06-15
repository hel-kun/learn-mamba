import random
from collections.abc import Iterator

import torch
from datasets import Dataset, DatasetDict, load_dataset
from torch import Tensor
from torch.utils.data import DataLoader, IterableDataset
from transformers import PreTrainedTokenizerBase

from config import TrainConfig


class TokenBlockDataset(IterableDataset[dict[str, Tensor]]):
    def __init__(
        self,
        dataset: Dataset,
        tokenizer: PreTrainedTokenizerBase,
        text_column: str,
        block_size: int,
        max_samples: int | None,
        shuffle: bool,
        seed: int,
    ) -> None:
        super().__init__()
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.text_column = text_column
        self.block_size = block_size
        self.max_samples = max_samples
        self.shuffle = shuffle
        self.seed = seed

    def __iter__(self) -> Iterator[dict[str, Tensor]]:
        indices = list(range(len(self.dataset)))
        if self.max_samples is not None:
            indices = indices[: self.max_samples]
        if self.shuffle:
            rng = random.Random(self.seed)
            rng.shuffle(indices)

        buffer: list[int] = []
        eos_token_id = self.tokenizer.eos_token_id
        for index in indices:
            text = self.dataset[index].get(self.text_column)
            if not text:
                continue

            token_ids = self.tokenizer(str(text), add_special_tokens=False)["input_ids"]
            if eos_token_id is not None:
                token_ids.append(eos_token_id)
            buffer.extend(token_ids)

            window_size = self.block_size + 1
            while len(buffer) >= window_size:
                item = torch.tensor(buffer[:window_size], dtype=torch.long)
                buffer = buffer[window_size:]
                yield {"input_ids": item[:-1], "labels": item[1:]}


def collate_batch(examples: list[dict[str, Tensor]]) -> dict[str, Tensor]:
    return {
        "input_ids": torch.stack([example["input_ids"] for example in examples]),
        "labels": torch.stack([example["labels"] for example in examples]),
    }


def build_dataloaders(
    config: TrainConfig,
    tokenizer: PreTrainedTokenizerBase,
    token: str | None = None,
) -> tuple[DataLoader[dict[str, Tensor]], DataLoader[dict[str, Tensor]] | None]:
    raw_dataset = load_dataset(config.dataset.name, config.dataset.config_name, token=token)
    if not isinstance(raw_dataset, DatasetDict):
        raise TypeError("Expected load_dataset to return a DatasetDict")
    if "train" not in raw_dataset:
        raise ValueError("Dataset must have a train split")

    eval_split = "validation" if "validation" in raw_dataset else "test" if "test" in raw_dataset else None
    train_dataset = TokenBlockDataset(
        raw_dataset["train"],
        tokenizer,
        config.dataset.text_column,
        config.model.block_size,
        config.dataset.max_train_samples,
        shuffle=True,
        seed=config.runtime.seed,
    )
    train_loader = DataLoader(train_dataset, batch_size=config.training.batch_size, collate_fn=collate_batch)

    eval_loader = None
    if eval_split is not None:
        eval_dataset = TokenBlockDataset(
            raw_dataset[eval_split],
            tokenizer,
            config.dataset.text_column,
            config.model.block_size,
            config.dataset.max_eval_samples,
            shuffle=False,
            seed=config.runtime.seed,
        )
        eval_loader = DataLoader(eval_dataset, batch_size=config.training.batch_size, collate_fn=collate_batch)
    return train_loader, eval_loader
