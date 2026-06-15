import pytest

from config import MambaLMConfig


class TinyDataset:
    def __init__(self, rows: list[dict[str, str]]) -> None:
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, str]:
        return self.rows[index]


class TinyTokenizer:
    eos_token_id = 0

    def __call__(self, text: str, add_special_tokens: bool = False) -> dict[str, list[int]]:
        del add_special_tokens
        return {"input_ids": [ord(char) % 16 + 1 for char in text]}


@pytest.fixture
def small_config() -> MambaLMConfig:
    return MambaLMConfig(vocab_size=32, d_model=16, d_state=4, num_layers=2, block_size=8)
