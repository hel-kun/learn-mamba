from torch import Tensor, nn


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.dropout = nn.Dropout(dropout)

    @property
    def weight(self) -> nn.Parameter:
        return self.embedding.weight

    def forward(self, input_ids: Tensor) -> Tensor:
        return self.dropout(self.embedding(input_ids))
