from torch import Tensor, nn


class LMHead(nn.Module):
    def __init__(self, d_model: int, vocab_size: int, bias: bool = False) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size, bias=bias)

    @property
    def weight(self) -> nn.Parameter:
        return self.proj.weight

    def tie_weight(self, value: nn.Parameter) -> None:
        self.proj.weight = value

    def forward(self, hidden_states: Tensor) -> Tensor:
        return self.proj(hidden_states)
