from torch import Tensor, nn
from torch.nn import functional as F

from config import MambaLMConfig
from models.embedding import TokenEmbedding
from models.lm_head import LMHead
from models.mamba import Mamba


class MambaBlock(nn.Module):
    def __init__(self, config: MambaLMConfig) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(config.d_model)
        self.mamba = Mamba(
            d_model=config.d_model,
            d_state=config.d_state,
            d_conv=config.d_conv,
            dt_rank=config.dt_rank,
            expand=config.expand,
        )
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, hidden_states: Tensor) -> Tensor:
        return hidden_states + self.dropout(self.mamba(self.norm(hidden_states)))

    def infer(self, hidden_states: Tensor) -> Tensor:
        return hidden_states + self.dropout(self.mamba.infer(self.norm(hidden_states)))


class MambaLanguageModel(nn.Module):
    def __init__(self, config: MambaLMConfig) -> None:
        super().__init__()
        self.config = config
        self.embedding = TokenEmbedding(config.vocab_size, config.d_model, config.dropout)
        self.layers = nn.ModuleList(MambaBlock(config) for _ in range(config.num_layers))
        self.norm = nn.LayerNorm(config.d_model)
        self.lm_head = LMHead(config.d_model, config.vocab_size, bias=False)

        if config.tie_embeddings:
            self.lm_head.tie_weight(self.embedding.weight)

    def forward(self, input_ids: Tensor, labels: Tensor | None = None) -> tuple[Tensor, Tensor | None]:
        hidden_states = self.embedding(input_ids)
        for layer in self.layers:
            hidden_states = layer(hidden_states)

        logits = self.lm_head(self.norm(hidden_states))
        if labels is None:
            return logits, None

        loss = F.cross_entropy(logits.contiguous().view(-1, logits.size(-1)), labels.contiguous().view(-1))
        return logits, loss

    def infer(self, input_ids: Tensor) -> Tensor:
        hidden_states = self.embedding(input_ids)
        for layer in self.layers:
            hidden_states = layer.infer(hidden_states)
        return self.lm_head(self.norm(hidden_states))
