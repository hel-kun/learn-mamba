import math

import pytest
import torch

from models.mamba import Mamba


def test_mamba_forward_preserves_batch_sequence_and_model_shape() -> None:
    torch.manual_seed(0)
    model = Mamba(d_model=16, d_state=4, d_conv=3, expand=2)
    hidden_states = torch.randn(2, 5, 16)

    output = model(hidden_states)

    assert output.shape == hidden_states.shape
    assert torch.isfinite(output).all()


def test_mamba_initializes_internal_parameter_shapes() -> None:
    model = Mamba(d_model=17, d_state=5, d_conv=4, expand=3, dt_rank=None)

    assert model.dt_rank == math.ceil(model.d_model / 16)
    assert model.d_inner == 51
    assert model.in_proj.weight.shape == (102, 17)
    assert model.conv1d.weight.shape == (51, 1, 4)
    assert model.x_proj.weight.shape == (model.dt_rank + 10, 51)
    assert model.dt_proj.weight.shape == (51, model.dt_rank)
    assert model.A_log.shape == (51, 5)
    assert model.D.shape == (51,)
    assert model.out_proj.weight.shape == (17, 51)


def test_mamba_backward_produces_finite_gradients() -> None:
    torch.manual_seed(0)
    model = Mamba(d_model=8, d_state=3, d_conv=2)
    hidden_states = torch.randn(2, 4, 8, requires_grad=True)

    loss = model(hidden_states).pow(2).mean()
    loss.backward()

    assert hidden_states.grad is not None
    assert torch.isfinite(hidden_states.grad).all()
    for parameter in model.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()


def test_mamba_rejects_invalid_dt_init() -> None:
    with pytest.raises(ValueError, match="Invalid dt_init"):
        Mamba(d_model=8, d_state=3, dt_init="bad")
