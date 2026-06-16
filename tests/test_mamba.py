import math

import pytest
import torch
from torch.testing import assert_close

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


@pytest.mark.parametrize("d_conv", [2, 3, 4])
def test_mamba_infer_matches_forward(d_conv: int) -> None:
    torch.manual_seed(0)
    model = Mamba(d_model=8, d_state=4, d_conv=d_conv)
    model.eval()
    hidden_states = torch.randn(2, 5, 8)

    with torch.no_grad():
        forward_out = model(hidden_states)
        infer_out = model.infer(hidden_states)

    assert infer_out.shape == forward_out.shape
    assert_close(infer_out, forward_out, rtol=1e-5, atol=1e-6)


def test_mamba_manual_step_loop_matches_infer() -> None:
    torch.manual_seed(0)
    model = Mamba(d_model=8, d_state=4, d_conv=3)
    model.eval()
    hidden_states = torch.randn(2, 5, 8)
    conv_state = torch.zeros(2, model.d_inner, model.d_conv)
    ssm_state = torch.zeros(2, model.d_inner, model.d_state)
    outputs = []

    with torch.no_grad():
        for t in range(hidden_states.shape[1]):
            out_t, conv_state, ssm_state = model.step(hidden_states[:, t], conv_state, ssm_state)
            outputs.append(out_t)
        manual_out = torch.stack(outputs, dim=1)
        infer_out = model.infer(hidden_states)

    assert_close(manual_out, infer_out, rtol=0, atol=0)
