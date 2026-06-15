import pytest
import torch

from models.selective_ssm import SelectiveSSMFn


def make_scan_inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    batch, dim, seqlen, state = 2, 3, 4, 5
    x = torch.randn(batch, dim, seqlen, requires_grad=True)
    delta = torch.randn(batch, dim, seqlen, requires_grad=True)
    A = (-torch.rand(dim, state)).requires_grad_()
    B = torch.randn(batch, state, seqlen, requires_grad=True)
    C = torch.randn(batch, state, seqlen, requires_grad=True)
    return x, delta, A, B, C


def test_selective_ssm_forward_shape_without_optional_terms() -> None:
    x, delta, A, B, C = make_scan_inputs()

    output = SelectiveSSMFn.apply(x, delta, A, B, C)

    assert output.shape == x.shape
    assert torch.isfinite(output).all()


def test_selective_ssm_forward_shape_with_optional_terms() -> None:
    x, delta, A, B, C = make_scan_inputs()
    D = torch.randn(x.shape[1], requires_grad=True)
    z = torch.randn_like(x, requires_grad=True)
    delta_bias = torch.randn(x.shape[1], requires_grad=True)

    output = SelectiveSSMFn.apply(x, delta, A, B, C, D, z, delta_bias, True)

    assert output.shape == x.shape
    assert torch.isfinite(output).all()


def test_selective_ssm_backward_gradients_match_input_shapes() -> None:
    x, delta, A, B, C = make_scan_inputs()
    D = torch.randn(x.shape[1], requires_grad=True)
    z = torch.randn_like(x, requires_grad=True)
    delta_bias = torch.randn(x.shape[1], requires_grad=True)

    output = SelectiveSSMFn.apply(x, delta, A, B, C, D, z, delta_bias, True)
    output.sum().backward()

    for tensor in [x, delta, A, B, C, D, z, delta_bias]:
        assert tensor.grad is not None
        assert tensor.grad.shape == tensor.shape
        assert torch.isfinite(tensor.grad).all()


@pytest.mark.parametrize(
    ("name", "mutate", "message"),
    [
        ("x", lambda x, delta, A, B, C: (x[:, 0], delta, A, B, C), "x must have shape"),
        ("delta", lambda x, delta, A, B, C: (x, delta[:, :-1], A, B, C), "delta must have the same shape"),
        ("A", lambda x, delta, A, B, C: (x, delta, A.unsqueeze(0), B, C), "A must have shape"),
        ("B", lambda x, delta, A, B, C: (x, delta, A, B[:, :-1], C), "B must have shape"),
        ("C", lambda x, delta, A, B, C: (x, delta, A, B, C[:, :-1]), "C must have shape"),
    ],
)
def test_selective_ssm_rejects_core_shape_mismatches(name: str, mutate, message: str) -> None:
    del name
    x, delta, A, B, C = make_scan_inputs()
    bad_x, bad_delta, bad_A, bad_B, bad_C = mutate(x, delta, A, B, C)

    with pytest.raises(ValueError, match=message):
        SelectiveSSMFn.apply(bad_x, bad_delta, bad_A, bad_B, bad_C)


@pytest.mark.parametrize(
    ("optional_args", "message"),
    [
        ((torch.randn(2), None, None), "D must have shape"),
        ((None, torch.randn(2, 3, 3), None), "z must have the same shape"),
        ((None, None, torch.randn(2)), "delta_bias must have shape"),
    ],
)
def test_selective_ssm_rejects_optional_shape_mismatches(optional_args, message: str) -> None:
    x, delta, A, B, C = make_scan_inputs()
    D, z, delta_bias = optional_args

    with pytest.raises(ValueError, match=message):
        SelectiveSSMFn.apply(x, delta, A, B, C, D, z, delta_bias, True)
