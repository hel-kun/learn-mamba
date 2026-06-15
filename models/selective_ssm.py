import torch
from torch import Tensor
from torch.nn import functional as F


def _check_shapes(x: Tensor, delta: Tensor, A: Tensor, B: Tensor, C: Tensor) -> None:
    if x.ndim != 3:
        raise ValueError(f"x must have shape [batch, dim, seqlen], got {tuple(x.shape)}")
    if delta.shape != x.shape:
        raise ValueError(f"delta must have the same shape as x, got {tuple(delta.shape)} and {tuple(x.shape)}")
    if A.ndim != 2:
        raise ValueError(f"A must have shape [dim, state], got {tuple(A.shape)}")

    batch, dim, seqlen = x.shape
    state = A.shape[1]
    if A.shape[0] != dim:
        raise ValueError(f"A dim mismatch: expected {dim}, got {A.shape[0]}")
    if B.shape != (batch, state, seqlen):
        raise ValueError(f"B must have shape {(batch, state, seqlen)}, got {tuple(B.shape)}")
    if C.shape != (batch, state, seqlen):
        raise ValueError(f"C must have shape {(batch, state, seqlen)}, got {tuple(C.shape)}")


def _compute_scan_factors(
    x: Tensor,
    delta: Tensor,
    A: Tensor,
    B: Tensor,
    delta_bias: Tensor | None,
    is_softplus: bool = False,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    delta_pre = delta + delta_bias.view(1, -1, 1) if delta_bias is not None else delta
    delta_eff = F.softplus(delta_pre) if is_softplus else delta_pre

    a = torch.exp(delta_eff.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(2))
    b = delta_eff.unsqueeze(-1) * x.unsqueeze(-1) * B.transpose(1, 2).unsqueeze(1)
    return a, b, delta_pre, delta_eff


def _parallel_scan(a: Tensor, b: Tensor) -> Tensor:
    a_prefix = a
    b_prefix = b
    stride = 1
    seqlen = a.shape[2]

    while stride < seqlen:
        prev_a = a_prefix[:, :, :-stride, :]
        prev_b = b_prefix[:, :, :-stride, :]
        curr_a = a_prefix[:, :, stride:, :]
        curr_b = b_prefix[:, :, stride:, :]

        next_a = a_prefix.clone()
        next_b = b_prefix.clone()
        next_a[:, :, stride:, :] = curr_a * prev_a
        next_b[:, :, stride:, :] = curr_a * prev_b + curr_b
        a_prefix = next_a
        b_prefix = next_b
        stride *= 2
    return b_prefix


def _selective_scan_forward(
    x: Tensor,
    delta: Tensor,
    A: Tensor,
    B: Tensor,
    C: Tensor,
    D: Tensor | None = None,
    z: Tensor|None = None,
    delta_bias: Tensor|None = None,
    is_softplus: bool = False,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    a, b, delta_pre, delta_eff = _compute_scan_factors(x, delta, A, B, delta_bias, is_softplus)
    h = _parallel_scan(a, b)

    c = C.transpose(1, 2).unsqueeze(1)
    y_state = (h * c).sum(dim=-1)
    y_pre_gate = y_state if D is None else y_state + x * D.view(1, -1, 1)
    y = y_pre_gate if z is None else y_pre_gate * F.silu(z)
    return y, y_pre_gate, h, a, delta_pre, delta_eff


class SelectiveSSMFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: Tensor,
        delta: Tensor,
        A: Tensor,
        B: Tensor,
        C: Tensor,
        D: Tensor|None = None,
        z: Tensor|None = None,
        delta_bias=None,  # deltaのバイアス項
        is_softplus: bool = False,  # deltaに活性化関数を適用するかどうか
    ) -> Tensor:
        _check_shapes(x, delta, A, B, C)
        if D is not None and D.shape != (x.shape[1],):
            raise ValueError(f"D must have shape {(x.shape[1],)}, got {tuple(D.shape)}")
        if z is not None and z.shape != x.shape:
            raise ValueError(f"z must have the same shape as x, got {tuple(z.shape)} and {tuple(x.shape)}")
        if delta_bias is not None and delta_bias.shape != (x.shape[1],):
            raise ValueError(f"delta_bias must have shape {(x.shape[1],)}, got {tuple(delta_bias.shape)}")

        y, _, _, _, _, _ = _selective_scan_forward(x, delta, A, B, C, D, z, delta_bias, is_softplus)

        empty = x.new_empty(0)
        ctx.save_for_backward(
            x,
            delta,
            A,
            B,
            C,
            D if D is not None else empty,
            z if z is not None else empty,
            delta_bias if delta_bias is not None else empty,
        )  # h, a, delta_effは保存しない(勾配再計算)
        ctx.has_D = D is not None
        ctx.has_z = z is not None
        ctx.has_delta_bias = delta_bias is not None
        ctx.is_softplus = is_softplus
        return y

    @staticmethod
    def backward(ctx, grad_y: Tensor):
        x, delta, A, B, C, D_saved, z_saved, delta_bias_saved = ctx.saved_tensors
        D = D_saved if ctx.has_D else None
        z = z_saved if ctx.has_z else None
        delta_bias = delta_bias_saved if ctx.has_delta_bias else None

        _, y_pre_gate, h, a, delta_pre, delta_eff = _selective_scan_forward(
            x,
            delta,
            A,
            B,
            C,
            D,
            z,
            delta_bias,
            ctx.is_softplus,
        )  # もう一回計算

        if z is None:
            grad_pre_gate = grad_y
            grad_z = None
        else:
            sigmoid_z = torch.sigmoid(z)
            silu_z = z * sigmoid_z
            grad_pre_gate = grad_y * silu_z
            grad_z = grad_y * y_pre_gate * sigmoid_z * (1 + z * (1 - sigmoid_z))

        grad_x = torch.zeros_like(x)
        if D is None:
            grad_D = None
        else:
            grad_x = grad_x + grad_pre_gate * D.view(1, -1, 1)
            grad_D = (grad_pre_gate * x).sum(dim=(0, 2))

        c = C.transpose(1, 2).unsqueeze(1)
        grad_h_direct = grad_pre_gate.unsqueeze(-1) * c
        grad_C = (grad_pre_gate.unsqueeze(-1) * h).sum(dim=1).transpose(1, 2).contiguous()

        grad_a = torch.zeros_like(a)
        grad_b = torch.zeros_like(h)
        grad_h_next = torch.zeros_like(h[:, :, 0, :])
        for t in range(x.shape[2] - 1, -1, -1):
            grad_h_t = grad_h_direct[:, :, t, :] + grad_h_next
            h_prev = h[:, :, t - 1, :] if t > 0 else torch.zeros_like(grad_h_t)
            grad_b[:, :, t, :] = grad_h_t
            grad_a[:, :, t, :] = grad_h_t * h_prev
            grad_h_next = grad_h_t * a[:, :, t, :]

        B_t = B.transpose(1, 2).unsqueeze(1)
        grad_delta_eff = (grad_a * a * A.unsqueeze(0).unsqueeze(2)).sum(dim=-1)
        grad_A = (grad_a * a * delta_eff.unsqueeze(-1)).sum(dim=(0, 2))

        grad_delta_eff = grad_delta_eff + (grad_b * x.unsqueeze(-1) * B_t).sum(dim=-1)
        grad_x = grad_x + (grad_b * delta_eff.unsqueeze(-1) * B_t).sum(dim=-1)
        grad_B = (grad_b * delta_eff.unsqueeze(-1) * x.unsqueeze(-1)).sum(dim=1).transpose(1, 2).contiguous()

        if ctx.is_softplus:
            grad_delta = grad_delta_eff * torch.sigmoid(delta_pre)
        else:
            grad_delta = grad_delta_eff
        grad_delta_bias = grad_delta.sum(dim=(0, 2)) if ctx.has_delta_bias else None

        return grad_x, grad_delta, grad_A, grad_B, grad_C, grad_D, grad_z, grad_delta_bias, None
