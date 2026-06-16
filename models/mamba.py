import math
from typing import Literal

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from models.selective_ssm import SelectiveSSMFn
from models.swish import Swish


class Mamba(nn.Module):
    def __init__(
        self,
        d_model: int,  # モデルの次元数
        d_state: int,  # SSMの状態の次元数
        d_conv: int = 4,  # convのKernelの次元数
        dt_rank: int | None = None,  # SSMの状態遷移行列のランク（低ランク近似）
        dt_min: float = 0.001,  # SSMにおいて状態遷移する際の要素の最小値
        dt_max: float = 0.1,  # SSMにおいて状態遷移する際の要素の最大値
        dt_init: str = "random",  # dtの初期化方法（一様分布か、定数か)
        dt_scale: float = 1.0,  # dt初期化の際の標準偏差スケール
        dt_init_floor: float = 1e-4,  # 初期dtの下限値
        expand: int = 2,  # FFNの拡大率
        bias: bool = False,  # in_projとout_projのバイアス項
        conv_bias: bool = True,  # convのバイアス項
        act: Literal["silu", "swish"] = "silu",  # 活性化関数の種類(論文ではSiLUを使用)
        device=None,
        dtype=None,
    ):
        super().__init__()

        factory_kwargs = {"device": device, "dtype": dtype}

        self.d_model: int = d_model
        self.d_state: int = d_state
        self.d_conv: int = d_conv
        self.dt_rank: int = math.ceil(self.d_model / 16) if dt_rank is None else dt_rank
        self.expand: int = expand
        self.d_inner: int = int(self.expand * self.d_model)

        self.act = nn.SiLU() if act == "silu" else Swish()
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        # 入力をd_inner * 2次元に射影（SSMの状態とFFNの入力を同時に生成）、 forwardとstepで分割して使用するため、`self.d_inner * 2`にしている
        self.conv1d = nn.Conv1d(
            self.d_inner,
            self.d_inner,
            kernel_size=self.d_conv,
            padding=d_conv - 1,
            bias=conv_bias,
            groups=self.d_inner,
            **factory_kwargs,
        )
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs)
        # conv後の出力をd_inner次元に射影
        # forwardでこいつの出力を、[dt, B, C]に分割して使用する
        self.dt_proj = nn.Linear(
            self.dt_rank, self.d_inner, bias=True, **factory_kwargs
        )  # dtの値を生成するための線形層（ランク分の値を生成）
        # self.selective_ssm = SelectiveSSM()
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)  # conv後の出力をd_model次元に射影

        # パラメータの初期化
        # A: SSMの状態遷移行列（d_inner x d_state）
        # [[1, 2, 3, ..., d_state],
        #  [1, 2, 3, ..., d_state],
        #  ...,
        #  [1, 2, 3, ..., d_state]]
        A: Tensor = torch.arange(1, self.d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log: nn.Parameter = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.D._no_weight_decay = True
        # B, Cは逐次生成らしい、本当か？

        # $\text{softplus}(z) = \log(1 + e^z)$
        # dt_proj.weightの初期化
        dt_init_std = self.dt_rank**-0.5 * dt_scale
        if dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        elif dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        else:
            raise ValueError(f"Invalid dt_init value: {dt_init}")
        # dt_proj.biasの初期化
        dt = torch.exp(torch.randn(self.d_inner) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min))
        dt = dt.clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))  # softplus^{-1}(dt)とsoftplusの逆関数を取る
        with torch.no_grad():
            self.dt_proj.bias.copy_(
                inv_dt
            )  # dt_projのバイアスを、初期化したdtの値に対応するように設定（dt_projの出力が初期化したdtの値になるように）
        self.dt_proj.bias._no_reinit = True  # bias初期化しないようにする

    def forward(self, h_states: Tensor) -> Tensor:
        batch, seqlen, dim = h_states.shape
        A = -torch.exp(self.A_log.float())  # [d_inner, d_state]

        xz = self.in_proj(h_states).transpose(1, 2).contiguous()  # [B, d_inner*2, T]
        x, z = xz.chunk(2, dim=1)

        x = self.act(self.conv1d(x)[..., :seqlen])  # [B, d_inner, T]

        x_dbl = self.x_proj(x.transpose(1, 2).contiguous())
        x_dbl = x_dbl.view(x_dbl.shape[0] * x_dbl.shape[1], -1)
        dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = (
            (self.dt_proj.weight @ dt.t())
            .transpose(0, 1)
            .contiguous()
            .view(batch, seqlen, self.d_inner)
            .transpose(1, 2)
            .contiguous()
        )
        B = B.view(batch, seqlen, self.d_state).transpose(1, 2).contiguous()
        C = C.view(batch, seqlen, self.d_state).transpose(1, 2).contiguous()

        ssm_out = SelectiveSSMFn.apply(
            x,
            dt,
            A,
            B,
            C,
            self.D.float(),
            z,
            self.dt_proj.bias.float(),
            True,
        )  # [B, d_inner, T]
        out = self.out_proj(ssm_out.transpose(1, 2).contiguous())  # [B, T, d_model]
        return out

    def step(self, h_states: Tensor, conv_state: Tensor, ssm_state: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        dtype = h_states.dtype
        x, z = self.in_proj(h_states).chunk(2, dim=-1)  # [B, d_inner] * 2

        # conv step (TODO: causal_conv1d_updateの実装によるconv処理の高速化)
        # example: [a, b, c] -> [b, c, a] -> [b, c, x]
        conv_state = torch.cat([conv_state[:, :, 1:], x.unsqueeze(2)], dim=2)  # [B, D, W]
        x = self.conv1d(conv_state)[:, :, -1]
        x = self.act(x).to(dtype=dtype)  # [B, d_inner]
        x_db = self.x_proj(x)  # [B, dt_rank + d_state * 2]
        dt, B, C = torch.split(
            x_db, [self.dt_rank, self.d_state, self.d_state], dim=-1
        )  # [B, dt_rank], [B, d_state], [B, d_state]
        dt = F.linear(dt, self.dt_proj.weight)  # [B, d_inner]
        A = -torch.exp(self.A_log.float())  # [d_inner, d_state]

        # ssm step (TODO: selective_state_update実装によるSSM処理の高速化)
        dt = F.softplus(dt + self.dt_proj.bias.to(dtype=dt.dtype))
        dA = torch.exp(dt.unsqueeze(-1) * A.unsqueeze(0))  # [B, d_inner, d_state]
        dB = dt.unsqueeze(-1) * B.unsqueeze(1)  # [B, d_inner, d_state]
        ssm_state = dA * ssm_state + dB * x.unsqueeze(-1)  # [B, d_inner, d_state]
        y = (ssm_state.to(dtype=dtype) * C.unsqueeze(1)).sum(dim=-1)  # [B, d_inner]
        y = y + self.D.to(dtype=dtype) * x  # [B, d_inner]
        y = y * self.act(z)

        out = self.out_proj(y)  # [B, d_model]
        return out, conv_state, ssm_state

    def infer(self, h_states: Tensor, inference_params=None) -> Tensor:
        batch, seqlen, dim = h_states.shape
        conv_state = torch.zeros(batch, self.d_inner, self.d_conv - 1, device=h_states.device, dtype=h_states.dtype)
        ssm_state = torch.zeros(batch, self.d_inner, self.d_state, device=h_states.device, dtype=h_states.dtype)

        out = torch.empty(batch, seqlen, self.d_model, device=h_states.device, dtype=h_states.dtype)
        for t in range(seqlen):
            out[:, t, :], conv_state, ssm_state = self.step(h_states[:, t, :], conv_state, ssm_state)
        return out
