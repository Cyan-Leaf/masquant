import torch
import torch.nn as nn

from quantize.quantizer import round_ste

E2M1_GRID = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
E2M1_MAX = 6.0
E4M3_MAX = 448.0
E4M3_MIN_NORMAL = 2.0 ** -6


def _round_to_e2m1(x):
    """把 |x|<=6 的张量舍入到 E2M1 可表示值（RTN, 直通梯度）。"""
    sign = torch.sign(x)
    a = x.abs().clamp(max=E2M1_MAX)
    q = torch.where(
        a < 2.0,
        torch.round(a * 2.0) / 2.0,
        torch.where(a < 4.0, torch.round(a), torch.round(a / 2.0) * 2.0),
    )
    q = torch.where(a < 0.25, torch.zeros_like(q), q)
    return (sign * q - x).detach() + x


def _round_to_e4m3(s):
    """FP8 E4M3 舍入（仅正数缩放因子），3 位尾数。"""
    s = s.clamp(min=2.0 ** -9, max=E4M3_MAX)
    exp = torch.floor(torch.log2(s)).clamp(min=-6)
    step = torch.pow(2.0, exp - 3)
    return torch.round(s / step) * step


class NVFP4Quantizer(nn.Module):
    """
    NVFP4 伪量化：每 block_size(=16) 个连续元素共享一个 FP8(E4M3) 缩放，
    元素用 E2M1 表示；外层再乘一个 per-tensor FP32 缩放，使 block scale 落在 E4M3 范围内。
    沿最后一维分块（对激活即 per-token 沿 hidden 维，对权重即沿 in_features 维）。
    接口兼容 UniformAffineQuantizer：forward(x) -> 反量化后的 x。
    """

    def __init__(self, block_size=16, enable=True, fp8_block_scale=True, **kwargs):
        super().__init__()
        self.block_size = block_size
        self.enable = enable
        self.fp8_block_scale = fp8_block_scale
        self.n_bits = 4

    def forward(self, x):
        if not self.enable:
            return x
        shape = x.shape
        dtype = x.dtype
        d = shape[-1]
        pad = (-d) % self.block_size
        xf = x.float()
        if pad:
            xf = torch.nn.functional.pad(xf, (0, pad))
        xb = xf.reshape(-1, self.block_size)

        amax_block = xb.abs().amax(dim=-1, keepdim=True)
        amax_tensor = xf.abs().amax().clamp(min=1e-8)
        global_scale = amax_tensor / (E2M1_MAX * E4M3_MAX)
        block_scale = amax_block / E2M1_MAX / global_scale
        if self.fp8_block_scale:
            block_scale = _round_to_e4m3(block_scale.detach()) + (block_scale - block_scale.detach())
        scale = (block_scale * global_scale).clamp(min=1e-12)

        q = _round_to_e2m1(xb / scale) * scale
        q = q.reshape(*shape[:-1], d + pad)
        if pad:
            q = q[..., :d]
        return q.to(dtype)

    def register_scales_and_zeros(self):
        pass


def nvfp4_sqnr(x, block_size=16):
    q = NVFP4Quantizer(block_size)(x)
    err = (x.float() - q.float()).pow(2).mean()
    return 10 * torch.log10(x.float().pow(2).mean() / err.clamp(min=1e-20)).item()
