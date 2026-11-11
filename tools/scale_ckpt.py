"""按因子 k 缩放 mas_parameters.pth 里的平滑尺度，用于 NVFP4 下的 k 扫描。

用法: python tools/scale_ckpt.py <in.pth> <out.pth> <k> [--only text_smooth_scale]
注意：k<1 缩小尺度；min 1e-2 防止除零（与 truncate_number 一致）。
"""
import sys

import torch

src, dst, k = sys.argv[1], sys.argv[2], float(sys.argv[3])
only = sys.argv[5] if len(sys.argv) > 4 and sys.argv[4] == "--only" else None

d = torch.load(src, map_location="cpu")
n = 0
for li in d:
    for key, v in d[li].items():
        if not isinstance(v, torch.Tensor) or v.numel() <= 1:
            continue
        if "smooth_scale" not in key:
            continue
        if only and only not in key:
            continue
        d[li][key] = (v.float() * k).clamp(min=1e-2).to(v.dtype)
        n += 1
torch.save(d, dst)
print(f"scaled {n} tensors by k={k} -> {dst}")
