"""从 act_scales 里抽出做图需要的少量数据，落成小 npz（17MB 的 .pt 不进仓库）。

用法:
  python tools/extract_fig_src.py <act_scales/*.pt> results/figs_src/act_scale_stats.npz

产出:
  l6_text / l6_vision     第 6 层 mlp.up_proj 的逐通道最大激活（2048,）
  layer_overlap           36 层里 top-32 离群通道的交集个数
  layer_ratio_med/tail    每层 vision/text 比值的中位数与最大值
"""
import sys

import numpy as np
import torch

src, dst = sys.argv[1], sys.argv[2]
d = torch.load(src, map_location="cpu")
P = "model.language_model.layers.{}.mlp.up_proj."
K = 32
ov, med, tail = [], [], []
for i in range(36):
    t = d[P.format(i) + "text_scale"].float().numpy()
    v = d[P.format(i) + "vision_scale"].float().numpy()
    ov.append(len(set(np.argsort(-t)[:K]) & set(np.argsort(-v)[:K])))
    r = v / np.maximum(t, 1e-12)
    med.append(float(np.median(r)))
    tail.append(float(r.max()))
t6 = d[P.format(6) + "text_scale"].float().numpy()
v6 = d[P.format(6) + "vision_scale"].float().numpy()
np.savez_compressed(dst, l6_text=t6, l6_vision=v6,
                    layer_overlap=np.array(ov), layer_ratio_med=np.array(med),
                    layer_ratio_tail=np.array(tail))
print("wrote", dst, "overlap mean", np.mean(ov))
