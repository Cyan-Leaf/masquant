"""校验 forward_mas_*_fast 与原实现数值等价，并测 prefill 速度。

python tools/test_fast_forward.py
"""
import os
import sys
import time

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("inference_mode", "split_scales")

import quantize.int_linear as il
from quantize.int_linear import QuantLinear

torch.manual_seed(0)
dev = "cuda"
D_IN, D_OUT = 2048, 11008
T, N_IMG, N_VID = 2048, 1200, 0

wq = {"n_bits": 4, "per_channel_axes": [0], "symmetric": True, "dynamic_method": "per_channel",
      "group_size": None, "lwc": False, "disable_zero_point": False}
aq = {"n_bits": 8, "per_channel_axes": [], "symmetric": True, "dynamic_method": "per_token"}


def make(mode):
    lin = nn.Linear(D_IN, D_OUT, bias=True).to(dev, torch.bfloat16)
    m = QuantLinear(lin, wq, aq, support_training=(mode == "train"), mode=mode).to(dev)
    m.inference_mode = "split_scales"
    with torch.no_grad():
        for p, lo in ((m.text_smooth_scale, 0.5), (m.vision_smooth_scale, 1.5),
                      (m.audio_smooth_scale, 0.8), (m.video_smooth_scale, 1.2)):
            p.copy_(torch.rand(D_IN, device=dev, dtype=torch.bfloat16) * lo + 0.2)
    m.set_quant_state(True, True)
    return m


def masks():
    img = torch.zeros(1, T, dtype=torch.bool, device=dev)
    img[0, 100:100 + N_IMG] = True
    txt = ~img
    e = lambda t: t.unsqueeze(-1).expand(1, T, D_IN)
    return (None, e(img), e(txt), None)


def bench(fn, iters=20):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    t = time.time()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.time() - t) / iters * 1000


x = (torch.randn(1, T, D_IN, device=dev) * 2).to(torch.bfloat16)
x[0, 100:100 + N_IMG] *= 8
mm = masks()

# ---- 训练路径：前向 + 对 scale 的梯度 ----
m = make("train")
y_old = m.forward_mas_train(x, mm)
y_new = m.forward_mas_train_fast(x, mm)
rel = ((y_old.float() - y_new.float()).norm() / y_old.float().norm()).item()
print(f"[train] fwd rel diff = {rel:.2e}  max abs = {(y_old - y_new).abs().max().item():.3e}")

g = torch.randn_like(y_old)
m.zero_grad(); (y_old.float() * g).sum().backward()
go = {n: p.grad.clone() for n, p in m.named_parameters() if p.grad is not None}
m.zero_grad(); (m.forward_mas_train_fast(x, mm).float() * g).sum().backward()
gn = {n: p.grad.clone() for n, p in m.named_parameters() if p.grad is not None}
for n in go:
    a = go[n].float()
    if n not in gn:
        print(f"[train] grad {n:28s} legacy |g|={a.abs().max().item():.2e}  fast: None (未参与前向)")
        continue
    b = gn[n].float()
    cos = torch.nn.functional.cosine_similarity(a, b, dim=0).item()
    r = ((a - b).norm() / a.norm().clamp(min=1e-12)).item()
    print(f"[train] grad {n:28s} cos = {cos:.5f}  rel diff = {r:.2e}")

# 用接近真实训练的目标（MAE 对 FP 输出）再比一次梯度
with torch.no_grad():
    m.set_quant_state(False, False)
    tgt = m.forward_mas_train(x, mm).detach()
    m.set_quant_state(True, True)
m.zero_grad(); (m.forward_mas_train(x, mm).float() - tgt.float()).abs().mean().backward()
go = {n: p.grad.clone() for n, p in m.named_parameters() if p.grad is not None}
m.zero_grad(); (m.forward_mas_train_fast(x, mm).float() - tgt.float()).abs().mean().backward()
gn = {n: p.grad.clone() for n, p in m.named_parameters() if p.grad is not None}
for n in gn:
    a, b = go[n].float(), gn[n].float()
    print(f"[train/MAE] grad {n:24s} cos = {torch.nn.functional.cosine_similarity(a, b, dim=0).item():.5f}")
m.zero_grad()

with torch.no_grad():
    t_old = bench(lambda: m.forward_mas_train(x, mm))
    t_new = bench(lambda: m.forward_mas_train_fast(x, mm))
    print(f"[train/no-grad prefill] legacy {t_old:.2f} ms   fast {t_new:.2f} ms   speedup {t_old / t_new:.2f}x")
    xd = x[:, :1]
    t_old = bench(lambda: m.forward_mas_train(xd, mm), 200)
    t_new = bench(lambda: m.forward_mas_train_fast(xd, mm), 200)
    print(f"[train/no-grad decode ] legacy {t_old:.3f} ms  fast {t_new:.3f} ms   speedup {t_old / t_new:.2f}x")

# ---- CMC 推理路径 ----
mi = make("infer")
mi.inference_mode = "merged_scales"  # 复现 infer_mas.py 未设环境变量的真实情况
with torch.no_grad():
    mi.q_weight = mi.weight_quantizer((mi.weight.to(torch.bfloat16) * mi.text_smooth_scale).to(torch.bfloat16))
    r = 102
    mi.Lv = (torch.randn(D_IN, r, device=dev) * 0.02).to(torch.bfloat16)
    mi.Rv = (torch.randn(r, D_OUT, device=dev) * 0.02).to(torch.bfloat16)
    y_old = mi.forward_mas_infer(x, mm)
    y_new = mi.forward_mas_infer_fast(x, mm)
    rel = ((y_old.float() - y_new.float()).norm() / y_old.float().norm()).item()
    print(f"[infer] fwd rel diff = {rel:.2e}")
    t_old = bench(lambda: mi.forward_mas_infer(x, mm))
    t_new = bench(lambda: mi.forward_mas_infer_fast(x, mm))
    print(f"[infer prefill] legacy {t_old:.2f} ms   fast {t_new:.2f} ms   speedup {t_old / t_new:.2f}x")
