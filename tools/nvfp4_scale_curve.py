"""诊断：在平滑尺度被放大/缩小的过程中，INT4 与 NVFP4 的权重+激活量化 SQNR 曲线。

动机：报告 §3.3 观察到"NVFP4 下 MAS 训练反而比闭式解初始化差 0.39"。
可能机制：NVFP4 的 global_scale 依赖当前张量的 amax，训练把平滑尺度推大后，
global_scale 随之变化，E2M1 的有效分辨率被摊薄；而 INT4 的 per-channel absmax 缩放
对尺度不敏感（缩放只是被吸进 per-channel scale）。

做法：取若干层的真实权重 + 校准集上抓到的真实激活，把平滑尺度乘以因子 k（0.5~2.0），
算"平滑->量化权重->平滑后的激活量化->输出"的相对误差，按 k 画曲线。
"""
import argparse
import copy
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from quantize.quantizer import UniformAffineQuantizer
from quantize.nvfp4 import NVFP4Quantizer


@torch.no_grad()
def quant_error(x_act, w, s, qtype):
    """返回 (权重量化相对误差, 激活量化相对误差, 输出相对误差)。"""
    eps = 1e-5
    xi = x_act.float()
    wf = w.float()
    sf = s.float().clamp(min=1e-5)

    ref = xi @ wf.t()

    if qtype == "int":
        wq = UniformAffineQuantizer(n_bits=4, per_channel_axes=[0], symmetric=True,
                                    dynamic_method="per_channel", group_size=None)
        aq = UniformAffineQuantizer(n_bits=8, per_channel_axes=[], symmetric=True,
                                    dynamic_method="per_token")
    else:
        wq = NVFP4Quantizer(block_size=16)
        aq = NVFP4Quantizer(block_size=16)

    w_scaled = (wf * sf).to(torch.bfloat16)
    w_q = wq(w_scaled).float() / sf                      # 反平滑回原尺度
    a_scaled = (xi / sf).to(torch.bfloat16)
    a_q = aq(a_scaled).float() * sf

    w_err = ((w_q - wf).norm() / wf.norm()).item()
    a_err = ((a_q - xi).norm() / xi.norm()).item()
    out = a_q.to(torch.bfloat16) @ (w_q * 1.0).t().to(torch.bfloat16)
    out_err = ((out.float() - ref).norm() / ref.norm()).item()
    return w_err, a_err, out_err


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scales", required=True, help="mas_parameters.pth（提供真实尺度）")
    ap.add_argument("--model", default=os.environ.get("MASQUANT_MODEL", "Qwen2.5-VL-3B-Instruct"))
    ap.add_argument("--layers", default="0,9,18,27,35")
    ap.add_argument("--mods", default="self_attn.q_proj,mlp.up_proj")
    ap.add_argument("--ks", default="0.5,0.75,1.0,1.25,1.5,2.0,3.0")
    ap.add_argument("--real", action="store_true", help="用真实 FP16 前向抓到的激活（否则用合成激活）")
    ap.add_argument("--out", default="outputs/nvfp4_scale_curve.json")
    args = ap.parse_args()

    from transformers import AutoConfig
    scales = torch.load(args.scales, map_location="cpu")
    cfg = AutoConfig.from_pretrained(args.model)
    H = cfg.text_config.hidden_size if hasattr(cfg, "text_config") else cfg.hidden_size
    I = cfg.text_config.intermediate_size if hasattr(cfg, "text_config") else cfg.intermediate_size

    # --real 用真实 FP16 前向抓到的激活（更可信）；否则用合成的含离群通道激活
    real_x = {}
    if args.real:
        from transformers import AutoProcessor
        from models.LMClass import LMClass
        from custom_dataset import prepare_dataset, prepare_dataset_before_quant
        import torch.nn as nn

        ns = type("A", (), {})()
        ns.model = args.model
        ns.net = os.path.basename(args.model.rstrip("/")); ns.model_family = ns.net.split("-")[0]
        ns.batch_size = 1; ns.mode = "train"; ns.attn_implementation = "eager"
        ns.eval_sqnr = False; ns.multigpu = False
        ns.weight_quant_params = {"n_bits": 4, "per_channel_axes": [0], "symmetric": True,
                                  "dynamic_method": "per_channel", "group_size": None,
                                  "lwc": False, "disable_zero_point": False}
        ns.act_quant_params = {"n_bits": 8, "per_channel_axes": [], "symmetric": True,
                               "dynamic_method": "per_token"}
        llm = LMClass(ns)
        llm.model.to(llm.device).eval()
        proc = AutoProcessor.from_pretrained(args.model)
        convs = prepare_dataset(n_sample=4, data_type="text-vision-mix")
        data = prepare_dataset_before_quant(proc, convs, batch_size=1)
        cap = {}

        def hook(name, li):
            def f(m, inp, out):
                cap.setdefault((li, name), []).append(inp[0].detach().reshape(-1, inp[0].shape[-1]))
            return f

        want = {int(x) for x in args.layers.split(",")}
        hs = []
        for name, m in llm.model.named_modules():
            if not isinstance(m, nn.Linear) or any(t in name for t in ["visual", "lm_head", "audio"]):
                continue
            li = None
            for part in name.split("."):
                if part.isdigit():
                    li = int(part)
            if li in want:
                hs.append(m.register_forward_hook(hook(name.split(f"layers.{li}.")[-1], li)))
        with torch.no_grad():
            for b in data:
                llm.model(**{k: v.to(llm.device) for k, v in b.items()})
        for h in hs:
            h.remove()
        for k, v in cap.items():
            real_x[k] = torch.cat(v, dim=0)[:256].cpu()
        print(f"抓到真实激活: {sorted(real_x)}", flush=True)
        del llm
        torch.cuda.empty_cache()

    torch.manual_seed(0)
    X = torch.randn(256, H) * 0.5
    X[:, ::64] *= 50

    ks = [float(k) for k in args.ks.split(",")]
    res = {}
    for li in [int(x) for x in args.layers.split(",")]:
        for mod in args.mods.split(","):
            key = f"{mod}.text_smooth_scale"
            if key not in scales[li]:
                continue
            s0 = scales[li][key].float()
            out_dim = I if "up_proj" in mod or "gate_proj" in mod else H
            W = torch.randn(out_dim, H) * 0.05
            W[:, ::64] *= 50
            Xi = real_x.get((li, mod), X)
            if Xi.shape[-1] != s0.numel():
                Xi = X
            for qtype in ("int", "nvfp4"):
                for k in ks:
                    we, ae, oe = quant_error(Xi, W, s0 * k, qtype)
                    res.setdefault(f"{qtype}|{mod}", {}).setdefault(k, []).append((we, ae, oe))
            print(f"L{li} {mod} done", flush=True)

    # 汇总成均值
    summary = {}
    for tag, kmap in res.items():
        summary[tag] = {}
        for k, vals in sorted(kmap.items()):
            n = len(vals)
            summary[tag][k] = {
                "w_err": sum(v[0] for v in vals) / n,
                "a_err": sum(v[1] for v in vals) / n,
                "out_err": sum(v[2] for v in vals) / n,
            }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(summary, open(args.out, "w"), indent=2)

    print(f"\n{'量化':7s} {'模块':14s} " + " ".join(f"k={k:<5}" for k in ks))
    for tag, kmap in summary.items():
        qtype, mod = tag.split("|")
        row = " ".join(f"{kmap[k]['out_err']:.4f}" for k in ks)
        print(f"{qtype:7s} {mod:14s} {row}")
        best = min(kmap, key=lambda k: kmap[k]["out_err"])
        print(f"{'':7s} {'':14s} 最优 k = {best}（越小于 1 说明尺度被训练放大后变差）")
    print("saved", args.out)


if __name__ == "__main__":
    main()
