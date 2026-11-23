"""
CMC（低秩跨模态补偿）的单层诊断。

在真实视觉 token 上逐层比较三种计算方式相对 FP16 输出 Y = X W^T 的 SQNR（dB）：

    nocomp : Q(X / s_t) · Q(s_t W)^T                         （所有 token 用文本尺度，只存一份权重）
    ideal  : Q(X / s_v) · Q(s_v W)^T                         （视觉单独一份权重，CMC 试图逼近的上界）
    cmc    : Q(X / s_v) · Q(s_t W)^T + (X / s_v) · L R        （论文方法：共享权重 + 低秩补偿）

另外把样本分成"白化校准用过的"（前 --calib-n 个）与"未见过的"两组分别统计，
用来判断低秩补偿在校准集外是否泛化。

用法：
    python tools/cmc_probe.py --model $MODEL --scales <mas_parameters.pth> \
        --adapters <low_rank_adapters_*.pt> --nsamples 48 --calib-n 32 --layer-step 3 \
        --out outputs/cmc_probe.json
"""
import argparse
import json
import os
import sys

import torch
import torch.nn as nn
from transformers import AutoProcessor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.LMClass import LMClass
from quantize.quantizer import UniformAffineQuantizer


def sqnr(ref, q):
    noise = (ref - q).pow(2).mean().clamp(min=1e-20)
    return 10 * torch.log10(ref.pow(2).mean() / noise).item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--scales", required=True)
    ap.add_argument("--adapters", required=True)
    ap.add_argument("--data-type", default="vision-only")
    ap.add_argument("--nsamples", type=int, default=48)
    ap.add_argument("--calib-n", type=int, default=32)
    ap.add_argument("--layer-step", type=int, default=3)
    ap.add_argument("--wbits", type=int, default=4)
    ap.add_argument("--abits", type=int, default=8)
    ap.add_argument("--out", default="outputs/cmc_probe.json")
    args = ap.parse_args()

    args.weight_quant_params = {"n_bits": args.wbits, "per_channel_axes": [0], "symmetric": True,
                                "dynamic_method": "per_channel", "group_size": 0,
                                "lwc": False, "disable_zero_point": False}
    args.act_quant_params = {"n_bits": args.abits, "per_channel_axes": [], "symmetric": True,
                             "dynamic_method": "per_token"}
    args.net = os.path.basename(args.model.rstrip("/"))
    args.model_family = args.net.split("-")[0]
    args.batch_size = 1
    args.mode = "train"
    args.attn_implementation = "eager"
    args.eval_sqnr = False
    args.multigpu = False

    llm = LMClass(args)
    llm.model.to(llm.device)
    llm.model.eval()
    processor = AutoProcessor.from_pretrained(args.model)

    from custom_dataset import prepare_dataset, prepare_dataset_before_quant
    convs = prepare_dataset(n_sample=args.nsamples, data_type=args.data_type)
    data = prepare_dataset_before_quant(processor, convs, batch_size=1)

    scales_all = torch.load(args.scales, map_location="cpu")
    adapters = torch.load(args.adapters, map_location="cpu")["vision"]
    aq = UniformAffineQuantizer(**args.act_quant_params)
    wq = UniformAffineQuantizer(**args.weight_quant_params)
    eps = 1e-5

    targets = {}
    for name, m in llm.model.named_modules():
        if not isinstance(m, nn.Linear) or "language_model.layers." not in name:
            continue
        if any(f in name for f in ["visual", "lm_head", "down_proj"]):
            continue
        li = int(name.split("layers.")[1].split(".")[0])
        if li % args.layer_step == 0 and name in adapters:
            targets[name] = (li, m)

    prepared = {}
    for name, (li, m) in targets.items():
        short = name.split(f"layers.{li}.")[-1]
        sc = scales_all[li]
        st = sc[f"{short}.text_smooth_scale"].float().cuda() + eps
        sv = sc[f"{short}.vision_smooth_scale"].float().cuda() + eps
        w = m.weight.detach().float()
        qwt = wq((w * st).to(torch.bfloat16)).float()
        qwv = wq((w * sv).to(torch.bfloat16)).float()
        L = adapters[name]["L"].float().cuda()
        R = adapters[name]["R"].float().cuda()
        prepared[name] = (st, sv, qwt, qwv, L, R, w)
    print(f"probing {len(prepared)} modules, rank of first: {next(iter(prepared.values()))[4].shape[1]}", flush=True)

    captured = {}

    def make_hook(name):
        def hook(mod, inp, out):
            captured[name] = inp[0].detach()
        return hook

    hooks = [m.register_forward_hook(make_hook(n)) for n, (li, m) in targets.items()]
    acc = {grp: {n: {"nocomp": [], "ideal": [], "cmc": [], "tokens": 0} for n in prepared}
           for grp in ("calib", "heldout")}

    img_id = llm.model.config.image_token_id
    for si in range(len(data)):
        grp = "calib" if si < args.calib_n else "heldout"
        batch = {k: v.to(llm.device) for k, v in data[si].items()}
        rows = (batch["input_ids"] == img_id).reshape(-1)
        if rows.sum() == 0:
            continue
        captured.clear()
        with torch.no_grad():
            llm.model(**batch)
            for name, (st, sv, qwt, qwv, L, R, w) in prepared.items():
                x = captured[name].reshape(-1, captured[name].shape[-1]).float()[rows]
                ref = x @ w.t()
                xt = aq((x / st).to(torch.bfloat16)).float()
                xv_raw = x / sv
                xv = aq(xv_raw.to(torch.bfloat16)).float()
                y_nocomp = xt @ qwt.t()
                y_ideal = xv @ qwv.t()
                y_cmc = xv @ qwt.t() + (xv_raw @ L) @ R
                a = acc[grp][name]
                a["nocomp"].append(sqnr(ref, y_nocomp))
                a["ideal"].append(sqnr(ref, y_ideal))
                a["cmc"].append(sqnr(ref, y_cmc))
                a["tokens"] += int(rows.sum())
        if si % 8 == 0:
            print(f"  sample {si} ({grp}) image tokens={int(rows.sum())}", flush=True)
    for h in hooks:
        h.remove()

    summary = {}
    for grp, per in acc.items():
        rows = []
        for name, a in per.items():
            if not a["nocomp"]:
                continue
            mean = lambda v: sum(v) / len(v)
            rows.append({"module": name, "nocomp": mean(a["nocomp"]), "ideal": mean(a["ideal"]),
                         "cmc": mean(a["cmc"]), "n_samples": len(a["nocomp"])})
        if not rows:
            continue
        agg = {k: sum(r[k] for r in rows) / len(rows) for k in ("nocomp", "ideal", "cmc")}
        worse = sum(r["cmc"] < r["nocomp"] for r in rows)
        summary[grp] = {"mean_db": agg, "modules": len(rows), "cmc_worse_than_nocomp": worse, "per_module": rows}
        print(f"\n[{grp}] modules={len(rows)}  nocomp {agg['nocomp']:.2f} dB  ideal {agg['ideal']:.2f} dB  "
              f"cmc {agg['cmc']:.2f} dB   (cmc < nocomp on {worse}/{len(rows)} modules)")
        worst = sorted(rows, key=lambda r: r["cmc"] - r["nocomp"])[:5]
        for r in worst:
            print(f"    {r['module']:55s} nocomp {r['nocomp']:6.2f}  ideal {r['ideal']:6.2f}  cmc {r['cmc']:6.2f}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(summary, open(args.out, "w"), indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
