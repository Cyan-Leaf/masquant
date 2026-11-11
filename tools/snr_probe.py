"""
按模态统计量化 SQNR 的探针脚本（轻量版）。

做法：跑一次 FP16 前向，用 hook 抓取每个线性层的输入 X，
然后用给定的平滑尺度做"平滑 + 伪量化 + 反平滑"，比较输出与 FP16 的差异，
按模态 mask 分别统计 SQNR(dB)。对应论文 Fig.2 / Theorem 1 的度量。

用法：
    python tools/snr_probe.py --model <model> --scales <mas_parameters.pth> \
        --data-type text-vision-video --nsamples 2 --wbits 4 --abits 8 \
        --modes mas,uniform,vision

三种 modes 的语义（均在同一批数据上比较）：
    mas     : 每个模态用自己的 smooth_scale（论文方法）
    uniform : 所有模态用 all_in_one_scale（SmoothQuant 式统一平滑）
    vision  : 视频/图像 token 复用 vision 尺度，文本用 text 尺度（缺少 video 尺度时的退路）
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


def build_batch(processor, data_type, nsamples):
    from custom_dataset import prepare_dataset, prepare_dataset_before_quant

    convs = prepare_dataset(n_sample=nsamples, data_type=data_type)
    return prepare_dataset_before_quant(processor, convs, batch_size=1)


def masked_sqnr(orig, quant, masks):
    out = {}
    o, q = orig.float(), quant.float()
    noise = (o - q).pow(2)
    for name, m in masks.items():
        if m is None:
            continue
        mm = m.reshape(-1).unsqueeze(-1).expand_as(o).bool()
        if mm.sum() == 0:
            continue
        sig = o[mm].pow(2).mean()
        noi = noise[mm].mean().clamp(min=1e-20)
        out[name] = round(10 * torch.log10(sig / noi).item(), 3)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--scales", required=True)
    ap.add_argument("--data-type", default="text-vision-video")
    ap.add_argument("--nsamples", type=int, default=2)
    ap.add_argument("--layer-step", type=int, default=12)
    ap.add_argument("--wbits", type=int, default=4)
    ap.add_argument("--abits", type=int, default=8)
    ap.add_argument("--modes", default="mas,uniform,vision")
    ap.add_argument("--measure", default="output", choices=["output", "act"],
                    help="output: 线性层输出 SQNR（激活+权重都量化，默认）；act: 旧口径，只量化激活")
    ap.add_argument("--out", default="outputs/snr_probe.json")
    args = ap.parse_args()

    args.weight_quant_params = {"n_bits": args.wbits, "per_channel_axes": [0], "symmetric": True,
                                "dynamic_method": "per_channel", "group_size": None,
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
    data = build_batch(processor, args.data_type, args.nsamples)
    scales_all = torch.load(args.scales, map_location="cpu")

    neg_stats = {}
    aq = UniformAffineQuantizer(**args.act_quant_params)
    wq = UniformAffineQuantizer(**args.weight_quant_params)

    modes = args.modes.split(",")
    captured = {}

    def make_hook(name, layer_idx):
        def hook(m, inp, out):
            captured.setdefault(name, []).append((inp[0].detach(), layer_idx, m))
        return hook

    layer_step = args.layer_step
    hooks = []
    for name, m in llm.model.named_modules():
        if not isinstance(m, nn.Linear):
            continue
        if any(f in name for f in ["visual", "lm_head", "audio"]):
            continue
        li = None
        for part in name.split("."):
            if part.isdigit():
                li = int(part)
        if li is None or li % layer_step != 0:
            continue
        hooks.append(m.register_forward_hook(make_hook(name, li)))

    print(f"captured {len(hooks)} linear layers", flush=True)
    results = {mode: {} for mode in modes}
    for si in range(len(data)):
        batch = {k: v.to(llm.device) for k, v in data[si].items()}
        # dataloader 里不含 mask，这里按 token id 现算
        ids = batch["input_ids"]
        cfg = llm.model.config
        image_m = (ids == cfg.image_token_id).unsqueeze(-1)
        video_m = (ids == cfg.video_token_id).unsqueeze(-1)
        audio_m = None
        text_m = ~(image_m | video_m).squeeze(-1)
        captured.clear()
        with torch.no_grad():
            llm.model(**batch)
        print(f"  sample {si}: captured {len(captured)} layers, "
              f"tokens text={int(text_m.sum())} image={int(image_m.sum())} video={int(video_m.sum())}", flush=True)
        if len(captured) == 0:
            continue
        masks = {"text": text_m.reshape(-1), "image": image_m.reshape(-1),
                 "audio": audio_m, "video": video_m.reshape(-1)}

        for name, entries in captured.items():
            inp, li, lin = entries[0]
            inp = inp.to(torch.bfloat16)
            w_fp = lin.weight.detach().float()
            x2 = inp.reshape(-1, inp.shape[-1]).float()
            ref_out = x2 @ w_fp.t() if args.measure == "output" else None
            key = f"model.language_model.layers.{li}.{name.split('.layers.')[-1]}"
            if key not in scales_all[li] and name.endswith(("q_proj",)):
                pass
            short = name.split(f"layers.{li}.")[-1]
            sc = scales_all[li]
            per_mode = {}
            for mode in modes:
                if mode == "uniform":
                    # checkpoint 里这条键叫 all_in_one_smooth_scale；早期 act_scales 用的是
                    # all_in_one_scale。两个都试，避免静默回退成文本尺度而把 uniform 算错。
                    s = sc.get(f"{short}.all_in_one_smooth_scale")
                    if s is None:
                        s = sc.get(f"{short}.all_in_one_scale")
                    if s is None:
                        s = sc.get(f"{short}.text_smooth_scale")
                    picks = {"text": s, "image": s, "audio": s, "video": s}
                elif mode == "rtn":
                    one = torch.ones_like(sc[f"{short}.text_smooth_scale"]) if f"{short}.text_smooth_scale" in sc else None
                    picks = {"text": one, "image": one, "audio": one, "video": one}
                elif mode == "vision":
                    st = sc.get(f"{short}.text_smooth_scale")
                    sv = sc.get(f"{short}.vision_smooth_scale")
                    picks = {"text": st, "image": sv, "audio": st, "video": sv}
                else:  # mas
                    picks = {
                        "text": sc.get(f"{short}.text_smooth_scale"),
                        "image": sc.get(f"{short}.vision_smooth_scale"),
                        "audio": sc.get(f"{short}.audio_smooth_scale"),
                        "video": sc.get(f"{short}.video_smooth_scale"),
                    }
                    if picks["video"] is None:
                        picks["video"] = picks["image"]
                    if picks["audio"] is None:
                        picks["audio"] = picks["text"]
                if picks["text"] is None:
                    continue
                if args.measure == "act":
                    # 旧口径：只量化激活（x/s -> Q -> *s），不量化权重。仅用于和历史结果对照。
                    out_q = torch.zeros_like(inp, dtype=torch.float32)
                    for mod, s in picks.items():
                        m = masks.get(mod)
                        if m is None or s is None:
                            continue
                        sb = s.to(inp.device, dtype=torch.float32).clamp(min=1e-5)
                        mmask = m.reshape(-1).unsqueeze(-1).expand_as(inp).bool()
                        if mmask.sum() == 0:
                            continue
                        dq = aq(inp.float() / sb) * sb
                        out_q[mmask] = dq[mmask].float()
                    per_mode[mode] = masked_sqnr(inp, out_q, masks)
                    continue
                # 输出口径（默认，对应论文 Theorem 1）：Y_q = Q(X/s) · Q(W·s)^T，与 Y = X·W^T 比较
                out_q = torch.zeros_like(ref_out)
                for mod, s in picks.items():
                    m = masks.get(mod)
                    if m is None or s is None:
                        continue
                    rows = m.reshape(-1).bool().to(x2.device)
                    if rows.sum() == 0:
                        continue
                    # 与 int_linear 的真实前向一致：除以 (s + eps)，不做 clamp。
                    # 注意训练出的尺度里有 <1% 的负值（见报告 §9），clamp 会把它们压成 1e-5
                    # 从而伪造出 0 dB，早期用 clamp 的版本因此给出过错误的 SQNR 结论。
                    sb = s.to(x2.device, dtype=torch.float32)
                    sb = sb + torch.tensor(1e-5, dtype=sb.dtype, device=sb.device)
                    n_neg = int((sb <= 0).sum().item())
                    if n_neg:
                        neg_stats[name] = neg_stats.get(name, 0) + n_neg
                    wq_s = wq((w_fp * sb).to(torch.bfloat16)).float()
                    xq = aq((x2[rows] / sb).to(torch.bfloat16)).float()
                    out_q[rows] = xq @ wq_s.t()
                per_mode[mode] = masked_sqnr(ref_out, out_q, masks)
            for mode, val in per_mode.items():
                results[mode].setdefault(li, []).append(dict(val, _layer=short))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(results, open(args.out, "w"), indent=2, ensure_ascii=False)

    # 打印汇总
    for mode in modes:
        print(f"\n=== mode: {mode} ===")
        agg = {}
        for li, vals in sorted(results[mode].items()):
            merged = {}
            for v in vals:
                for k, x in v.items():
                    if k.startswith("_"):
                        continue
                    merged.setdefault(k, []).append(x)
            avg = {k: round(sum(x) / len(x), 2) for k, x in merged.items()}
            print(f"  layer {li}: {avg}")
            for k, x in avg.items():
                agg.setdefault(k, []).append(x)
        print(f"  MEAN: { {k: round(sum(v)/len(v), 2) for k, v in agg.items()} }")
    if neg_stats:
        print(f"\n[warning] {(sum(neg_stats.values()))} 个通道的平滑因子 <= 0（按真实前向除以 s+eps，结果会失真）：",
              {k: v for k, v in list(neg_stats.items())[:6]})
    print("saved", args.out)


if __name__ == "__main__":
    main()
