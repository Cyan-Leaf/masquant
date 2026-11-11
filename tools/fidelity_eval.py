"""视频/图像输入的量化保真度（无标签评测）。

思路：同一批样本，分别用 FP16 模型和量化模型在 teacher-forcing 下取下一个 token 的
分布，比较：
  - KL(FP16 || quant)  越低越好
  - top-1 一致率        越高越好
按模态分组：含视频的样本 vs 只含图像的样本。

这几个量对量化误差非常敏感，又不需要任何标注/判分模型，
适合在 benchmark 跑不动时做快速对比。

用法：
  python tools/fidelity_eval.py --model <path> --mas <mas_parameters.pth> \
      --mode mas --data-type text-vision-video --nsamples 8 --out outputs/fid.json
  --mode: fp16 | uniform | mas | cmc
"""
import argparse
import json
import os
import sys

import torch
import torch.nn.functional as F
from transformers import AutoProcessor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.LMClass import LMClass


def build(llm, args, mode):
    if mode == "fp16":
        return llm.model
    import copy
    from models.int_qwen_vl_layer import QuantQwenDecoderLayerV2

    # cmc 模式走 infer_mas.py 的真实推理路径（mode=infer + MAS_NO_CMC=1）：
    # 所有 token 都除以文本尺度，共享一份 Q(S_t W)，≡ uniform 的设计
    args.mode = "infer" if mode == "cmc" else "train"

    model = llm.model
    layers = model.model.language_model.layers
    scales = torch.load(args.mas, map_location="cpu") if args.mas else None
    for i in range(len(layers)):
        layer = layers[i]
        qlayer = QuantQwenDecoderLayerV2(model.config, copy.deepcopy(layer), args, layer_idx=i)
        qlayer.let = True
        qlayer.set_quant_state(weight_quant=True, act_quant=True)
        sub_qlinear = [(n, m) for n, m in qlayer.named_modules()
                       if m.__class__.__name__ == "QuantLinear"]
        if scales is not None:
            with torch.no_grad():
                for name, m in sub_qlinear:
                    short = name.split(f"layers.{i}.")[-1]
                    for mod, buf in (("text", "text_smooth_scale"), ("vision", "vision_smooth_scale"),
                                     ("audio", "audio_smooth_scale"), ("video", "video_smooth_scale")):
                        key = f"{short}.{mod}_smooth_scale"
                        if key in scales[i]:
                            getattr(m, buf).data.copy_(scales[i][key].to(getattr(m, buf).dtype))
                        elif mod == "video" and f"{short}.vision_smooth_scale" in scales[i]:
                            m.video_smooth_scale.data.copy_(
                                scales[i][f"{short}.vision_smooth_scale"].to(m.video_smooth_scale.dtype))
        # 必须在尺度加载之后再设置：uniform 模式把所有模态的尺度统一成文本尺度
        for _, sub in sub_qlinear:
            if mode == "uniform":
                with torch.no_grad():
                    sub.all_in_one_smooth_scale.data.copy_(sub.text_smooth_scale.data)
                sub.all_in_one_smooth_scale.requires_grad_(False)
            sub.inference_mode = "merged_scales" if mode == "uniform" else "split_scales"
        layers[i] = qlayer.to(llm.device)
        if mode == "cmc":
            # q_weight 依赖权重和尺度，必须在搬到 GPU 之后再构造（同 mas_quantize_model）
            with torch.no_grad():
                for _, sub in sub_qlinear:
                    sub.q_weight = sub.weight_quantizer(
                        (sub.weight.to(torch.bfloat16) * sub.text_smooth_scale).to(sub.weight.dtype))
                    sub.Lv = None
                    sub.Rv = None
                    sub.La = None
                    sub.Ra = None
    return model


@torch.no_grad()
def get_logits(model, batch, n_keep=64):
    out = model(**{k: v.to(model.device) if torch.is_tensor(v) else v for k, v in batch.items()})
    logits = out.logits if hasattr(out, "logits") else out[0]
    lp = F.log_softmax(logits[:, -(n_keep + 1):-1, :].float(), dim=-1)
    return lp


def metrics(lp_ref, lp_q):
    kl = F.kl_div(lp_q, lp_ref, log_target=True, reduction="none").sum(-1).mean().item()
    agree = (lp_ref.argmax(-1) == lp_q.argmax(-1)).float().mean().item()
    return kl, agree


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--mas", default=None)
    ap.add_argument("--modes", default="uniform,mas")
    ap.add_argument("--data-type", default="text-vision-video")
    ap.add_argument("--compare-type", default=None, help="对照的另一种数据类型（如 text-vision）")
    ap.add_argument("--nsamples", type=int, default=8)
    ap.add_argument("--wbits", type=int, default=4)
    ap.add_argument("--abits", type=int, default=8)
    ap.add_argument("--out", default="outputs/fidelity.json")
    args = ap.parse_args()

    args.weight_quant_params = {"n_bits": args.wbits, "per_channel_axes": [0], "symmetric": True,
                                "dynamic_method": "per_channel", "group_size": None,
                                "lwc": False, "disable_zero_point": False}
    args.act_quant_params = {"n_bits": args.abits, "per_channel_axes": [], "symmetric": True,
                             "dynamic_method": "per_token"}
    args.q_quant_params = {"n_bits": args.abits, "per_channel_axes": [], "symmetric": False,
                           "dynamic_method": "per_token"}
    args.k_quant_params = dict(args.q_quant_params)
    args.v_quant_params = dict(args.q_quant_params)
    args.p_quant_params = {"n_bits": 16, "metric": "fix0to1"}
    args.net = os.path.basename(args.model.rstrip("/"))
    args.model_family = args.net.split("-")[0]
    args.batch_size = 1
    args.mode = "train"
    os.environ.setdefault("MAS_NO_CMC", "1")
    args.attn_implementation = "eager"
    args.eval_sqnr = False
    args.multigpu = False
    os.environ.setdefault("inference_mode", "split_scales")

    from custom_dataset import prepare_dataset, prepare_dataset_before_quant
    processor = AutoProcessor.from_pretrained(args.model)

    # 参考：FP16 logits
    llm = LMClass(args)
    llm.model.to(llm.device).eval()
    data_types = [d for d in (args.data_type, args.compare_type) if d]
    ref = {}
    batches = {}
    for dt in data_types:
        convs = prepare_dataset(n_sample=args.nsamples, data_type=dt)
        data = prepare_dataset_before_quant(processor, convs, batch_size=1)
        batches[dt] = data
        ref[dt] = []
        for b in data:
            ref[dt].append(get_logits(llm.model, b).cpu())
        print(f"[fp16] {dt}: {len(ref[dt])} samples", flush=True)
    del llm
    torch.cuda.empty_cache()

    results = {}
    for mode in args.modes.split(","):
        llm = LMClass(args)
        llm.model.to(llm.device).eval()
        build(llm, args, mode)
        for dt in data_types:
            kl_s, ag_s = [], []
            for b, r in zip(batches[dt], ref[dt]):
                lp = get_logits(llm.model, b).cpu()
                kl, ag = metrics(r, lp)
                kl_s.append(kl)
                ag_s.append(ag)
            results[f"{mode}/{dt}"] = {"kl": sum(kl_s) / len(kl_s), "agree": sum(ag_s) / len(ag_s)}
            print(f"[{mode}] {dt}: KL={results[f'{mode}/{dt}']['kl']:.4f} "
                  f"top1-agree={100*results[f'{mode}/{dt}']['agree']:.2f}%", flush=True)
        del llm
        torch.cuda.empty_cache()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(results, open(args.out, "w"), indent=2)
    print("saved", args.out)


if __name__ == "__main__":
    main()
