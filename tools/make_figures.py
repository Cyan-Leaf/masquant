"""生成报告用的 4 张图（全部基于已落盘的实验数字，不重新跑实验）。

  fig1_calib_ratio.png    校准集纯文本占比 vs PPL（wikitext2 + C4 双领域，带误差棒）
  fig2_main_bench.png     全量 benchmark 主结果（带 95% CI）
  fig3_breakdown.png      ScienceQA 按"是否必须读图"拆分的量化损失
  fig4_sqnr.png           修复后的分模态 SQNR（288 配对）

图中统一用英文（服务器无中文字体）。
用法: python tools/make_figures.py --outdir docs/figs
"""
import argparse
import json
import os
import statistics as st

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "savefig.dpi": 160, "savefig.bbox": "tight"})


def ms(v):
    return st.mean(v), (st.stdev(v) if len(v) > 1 else 0.0)


def fig1(outdir):
    # wikitext：匹配激活统计的比例扫描（报告 §2.1.1）
    wiki = {0.0: [17.03, 18.46, 19.89], 0.125: [18.754, 20.728, 21.296], 0.25: [22.373, 17.043, 16.603],
            0.375: [17.045, 16.145, 17.028], 0.5: [16.082, 17.302, 16.671, 15.957, 16.220, 15.879],
            0.75: [15.855, 16.525, 16.176], 0.875: [15.769, 16.563, 16.533]}
    # C4：跨领域验证（报告 §2.1.3）
    c4 = {0.0: [20.249, 20.007, 20.069, 20.526], 0.5: [18.429, 18.649, 18.500], 0.75: [18.414, 19.339, 18.547]}
    tok = {0.0: 8.1, 0.125: 18.3, 0.25: 28.9, 0.375: 40.2, 0.5: 51.4, 0.75: 74.9, 0.875: 87.1}

    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    for a, data, name, fp in ((ax[0], wiki, "WikiText-2", 11.11), (ax[1], c4, "C4 (out-of-domain)", 14.05)):
        xs = sorted(data)
        m = [ms(data[x])[0] for x in xs]
        s = [ms(data[x])[1] for x in xs]
        a.errorbar([tok[x] for x in xs], m, yerr=s, marker="o", capsize=4, color="#1f77b4", label="MAS (W4A8)")
        a.axhline(fp, ls="--", color="gray", lw=1, label=f"FP16 ({fp})")
        a.set_xlabel("text-token share in calibration set (%)")
        a.set_ylabel("perplexity (lower is better)")
        a.set_title(name)
        a.legend(frameon=False, fontsize=8)
    ax[0].axvspan(40, 90, color="#2ca02c", alpha=0.08)
    ax[0].text(62, max(ms(wiki[0.25])[0], 19) + 0.5, "plateau", ha="center", color="#2ca02c", fontsize=9)
    fig.suptitle("Fig.1  Modality-balanced calibration: gain holds out of domain (C4: -1.69, p=1e-4)", fontsize=10, y=1.04)
    p = os.path.join(outdir, "fig1_calib_ratio.png")
    fig.savefig(p)
    plt.close(fig)
    return p


def fig2(outdir, boot_path):
    d = json.load(open(boot_path))
    order = [("fp16", "FP16"), ("sq", "SmoothQuant"), ("mas05", "MAS (r=0.5)"), ("mas075", "MAS (r=0.75)"),
             ("uniform05", "Uniform (r=0.5)"), ("uniform075", "Uniform (r=0.75)"), ("mas_lwc", "MAS+LWC")]
    colors = ["#7f7f7f", "#d62728", "#1f77b4", "#1f77b4", "#ff7f0e", "#ff7f0e", "#9467bd"]
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
    for a, task, title in ((ax[0], "scienceqa_img", "ScienceQA-IMG (n=2017)"), (ax[1], "textvqa_val", "TextVQA (n=2000)")):
        t = d[task]
        accs = [t[k]["acc"] * 100 for k, _ in order]
        lo = [accs[i] - t[k]["ci"][0] * 100 for i, (k, _) in enumerate(order)]
        hi = [t[k]["ci"][1] * 100 - accs[i] for i, (k, _) in enumerate(order)]
        x = np.arange(len(order))
        a.bar(x, accs, yerr=[lo, hi], color=colors, capsize=3, alpha=0.85)
        a.set_xticks(x)
        a.set_xticklabels([n for _, n in order], rotation=30, ha="right", fontsize=8)
        a.set_ylim(min(accs) - 6, max(accs) + 3)
        a.set_ylabel("accuracy (%)")
        a.set_title(title)
        for i, v in enumerate(accs):
            a.text(i, v + hi[i] + 0.3, f"{v:.1f}", ha="center", fontsize=7)
    fig.suptitle("Fig.2  W4A8 full-set results with 95% bootstrap CI  (MAS > SQ significant; MAS = Uniform)", fontsize=10, y=1.03)
    p = os.path.join(outdir, "fig2_main_bench.png")
    fig.savefig(p)
    plt.close(fig)
    return p


def fig3(outdir, bd_path):
    d = json.load(open(bd_path))["是否依赖读图"]
    groups = [("有文字提示", "text hint given\n(n=1257)"), ("无文字提示（必须读图）", "no hint: must read image\n(n=760)")]
    cfg = [("fp16", "FP16", "#7f7f7f"), ("sq", "SmoothQuant", "#d62728"), ("mas", "MAS (r=0.5)", "#1f77b4"), ("uni", "Uniform (r=0.5)", "#ff7f0e")]
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6), gridspec_kw={"width_ratios": [1.4, 1]})
    w = 0.2
    for gi, (g, lab) in enumerate(groups):
        for ci, (k, name, c) in enumerate(cfg):
            ax[0].bar(gi + (ci - 1.5) * w, d[g]["acc"][k], w, color=c, label=name if gi == 0 else None, alpha=0.85)
    ax[0].set_xticks([0, 1])
    ax[0].set_xticklabels([lab for _, lab in groups])
    ax[0].set_ylim(65, 92)
    ax[0].set_ylabel("accuracy (%)")
    ax[0].legend(frameon=False, fontsize=8, ncol=2)
    ax[0].set_title("accuracy by question type")

    # 右图：FP16 - MAS（量化剩余损失）与 MAS - SQ
    labs = ["text hint", "must read image"]
    for i, (key, name, c) in enumerate((("fp16_minus_mas", "FP16 − MAS (remaining loss)", "#1f77b4"),
                                         ("mas_minus_sq", "MAS − SQ (gain over SQ)", "#2ca02c"))):
        for gi, (g, _) in enumerate(groups):
            v, lo, hi = d[g][key]
            ax[1].errorbar(gi + (i - 0.5) * 0.25, v, yerr=[[v - lo], [hi - v]], fmt="o", color=c, capsize=4,
                           label=name if gi == 0 else None)
    ax[1].axhline(0, color="black", lw=0.8)
    ax[1].set_xticks([0, 1])
    ax[1].set_xticklabels(labs)
    ax[1].set_ylabel("accuracy difference (pts, 95% CI)")
    ax[1].legend(frameon=False, fontsize=8)
    ax[1].set_title("where is the quantization loss?")
    fig.suptitle("Fig.3  Remaining W4A8 loss concentrates on image-dependent questions (ScienceQA)", fontsize=10, y=1.03)
    p = os.path.join(outdir, "fig3_breakdown.png")
    fig.savefig(p)
    plt.close(fig)
    return p


def fig4(outdir, snr_path):
    d = json.load(open(snr_path))
    mods = ["text", "image", "video"]
    modes = [("rtn", "RTN (no smoothing)", "#7f7f7f"), ("uniform", "Uniform scale", "#ff7f0e"), ("mas", "MAS", "#1f77b4")]
    vals = {}
    for mode, _, _ in modes:
        per = {m: [] for m in mods}
        for li, items in d[mode].items():
            for v in items:
                for m in mods:
                    if m in v:
                        per[m].append(v[m])
        vals[mode] = {m: (np.mean(per[m]), np.std(per[m]) / np.sqrt(len(per[m]))) for m in mods}
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    w = 0.26
    x = np.arange(len(mods))
    for i, (mode, name, c) in enumerate(modes):
        ax.bar(x + (i - 1) * w, [vals[mode][m][0] for m in mods], w, yerr=[vals[mode][m][1] for m in mods],
               color=c, capsize=3, label=name, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(mods)
    ax.set_ylim(14.5, 17.8)
    ax.set_ylabel("output SQNR (dB, higher is better)")
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("Fig.4  Per-modality output SQNR, W4A8 (288 layer x sample pairs)\n"
                 "error bar = unpaired s.e.m.; paired 95% CI of MAS-Uniform excludes 0 for all modalities", fontsize=8)
    p = os.path.join(outdir, "fig4_sqnr.png")
    fig.savefig(p)
    plt.close(fig)
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="docs/figs")
    ap.add_argument("--boot", default="outputs_bench/full_bootstrap.json")
    ap.add_argument("--breakdown", default="outputs/sqa_breakdown.json")
    ap.add_argument("--snr", default="outputs/snr_big_075.json")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    for f in (fig1(args.outdir), fig2(args.outdir, args.boot), fig3(args.outdir, args.breakdown),
              fig4(args.outdir, args.snr)):
        print("saved", f)


if __name__ == "__main__":
    main()
