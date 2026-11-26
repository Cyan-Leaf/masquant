"""汇报 PPT 用的图（全部来自已落盘的数字，不重跑实验）。

  fig7_why_one_scale.png   为什么"一套 scale"不够：真实 act_scales，视觉/文本的离群通道不是同一批（P2/P3）
  fig8_three_ties.png      三处测量、同一个结论：分模态 vs 统一 scale 打平（P5）
  fig9_audio_ladder.png    音频主结果阶梯 + SQ/RTN 崩溃（P5 或单页）
  fig10_audio_calib.png    音频校准集：纯音频校准反而更差（P6 补充）
  fig11_two_knobs.png      量化只有两个自由度（P7 骨架，替代 ASCII 图）

用法: python tools/make_slides_figs.py --src results/figs_src --outdir docs/figs
"""
import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "savefig.dpi": 160, "savefig.bbox": "tight"})
C_MAIN, C_ALT, C_FP16, C_BAD = "#1f77b4", "#ff7f0e", "#7f7f7f", "#d62728"


def fig7(src, outdir):
    """一套 scale 不能同时服务两个模态：真实 act_scales 的第 6 层 mlp.up_proj。"""
    d = np.load(os.path.join(src, "act_scale_stats.npz"))
    t, v = d["l6_text"].astype(float), d["l6_vision"].astype(float)
    keep = (t > 0.1) & (v > 0.1)  # 丢掉该模态根本不激活的通道（否则 log 轴被拉到 1e-3）
    r = v[keep] / t[keep]
    ov = d["layer_overlap"]
    share = len(set(np.argsort(-t)[:32]) & set(np.argsort(-v)[:32]))

    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8), gridspec_kw={"width_ratios": [1.3, 1]})
    a = ax[0]
    x = np.arange(t.size)
    a.plot(x, np.where(keep, t, np.nan), lw=0.7, color=C_MAIN, label="text tokens")
    a.plot(x, np.where(keep, v, np.nan), lw=0.7, color=C_ALT, alpha=0.85, label="vision tokens")
    for idx, c in ((np.argsort(-t)[:32], C_MAIN), (np.argsort(-v)[:32], C_ALT)):
        a.scatter(idx, np.minimum(t[idx] if c == C_MAIN else v[idx], 300), s=7, color=c, zorder=3)
    a.set_yscale("log")
    a.set_ylim(0.1, 300)
    a.set_xlabel("channel index (layer 6, mlp.up_proj)")
    a.set_ylabel("per-channel max |activation|  (log)")
    a.set_title("Peaks sit on different channels", fontsize=10)
    a.legend(frameon=False, fontsize=8, loc="upper right")

    a = ax[1]
    lg = np.log2(r)
    a.hist(lg, bins=60, color="#7f9fc4", edgecolor="none")
    a.axvline(0, ls="--", lw=1, color="k")
    a.axvline(np.median(lg), lw=1.4, color=C_BAD)
    a.text(np.median(lg) + 0.08, a.get_ylim()[1] * 0.92, f"median {np.median(r):.2f}x",
           fontsize=8.5, color=C_BAD)
    a.set_xlabel("log2( vision / text ) per channel")
    a.set_ylabel("channels")
    a.set_title(f"Gap per channel: {r.min():.2f}x to {r.max():.1f}x", fontsize=10)

    fig.suptitle("One smoothing scale cannot serve both modalities: "
                 f"only {share}/32 top outlier channels are shared in this layer "
                 f"({ov.mean():.0f}/32 on average over 36 layers)", fontsize=10, y=1.03)
    p = os.path.join(outdir, "fig7_why_one_scale.png")
    fig.savefig(p)
    plt.close(fig)
    return p


def fig8(outdir):
    fig, ax = plt.subplots(1, 3, figsize=(11.5, 3.6))

    # (a) TextVQA，带 95% bootstrap CI
    a = ax[0]
    vals = [("MAS\n(per-modality)", 63.47, [61.46, 65.47], C_MAIN),
            ("Uniform\n(one scale)", 64.39, [62.38, 66.40], C_ALT)]
    for i, (lab, m, ci, c) in enumerate(vals):
        a.bar(i, m, 0.62, color=c, yerr=[[m - ci[0]], [ci[1] - m]], capsize=4)
        a.text(i, m - 2.2, f"{m:.1f}", ha="center", color="white", fontsize=9)
    a.axhline(75.85, ls="--", lw=1, color=C_FP16)
    a.text(1.45, 76.4, "FP16 75.9", ha="right", fontsize=8, color=C_FP16)
    a.set_xticks([0, 1])
    a.set_xticklabels([v[0] for v in vals], fontsize=8)
    a.set_ylim(55, 80)
    a.set_ylabel("TextVQA (n=2000)")
    a.set_title("(a) downstream task", fontsize=9)

    # (b) C4 PPL，B200 口径（14 vs 12 个 checkpoint，p=0.20）
    a = ax[1]
    rows = [("MAS", 18.88, 0.50, 14, C_MAIN), ("Uniform", 19.16, 0.69, 12, C_ALT)]
    for i, (lab, m, sd, n, c) in enumerate(rows):
        a.bar(i, m, 0.62, color=c, yerr=sd, capsize=4)
        a.text(i, 0.6, f"{m:.1f}\n\u00b1{sd:.1f}\n(n={n})", ha="center", color="white", fontsize=8)
    a.set_xticks([0, 1])
    a.set_xticklabels(["MAS", "Uniform"], fontsize=8)
    a.set_xlim(-0.5, 1.5)
    a.set_ylim(0, 23)
    a.set_ylabel("C4 PPL over checkpoints")
    a.set_title("(b) text PPL (n.s., p=0.20)", fontsize=9)
    a.text(0.5, 21.4, "FP16 = 14.1", ha="center", fontsize=7.5, color=C_FP16)

    # (c) 音频 WER
    a = ax[2]
    av = [("MAS", 6.58, 0.38, C_MAIN), ("Uniform", 6.58, 0.03, C_ALT)]
    for i, (lab, m, s, c) in enumerate(av):
        a.bar(i, m, 0.62, color=c, yerr=s, capsize=4)
        a.text(i, 0.35, f"{m:.2f}\n\u00b1{s:.2f}", ha="center", color="white", fontsize=8.5)
    a.axhline(4.78, ls="--", lw=1, color=C_FP16)
    a.text(1.42, 5.05, "FP16 4.78", ha="right", fontsize=8, color=C_FP16,
           bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none"))
    a.set_xticks([0, 1])
    a.set_xticklabels(["MAS", "Uniform"], fontsize=8)
    a.set_ylim(0, 8.2)
    a.set_ylabel("LibriSpeech WER % (lower better)")
    a.set_title("(c) audio ASR, n=2676", fontsize=9)

    fig.suptitle("Three different measurements, one verdict: per-modality scales add nothing "
                 "(60/40 downstream split, PPL n.s., 6.58 vs 6.58 on ASR)", fontsize=10, y=1.04)
    p = os.path.join(outdir, "fig8_three_ties.png")
    fig.savefig(p)
    plt.close(fig)
    return p


def fig9(outdir):
    rows = [("FP16 (reference)", 4.78, C_FP16),
            ("GPTQ + clip + cd", 5.17, "#2ca02c"),
            ("trained + GPTQ", 5.49, "#2ca02c"),
            ("GPTQ (closed-form)", 5.92, "#2ca02c"),
            ("uniform + trained", 6.58, C_ALT),
            ("MAS + trained", 6.58, C_MAIN),
            ("MAS closed-form", 7.02, C_MAIN)]
    fig, ax = plt.subplots(figsize=(8.6, 3.9))
    y = np.arange(len(rows))[::-1]
    for yi, (lab, v, c) in zip(y, rows):
        ax.barh(yi, v, height=0.62, color=c, alpha=0.9)
        ax.text(v + 0.1, yi, f"{v:.2f}", va="center", fontsize=9)
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows], fontsize=9)
    ax.axvline(4.78, ls="--", lw=1, color="k")
    ax.set_xlim(0, 9.4)
    ax.set_xlabel("LibriSpeech test-other WER % (lower is better, n=2676)")
    ax.text(7.35, y[0] - 0.05, "SmoothQuant 301%\nRTN 114%\n(off scale, crash)", fontsize=8.5,
            color=C_BAD, va="top", ha="center",
            bbox=dict(boxstyle="round,pad=0.35", fc="#fdeaea", ec=C_BAD, lw=0.8))
    ax.set_title("Audio (Qwen2.5-Omni-3B, W4A8): SQ/RTN collapse, every MAS variant survives", fontsize=10)
    p = os.path.join(outdir, "fig9_audio_ladder.png")
    fig.savefig(p)
    plt.close(fig)
    return p


def fig10(outdir):
    rows = [("original calib\n(audio 19%)", 8.03, 1.61),
            ("mixed 0.5\n(audio 11%)", 6.58, 0.38),
            ("pure audio\n(audio 65%)", 8.59, 3.84)]
    fig, ax = plt.subplots(figsize=(5.6, 3.7))
    for i, (lab, m, s) in enumerate(rows):
        c = C_ALT if i == 1 else "#8c8c8c"
        ax.bar(i, m, 0.6, color=c, yerr=s, capsize=5)
        ax.text(i + 0.34, m + s + 0.15, f"{m:.2f}\u00b1{s:.2f}", ha="left", va="center", fontsize=8.5)
    ax.axhline(4.78, ls="--", lw=1, color="k")
    ax.text(-0.42, 5.0, "FP16 4.78", fontsize=8)
    ax.set_xticks(range(3))
    ax.set_xticklabels([r[0] for r in rows], fontsize=8.5)
    ax.set_xlim(-0.6, 2.85)
    ax.set_ylim(0, 13.2)
    ax.set_ylabel("LibriSpeech WER %")
    ax.set_title("More audio is not better: diversity is what matters", fontsize=10)
    ax.annotate("more audio tokens,\nworse and less stable", xy=(2.0, 10.4), xytext=(1.35, 11.8),
                fontsize=8, color=C_BAD, ha="center",
                arrowprops=dict(arrowstyle="->", color=C_BAD, lw=0.9))
    p = os.path.join(outdir, "fig10_audio_calib.png")
    fig.savefig(p)
    plt.close(fig)
    return p


def fig11(outdir):
    fig, ax = plt.subplots(figsize=(11, 4.6))
    ax.set_axis_off()
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4.8)

    def box(x, y, w, h, text, fc="#eef4fb", ec="#1f77b4", fs=8.6):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05", fc=fc, ec=ec, lw=1.1))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs)

    def arrow(x0, y0, x1, y1):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=12,
                                     lw=1.1, color="#555555"))

    ax.text(2.35, 4.45, "① step size  —  this row's grid pitch", fontsize=10.5, ha="center", weight="bold")
    ax.text(7.65, 4.45, "② code  —  which grid point a weight lands on", fontsize=10.5, ha="center", weight="bold")

    W, H = 3.5, 0.66
    box(0.6, 3.35, W, H, "default:  absmax / 7\n“the largest value must be lossless”")
    box(0.6, 2.25, W, H, "clip:  γ·absmax / 7\nshave the peak → finer grid for the rest",
        fc="#eaf7ea", ec="#2ca02c")
    arrow(2.35, 3.32, 2.35, 2.95)
    ax.text(2.35, 1.85, "smoothing (the paper, and ours)\nchanges the coordinate system —\nneither knob",
            fontsize=8.4, ha="center", va="top", color="#666666")

    box(5.9, 3.35, W, H, "default:  round to nearest (RTN)\nevery weight decided alone")
    box(5.9, 2.25, W, H, "GPTQ:  error feedback (dithering)\nkernel = inverse Hessian",
        fc="#eaf7ea", ec="#2ca02c")
    box(5.9, 1.15, W, H, "+ coordinate descent → 1-opt\nstructurally final · off by default (no gain)",
        fc="#f2f2f2", ec="#8c8c8c")
    arrow(7.65, 3.32, 7.65, 2.95)
    arrow(7.65, 2.22, 7.65, 1.85)

    ax.add_patch(FancyBboxPatch((0.6, 0.05), 8.8, 0.8, boxstyle="round,pad=0.05",
                                fc="#fff6e5", ec="#ff7f0e", lw=1.1))
    ax.text(5.0, 0.45,
            "GPTQ alone +7.2   ·   + clip → +7.5   (TextVQA 63.5 → 71.0, gap to FP16 12.4 → 4.9)"
            "\ncoordinate descent: no downstream gain, kept off  ·  same format, zero inference cost",
            ha="center", va="center", fontsize=8.6)
    p = os.path.join(outdir, "fig11_two_knobs.png")
    fig.savefig(p)
    plt.close(fig)
    return p


def fig12(outdir, summary="results/repro_b200/ppl_b200_summary.json"):
    """P6 主图（汇报口径 = 4×B200）：原校准 vs 混合校准，wikitext2 与 C4 两栏。
    与报告里的 fig1_calib_ratio.png 同结论、不同机器与口径，所以单独出一张，避免台上数字对不上。"""
    d = json.load(open(summary))
    panels = [("WikiText-2 (in-domain)", "wiki"), ("C4 (out-of-domain)", "c4")]
    fig, ax = plt.subplots(1, 2, figsize=(8.6, 3.6), sharey=True)
    for a, (title, ds) in zip(ax, panels):
        for i, (k, lab, c) in enumerate((("mas_orig", "original calib\n(text tokens 8%)", "#8c8c8c"),
                                        ("mas_mix", "modality-balanced\n(text about half)", C_ALT))):
            v = d[k][ds]
            a.bar(i, v["mean"], 0.6, color=c, yerr=v["std"], capsize=6)
            a.text(i + 0.33, v["mean"], f"{v['mean']:.1f}", ha="left", va="center", fontsize=10)
        a.set_xticks([0, 1])
        a.set_xticklabels([lab for _, lab, _ in
                           (("mas_orig", "original calib\n(text tokens 8%)", 0),
                            ("mas_mix", "modality-balanced\n(text about half)", 0))], fontsize=8.5)
        a.set_title(title, fontsize=10)
        a.set_xlim(-0.6, 1.75)
    ax[0].set_ylabel("perplexity (lower is better)")
    ax[0].set_ylim(0, 24)
    fig.suptitle("Same gain outside the calibration domain: calibration diversity, not domain matching",
                 fontsize=10, y=1.03)
    p = os.path.join(outdir, "fig12_calib_b200.png")
    fig.savefig(p)
    plt.close(fig)
    return p


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="results/figs_src")
    ap.add_argument("--outdir", default="docs/figs")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    for f in (lambda: fig7(a.src, a.outdir), lambda: fig8(a.outdir), lambda: fig9(a.outdir),
              lambda: fig10(a.outdir), lambda: fig11(a.outdir), lambda: fig12(a.outdir)):
        print(f())
