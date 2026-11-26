"""汇报 PPT 的"口径 A（4×B200、SQA 2000 + TextVQA 2000、doc_id<2000）"图。

这些图与 docs/figs/fig2_main_bench.png（报告版，8×RTX、SQA 2017）**不是同一口径**，
所以单独出，不要混用。

  fig13_bench_b200.png     P4：复现（FP16 / SQ / MAS / 统一 scale，全量 2000 题 + 95% CI）
  fig14_solver_textvqa.png P7 钱图：同一 checkpoint 上 RTN → GPTQ → +clip 的 TextVQA（含 FP16 线）

数据来源（都已落盘，脚本只读不再训练）：
  results/bench_b200/boot_b200_4.json      FP16 / SQ / MAS / 统一 scale
  results/gptq/boot_solver_2000.json       RTN vs GPTQ vs clip（doc_id<2000）
  results/gptq/bench/*/samples_*.jsonl     GPTQ-only 的逐样本（用于 2000 子集）

用法: python tools/make_slides_figs_b200.py --outdir docs/figs
"""
import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "savefig.dpi": 160, "savefig.bbox": "tight"})
C_MAIN, C_ALT, C_FP16, C_SOLVER = "#1f77b4", "#ff7f0e", "#7f7f7f", "#2ca02c"


def acc_subset(path, lim=2000):
    """从逐样本 jsonl 算 doc_id<lim 的准确率。"""
    n = s = 0
    for line in open(path):
        d = json.loads(line)
        if int(d["doc_id"]) >= lim:
            continue
        n += 1
        s += float(d["exact_match"])
    return 100 * s / n


def fig13(outdir):
    d = json.load(open("results/bench_b200/boot_b200_4.json"))
    rows = [("fp16", "FP16", C_FP16), ("sq", "SmoothQuant\n+ RTN", "#d62728"),
            ("mas", "MAS\n(per-modality)", C_MAIN), ("uni", "Uniform scale\n(one scale)", C_ALT)]
    fig, ax = plt.subplots(1, 2, figsize=(9.6, 3.8))
    for a, task, title in zip(ax, ("scienceqa_img", "textvqa_val"),
                              ("ScienceQA-IMG (n=2000)", "TextVQA (n=2000)")):
        t = d[task]
        for i, (k, lab, c) in enumerate(rows):
            v = t[k]
            m, ci = 100 * v["acc"], [100 * x for x in v["ci"]]
            a.bar(i, m, 0.62, color=c, yerr=[[m - ci[0]], [ci[1] - m]], capsize=4)
            a.text(i, ci[0] - 1.5, f"{m:.1f}", ha="center", fontsize=9)
        a.set_xticks(range(len(rows)))
        a.set_xticklabels([r[1] for r in rows], fontsize=8)
        a.set_title(title, fontsize=10)
        a.set_ylim(min(100 * t[k]["ci"][0] for k, _, _ in rows) - 5, 86 if task == "scienceqa_img" else 82)
        if task == "scienceqa_img":
            a.set_ylabel("accuracy %")
    fig.suptitle("W4A8 on 4×B200, full 2000-item test sets (paired bootstrap CI)",
                 fontsize=10, y=1.03)
    for a, lo in zip(ax, (68, 55)):
        a.text(0.01, 0.02, f"y-axis from {lo}", transform=a.transAxes, fontsize=7, color="#777777")
    fig.text(0.5, -0.06,
             "MAS beats SmoothQuant by +3.4 / +2.4 (both significant)   ·   "
             "but W4A8 is still behind FP16 by 2.2 / 12.4 (both significant)",
             ha="center", fontsize=9)
    p = os.path.join(outdir, "fig13_bench_b200.png")
    fig.savefig(p)
    plt.close(fig)
    return p


def fig14(outdir):
    """P7 钱图：同一个 MAS checkpoint，只换权重求解器。全部 doc_id<2000。"""
    boot = json.load(open("results/gptq/boot_solver_2000.json"))["textvqa_val"]
    mas_gptq = acc_subset("results/gptq/bench/mas_gptq/samples_textvqa_val.jsonl")
    fp16 = 100 * boot["fp16"]["acc"]
    mas_rtn = 100 * boot["mas_rtn"]["acc"]
    mas_clip = 100 * boot["mas_clip"]["acc"]
    uni_rtn = 100 * boot["uni_rtn"]["acc"]
    uni_clip = 100 * boot["uni_clip"]["acc"]

    bars = [("MAS\n+ RTN", mas_rtn, "#9fc0dc"), ("MAS\n+ GPTQ", mas_gptq, C_SOLVER),
            ("MAS\n+ GPTQ\n+ clip", mas_clip, "#1a7a1a"),
            ("Uniform\n+ RTN", uni_rtn, "#f7c39a"), ("Uniform\n+ GPTQ\n+ clip", uni_clip, "#d9600c")]
    fig, ax = plt.subplots(figsize=(8.6, 4.0))
    for i, (lab, m, c) in enumerate(bars):
        ax.bar(i, m, 0.64, color=c, edgecolor="none")
        ax.text(i, m + 0.7, f"{m:.1f}", ha="center", fontsize=9.5)
    ax.axhline(fp16, ls="--", lw=1, color="k")
    ax.text(4.45, fp16 + 0.6, f"FP16 {fp16:.1f}", ha="right", fontsize=8.5)
    ax.set_xticks(range(len(bars)))
    ax.set_xticklabels([b[0] for b in bars], fontsize=8.5)
    ax.set_ylim(55, 82)
    ax.set_ylabel("TextVQA accuracy % (doc_id < 2000)")
    # 归因：GPTQ 是主贡献，clip 几乎不贡献（箭头画在柱子上方，不压柱）
    ytop = mas_gptq + 4.2
    ax.annotate("", xy=(1, ytop), xytext=(0, ytop),
                arrowprops=dict(arrowstyle="-|>", color=C_SOLVER, lw=1.6))
    ax.text(0.5, ytop + 0.8, f"GPTQ {mas_gptq - mas_rtn:+.1f}", ha="center",
            color=C_SOLVER, fontsize=11, weight="bold")
    ax.annotate("", xy=(2, ytop), xytext=(1, ytop),
                arrowprops=dict(arrowstyle="-|>", color="#555555", lw=1.2))
    ax.text(1.5, ytop - 1.6, f"clip {mas_clip - mas_gptq:+.1f}", ha="center", fontsize=9, color="#333333")
    fig.text(0.5, -0.05,
             f"MAS: {mas_rtn:.1f} → {mas_clip:.1f}   (GPTQ {mas_gptq - mas_rtn:+.1f}, clip {mas_clip - mas_gptq:+.1f})"
             f"   ·   gap to FP16 {fp16 - mas_rtn:.1f} → {fp16 - mas_clip:.1f}"
             f"   ·   y-axis from 55",
             ha="center", fontsize=8.8)
    ax.set_title("Same checkpoint, only the weight solver changes — GPTQ is where the gain comes from",
                 fontsize=10)
    p = os.path.join(outdir, "fig14_solver_textvqa.png")
    fig.savefig(p)
    plt.close(fig)
    return p, dict(mas_rtn=mas_rtn, mas_gptq=mas_gptq, mas_clip=mas_clip, uni_rtn=uni_rtn,
                   uni_clip=uni_clip, fp16=fp16)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="docs/figs")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    print(fig13(a.outdir))
    p, nums = fig14(a.outdir)
    print(p, {k: round(v, 2) for k, v in nums.items()})
