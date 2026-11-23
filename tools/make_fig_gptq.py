"""Fig.5 / Fig.6：平滑之后换权重求解器（RTN → GPTQ → GPTQ+clip）。数据由 tools/gptq_summary.py --json 生成。

fig5_solver_ppl.png    各平滑方式 × 求解器的 C4 PPL（W4A8，混合校准）
fig6_calib_gptq.png    原校准(文本 8%) vs 混合校准下 RTN/GPTQ 的 C4 PPL：GPTQ 只在模态平衡的校准集上有效

用法: python tools/make_fig_gptq.py results/gptq/ppl_summary.json docs/figs
"""
import json
import statistics as st
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "savefig.dpi": 160, "savefig.bbox": "tight"})
runs = json.load(open(sys.argv[1]))["runs"]
out = sys.argv[2]


def vals(prefix, suffix, ds="c4"):
    return [r[ds] for k, r in runs.items() if k.startswith(prefix) and k.endswith("_" + suffix) and ds in r
            and not k.startswith("mas_val_mas_mix_r1")]


def ms(v):
    return (st.mean(v), st.stdev(v) if len(v) > 1 else 0.0) if v else (np.nan, 0)


rows = [("No smoothing", lambda s: [runs[f"plain_{s}"]["c4"]] if f"plain_{s}" in runs else []),
        ("SmoothQuant", lambda s: [runs[f"sq_{s}"]["c4"]] if f"sq_{s}" in runs else []),
        ("MAS closed-form", lambda s: [runs[f"masinit_{s}"]["c4"]] if f"masinit_{s}" in runs else []),
        ("Uniform trained", lambda s: vals("uni_", s)),
        ("MAS trained", lambda s: vals("mas_", s))]
solvers = [("rtn", "RTN"), ("gptq", "GPTQ"), ("clip", "GPTQ+clip")]
fig, ax = plt.subplots(figsize=(8, 3.6))
w = 0.26
for j, (s, lab) in enumerate(solvers):
    m, e = zip(*[ms(f(s)) for _, f in rows])
    m = [min(x, 26) for x in m]
    ax.bar(np.arange(len(rows)) + (j - 1) * w, m, w, yerr=e, capsize=2, label=lab)
ax.axhline(14.05, ls="--", c="k", lw=0.8, label="FP16 (14.05)")
ax.set_xticks(np.arange(len(rows)))
ax.set_xticklabels([r[0] for r in rows])
ax.set_ylim(12, 26)
ax.set_ylabel("C4 PPL (W4A8)")
ax.text(-0.26, 25.3, "5004", ha="center", fontsize=8)
ax.legend(ncol=4, fontsize=8, loc="upper right")
ax.set_title("Weight solver after smoothing (mixed calibration, trained: mean±std over checkpoints)")
fig.savefig(f"{out}/fig5_solver_ppl.png")

fig, ax = plt.subplots(figsize=(5.5, 3.4))
cats = [("Original calib\n(text 8%)", "orig_"), ("Mixed calib\n(text 51%)", "mas_")]
for j, (s, lab) in enumerate([("rtn", "RTN"), ("gptq", "GPTQ"), ("clip", "GPTQ+clip")]):
    m, e = zip(*[ms(vals(p, s)) for _, p in cats])
    ax.bar(np.arange(2) + (j - 1) * w, m, w, yerr=e, capsize=3, label=lab)
ax.set_xticks(range(2))
ax.set_xticklabels([c[0] for c in cats])
ax.set_ylim(14, 24)
ax.set_ylabel("C4 PPL (trained MAS, W4A8)")
ax.legend(fontsize=8)
ax.set_title("GPTQ needs modality-balanced calibration")
fig.savefig(f"{out}/fig6_calib_gptq.png")
print("ok")
