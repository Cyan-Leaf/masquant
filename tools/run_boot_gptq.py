"""把本次（GPTQ 求解器）的 benchmark 结果与既有基线一起做配对 bootstrap 并落盘。

用法: MASQUANT_RUN_ROOT=<跑实验的目录，含 outputs_bench/ outputs_gptq/> python tools/run_boot_gptq.py
      （默认就是仓库根目录）
输出: results/gptq/boot_*.json + 终端表格
"""
import glob
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.environ.get("MASQUANT_RUN_ROOT", ROOT)


def d(pat):
    g = glob.glob(os.path.join(R, pat))
    return os.path.dirname(sorted(g)[-1]) if g else None


runs = {
    "fp16": d("outputs_bench/b200_fp16/*/samples_scienceqa_img.jsonl"),
    "mas_rtn": d("outputs_bench/b200_mas_mix/*/samples_scienceqa_img.jsonl"),
    "mas_gptq": d("outputs_gptq/bench_mas_gptq/*/samples_scienceqa_img.jsonl"),
    "mas_clip": d("outputs_gptq/bench_mas_clip/*/samples_scienceqa_img.jsonl"),
    "single_clip": d("outputs_gptq/bench_single_clip/*/samples_scienceqa_img.jsonl"),
    "sq_clip": d("outputs_gptq/bench_sq_clip/*/samples_scienceqa_img.jsonl"),
    "sq_rtn": d("outputs_bench/b200_sq/*/samples_scienceqa_img.jsonl"),
    "uni_rtn": d("outputs_bench/b200_uni_mix/*/samples_scienceqa_img.jsonl"),
}
for n in ("plain_clip", "masinit_clip", "uni_clip", "plain_gptq"):
    p = d(f"outputs_gptq/bench_{n}/*/samples_scienceqa_img.jsonl")
    if p:
        runs[n] = p
if "--only" in sys.argv:
    keep = sys.argv[sys.argv.index("--only") + 1].split(",")
    runs = {k: v for k, v in runs.items() if k in keep}
runs = {k: v for k, v in runs.items() if v}
print(json.dumps(runs, indent=1))
out = os.path.join(ROOT, "results/gptq")
os.makedirs(out, exist_ok=True)


def boot(names, ref, tag):
    args = [sys.executable, os.path.join(ROOT, "tools/paired_bootstrap.py"),
            "--runs"] + [f"{n}={runs[n]}" for n in names] + [
            "--tasks", "scienceqa_img", "textvqa_val", "--ref", ref,
            "--out", os.path.join(out, f"boot_{tag}.json")]
    r = subprocess.run(args, capture_output=True, text=True, cwd=ROOT)
    print(r.stdout, r.stderr)


main = [n for n in ("fp16", "mas_rtn", "mas_gptq", "mas_clip", "single_clip") if n in runs]
boot(main, "mas_rtn", "solver_full")
sub = [n for n in ("fp16", "mas_rtn", "mas_clip", "plain_clip", "masinit_clip", "uni_clip", "sq_rtn",
                   "sq_clip", "uni_rtn", "plain_gptq") if n in runs]
boot(sub, "mas_clip", "solver_2000")
boot([n for n in sub if n != "mas_rtn"], "plain_clip", "solver_2000_vs_plain")
