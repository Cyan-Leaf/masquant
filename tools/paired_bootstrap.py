"""对多组评测结果做配对 bootstrap，给出均值、95% CI 与配对差异的显著性。

python tools/paired_bootstrap.py --runs fp16=outputs_vl/bench_fp16 sq=outputs_vl/bench_sq ... \
    --tasks scienceqa_img textvqa_val --ref mas
"""
import argparse
import glob
import json
import os

import numpy as np


def load(run_dir, task, metric):
    paths = glob.glob(os.path.join(run_dir, "**", f"samples_{task}.jsonl"), recursive=True)
    if not paths:
        return None
    path = max(paths, key=os.path.getmtime)
    out = {}
    for line in open(path):
        r = json.loads(line)
        if metric in r:
            out[r["doc_id"]] = float(r[metric])
    return out


def bootstrap_mean(x, n_boot, rng):
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    return x[idx].mean(axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="name=dir")
    ap.add_argument("--tasks", nargs="+", default=["scienceqa_img", "textvqa_val"])
    ap.add_argument("--metric", default="exact_match")
    ap.add_argument("--ref", default=None, help="与哪个配置做配对比较")
    ap.add_argument("--n_boot", type=int, default=10000)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rng = np.random.default_rng(0)
    runs = dict(r.split("=", 1) for r in args.runs)
    report = {}
    for task in args.tasks:
        data = {k: load(v, task, args.metric) for k, v in runs.items()}
        data = {k: v for k, v in data.items() if v}
        if not data:
            continue
        common = sorted(set.intersection(*[set(v) for v in data.values()]))
        print(f"\n=== {task}  (配对样本数 n={len(common)}) ===")
        print(f"{'config':14s} {'acc':>7s} {'95% CI':>17s}")
        arr = {k: np.array([v[i] for i in common]) for k, v in data.items()}
        report[task] = {"n": len(common)}
        for k, x in arr.items():
            b = bootstrap_mean(x, args.n_boot, rng)
            lo, hi = np.percentile(b, [2.5, 97.5])
            print(f"{k:14s} {100*x.mean():7.2f} [{100*lo:6.2f}, {100*hi:6.2f}]")
            report[task][k] = {"acc": x.mean(), "ci": [lo, hi]}
        if args.ref and args.ref in arr:
            print(f"-- 配对差异（相对 {args.ref}）--")
            r = arr[args.ref]
            for k, x in arr.items():
                if k == args.ref:
                    continue
                d = r - x
                b = bootstrap_mean(d, args.n_boot, rng)
                lo, hi = np.percentile(b, [2.5, 97.5])
                p = 2 * min((b <= 0).mean(), (b >= 0).mean())
                sig = "显著" if (lo > 0 or hi < 0) else "不显著"
                print(f"{args.ref} - {k:10s} = {100*d.mean():+6.2f} [{100*lo:+6.2f}, {100*hi:+6.2f}]  p≈{p:.3f}  {sig}")
                report[task][f"{args.ref}-{k}"] = {"diff": d.mean(), "ci": [lo, hi], "p": p}
    if args.out:
        json.dump(report, open(args.out, "w"), indent=2, default=float)


if __name__ == "__main__":
    main()
