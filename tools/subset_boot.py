"""在同一个 doc_id 子集（默认前 2000 题）上重算配对 bootstrap，使全长跑与只跑了前 2000 题的配置可比。

用法: python tools/subset_boot.py 2000 fp16=... mas_clip=... --ref plain_clip
"""
import glob
import json
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paired_bootstrap import bootstrap_mean, load  # noqa: E402

n = int(sys.argv[1])
rest = sys.argv[sys.argv.index("--ref") + 2:]
ref = sys.argv[sys.argv.index("--ref") + 1]
runs = dict(r.split("=", 1) for r in sys.argv[2:sys.argv.index("--ref")])
for task in ("scienceqa_img", "textvqa_val"):
    data = {k: {d: v for d, v in load(p, task, "exact_match").items() if int(d) < n} for k, p in runs.items()}
    data = {k: v for k, v in data.items() if v}
    common = sorted(set.intersection(*[set(v) for v in data.values()]), key=int)
    arr = {k: np.array([v[i] for i in common]) for k, v in data.items()}
    rng = np.random.default_rng(0)
    print(f"\n=== {task}  n={len(common)} ===")
    for k, x in arr.items():
        b = bootstrap_mean(x, 10000, rng)
        print(f"{k:14s} {100*x.mean():7.2f}")
    r = arr[ref]
    for k, x in arr.items():
        if k == ref:
            continue
        d = r - x
        b = bootstrap_mean(d, 10000, rng)
        lo, hi = np.percentile(b, [2.5, 97.5])
        p = 2 * min((b <= 0).mean(), (b >= 0).mean())
        print(f"{ref} - {k:10s} = {100*d.mean():+6.2f} [{100*lo:+6.2f}, {100*hi:+6.2f}] p≈{p:.3f} "
              f"{'显著' if (lo > 0 or hi < 0) else '不显著'}")
