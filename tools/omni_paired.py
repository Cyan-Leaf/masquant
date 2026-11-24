"""Omni LibriSpeech WER 的配对比较：按音频对齐、只在共同子集上算，配对 bootstrap 给 95% CI 与双侧 p。

用法：
  python tools/omni_paired.py --ref A.jsonl B.jsonl C.jsonl ...      # 每个文件都与 ref 配对
  python tools/omni_paired.py --group mas=r1.jsonl,r2.jsonl,r3.jsonl --group mix=... [--ref fp16.jsonl]
    --group：同一配置的多次重复，报 mean ± std，并把各重复的逐条错误数取平均后与其他组配对比较。
WER 是语料级（总错误词数 / 总参考词数），bootstrap 以"条"为单位重采样。
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from omni_wer import _open, edit, norm, ref_of  # noqa: E402


def load(path):
    out = {}
    for line in _open(path):
        d = json.loads(line)
        audio = next(c["audio"] for m in d["prompt"] if m["role"] == "user"
                     for c in m["content"] if c["type"] == "audio")
        uid = os.path.basename(audio).rsplit(".", 1)[0]
        r, h = norm(ref_of(audio)), norm(d["response"])
        out[uid] = (edit(r, h), len(r))
    return out


def boot(ea, eb, n_ref, iters=10000, seed=0):
    rng = np.random.default_rng(seed)
    n = len(n_ref)
    idx = rng.integers(0, n, size=(iters, n))
    da = ea[idx].sum(1) / n_ref[idx].sum(1)
    db = eb[idx].sum(1) / n_ref[idx].sum(1)
    d = 100 * (da - db)
    lo, hi = np.percentile(d, [2.5, 97.5])
    p = 2 * min((d <= 0).mean(), (d >= 0).mean())
    return lo, hi, max(p, 1.0 / iters)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="*")
    ap.add_argument("--ref", default=None)
    ap.add_argument("--group", action="append", default=[])
    ap.add_argument("--iters", type=int, default=10000)
    a = ap.parse_args()

    groups = {}
    for g in a.group:
        name, fs = g.split("=", 1)
        groups[name] = [load(f) for f in fs.split(",")]
    for f in a.files:
        groups[os.path.basename(f).rsplit(".", 1)[0]] = [load(f)]
    if a.ref:
        groups = {"REF": [load(a.ref)], **groups}

    common = set.intersection(*[set(r) for rs in groups.values() for r in rs])
    uids = sorted(common)
    nref = np.array([groups[next(iter(groups))][0][u][1] for u in uids], dtype=float)
    print(f"共同子集 n={len(uids)}，参考词数 {int(nref.sum())}")

    errs = {}
    for name, rs in groups.items():
        per = np.array([[r[u][0] for u in uids] for r in rs], dtype=float)
        wers = 100 * per.sum(1) / nref.sum()
        errs[name] = per.mean(0)
        s = f"{wers.mean():.2f}" + (f" ± {wers.std(ddof=1):.2f}（{len(rs)} 次: " + ", ".join(f"{w:.2f}" for w in wers) + "）" if len(rs) > 1 else "")
        print(f"  {name:<20s} WER = {s}")

    names = list(groups)
    print("配对差异（行 − 列，WER 点，95% CI，p）：")
    for i, x in enumerate(names):
        for y in names[i + 1:]:
            d = 100 * (errs[x].sum() - errs[y].sum()) / nref.sum()
            lo, hi, p = boot(errs[x], errs[y], nref, a.iters)
            print(f"  {x} − {y}: {d:+.2f} [{lo:+.2f}, {hi:+.2f}] p={p:.4f}")


if __name__ == "__main__":
    main()
