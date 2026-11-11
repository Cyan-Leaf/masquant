"""按题目属性拆分量化造成的准确率损失（ScienceQA-IMG）。

核心问题：量化掉的分，是掉在"需要读图"的题上，还是"靠文字就能答"的题上？
如果 MAS 的分模态平滑真的保护了视觉，那它相对 SQ 的优势应该集中在"需要读图"的题上。

题目是否依赖读图，用 ScienceQA 自带的 hint 字段判断：hint 为空 => 文字上下文缺失，必须读图。
另外按 subject / category 拆分，找量化最伤的题型。
"""
import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np


def load_em(run_dir, task):
    paths = glob.glob(os.path.join(run_dir, "**", f"samples_{task}.jsonl"), recursive=True)
    path = max(paths, key=os.path.getmtime)
    return {json.loads(l)["doc_id"]: float(json.loads(l)["exact_match"]) for l in open(path)}


def boot_diff(a, b, n=5000, seed=0):
    """配对 bootstrap：a-b 的均值差及 95% CI。"""
    a, b = np.asarray(a), np.asarray(b)
    d = a - b
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n, len(d)))
    m = d[idx].mean(axis=1)
    return d.mean() * 100, np.percentile(m, 2.5) * 100, np.percentile(m, 97.5) * 100


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--parquet", default=os.environ.get("SQA_PARQUET", ""))
    ap.add_argument("--out", default="outputs/sqa_breakdown.json")
    args = ap.parse_args()

    from datasets import load_dataset
    ds = load_dataset("parquet", data_files=args.parquet, split="train").remove_columns(["image"])
    meta = ds.to_list()

    runs = {}
    for spec in args.runs:
        name, d = spec.split("=", 1)
        runs[name] = load_em(d, "scienceqa_img")
    names = list(runs)
    common = sorted(set.intersection(*[set(v) for v in runs.values()]))

    def group_key_hint(m):
        h = (m.get("hint") or "").strip()
        return "无文字提示（必须读图）" if not h else "有文字提示"

    groupings = {
        "是否依赖读图": group_key_hint,
        "subject": lambda m: m.get("subject") or "?",
    }
    out = {}
    for gname, fn in groupings.items():
        buckets = defaultdict(list)
        for k in common:
            buckets[fn(meta[k])].append(k)
        print(f"\n===== 按 {gname} 拆分 =====")
        hdr = f"{'分组':18s} {'n':>5s} " + " ".join(f"{n:>8s}" for n in names)
        print(hdr)
        out[gname] = {}
        for g, ks in sorted(buckets.items(), key=lambda x: -len(x[1])):
            accs = {n: np.mean([runs[n][k] for k in ks]) * 100 for n in names}
            print(f"{g:18s} {len(ks):5d} " + " ".join(f"{accs[n]:8.2f}" for n in names))
            out[gname][g] = {"n": len(ks), "acc": accs}
            # 关键对比：MAS vs SQ、FP16 vs MAS 在这个分组里的差
            if "mas" in runs and "sq" in runs:
                d, lo, hi = boot_diff([runs["mas"][k] for k in ks], [runs["sq"][k] for k in ks])
                out[gname][g]["mas_minus_sq"] = [d, lo, hi]
                print(f"{'':18s}        MAS−SQ = {d:+.2f} [{lo:+.2f},{hi:+.2f}]"
                      + ("  显著" if lo > 0 or hi < 0 else ""))
            if "fp16" in runs and "mas" in runs:
                d, lo, hi = boot_diff([runs["fp16"][k] for k in ks], [runs["mas"][k] for k in ks])
                out[gname][g]["fp16_minus_mas"] = [d, lo, hi]
                print(f"{'':18s}        FP16−MAS = {d:+.2f} [{lo:+.2f},{hi:+.2f}]"
                      + ("  显著" if lo > 0 or hi < 0 else ""))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2, ensure_ascii=False)
    print("\nsaved", args.out)


if __name__ == "__main__":
    main()
