"""从逐样本 jsonl 直接算某个配置在 TextVQA/SQA 上的准确率（同一批 doc_id、同一口径）。

用法: python tools/acc_from_samples.py results/gptq/bench/mas_gptq/samples_textvqa_val.jsonl
"""
import json
import sys


def score(x):
    for k in ("exact_match", "acc", "score", "correct", "accuracy"):
        if k in x:
            v = x[k]
            if isinstance(v, bool):
                return float(v)
            if isinstance(v, (int, float)):
                return float(v)
            if isinstance(v, (list, tuple)) and v and isinstance(v[0], (int, float, bool)):
                return float(v[0])
            if isinstance(v, str):
                try:
                    return float(json.loads(v)[0])
                except Exception:
                    return float(v) if v.replace(".", "").isdigit() else 0.0
    return 0.0


for p in sys.argv[1:]:
    n = s = 0
    for line in open(p):
        n += 1
        s += score(json.loads(line))
    print(f"{p}\tn={n}\tacc={100 * s / n:.2f}")
