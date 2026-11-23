"""汇总 gptq_ppl.sh 的日志：logs/gptq_<tag>.log -> 表格 + 配对检验（同一 checkpoint 上 RTN vs GPTQ）。
用法: python tools/gptq_summary.py [logs 目录] [--json out.json]
"""
import glob, gzip, json, math, os, re, sys

LOGDIR = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else "logs"
res = {}
for f in sorted(glob.glob(os.path.join(LOGDIR, "gptq_*.log")) + glob.glob(os.path.join(LOGDIR, "gptq_*.log.gz"))):
    base = os.path.basename(f)
    tag = base[5:-7] if base.endswith(".gz") else base[5:-4]
    txt = (gzip.open(f, "rt", errors="ignore") if f.endswith(".gz") else open(f, errors="ignore")).read()
    r = {}
    for ds in ("wikitext2", "c4"):
        m = re.findall(rf"INFO {ds} : ([0-9.e+]+)", txt)
        if m:
            r[ds] = float(m[-1])
    rec = [float(x) for x in re.findall(r"rec_final=([0-9.\-]+)%", txt)]
    if rec:
        r["rec_mean"] = sum(rec) / len(rec)
    if r:
        res[tag] = r


def paired(prefix, keys, a, b, ds):
    d = []
    for k in keys:
        ra, rb = res.get(f"{prefix}_{k}_{a}"), res.get(f"{prefix}_{k}_{b}")
        if ra and rb and ds in ra and ds in rb:
            d.append((ra[ds], rb[ds]))
    if len(d) < 2:
        return None
    x = [p[0] for p in d]
    y = [p[1] for p in d]
    diff = [q - p for p, q in d]
    n = len(d)
    md = sum(diff) / n
    sd = math.sqrt(sum((t - md) ** 2 for t in diff) / (n - 1))
    t = md / (sd / math.sqrt(n)) if sd > 0 else float("inf")
    mean = lambda v: sum(v) / len(v)
    std = lambda v: math.sqrt(sum((t_ - mean(v)) ** 2 for t_ in v) / (len(v) - 1))
    return dict(n=n, a_mean=mean(x), a_std=std(x), b_mean=mean(y), b_std=std(y), diff=md, diff_sd=sd, t=t,
                wins=sum(1 for v in diff if v < 0))


if __name__ == "__main__":
    for k in sorted(res):
        r = res[k]
        print(f"{k:40s} " + " ".join(f"{d}={r[d]:.3f}" for d in ("wikitext2", "c4") if d in r)
              + (f" rec={r['rec_mean']:.1f}%" if "rec_mean" in r else ""))
    groups = {}
    for k in res:
        m = re.match(r"(mas|uni|orig|single|xcal)_(.+)_(rtn|gptq|clip|clipcd|actq|bal|rtnclip|hall)$", k)
        if m:
            groups.setdefault(m.group(1), set()).add(m.group(2))
    out = {"runs": res, "paired": {}}
    print("\n== paired (b - a), t = paired t statistic ==")
    for g, keys in groups.items():
        for a, b in (("rtn", "gptq"), ("rtn", "rtnclip"), ("rtnclip", "clip"), ("gptq", "hall"), ("rtn", "hall"), ("gptq", "clip"), ("gptq", "clipcd"), ("gptq", "actq"), ("gptq", "bal"), ("rtn", "bal")):
            for ds in ("wikitext2", "c4"):
                p = paired(g, sorted(keys), a, b, ds)
                if p:
                    out["paired"][f"{g}:{b}-{a}:{ds}"] = p
                    print(f"{g:7s} {b:7s}-{a:4s} {ds:9s} n={p['n']:2d} {p['a_mean']:.2f}±{p['a_std']:.2f} -> "
                          f"{p['b_mean']:.2f}±{p['b_std']:.2f}  Δ={p['diff']:+.3f}±{p['diff_sd']:.3f} t={p['t']:.2f} "
                          f"wins={p['wins']}/{p['n']}")
    if "--json" in sys.argv:
        json.dump(out, open(sys.argv[sys.argv.index("--json") + 1], "w"), indent=1)
