"""从逐样本评测结果里提取定性错误分析：找到量化后翻转的样本，按"是否依赖读图"分类。

用法:
  python tools/error_analysis.py --runs fp16=outputs_bench/full_fp16 mas=outputs_bench/full_mas075 \
      --task scienceqa_img --out outputs/err_sqa.json --topk 8
"""
import argparse
import glob
import json
import os
import re


def load(run_dir, task):
    paths = glob.glob(os.path.join(run_dir, "**", f"samples_{task}.jsonl"), recursive=True)
    if not paths:
        return {}
    path = max(paths, key=os.path.getmtime)
    out = {}
    for line in open(path):
        r = json.loads(line)
        out[r["doc_id"]] = r
    return out


def load_dataset_index(task, n_expected=None):
    """逐样本 jsonl 只存了 doc_id/exact_match/resp，题目要从数据集按 index 取。

    本地 local_datasets 下是哈希文件名，无法按名字判断，所以按样本数匹配：
    SQA-IMG test = 2017 条（c40f...），TextVQA = 5000 条分 3 片。
    """
    from datasets import load_dataset
    cands = sorted(glob.glob(os.environ.get("LOCAL_DATASETS_GLOB",
                             os.path.join(os.environ.get("MASQUANT_DATA_ROOT", "data"),
                                          "hf_datasets", "local_datasets", "*.parquet"))))
    # 按任务挑候选
    if "scienceqa" in task:
        cands = [c for c in cands if "c40f3b135be2f32f" in c] + cands
    elif "textvqa" in task:
        cands = [c for c in cands if "validation-00000-of-00003" in c] + cands
    for p in cands:
        try:
            ds = load_dataset("parquet", data_files=p, split="train")
        except Exception:
            continue
        if n_expected is not None and len(ds) != n_expected:
            continue
        if "question" in ds.column_names or "text" in ds.column_names:
            print(f"题目来源: {os.path.basename(p)} ({len(ds)} 条)")
            return ds
    return None


def brief(r, doc=None, n=110):
    """把样本记录压成一行可读文本。"""
    doc = doc or {}
    q = doc.get("question") or doc.get("text") or ""
    choices = doc.get("choices") or doc.get("options") or []
    if isinstance(choices, list) and choices and isinstance(choices[0], (list, tuple)):
        choices = [c[0] if isinstance(c, (list, tuple)) else c for c in choices]
    hint = doc.get("hint") or ""
    if isinstance(hint, list):
        hint = " ".join(hint)
    resp = r.get("resp") or r.get("resps") or r.get("filtered_resps") or r.get("response") or ""
    if isinstance(resp, list):
        resp = resp[0] if resp else ""
    resp = re.sub(r"\s+", " ", str(resp))[:n]
    return {
        "question": str(q)[:200],
        "hint": str(hint)[:160],
        "choices": [str(c)[:60] for c in (choices or [])][:6],
        "response": resp,
        "target": str(doc.get("answer", ""))[:40],
        "correct": float(r.get("exact_match", 0)),
    }


DOCS = None


def main():
    global DOCS
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="name=dir")
    ap.add_argument("--task", default="scienceqa_img")
    ap.add_argument("--ref", default=None, help="参照配置（默认第一个）")
    ap.add_argument("--topk", type=int, default=8)
    ap.add_argument("--out", default="outputs/error_analysis.json")
    args = ap.parse_args()

    runs = {}
    for spec in args.runs:
        name, d = spec.split("=", 1)
        runs[name] = load(d, args.task)
    ref = args.ref or list(runs)[0]

    n_exp = 2017 if "scienceqa" in args.task else (5000 if "textvqa" in args.task else None)
    DOCS = load_dataset_index(args.task, n_exp)
    common = set(runs[ref])
    for name, data in runs.items():
        common &= set(data)
    print(f"{args.task}: {len(common)} 个共有样本，参照={ref}")

    acc = {name: sum(d[k]["exact_match"] for k in common) / len(common) for name, d in runs.items()}
    for name, a in acc.items():
        print(f"  {name:12s} acc={a*100:.2f}")

    # 找出"参照答对、其他配置答错"的样本
    flips = {}
    for name, data in runs.items():
        if name == ref:
            continue
        # 名字和含义要对上：me_wrong = 参照对、本配置错
        me_wrong = [k for k in common if runs[ref][k]["exact_match"] >= 1 and data[k]["exact_match"] < 1]
        ref_wrong = [k for k in common if runs[ref][k]["exact_match"] < 1 and data[k]["exact_match"] >= 1]
        flips[name] = {"ref_right_me_wrong": len(me_wrong), "ref_wrong_me_right": len(ref_wrong),
                       "_bad": me_wrong, "_good": ref_wrong}
        print(f"  {ref} 对 / {name} 错: {len(me_wrong)}   "
              f"{ref} 错 / {name} 对: {len(ref_wrong)}   （共切换 {len(me_wrong)+len(ref_wrong)} 题）")

    # 导出可读案例
    out = {"task": args.task, "n": len(common), "acc": acc, "flips": {}}
    for name, f in flips.items():
        bad = f["_bad"][: args.topk]
        out["flips"][name] = {
            "ref_right_me_wrong": f["ref_right_me_wrong"],
            "ref_wrong_me_right": f["ref_wrong_me_right"],
            "cases": [],
        }
        for k in bad:
            doc = DOCS[k] if DOCS is not None and k < len(DOCS) else None
            c = {"doc_id": k, ref: brief(runs[ref][k], doc)}
            for other in runs:
                if other != ref:
                    c[other] = brief(runs[other][k], doc)
            out["flips"][name]["cases"].append(c)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2, ensure_ascii=False)
    print("saved", args.out)

    # 打印前几个案例
    for name, v in out["flips"].items():
        print(f"\n===== {ref} 对、{name} 错（前 {min(3, len(v['cases']))} 例）=====")
        for c in v["cases"][:3]:
            print(f"[{c['doc_id']}] Q: {c[ref]['question'][:100]}")
            print(f"   target={c[ref]['target']}")
            for who in runs:
                print(f"   {who:8s}: {c[who]['response'][:90]}")


if __name__ == "__main__":
    main()
