"""为 Qwen2.5-Omni 构造校准集与 LibriSpeech 评测集（jsonl，每行 {"prompt": [对话]}）。

输入：
  $MASQUANT_DATA_ROOT/librispeech/LibriSpeech/test-other/<spk>/<chap>/*.flac + *.trans.txt
  $MASQUANT_DATA_ROOT/coco/train2017/*.jpg 与仓库内 ShareGPT4V-COCO 标注（图像描述）
  $MASQUANT_DATA_ROOT/wikitext/train-*.parquet（仅 --text_ratio > 0 时用）

输出（$MASQUANT_DATA_ROOT/jsonls/）：
  libri_test_other_eval.jsonl   评测集：按说话人切分，与校准集无重叠
  libri_test_other_calib.jsonl  音频校准池（ASR 对话，含参考转写作为 assistant 回答）
  omnibench.jsonl               三模态校准集（原口径近似）：音频 ASR / 图像描述 / 音频+图像 三类轮换
  omnibench_mix{r}.jsonl        混合校准：在上面基础上把比例 r 的样本替换为纯文本（wikitext）

评测集不带 system prompt：Omni 的默认"虚拟人" system prompt 会让约 15% 的短句被当成闲聊来回答
（FP16 下 WER 从 ~5% 被拉到 ~43%），去掉后即为正常的转写行为。校准集保留 system prompt（Omni 的标准对话格式）。

说明：论文代码的 audio 校准默认读 libri_test_other.jsonl，即与评测集同源；这里按说话人切开，
避免校准与评测重叠。官方 omnibench.jsonl 未公开，本脚本按其代码中的读取格式构造一个近似版本。
"""
import argparse
import glob
import json
import os
import random

DATA_ROOT = os.environ.get("MASQUANT_DATA_ROOT", os.path.join(os.path.dirname(__file__), "..", "data"))
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SHAREGPT4V = os.path.join(REPO, "dataset", "sharegpt4v_instruct_gpt4-vision_cap100k_filtered_coco_image.json")

SYSTEM = ("You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, "
          "capable of perceiving auditory and visual inputs, as well as generating text and speech.")
ASR_PROMPT = "Transcribe the English audio into text without any punctuation marks."


def sys_msg():
    return {"role": "system", "content": [{"type": "text", "text": SYSTEM}]}


def load_libri(split="test-other"):
    root = os.path.join(DATA_ROOT, "librispeech", "LibriSpeech", split)
    utts = []
    for trans in sorted(glob.glob(os.path.join(root, "*", "*", "*.trans.txt"))):
        d = os.path.dirname(trans)
        spk = os.path.basename(os.path.dirname(d))
        for line in open(trans):
            uid, text = line.strip().split(" ", 1)
            utts.append({"id": uid, "spk": spk, "audio": os.path.join(d, uid + ".flac"), "text": text})
    return utts


def asr_item(u, with_answer, system=True):
    conv = ([sys_msg()] if system else []) + [{"role": "user", "content": [
        {"type": "audio", "audio": "file://" + os.path.abspath(u["audio"])},
        {"type": "text", "text": ASR_PROMPT}]}]
    if with_answer:
        conv.append({"role": "assistant", "content": [{"type": "text", "text": u["text"].lower()}]})
    return conv


def load_captions(n):
    data = json.load(open(SHAREGPT4V))
    out = []
    for item in data:
        img = os.path.join(DATA_ROOT, item["image"])
        if not os.path.exists(img):
            continue
        q = a = None
        for c in item["conversations"]:
            if c["from"] == "human":
                q = c["value"].replace("<image>", "").replace("\n", "")
            elif c["from"] == "gpt":
                a = c["value"]
        out.append({"image": img, "q": q, "a": a})
        if len(out) >= n:
            break
    return out


def image_item(c):
    return [sys_msg(),
            {"role": "user", "content": [{"type": "image", "image": "file://" + os.path.abspath(c["image"])},
                                         {"type": "text", "text": c["q"]}]},
            {"role": "assistant", "content": [{"type": "text", "text": c["a"]}]}]


def av_item(u, c):
    return [sys_msg(),
            {"role": "user", "content": [
                {"type": "image", "image": "file://" + os.path.abspath(c["image"])},
                {"type": "audio", "audio": "file://" + os.path.abspath(u["audio"])},
                {"type": "text", "text": "First transcribe the audio, then describe the image."}]},
            {"role": "assistant", "content": [{"type": "text", "text": u["text"].lower() + "\n" + c["a"]}]}]


def text_item(t):
    return [sys_msg(),
            {"role": "user", "content": [{"type": "text", "text": "Continue the following passage."}]},
            {"role": "assistant", "content": [{"type": "text", "text": t}]}]


def wikitext_chunks(n, words=256, seed=0):
    import pandas as pd
    f = sorted(glob.glob(os.path.join(DATA_ROOT, "wikitext", "train-*.parquet")))[0]
    lines = [l for l in pd.read_parquet(f)["text"].tolist() if len(l.split()) > 20 and not l.strip().startswith("=")]
    rng = random.Random(seed)
    out = []
    while len(out) < n:
        i = rng.randrange(len(lines))
        buf = []
        while len(buf) < words and i < len(lines):
            buf += lines[i].split()
            i += 1
        out.append(" ".join(buf[:words]))
    return out


def dump(path, convs):
    with open(path, "w") as f:
        for c in convs:
            f.write(json.dumps({"prompt": c}, ensure_ascii=False) + "\n")
    print(f"wrote {len(convs):5d} -> {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_calib", type=int, default=128)
    ap.add_argument("--calib_speakers", type=int, default=3, help="前 K 个说话人留作校准，其余做评测")
    ap.add_argument("--text_ratio", type=float, nargs="*", default=[0.5])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out = os.path.join(DATA_ROOT, "jsonls")
    os.makedirs(out, exist_ok=True)

    utts = load_libri()
    spks = sorted({u["spk"] for u in utts})
    cal_spk = set(spks[:args.calib_speakers])
    cal = [u for u in utts if u["spk"] in cal_spk]
    ev = [u for u in utts if u["spk"] not in cal_spk]
    print(f"librispeech test-other: {len(utts)} utts / {len(spks)} spk -> calib {len(cal)} ({len(cal_spk)} spk), eval {len(ev)}")
    dump(os.path.join(out, "libri_test_other_eval.jsonl"), [asr_item(u, False, system=False) for u in ev])
    dump(os.path.join(out, "libri_test_other_calib.jsonl"), [asr_item(u, True) for u in cal])

    rng = random.Random(args.seed)
    rng.shuffle(cal)
    caps = load_captions(args.n_calib)
    base = []
    for i in range(args.n_calib):
        u, c = cal[i % len(cal)], caps[i % len(caps)]
        base.append([asr_item(u, True), image_item(c), av_item(u, c)][i % 3])
    dump(os.path.join(out, "omnibench.jsonl"), base)

    for r in args.text_ratio:
        k = int(round(r * args.n_calib))
        texts = wikitext_chunks(k, seed=args.seed)
        idx = sorted(random.Random(args.seed + 1).sample(range(args.n_calib), k))
        mix = list(base)
        for j, i in enumerate(idx):
            mix[i] = text_item(texts[j])
        dump(os.path.join(out, f"omnibench_mix{r}.jsonl"), mix)


if __name__ == "__main__":
    main()
