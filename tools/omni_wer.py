"""对 main.py --eval_omni_task 的输出计算 LibriSpeech WER。

用法：python tools/omni_wer.py <output.jsonl> [<output2.jsonl> ...]
仓库 results/omni/ 里的文件已把数据路径脱敏为 <DATA_ROOT>，复算时设置 MASQUANT_DATA_ROOT 即可。
参考转写从输入 prompt 的音频路径回查 *.trans.txt（评测 jsonl 里不含答案，避免泄漏到生成）。
归一化：小写、去标点、合并空白（与 Whisper/Qwen 报 LibriSpeech WER 的常用口径一致）。
"""
import json
import os
import re
import sys

_cache = {}


def ref_of(audio_path):
    p = audio_path.replace("file://", "")
    p = p.replace("<DATA_ROOT>", os.environ.get("MASQUANT_DATA_ROOT", "data"))
    d, uid = os.path.dirname(p), os.path.basename(p).rsplit(".", 1)[0]
    if d not in _cache:
        tf = [f for f in os.listdir(d) if f.endswith(".trans.txt")][0]
        _cache[d] = dict(l.strip().split(" ", 1) for l in open(os.path.join(d, tf)))
    return _cache[d][uid]


def norm(s):
    s = s.lower().replace("’", "'")
    s = re.sub(r"[^a-z0-9' ]+", " ", s)
    return s.split()


def edit(a, b):
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, y in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y))
        prev = cur
    return prev[-1]


def wer_file(path):
    err = tot = n = 0
    for line in open(path):
        d = json.loads(line)
        audio = next(c["audio"] for m in d["prompt"] if m["role"] == "user"
                     for c in m["content"] if c["type"] == "audio")
        r, h = norm(ref_of(audio)), norm(d["response"])
        err += edit(r, h)
        tot += len(r)
        n += 1
    return 100.0 * err / max(tot, 1), n


if __name__ == "__main__":
    for p in sys.argv[1:]:
        w, n = wer_file(p)
        print(f"{p}\tWER={w:.2f}\tn={n}")
