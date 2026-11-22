import os
import sys
import time
import shutil
from concurrent.futures import ThreadPoolExecutor

import requests

PROXY = os.environ.get("https_proxy") or os.environ.get("http_proxy") or None
MS = "https://www.modelscope.cn/models/Qwen/Qwen2.5-VL-3B-Instruct/resolve/master"
_REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
OUT = os.environ.get("MASQUANT_MODEL", os.path.join(_REPO, "models", "Qwen2.5-VL-3B-Instruct"))
SMALL = [
    ".gitattributes", "LICENSE", "README.md", "chat_template.json", "config.json",
    "generation_config.json", "merges.txt", "model.safetensors.index.json",
    "preprocessor_config.json", "tokenizer.json", "tokenizer_config.json", "vocab.json",
]
BIG = ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"]
NTHREAD = 24
CHUNK = 24 * 1024 * 1024


def sess():
    s = requests.Session()
    if PROXY:
        s.proxies = {"http": PROXY, "https": PROXY}
    return s


def fetch_small(fname):
    path = os.path.join(OUT, fname)
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return
    r = sess().get(f"{MS}/{fname}", timeout=120)
    r.raise_for_status()
    with open(path, "wb") as f:
        f.write(r.content)
    print(f"ok {fname} {len(r.content)}", flush=True)


def cdn_url(fname):
    r = sess().get(f"{MS}/{fname}", allow_redirects=False, timeout=60)
    loc = r.headers.get("location")
    if loc:
        return loc
    r = sess().get(f"{MS}/{fname}", stream=True, timeout=60)
    r.raise_for_status()
    return r.url


def part(url, start, end, path):
    need = end - start + 1
    for attempt in range(8):
        done = os.path.getsize(path) if os.path.exists(path) else 0
        if done >= need:
            return True
        try:
            h = {"Range": f"bytes={start + done}-{end}"}
            with sess().get(url, headers=h, stream=True, timeout=180) as r:
                r.raise_for_status()
                with open(path, "ab") as f:
                    for c in r.iter_content(8 * 1024 * 1024):
                        f.write(c)
        except Exception as e:
            print(f"  retry{attempt} {os.path.basename(path)}: {e}", flush=True)
            time.sleep(1 + attempt)
    return os.path.getsize(path) >= need


def fetch_big(fname):
    target = os.path.join(OUT, fname)
    url = cdn_url(fname)
    r = sess().get(url, headers={"Range": "bytes=0-0"}, stream=True, timeout=60)
    cr = r.headers.get("content-range", "")
    size = int(cr.split("/")[-1])
    r.close()
    print(f"== {fname} {size/1e9:.2f}GB", flush=True)
    if os.path.exists(target) and os.path.getsize(target) == size:
        print("   already complete", flush=True)
        return

    tmp = os.path.join(OUT, ".parts", fname)
    os.makedirs(tmp, exist_ok=True)
    ranges = [(p, min(p + CHUNK - 1, size - 1)) for p in range(0, size, CHUNK)]
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=NTHREAD) as ex:
        futs = [
            ex.submit(part, url, s, e, os.path.join(tmp, f"p{i:04d}"))
            for i, (s, e) in enumerate(ranges)
        ]
        for i, fu in enumerate(futs):
            fu.result()
            if (i + 1) % 16 == 0:
                got = sum(os.path.getsize(os.path.join(tmp, p)) for p in os.listdir(tmp))
                print(f"   {got/1e9:.2f}/{size/1e9:.2f}GB {(got/(time.time()-t0))/1e6:.1f}MB/s", flush=True)

    with open(target, "wb") as out:
        for i in range(len(ranges)):
            with open(os.path.join(tmp, f"p{i:04d}"), "rb") as f:
                shutil.copyfileobj(f, out, 16 * 1024 * 1024)
    print(f"== {fname} done {os.path.getsize(target)} / {size}", flush=True)
    if os.path.getsize(target) == size:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    with ThreadPoolExecutor(max_workers=8) as ex:
        list(ex.map(fetch_small, SMALL))
    for f in BIG:
        fetch_big(f)
