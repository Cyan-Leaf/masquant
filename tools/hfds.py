import os
import sys
import time
import subprocess

from huggingface_hub import HfApi

PROXY = os.environ.get("https_proxy") or os.environ.get("http_proxy") or None
MIRROR = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com")
_REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
ROOT = os.path.join(os.environ.get("MASQUANT_DATA_ROOT", os.path.join(_REPO, "data")), "hf_datasets")


def list_files(repo):
    api = HfApi(endpoint=MIRROR)
    for attempt in range(6):
        try:
            return [s.rfilename for s in api.dataset_info(repo).siblings]
        except Exception as e:
            print(f"list retry {attempt}: {e}", flush=True)
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"cannot list {repo}")


def fetch(repo, fname, dst):
    if os.path.exists(dst) and os.path.getsize(dst) > 0:
        return True
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    url = f"{MIRROR}/datasets/{repo}/resolve/main/{fname}"
    env = dict(os.environ)
    for attempt in range(8):
        r = subprocess.run(
            ["curl", "-sSL", "--retry", "3", "-o", dst + ".tmp", "-w", "%{http_code}", url],
            capture_output=True, text=True, env=env,
        )
        if r.stdout.strip() == "200":
            os.replace(dst + ".tmp", dst)
            print(f"  ok {fname} {os.path.getsize(dst)/1e6:.1f}MB", flush=True)
            return True
        print(f"  retry{attempt} {fname} code={r.stdout.strip()}", flush=True)
        time.sleep(5 * (attempt + 1))
    return False


def main():
    repo = sys.argv[1]
    patterns = sys.argv[2:]
    files = list_files(repo)
    sel = [f for f in files if (not patterns or any(p in f for p in patterns))]
    print(f"{repo}: {len(sel)}/{len(files)} files", flush=True)
    out = os.path.join(ROOT, repo)
    for f in sel:
        fetch(repo, f, os.path.join(out, f))
    print("done", out)


if __name__ == "__main__":
    main()
