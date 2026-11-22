"""下载校准用的 COCO train2017 图像（只取 ShareGPT4V-COCO 标注里前 N 条引用的图片，默认 256 张）。

用法: python tools/fetch_coco.py [N]
输出: $MASQUANT_DATA_ROOT/coco/train2017/*.jpg（默认 <repo>/data/coco/train2017）
"""
import json
import os
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor

_REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
DATA = os.environ.get("MASQUANT_DATA_ROOT", os.path.join(_REPO, "data"))
SRC_JSON = os.path.join(_REPO, "dataset",
                        "sharegpt4v_instruct_gpt4-vision_cap100k_filtered_coco_image.json")
URL = "http://images.cocodataset.org/train2017/{}"


def fetch(rel):
    name = os.path.basename(rel)
    dst = os.path.join(DATA, "coco", "train2017", name)
    if os.path.exists(dst) and os.path.getsize(dst) > 0:
        return
    urllib.request.urlretrieve(URL.format(name), dst + ".tmp")
    os.replace(dst + ".tmp", dst)


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 256
    os.makedirs(os.path.join(DATA, "coco", "train2017"), exist_ok=True)
    items = json.load(open(SRC_JSON))[:n]
    with ThreadPoolExecutor(16) as ex:
        list(ex.map(fetch, [it["image"] for it in items]))
    print(f"done: {n} images -> {os.path.join(DATA, 'coco', 'train2017')}")


if __name__ == "__main__":
    main()
