"""由 COCO 图像合成短视频，用于验证 MASQuant 的视频模态扩展。

每段视频由 8 张连续 COCO 图像各重复 2 帧构成（16 帧，fps=4，约 4 秒），
这样视频 token 数可控，同时具有真实的视觉统计特性。
"""
import json
import os

import cv2

_REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
DATA = os.environ.get("MASQUANT_DATA_ROOT", os.path.join(_REPO, "data"))
SRC_JSON = os.path.join(_REPO, "dataset", "sharegpt4v_instruct_gpt4-vision_cap100k_filtered_coco_image.json")
OUT_DIR = os.path.join(DATA, "video")
N_VIDEOS = 32
FRAMES_PER_VIDEO = 8
REPEAT = 2
FPS = 4
SIZE = (448, 448)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    meta = json.load(open(SRC_JSON))
    made = []
    for v in range(N_VIDEOS):
        out = os.path.join(OUT_DIR, f"clip_{v:03d}.mp4")
        if os.path.exists(out):
            made.append(out)
            continue
        writer = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), FPS, SIZE)
        for f in range(FRAMES_PER_VIDEO):
            idx = (v * FRAMES_PER_VIDEO + f) % len(meta)
            img_path = os.path.join(DATA, meta[idx]["image"])
            img = cv2.imread(img_path)
            if img is None:
                continue
            img = cv2.resize(img, SIZE)
            for _ in range(REPEAT):
                writer.write(img)
        writer.release()
        made.append(out)
    print(f"created {len(made)} clips in {OUT_DIR}")


if __name__ == "__main__":
    main()
