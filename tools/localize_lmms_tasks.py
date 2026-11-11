"""
把 lmms-eval 的内置任务改成从本地 parquet 加载（离线评测）。

用法：
    python tools/localize_lmms_tasks.py [--revert]

原理：把任务 yaml 里的 dataset_path/dataset_name/dataset_kwargs 替换为
parquet 加载方式，并记录原始 yaml 到 .orig 备份，方便回滚。
"""

import argparse
import os
import shutil

import lmms_eval as _lmms

# 在哪装的就改哪里的任务 yaml；也可用环境变量覆盖
TASKS_DIR = os.environ.get("LMMS_TASKS_DIR", os.path.join(os.path.dirname(_lmms.__file__), "tasks"))
DATA = os.environ.get("MASQUANT_HF_DATASETS",
                      os.path.join(os.environ.get("MASQUANT_DATA_ROOT", "data"), "hf_datasets"))

PATCHES = {
    "scienceqa/scienceqa_img.yaml": [
        ("dataset_path: lmms-lab/ScienceQA\ndataset_name: ScienceQA-IMG\n",
         f"dataset_path: parquet\ndataset_kwargs:\n  data_files:\n    test: {DATA}/lmms-lab/ScienceQA/ScienceQA-IMG/test-00000-of-00001.parquet\n"),
    ],
    "ocrbench/ocrbench.yaml": [
        ("dataset_path: echo840/OCRBench\n",
         f"dataset_path: parquet\ndataset_kwargs:\n  data_files:\n    test: {DATA}/echo840/OCRBench/data/test-00000-of-00001.parquet\n"),
    ],
    "mmmu/mmmu_val.yaml": [
        ("dataset_path: lmms-lab/MMMU\n",
         f"dataset_path: parquet\ndataset_kwargs:\n  data_files:\n    validation: {DATA}/lmms-lab/MMMU/data/validation-00000-of-00001.parquet\n"),
    ],
    "textvqa/_textvqa.yaml": [
        ("dataset_path: lmms-lab/textvqa\n",
         f"dataset_path: parquet\ndataset_kwargs:\n  data_files:\n    validation:\n"
         f"      - {DATA}/lmms-lab/textvqa/data/validation-00000-of-00003.parquet\n"
         f"      - {DATA}/lmms-lab/textvqa/data/validation-00001-of-00003.parquet\n"
         f"      - {DATA}/lmms-lab/textvqa/data/validation-00002-of-00003.parquet\n"),
    ],
}


def patch(revert=False):
    for rel, subs in PATCHES.items():
        path = os.path.join(TASKS_DIR, rel)
        bak = path + ".orig"
        if not os.path.exists(path):
            print(f"skip missing {rel}")
            continue
        if revert:
            if os.path.exists(bak):
                shutil.copy2(bak, path)
                print(f"reverted {rel}")
            continue
        if not os.path.exists(bak):
            shutil.copy2(path, bak)
        text = open(bak if os.path.exists(bak) else path).read()
        orig = text
        for old, new in subs:
            text = text.replace(old, new)
        # 去掉残留的 token: True（本地加载不需要鉴权）
        text = text.replace("dataset_kwargs:\n  token: True\n", "")
        if text == orig:
            print(f"WARN no change for {rel}")
        open(path, "w").write(text)
        print(f"patched {rel}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--revert", action="store_true")
    patch(ap.parse_args().revert)
