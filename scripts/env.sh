#!/bin/bash
# 公共环境变量。其他脚本会 source 本文件；可在外部预先 export 覆盖任意一项。
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export REPO_ROOT
export PY=${PY:-python}
export MODEL=${MODEL:-$REPO_ROOT/models/Qwen2.5-VL-3B-Instruct}
export MASQUANT_DATA_ROOT=${MASQUANT_DATA_ROOT:-$REPO_ROOT/data}
export MASQUANT_WIKITEXT_DIR=${MASQUANT_WIKITEXT_DIR:-$MASQUANT_DATA_ROOT/wikitext}
export HF_HOME=${HF_HOME:-$REPO_ROOT/hf_cache}
export LMMS_EVAL_BATCH_SIZE=${LMMS_EVAL_BATCH_SIZE:-1}
export LMMS_MAX_PIXELS=${LMMS_MAX_PIXELS:-401408}
export LMMS_PADDING_SIDE=${LMMS_PADDING_SIDE:-left}
COMMON_Q="--symmetric --group_size 0 --cache_dir ./cache --nsamples 128"
