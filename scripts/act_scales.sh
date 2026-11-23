#!/bin/bash
# 统计分模态激活尺度（MAS / SmoothQuant 闭式解初始化用）
# 用法: bash scripts/act_scales.sh <gpu> <dataset-type>
#   例: bash scripts/act_scales.sh 0 text-vision
#       MAS_CALI_TEXT_RATIO=0.75 bash scripts/act_scales.sh 0 text-vision-mix
set -eu
source "$(dirname "$0")/env.sh"
cd "$REPO_ROOT"
mkdir -p cache act_scales
GPU=$1
DT=$2
CUDA_VISIBLE_DEVICES=$GPU $PY generate_act_scale_shift.py \
  --model "$MODEL" --dataset-type "$DT" --nsamples 128 --cache_dir ./cache
