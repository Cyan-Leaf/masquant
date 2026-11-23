#!/bin/bash
# 用已训练好的 checkpoint 评测多模态 benchmark（逐样本结果落盘为 samples_<task>.jsonl）
# 用法: bash scripts/bench_ckpt.sh <tag> <gpu> <split_scales|merged_scales|fp16> <mas_parameters.pth|none> [额外参数]
#   例: bash scripts/bench_ckpt.sh fp16 0 fp16 none
#       bash scripts/bench_ckpt.sh mas_mix 1 split_scales outputs/mas_mix/*/mas_parameters.pth
set -eu
source "$(dirname "$0")/env.sh"
cd "$REPO_ROOT"
mkdir -p cache act_scales
TAG=$1; GPU=$2; MODE=$3; CKPT=$4; shift 4
TASKS=${TASKS-scienceqa_img,textvqa_val}
LIMIT=${LIMIT:-1.0}
W=${W:-4}; A=${A:-8}
if [ "$MODE" = fp16 ]; then
  export inference_mode=merged_scales; W=16; A=16; EXTRA="--epochs 0"
else
  export inference_mode=$MODE; EXTRA="--epochs 0 --let"
  [ "$CKPT" != none ] && EXTRA="$EXTRA --resume $CKPT"
fi
mkdir -p logs outputs_bench
CUDA_VISIBLE_DEVICES=$GPU $PY -u main.py --model "$MODEL" --mode train \
  --wbits "$W" --abits "$A" $COMMON_Q --dataset-type "${DT:-text-vision}" \
  ${TASKS:+--tasks_multimodal "$TASKS" --limit_multimodal "$LIMIT"} \
  --output_dir "./outputs_bench/$TAG" $EXTRA "$@" 2>&1 | tee "logs/bench_$TAG.log"
