#!/bin/bash
# 多模态 benchmark 评测：Qwen2.5-VL-3B
# 用法: bash scripts/run_bench.sh <tag> <gpu> [wbits] [abits] [extra args...]
#   tag=fp16       -> 不量化
#   tag=sq         -> SmoothQuant 统一 scale
#   tag=mas        -> MASQuant（需先训练出 mas_parameters.pth）
set -u
cd "$(dirname "$0")/.."

TAG=$1
GPU=$2
W=${3:-16}
A=${4:-8}
shift 4 2>/dev/null || shift 3

PY=${PY:-python3}
# 模型目录自己指定：export MODEL=/path/to/Qwen2.5-VL-3B-Instruct
MODEL=${MODEL:?set MODEL to a Qwen2.5-VL-3B-Instruct directory}
TASKS=${TASKS:-"scienceqa_img,textvqa_val,ocrbench,mmmu_val"}
LIMIT=${LIMIT:-1.0}

export HF_DATASETS_OFFLINE=0 HF_HUB_OFFLINE=0
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
export HF_TOKEN=${HF_TOKEN:-hf_placeholder}

case "$TAG" in
  fp16)
    export inference_mode=merged_scales
    EXTRA="--epochs 0"
    ;;
  sq)
    export inference_mode=merged_scales
    EXTRA="--epochs 0 --let"
    ;;
  mas)
    export inference_mode=split_scales
    RESUME=$(ls -d outputs_vl/w4a8/*/ | tail -1)mas_parameters.pth
    EXTRA="--epochs 0 --let --resume $RESUME"
    ;;
  uniformopt)
    export inference_mode=merged_scales
    EXTRA="--epochs 2 --let --loss_multi_modal_mae_alpha"
    ;;
  masnoopt)
    export inference_mode=split_scales
    EXTRA="--epochs 0 --let"
    ;;
  *)
    echo "unknown tag $TAG"; exit 1;;
esac

mkdir -p logs
CUDA_VISIBLE_DEVICES=$GPU $PY -u main.py \
  --model $MODEL --mode train --wbits $W --abits $A \
  --symmetric --group_size 0 --cache_dir ./cache \
  --dataset-type text-vision --nsamples 128 \
  --tasks_multimodal "$TASKS" --limit_multimodal "$LIMIT" \
  --output_dir ./outputs_vl/bench_${TAG}_w${W}a${A} \
  --tasks_multimodal "$TASKS" --limit_multimodal "$LIMIT" \
  $EXTRA "$@" > logs/bench_${TAG}_w${W}a${A}.log 2>&1
grep -A3 "tasks_multimodal:" logs/bench_${TAG}_w${W}a${A}.log | tail -5
