#!/bin/bash

# 通用路径（按需修改或 export 覆盖）
EXP=${EXP:-$(cd "$(dirname "$0")" && pwd)}
REPO=${REPO:-$(cd "$EXP/.." && pwd)}
PY=${PY:-python3}
MODEL=${MODEL:?set MODEL to Qwen2.5-VL-3B-Instruct}
DATA_ROOT=${DATA_ROOT:-$REPO/data}
HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
PYCACHE=${PYCACHE:-$REPO/pycache}

set -u
cd "$REPO"
export MASQUANT_DATA_ROOT=$DATA_ROOT MASQUANT_WIKITEXT_DIR="$DATA_ROOT"/wikitext
export HF_HOME=$HF_HOME HF_ENDPOINT=https://hf-mirror.com HF_TOKEN=hf_placeholder
export PYTHONPYCACHEPREFIX=$PYCACHE CUDA_VISIBLE_DEVICES=${GPU:-7}
export inference_mode=merged_scales LMMS_EVAL_BATCH_SIZE=1 LMMS_MAX_PIXELS=401408
"$PY" -u main.py --model "$MODEL" \
  --mode train --epochs 0 --symmetric --group_size 0 --cache_dir ./cache \
  --wbits 16 --abits 16 --dataset-type text-vision --nsamples 128 \
  --tasks_multimodal scienceqa_img --limit_multimodal 1.0 \
  --output_dir "$REPO"/outputs_bench/full_fp16 >> logs/benchfull_fp16.log 2>&1
"$PY" -u main.py --model "$MODEL" \
  --mode train --epochs 0 --symmetric --group_size 0 --cache_dir ./cache \
  --wbits 16 --abits 16 --dataset-type text-vision --nsamples 128 \
  --tasks_multimodal textvqa_val --limit_multimodal 0.4 \
  --output_dir "$REPO"/outputs_bench/full_fp16 >> logs/benchfull_fp16.log 2>&1
echo "fp16 done"
