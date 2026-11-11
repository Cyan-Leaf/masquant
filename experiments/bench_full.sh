#!/bin/bash

# 通用路径（按需修改或 export 覆盖）
EXP=${EXP:-$(cd "$(dirname "$0")" && pwd)}
REPO=${REPO:-$(cd "$EXP/.." && pwd)}
PY=${PY:-python3}
MODEL=${MODEL:?set MODEL to Qwen2.5-VL-3B-Instruct}
DATA_ROOT=${DATA_ROOT:-$REPO/data}
HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
PYCACHE=${PYCACHE:-$REPO/pycache}

# 主表全量评测：SQA 全量(2017) + TextVQA 前 2000 题。每组一张卡。
# 用法: LIMIT_TVQA=0.4 bash bench_full.sh
set -u
cd "$REPO"
LIMIT_TVQA=${LIMIT_TVQA:-0.4}      # textvqa: 0.4*5000 = 2000 题
LIMIT_SQA=${LIMIT_SQA:-1.0}        # scienceqa: 1.0 = 2017 题全量
CKMIX=$(ls -d outputs_par/C_mas_mix_r1/*/ | head -1)mas_parameters.pth
CK075=$(ls -d outputs_par/M_ratio075_r1/*/ | head -1)mas_parameters.pth
CKUNI=$(ls -d outputs_par/C_uniformopt_mix_r1/*/ | head -1)mas_parameters.pth
CKU075=$(ls -d outputs_par/MU_ratio075_r1/*/ | head -1)mas_parameters.pth
CKLWC=$(ls -d outputs_par/C_mas_lwc_r1/*/ | head -1)mas_parameters.pth
export LMMS_EVAL_BATCH_SIZE=1 LMMS_MAX_PIXELS=${LMMS_MAX_PIXELS:-401408}

# 不同任务用不同 limit，所以拆成两次调用
run() { # name gpu mode ckpt extra...
  local NAME=$1 GPU=$2 MODE=$3 CK=$4; shift 4
  local RES=""; [ "$CK" != "none" ] && RES="--resume $CK"
  env --chdir="$REPO" \
    MASQUANT_DATA_ROOT=$DATA_ROOT MASQUANT_WIKITEXT_DIR="$DATA_ROOT"/wikitext \
    HF_HOME=$HF_HOME HF_ENDPOINT=https://hf-mirror.com HF_TOKEN=hf_placeholder \
    PYTHONPYCACHEPREFIX=$PYCACHE CUDA_VISIBLE_DEVICES=$GPU inference_mode=$MODE \
    LMMS_EVAL_BATCH_SIZE=1 LMMS_MAX_PIXELS=$LMMS_MAX_PIXELS \
    "$PY" -u main.py --model "$MODEL" \
    --mode train --epochs 0 --symmetric --group_size 0 --cache_dir ./cache \
    --wbits $WB --abits $AB --dataset-type text-vision --nsamples 128 \
    --tasks_multimodal $TASKS --limit_multimodal $LIM \
    --output_dir "$REPO"/outputs_bench/full_$NAME $RES "$@" \
    >> logs/benchfull_$NAME.log 2>&1
  echo "  done $NAME"
}

launch() { # name gpu mode ckpt
  local NAME=$1 GPU=$2 MODE=$3 CK=$4
  (
    WB=4; AB=8; TASKS=scienceqa_img;    LIM=$LIMIT_SQA   run "$NAME" "$GPU" "$MODE" "$CK" --let
    TASKS=textvqa_val;                  LIM=$LIMIT_TVQA  run "$NAME" "$GPU" "$MODE" "$CK" --let
  ) &
  echo "launched gpu$GPU $NAME"
  sleep 4
}

launch fp16        0 merged_scales none
launch sq          1 merged_scales none
launch mas05       2 split_scales   "$CKMIX"
launch mas075      3 split_scales   "$CK075"
launch uniform05   4 merged_scales  "$CKUNI"
launch uniform075  5 merged_scales  "$CKU075"
launch mas_lwc     6 split_scales   "$CKLWC"
echo "7 jobs launched; gpu7 free"
