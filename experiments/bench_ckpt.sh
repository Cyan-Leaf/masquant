#!/bin/bash

# 通用路径（按需修改或 export 覆盖）
EXP=${EXP:-$(cd "$(dirname "$0")" && pwd)}
REPO=${REPO:-$(cd "$EXP/.." && pwd)}
PY=${PY:-python3}
MODEL=${MODEL:?set MODEL to Qwen2.5-VL-3B-Instruct}
DATA_ROOT=${DATA_ROOT:-$REPO/data}
HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
PYCACHE=${PYCACHE:-$REPO/pycache}

# 对指定 checkpoint 做多模态 benchmark 评测（逐样本落盘，供配对 bootstrap）
# 用法: bash "$EXP"/bench_ckpt.sh <name> <gpu> <inference_mode> <ckpt|none> [extra main.py args]
#   ckpt=none 表示不加载（fp16 或闭式解初始化），用 extra 指定 --let 等
#   WBITS/ABITS 环境变量可覆盖，默认 4/8（== 主表口径）。
#   注意：main.py 的 --abits 默认是 16，曾经漏传跑成 w4a16 白跑一轮，所以这里写死默认值。
#   评 MAS 训练出的 checkpoint 记得传 --let，否则尺度不参与计算。
set -u
NAME=$1; GPU=$2; MODE=$3; CK=$4; shift 4
cd "$REPO"
mkdir -p logs
RES=""
if [ "$CK" != "none" ]; then RES="--resume $CK"; fi
WBITS=${WBITS:-4}
ABITS=${ABITS:-8}
env --chdir="$REPO" \
    MASQUANT_DATA_ROOT=$DATA_ROOT MASQUANT_WIKITEXT_DIR="$DATA_ROOT"/wikitext \
    HF_HOME=$HF_HOME HF_ENDPOINT=https://hf-mirror.com HF_TOKEN=hf_placeholder \
    PYTHONPYCACHEPREFIX=$PYCACHE CUDA_VISIBLE_DEVICES=$GPU inference_mode=$MODE \
    LMMS_EVAL_BATCH_SIZE=1 LMMS_MAX_PIXELS=${LMMS_MAX_PIXELS:-401408} \
    "$PY" -u main.py --model "$MODEL" \
    --mode train --epochs 0 --symmetric --group_size 0 --cache_dir ./cache \
    --wbits $WBITS --abits $ABITS \
    --dataset-type text-vision --nsamples 128 \
    --tasks_multimodal ${TASKS:-scienceqa_img,textvqa_val} --limit_multimodal ${LIMIT:-0.2} \
    --output_dir "$REPO"/outputs_bench/$NAME $RES "$@" \
    > logs/bench2_$NAME.log 2>&1
echo "exit=$? $NAME"
