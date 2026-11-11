#!/bin/bash

# 通用路径（按需修改或 export 覆盖）
EXP=${EXP:-$(cd "$(dirname "$0")" && pwd)}
REPO=${REPO:-$(cd "$EXP/.." && pwd)}
PY=${PY:-python3}
MODEL=${MODEL:?set MODEL to Qwen2.5-VL-3B-Instruct}
DATA_ROOT=${DATA_ROOT:-$REPO/data}
HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
PYCACHE=${PYCACHE:-$REPO/pycache}

# CMC 多模态评测：同一 checkpoint 下比较
#   nocomp     : 只存 Q(S_t W)，非文本模态不做任何补偿
#   cmc        : CMC（论文方法），固定 rank 比例
#   cmc_adapt  : CMC + 自适应 rank（同预算）
#   fullsplit  : 上界参考——每个模态用自己的量化权重（不共享，靠 env 开关做不到，
#                因此这里用 MAS 的 split_scales 训练/评测路径代替，见 run_bench.sh mas）
#
# 用法: bash "$EXP"/cmc_eval.sh <cfg> <gpu> [extra infer_mas args...]
set -u
CFG=$1; GPU=$2; shift 2
cd "$REPO"
mkdir -p logs
CK=$(ls -d outputs_vl/w4a8/*/ | head -1)mas_parameters.pth
BASE="--mode infer --net qwen2.5-vl-3b --scales_path $CK --wbits 4 --abits 8 \
      --LR --quantize --n_cali_samples 32 --cali_data_type vision-audio-only --quant_cmc 0"

case $CFG in
  nocomp)
    ENVS="MAS_NO_CMC=1"
    EXTRA="--rank ${RANK:-0.05}"
    ;;
  cmc)
    ENVS="MAS_NO_CMC=0"
    EXTRA="--rank ${RANK:-0.05}"
    ;;
  cmc_adapt)
    ENVS="MAS_NO_CMC=0"
    EXTRA="--rank ${RANK:-0.05} --adaptive_rank greedy --rank_min 32 --rank_cap ${RANK:-0.05}"
    ;;
  *) echo "unknown cfg"; exit 1;;
esac

env --chdir="$REPO" \
    MASQUANT_DATA_ROOT=$DATA_ROOT MASQUANT_WIKITEXT_DIR="$DATA_ROOT"/wikitext \
    HF_HOME=$HF_HOME HF_ENDPOINT=https://hf-mirror.com HF_TOKEN=hf_placeholder \
    PYTHONPYCACHEPREFIX=$PYCACHE CUDA_VISIBLE_DEVICES=$GPU $ENVS \
    "$PY" -u infer_mas.py --model "$MODEL" \
    $BASE $EXTRA --eval_ppl --output_dir ./outputs_cmc/$CFG "$@" \
    > logs/cmc_$CFG.log 2>&1
echo "exit=$? log=logs/cmc_$CFG.log"
grep -a "wikitext2 :" logs/cmc_$CFG.log | tail -1
