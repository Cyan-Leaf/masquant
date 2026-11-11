#!/bin/bash

# 通用路径（按需修改或 export 覆盖）
EXP=${EXP:-$(cd "$(dirname "$0")" && pwd)}
REPO=${REPO:-$(cd "$EXP/.." && pwd)}
PY=${PY:-python3}
MODEL=${MODEL:?set MODEL to Qwen2.5-VL-3B-Instruct}
DATA_ROOT=${DATA_ROOT:-$REPO/data}
HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
PYCACHE=${PYCACHE:-$REPO/pycache}

# CMC 的多模态评测
set -u
CFG=$1; GPU=$2
cd "$REPO"
mkdir -p logs
CK=$(ls -d outputs_vl/w4a8/*/ | head -1)mas_parameters.pth
case $CFG in
  nocomp)    ENVS="MAS_NO_CMC=1"; EXTRA="--rank 0.05";;
  cmc)       ENVS="MAS_NO_CMC=0"; EXTRA="--rank 0.05";;
  cmc_adapt) ENVS="MAS_NO_CMC=0"; EXTRA="--rank 0.05 --adaptive_rank greedy --rank_min 32 --rank_cap 0.05 --save_low_rank_adapters ./low_rank_adapters/adaptive_greedy_r0.05.pt";;
  *) echo bad; exit 1;;
esac
env --chdir="$REPO" \
   MASQUANT_DATA_ROOT=$DATA_ROOT MASQUANT_WIKITEXT_DIR="$DATA_ROOT"/wikitext \
   HF_HOME=$HF_HOME HF_ENDPOINT=https://hf-mirror.com HF_TOKEN=hf_placeholder \
   PYTHONPYCACHEPREFIX=$PYCACHE CUDA_VISIBLE_DEVICES=$GPU $ENVS \
   LMMS_EVAL_BATCH_SIZE=1 LMMS_MAX_PIXELS=401408 \
   "$PY" -u infer_mas.py --mode infer --model "$MODEL" \
   --net qwen2.5-vl-3b --scales_path $CK --wbits 4 --abits 8 --LR --quantize \
   --n_cali_samples 32 --cali_data_type vision-audio-only --quant_cmc 0 $EXTRA \
   --tasks_multimodal ${TASKS:-scienceqa_img,textvqa_val} --limit_multimodal ${LIMIT:-0.2} \
   --output_dir ./outputs_cmc/bench_$CFG > logs/cmcbench_$CFG.log 2>&1
echo "exit=$? $CFG"
