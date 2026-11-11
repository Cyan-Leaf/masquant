#!/bin/bash

# 通用路径（按需修改或 export 覆盖）
EXP=${EXP:-$(cd "$(dirname "$0")" && pwd)}
REPO=${REPO:-$(cd "$EXP/.." && pwd)}
PY=${PY:-python3}
MODEL=${MODEL:?set MODEL to Qwen2.5-VL-3B-Instruct}
DATA_ROOT=${DATA_ROOT:-$REPO/data}
HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
PYCACHE=${PYCACHE:-$REPO/pycache}

# 串行重复实验：每种配置在同一张卡上顺序跑 N 次，量出 PPL 的噪声底。
# 用法: bash "$EXP"/repeat_ppl.sh <repeats> <out_tag> [configs...]
set -u
REPS=${1:-3}
TAG=${2:-rep}
shift 2
CONFIGS=${*:-"rtn sq mas_noopt mas uniformopt"}

cd "$REPO"
mkdir -p logs
COMMON="--mode train --wbits 4 --abits 8 --symmetric --group_size 0 --cache_dir ./cache \
        --dataset-type text-vision --nsamples 128 --eval_ppl"

for cfg in $CONFIGS; do
  case $cfg in
    rtn)        ENVS="inference_mode=merged_scales"; EXTRA="--epochs 0";;
    sq)         ENVS="inference_mode=merged_scales"; EXTRA="--epochs 0 --let";;
    mas_noopt)  ENVS="inference_mode=split_scales";  EXTRA="--epochs 0 --let";;
    mas)        ENVS="inference_mode=split_scales";  EXTRA="--epochs 2 --let --loss_multi_modal_mae_alpha";;
    uniformopt) ENVS="inference_mode=merged_scales"; EXTRA="--epochs 2 --let --loss_multi_modal_mae_alpha";;
    nvfp4_mas)  ENVS="inference_mode=split_scales";  EXTRA="--epochs 2 --let --loss_multi_modal_mae_alpha --wbits 4 --abits 4 --wfmt nvfp4 --afmt nvfp4";;
    nvfp4_rtn)  ENVS="inference_mode=merged_scales"; EXTRA="--epochs 0 --wbits 4 --abits 4 --wfmt nvfp4 --afmt nvfp4";;
    *) echo "unknown cfg $cfg"; continue;;
  esac
  for r in $(seq 1 $REPS); do
    LG="rep_${TAG}_${cfg}_r${r}"
    echo "[$(date +%H:%M:%S)] $cfg rep $r"
    env --chdir="$REPO" \
        MASQUANT_DATA_ROOT=$DATA_ROOT MASQUANT_WIKITEXT_DIR="$DATA_ROOT"/wikitext \
        HF_HOME=$HF_HOME PYTHONPYCACHEPREFIX=$PYCACHE \
        CUDA_VISIBLE_DEVICES=${GPU:-0} MAS_DETERMINISTIC=${MAS_DETERMINISTIC:-0} $ENVS \
        "$PY" -u main.py --model "$MODEL" \
        $COMMON $EXTRA --output_dir "$REPO"/outputs_rep/${TAG}_${cfg}_r${r} \
        > logs/$LG.log 2>&1
    PPL=$(grep -a "wikitext2 :" logs/$LG.log | tail -1 | sed 's/.*wikitext2 : //')
    echo "  -> PPL $PPL"
  done
done
echo "=== summary ==="
grep -a -H "wikitext2 :" logs/rep_${TAG}_*.log | sed 's#logs/rep_'"$TAG"'_##; s#\.log:.*wikitext2 : # #'
