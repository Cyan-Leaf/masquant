#!/bin/bash

# 通用路径（按需修改或 export 覆盖）
EXP=${EXP:-$(cd "$(dirname "$0")" && pwd)}
REPO=${REPO:-$(cd "$EXP/.." && pwd)}
PY=${PY:-python3}
MODEL=${MODEL:?set MODEL to Qwen2.5-VL-3B-Instruct}
DATA_ROOT=${DATA_ROOT:-$REPO/data}
HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
PYCACHE=${PYCACHE:-$REPO/pycache}

# 扫混合校准的纯文本占比（统一 scale 版）。用法: bash sweep_ratio.sh <tag> "<gpu:ratio:rep> ..."
set -u
TAG=$1; shift
cd "$REPO"
mkdir -p logs
for job in $*; do
  GPU=${job%%:*}; rest=${job#*:}; RATIO=${rest%%:*}; REP=${rest##*:}
  RTAG=$(echo "$RATIO" | tr -d '.')
  OUT="$REPO"/outputs_par/${TAG}_uniratio${RTAG}_r${REP}
  LG="$REPO"/logs/par_${TAG}_uniratio${RTAG}_r${REP}.log
  setsid nohup env --chdir="$REPO" \
      MASQUANT_DATA_ROOT=$DATA_ROOT MASQUANT_WIKITEXT_DIR="$DATA_ROOT"/wikitext \
      HF_HOME=$HF_HOME PYTHONPYCACHEPREFIX=$PYCACHE \
      CUDA_VISIBLE_DEVICES=$GPU inference_mode=merged_scales \
      MAS_CALI_TEXT_RATIO=$RATIO \
      "$PY" -u main.py --model "$MODEL" \
      --mode train --wbits 4 --abits 8 --symmetric --group_size 0 --cache_dir ./cache \
      --dataset-type text-vision-mix --nsamples 128 --eval_ppl --output_dir "$OUT" \
      --epochs 2 --let --loss_multi_modal_mae_alpha \
      > "$LG" 2>&1 < /dev/null &
  echo "launched gpu$GPU ratio=$RATIO r$REP -> $LG"
  sleep 3
done
