#!/bin/bash

# 通用路径（按需修改或 export 覆盖）
EXP=${EXP:-$(cd "$(dirname "$0")" && pwd)}
REPO=${REPO:-$(cd "$EXP/.." && pwd)}
PY=${PY:-python3}
MODEL=${MODEL:?set MODEL to Qwen2.5-VL-3B-Instruct}
DATA_ROOT=${DATA_ROOT:-$REPO/data}
HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
PYCACHE=${PYCACHE:-$REPO/pycache}

# 扫混合校准的纯文本占比。用法: bash sweep_ratio.sh <tag> "<gpu:ratio:rep> ..."
# 可选环境变量：
#   MODE=split_scales|merged_scales   (默认 split_scales)
#   ACT=<act_scales 路径>              (默认 0.5 混合的统计，与上一轮扫描保持一致)
#   POS=1                              (开启平滑因子正值约束 MAS_SCALE_POS)
set -u
TAG=$1; shift
MODE=${MODE:-split_scales}
ACT=${ACT:-./act_scales/Qwen2.5-VL-3B-Instruct-text-vision-mix-128.pt}
POS=${POS:-0}
cd "$REPO"
mkdir -p logs
for job in $*; do
  GPU=${job%%:*}; rest=${job#*:}; RATIO=${rest%%:*}; REP=${rest##*:}
  RTAG=$(echo "$RATIO" | tr -d '.')
  OUT="$REPO"/outputs_par/${TAG}_ratio${RTAG}_r${REP}
  LG="$REPO"/logs/par_${TAG}_ratio${RTAG}_r${REP}.log
  setsid nohup env --chdir="$REPO" \
      MASQUANT_DATA_ROOT=$DATA_ROOT MASQUANT_WIKITEXT_DIR="$DATA_ROOT"/wikitext \
      HF_HOME=$HF_HOME PYTHONPYCACHEPREFIX=$PYCACHE \
      CUDA_VISIBLE_DEVICES=$GPU inference_mode=$MODE \
      MAS_CALI_TEXT_RATIO=$RATIO MAS_SCALE_POS=$POS \
      "$PY" -u main.py --model "$MODEL" \
      --mode train --wbits 4 --abits 8 --symmetric --group_size 0 --cache_dir ./cache \
      --dataset-type text-vision-mix --nsamples 128 --eval_ppl --output_dir "$OUT" \
      --act-scales "$ACT" \
      --epochs 2 --let --loss_multi_modal_mae_alpha \
      > "$LG" 2>&1 < /dev/null &
  echo "launched gpu$GPU ratio=$RATIO r$REP mode=$MODE pos=$POS act=$ACT -> $LG"
  sleep 3
done
