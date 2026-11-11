#!/bin/bash

# 通用路径（按需修改或 export 覆盖）
EXP=${EXP:-$(cd "$(dirname "$0")" && pwd)}
REPO=${REPO:-$(cd "$EXP/.." && pwd)}
PY=${PY:-python3}
MODEL=${MODEL:?set MODEL to Qwen2.5-VL-3B-Instruct}
DATA_ROOT=${DATA_ROOT:-$REPO/data}
HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
PYCACHE=${PYCACHE:-$REPO/pycache}

# 并行跑多组训练（非确定性，用于量方差）。每个 job 一张卡。
# 用法: bash "$EXP"/par_train.sh <tag> "<gpu:cfg:rep> ..."
#   cfg ∈ mas | uniformopt | mas_mix | uniformopt_mix | mas_lv1 | mas_lwc | mas_w4a6 | sq_w4a6 | nvfp4_mas
set -u
TAG=$1; shift
cd "$REPO"
mkdir -p logs
for job in $*; do
  GPU=${job%%:*}; rest=${job#*:}; CFG=${rest%%:*}; REP=${rest##*:}
  DS=text-vision; MODE=split_scales; EXTRA="--epochs 2 --let --loss_multi_modal_mae_alpha"; WA="--wbits 4 --abits 8"; ENVX=""
  case $CFG in
    mas)            ;;
    uniformopt)     MODE=merged_scales ;;
    mas_mix)        DS=text-vision-mix ;;
    uniformopt_mix) DS=text-vision-mix; MODE=merged_scales ;;
    mas_lv1)        ENVX="MAS_VISION_GRAD=1.0" ;;
    mas_mix_lv1)    DS=text-vision-mix; ENVX="MAS_VISION_GRAD=1.0" ;;
    mas_lwc)        EXTRA="$EXTRA --lwc" ;;
    mas_mix_lwc)    DS=text-vision-mix; EXTRA="$EXTRA --lwc" ;;
    mas_w4a6)       WA="--wbits 4 --abits 6" ;;
    sq_w4a6)        WA="--wbits 4 --abits 6"; MODE=merged_scales; EXTRA="--epochs 0 --let" ;;
    mas_mix_w4a6)   DS=text-vision-mix; WA="--wbits 4 --abits 6" ;;
    nvfp4_mas)      WA="--wbits 4 --abits 4 --wfmt nvfp4 --afmt nvfp4" ;;
    nvfp4_mas_mix)  DS=text-vision-mix; WA="--wbits 4 --abits 4 --wfmt nvfp4 --afmt nvfp4" ;;
    *) echo "unknown $CFG"; continue;;
  esac
  OUT="$REPO"/outputs_par/${TAG}_${CFG}_r${REP}
  LG=logs/par_${TAG}_${CFG}_r${REP}.log
  setsid nohup env --chdir="$REPO" \
      MASQUANT_DATA_ROOT=$DATA_ROOT MASQUANT_WIKITEXT_DIR="$DATA_ROOT"/wikitext \
      HF_HOME=$HF_HOME PYTHONPYCACHEPREFIX=$PYCACHE \
      CUDA_VISIBLE_DEVICES=$GPU inference_mode=$MODE $ENVX \
      "$PY" -u main.py --model "$MODEL" \
      --mode train $WA --symmetric --group_size 0 --cache_dir ./cache \
      --dataset-type $DS --nsamples 128 --eval_ppl --output_dir $OUT $EXTRA \
      > $LG 2>&1 < /dev/null &
  echo "launched gpu$GPU $CFG r$REP -> $LG"
  sleep 3
done
