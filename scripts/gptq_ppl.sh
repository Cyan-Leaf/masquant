#!/bin/bash
# 平滑 + GPTQ：在冻结的平滑尺度上用 GPTQ（及改进）代替 RTN，评 PPL（可选再评多模态 benchmark）。
# 用法: bash scripts/gptq_ppl.sh <tag> <gpu> <split_scales|merged_scales> <mas_parameters.pth|none> [额外 main.py 参数]
#   ckpt=none 时用闭式解尺度（split_scales=MAS 仅初始化，merged_scales=SmoothQuant）
#   不加 --gptq 即为 RTN 对照；常用: --gptq [--gptq_clip 1.0,0.95,0.9,0.85] [--gptq_cd 2] [--gptq_hmode balanced]
#   LET=0: 不平滑（纯 GPTQ / 纯 RTN 对照）
#   环境变量: W/A 位宽(4/8)、PPL_SETS(默认 wikitext2,c4)、TASKS/LIMIT(多模态评测)、DT(校准集，默认 text-vision-mix)
set -eu
source "$(dirname "$0")/env.sh"
cd "$REPO_ROOT"
mkdir -p cache act_scales
TAG=$1; GPU=$2; MODE=$3; CKPT=$4; shift 4
W=${W:-4}; A=${A:-8}
export inference_mode=$MODE
export MAS_PPL_DATASETS=${PPL_SETS:-wikitext2,c4}
EXTRA="--epochs 0"; [ "${LET:-1}" = 1 ] && EXTRA="$EXTRA --let"
[ "$CKPT" != none ] && EXTRA="$EXTRA --resume $CKPT"
BENCH=""
[ -n "${TASKS:-}" ] && BENCH="--tasks_multimodal $TASKS --limit_multimodal ${LIMIT:-1.0}"
mkdir -p logs outputs_gptq
CUDA_VISIBLE_DEVICES=$GPU $PY -u main.py --model "$MODEL" --mode train \
  --wbits "$W" --abits "$A" $COMMON_Q --dataset-type "${DT:-text-vision-mix}" --eval_ppl \
  --output_dir "./outputs_gptq/$TAG" $EXTRA $BENCH "$@" > "logs/gptq_$TAG.log" 2>&1
grep -E "wikitext2 :|c4 :" "logs/gptq_$TAG.log" | sed "s/^/[$TAG] /"
