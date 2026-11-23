#!/bin/bash
# 训练（或仅初始化）并评测 PPL，可选再评多模态 benchmark。
# 用法: bash scripts/train_eval.sh <tag> <gpu> <method> <dataset-type> [额外 main.py 参数]
#   method:
#     rtn       不平滑，直接量化
#     sq        SmoothQuant：统一 scale，闭式解，不训练
#     mas_init  MAS：分模态闭式解，不训练
#     mas       MAS：分模态，训练 2 epoch（论文方法）
#     uniform   统一 scale，训练 2 epoch（"分模态"的对照组）
#   环境变量:
#     W / A                位宽（默认 4 / 8）
#     PPL_SETS             PPL 测试集，逗号分隔（默认 wikitext2；可设 wikitext2,c4）
#     TASKS                多模态任务（默认空 = 不评；如 scienceqa_img,textvqa_val）
#     LIMIT                多模态评测比例（默认 1.0 = 全量）
#     MAS_CALI_TEXT_RATIO  text-vision-mix 中纯文本样本占比（默认 0.5）
# 例:
#   bash scripts/train_eval.sh mas_orig 0 mas text-vision
#   bash scripts/train_eval.sh mas_mix  1 mas text-vision-mix
#   W=4 A=4 bash scripts/train_eval.sh nvfp4_mas 2 mas text-vision --wfmt nvfp4 --afmt nvfp4
#   TASKS=scienceqa_img,textvqa_val bash scripts/train_eval.sh mas_mix_bench 3 mas text-vision-mix
set -eu
source "$(dirname "$0")/env.sh"
cd "$REPO_ROOT"
mkdir -p cache act_scales
TAG=$1; GPU=$2; METHOD=$3; DT=$4; shift 4
W=${W:-4}; A=${A:-8}
export MAS_PPL_DATASETS=${PPL_SETS:-wikitext2}

case "$METHOD" in
  rtn)      export inference_mode=merged_scales; EXTRA="--epochs 0" ;;
  sq)       export inference_mode=merged_scales; EXTRA="--epochs 0 --let" ;;
  mas_init) export inference_mode=split_scales;  EXTRA="--epochs 0 --let" ;;
  mas)      export inference_mode=split_scales;  EXTRA="--epochs 2 --let --loss_multi_modal_mae_alpha" ;;
  uniform)  export inference_mode=merged_scales; EXTRA="--epochs 2 --let --loss_multi_modal_mae_alpha" ;;
  *) echo "unknown method $METHOD"; exit 1 ;;
esac

BENCH=""
if [ -n "${TASKS:-}" ]; then
  BENCH="--tasks_multimodal $TASKS --limit_multimodal ${LIMIT:-1.0}"
fi

mkdir -p logs outputs
CUDA_VISIBLE_DEVICES=$GPU $PY -u main.py --model "$MODEL" --mode train \
  --wbits "$W" --abits "$A" $COMMON_Q --dataset-type "$DT" --eval_ppl \
  --output_dir "./outputs/$TAG" $EXTRA $BENCH "$@" 2>&1 | tee "logs/$TAG.log"
grep -E "wikitext2 :|c4 :" "logs/$TAG.log" | tail -2 || true
