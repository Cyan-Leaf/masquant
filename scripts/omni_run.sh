#!/bin/bash
# Qwen2.5-Omni 的量化 + LibriSpeech 评测一键入口。
# 用法: bash scripts/omni_run.sh <tag> <gpu> <method> [额外 main.py 参数]
#   method: fp16 | rtn | sq | mas_init | mas | uniform（含义同 scripts/train_eval.sh）
#   环境变量:
#     W / A                      位宽（默认 4 / 8）
#     MASQUANT_OMNI_CALIB_JSON   校准 jsonl（默认 data/jsonls/omnibench.jsonl；混合校准用 omnibench_mix0.5.jsonl）
#                                换文件前须先用同一变量跑一次 generate_act_scale_shift.py（见 docs/Omni_音频实验.md）
#     EVAL_JSON                  评测 jsonl（默认 data/jsonls/libri_test_other_eval.jsonl，2676 条）
#     EVAL_N                     只评前 N 条（冒烟测试用；默认全量）
#     BS                         评测 batch（默认 2；量化层对批量 padding 的 mask 已做转换，BS=1 可作对照）
#     PPL_SETS                   PPL 测试集（默认 wikitext2,c4；设为空则不测）
# 产物: outputs_omni/<tag>/（训练出的 mas_parameters.pth）、outputs_omni/<tag>.jsonl（逐条转写）、logs/omni_<tag>.log
# 例:
#   bash scripts/omni_run.sh fp16      0 fp16
#   bash scripts/omni_run.sh sq_orig   1 sq
#   bash scripts/omni_run.sh mas_orig  2 mas
#   MASQUANT_OMNI_CALIB_JSON=$MASQUANT_DATA_ROOT/jsonls/omnibench_mix0.5.jsonl bash scripts/omni_run.sh mas_mix 3 mas
set -eu
source "$(dirname "$0")/env.sh"
cd "$(dirname "$0")/.."
MODEL=${OMNI_MODEL:-$REPO_ROOT/models/Qwen2.5-Omni-3B}
TAG=$1; GPU=$2; METHOD=$3; shift 3
W=${W:-4}; A=${A:-8}
export MAS_PPL_DATASETS=${PPL_SETS-wikitext2,c4}
EVAL_JSON=${EVAL_JSON:-$MASQUANT_DATA_ROOT/jsonls/libri_test_other_eval.jsonl}

case "$METHOD" in
  fp16)     export inference_mode=merged_scales; W=16; A=16; EXTRA="--epochs 0" ;;
  rtn)      export inference_mode=merged_scales; EXTRA="--epochs 0" ;;
  sq)       export inference_mode=merged_scales; EXTRA="--epochs 0 --let" ;;
  mas_init) export inference_mode=split_scales;  EXTRA="--epochs 0 --let" ;;
  mas)      export inference_mode=split_scales;  EXTRA="--epochs 2 --let --loss_multi_modal_mae_alpha" ;;
  uniform)  export inference_mode=merged_scales; EXTRA="--epochs 2 --let --loss_multi_modal_mae_alpha" ;;
  *) echo "unknown method $METHOD"; exit 1 ;;
esac

mkdir -p logs outputs_omni cache act_scales
if [ -n "${EVAL_N:-}" ]; then
  head -n "$EVAL_N" "$EVAL_JSON" > "outputs_omni/.eval_head_${TAG}.jsonl"
  EVAL_JSON="outputs_omni/.eval_head_${TAG}.jsonl"
fi
PPL_FLAG=""; [ -n "$MAS_PPL_DATASETS" ] && PPL_FLAG="--eval_ppl"

CUDA_VISIBLE_DEVICES=$GPU $PY -u main.py --model "$MODEL" --mode train \
  --wbits "$W" --abits "$A" $COMMON_Q --dataset-type text-audio-vision \
  --output_dir "./outputs_omni/$TAG" $EXTRA $PPL_FLAG \
  --eval_omni_task --input_file "$EVAL_JSON" --output_file "outputs_omni/$TAG.jsonl" --batch_size ${BS:-2} \
  "$@" 2>&1 | tee "logs/omni_$TAG.log"
grep -E "wikitext2 :|c4 :" "logs/omni_$TAG.log" | tail -2 || true
$PY tools/omni_wer.py "outputs_omni/$TAG.jsonl"
