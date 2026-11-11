#!/bin/bash

# 通用路径（按需修改或 export 覆盖）
EXP=${EXP:-$(cd "$(dirname "$0")" && pwd)}
REPO=${REPO:-$(cd "$EXP/.." && pwd)}
PY=${PY:-python3}
MODEL=${MODEL:?set MODEL to Qwen2.5-VL-3B-Instruct}
DATA_ROOT=${DATA_ROOT:-$REPO/data}
HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
PYCACHE=${PYCACHE:-$REPO/pycache}

# 用法: bash c4_batch.sh <gpu> <tag:mode> ...   tag 对应 outputs_rep/<tag> 或 outputs_par/<tag>
GPU=$1; shift
cd "$REPO"
for job in "$@"; do
  TAG=${job%%:*}; MODE=${job##*:}
  D=$(ls -d outputs_rep/$TAG/*/ outputs_par/$TAG/*/ 2>/dev/null | head -1)
  bash "$EXP"/c4_ppl.sh "$TAG" "$GPU" "$MODE" "$REPO/${D}mas_parameters.pth" --let
done
