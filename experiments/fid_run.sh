#!/bin/bash

# 通用路径（按需修改或 export 覆盖）
EXP=${EXP:-$(cd "$(dirname "$0")" && pwd)}
REPO=${REPO:-$(cd "$EXP/.." && pwd)}
PY=${PY:-python3}
MODEL=${MODEL:?set MODEL to Qwen2.5-VL-3B-Instruct}
DATA_ROOT=${DATA_ROOT:-$REPO/data}
HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
PYCACHE=${PYCACHE:-$REPO/pycache}

# 用法: bash fid_run.sh <logname> <gpu> KEY=VAL ... -- <fidelity_eval.py args...>
set -u
LOG=$1; GPU=$2; shift 2
ENVS=()
while [ "$1" != "--" ]; do ENVS+=("$1"); shift; done
shift
cd "$REPO"
mkdir -p logs
env --chdir="$REPO" \
    MASQUANT_DATA_ROOT=$DATA_ROOT \
    MASQUANT_WIKITEXT_DIR="$DATA_ROOT"/wikitext \
    HF_HOME=$HF_HOME \
    HF_ENDPOINT=https://hf-mirror.com \
    HF_TOKEN=hf_placeholder \
    PYTHONPYCACHEPREFIX=$PYCACHE \
    TOKENIZERS_PARALLELISM=false \
    CUDA_VISIBLE_DEVICES=$GPU \
    "${ENVS[@]}" \
    "$PY" -u tools/fidelity_eval.py --model "$MODEL" "$@" \
    > logs/$LOG.log 2>&1
echo "exit=$? log=logs/$LOG.log"
