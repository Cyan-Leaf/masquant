#!/bin/bash

# 通用路径（按需修改或 export 覆盖）
EXP=${EXP:-$(cd "$(dirname "$0")" && pwd)}
REPO=${REPO:-$(cd "$EXP/.." && pwd)}
PY=${PY:-python3}
MODEL=${MODEL:?set MODEL to Qwen2.5-VL-3B-Instruct}
DATA_ROOT=${DATA_ROOT:-$REPO/data}
HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
PYCACHE=${PYCACHE:-$REPO/pycache}

# 视频尺度实验评测：保真度（KL/top-1）+ 输出口径 SQNR
# 用法: bash video_eval.sh <run_dir_name> <gpu> <share:0|1>
set -u
RUN=$1; GPU=$2; SHARE=$3
cd "$REPO"
CK=$(ls -d outputs_par/$RUN/*/ | head -1)mas_parameters.pth
if [ "$SHARE" = "1" ]; then SNR_MODE=vision; else SNR_MODE=mas; fi
ENV="MASQUANT_DATA_ROOT=$DATA_ROOT MASQUANT_WIKITEXT_DIR="$DATA_ROOT"/wikitext HF_HOME=$HF_HOME PYTHONPYCACHEPREFIX=$PYCACHE CUDA_VISIBLE_DEVICES=$GPU MAS_VIDEO_SHARE_VISION=$SHARE"
env --chdir="$REPO" $ENV "$PY" -u tools/fidelity_eval.py \
    --model "$MODEL" --mas "$CK" --modes mas \
    --data-type text-vision-video --compare-type text-vision --nsamples 16 \
    --out outputs/vid_fid_$RUN.json > logs/vid_fid_$RUN.log 2>&1
env --chdir="$REPO" $ENV "$PY" -u tools/snr_probe.py \
    --model "$MODEL" --scales "$CK" --modes $SNR_MODE,uniform \
    --data-type text-vision-video --nsamples 4 --layer-step 9 \
    --out outputs/vid_snr_$RUN.json > logs/vid_snr_$RUN.log 2>&1
echo "== $RUN (share=$SHARE)"
grep -E "^\[mas\]" logs/vid_fid_$RUN.log
grep -A1 "mode: $SNR_MODE\|mode: uniform" logs/vid_snr_$RUN.log | grep -E "mode|MEAN"
