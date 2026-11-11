#!/bin/bash

# 通用路径（按需修改或 export 覆盖）
EXP=${EXP:-$(cd "$(dirname "$0")" && pwd)}
REPO=${REPO:-$(cd "$EXP/.." && pwd)}
PY=${PY:-python3}
MODEL=${MODEL:?set MODEL to Qwen2.5-VL-3B-Instruct}
DATA_ROOT=${DATA_ROOT:-$REPO/data}
HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
PYCACHE=${PYCACHE:-$REPO/pycache}

# 批量多模态评测（每组一张卡，逐样本落盘）
cd "$REPO"
M="$MODEL"
CKV=$(ls -d outputs_vl/w4a8/*/ | head -1)mas_parameters.pth
CKMIX=$(ls -d outputs_par/C_mas_mix_r1/*/ | head -1)mas_parameters.pth
CKUNI=$(ls -d outputs_par/C_uniformopt_mix_r1/*/ | head -1)mas_parameters.pth
CKLWC=$(ls -d outputs_par/C_mas_lwc_r1/*/ | head -1)mas_parameters.pth

export LIMIT=${LIMIT:-0.2} TASKS=${TASKS:-scienceqa_img,textvqa_val}

bash "$EXP"/bench_ckpt.sh fp16       0 merged_scales none --wbits 16 --abits 16 &
sleep 5
bash "$EXP"/bench_ckpt.sh sq         1 merged_scales none --wbits 4 --abits 8 --let &
sleep 5
bash "$EXP"/bench_ckpt.sh mas        2 split_scales   $CKV --wbits 4 --abits 8 --let &
sleep 5
bash "$EXP"/bench_ckpt.sh mas_noopt  3 split_scales   none  --wbits 4 --abits 8 --let &
sleep 5
bash "$EXP"/bench_ckpt.sh mas_mix    4 split_scales   $CKMIX --wbits 4 --abits 8 --let &
sleep 5
bash "$EXP"/bench_ckpt.sh uniform_mix 5 merged_scales $CKUNI --wbits 4 --abits 8 --let &
sleep 5
bash "$EXP"/bench_ckpt.sh mas_lwc    6 split_scales   $CKLWC --wbits 4 --abits 8 --let &
sleep 5
echo "7 jobs launched (gpu0-6); gpu7 kept free for CMC"
