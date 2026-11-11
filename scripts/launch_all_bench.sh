#!/bin/bash
# 依次在指定 GPU 上启动 4 组多模态评测（fp16 / SQ / MAS / MAS-noopt）
cd "$(dirname "$0")/.."
export LIMIT=${LIMIT:-0.1}
export TASKS=${TASKS:-"scienceqa_img,textvqa_val"}
export LMMS_EVAL_BATCH_SIZE=${LMMS_EVAL_BATCH_SIZE:-4}
export LMMS_MAX_PIXELS=${LMMS_MAX_PIXELS:-401408}
export LMMS_PADDING_SIDE=${LMMS_PADDING_SIDE:-left}
mkdir -p logs

nohup bash scripts/run_bench.sh fp16     1 16 8 > /tmp/b_fp16.log 2>&1 &
sleep 8
nohup bash scripts/run_bench.sh sq       2 4 8  > /tmp/b_sq.log 2>&1 &
sleep 8
nohup bash scripts/run_bench.sh mas      3 4 8  > /tmp/b_mas.log 2>&1 &
sleep 8
nohup bash scripts/run_bench.sh masnoopt 4 4 8  > /tmp/b_masnoopt.log 2>&1 &
echo "launched fp16(gpu1) sq(gpu2) mas(gpu3) masnoopt(gpu4), LIMIT=$LIMIT TASKS=$TASKS"
