#!/bin/bash
# PPL 复现：Qwen2.5-VL-3B, W4A8，对应论文 Table 3/5 的各行
set -e
cd "$(dirname "$0")/.."
PY=${PY:-python3}
# 模型目录自己指定：export MODEL=/path/to/Qwen2.5-VL-3B-Instruct
MODEL=${MODEL:?set MODEL to a Qwen2.5-VL-3B-Instruct directory}
W=${W:-4}
A=${A:-8}
COMMON="--model $MODEL --mode train --wbits $W --abits $A --symmetric --group_size 0 --cache_dir ./cache --dataset-type text-vision --nsamples 128 --eval_ppl"
mkdir -p logs

# RTN：不做平滑
inference_mode=merged_scales CUDA_VISIBLE_DEVICES=2 nohup $PY -u main.py $COMMON --epochs 0 \
  --output_dir ./outputs_vl/rtn_w${W}a${A} > logs/ppl_rtn_w${W}a${A}.log 2>&1 &

# SmoothQuant：统一 scale，闭式解初始化，不训练
inference_mode=merged_scales CUDA_VISIBLE_DEVICES=3 nohup $PY -u main.py $COMMON --epochs 0 --let \
  --output_dir ./outputs_vl/sq_w${W}a${A} > logs/ppl_sq_w${W}a${A}.log 2>&1 &

# MAS 不训练：分模态闭式解初始化
inference_mode=split_scales CUDA_VISIBLE_DEVICES=4 nohup $PY -u main.py $COMMON --epochs 0 --let \
  --output_dir ./outputs_vl/mas_noopt_w${W}a${A} > logs/ppl_mas_noopt_w${W}a${A}.log 2>&1 &

# Uniform + 可学习优化（OmniQuant 式统一 scale）
inference_mode=merged_scales CUDA_VISIBLE_DEVICES=5 nohup $PY -u main.py $COMMON --epochs 2 --let \
  --loss_multi_modal_mae_alpha --output_dir ./outputs_vl/uni_opt_w${W}a${A} > logs/ppl_uni_opt_w${W}a${A}.log 2>&1 &

wait
grep -H "wikitext2 :" logs/ppl_*_w${W}a${A}.log
