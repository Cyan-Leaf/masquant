# results/omni —— Qwen2.5-Omni-3B · W4A8 · LibriSpeech test-other（全量）

逐条转写（`main.py --eval_omni_task` 的原始输出），数据路径已脱敏为 `<DATA_ROOT>`，gzip 存放。
WER 全在同一批 2676 条（47790 参考词）上算，可直接离线复算。

```bash
export MASQUANT_DATA_ROOT=/path/to/data
python tools/omni_paired.py --ref results/omni/fp16_full.jsonl.gz \
  --group mas=results/omni/mas_r1.jsonl.gz,results/omni/mas_r2.jsonl.gz,results/omni/mas_r3.jsonl.gz \
  --group mix=results/omni/mas_mix_r1.jsonl.gz,results/omni/mas_mix_r2.jsonl.gz,results/omni/mas_mix_r3.jsonl.gz \
  --group libri=results/omni/mas_libri_r1.jsonl.gz,results/omni/mas_libri_r2.jsonl.gz,results/omni/mas_libri_r3.jsonl.gz \
  --group masinit=results/omni/mas_init.jsonl.gz --group masinit_mix=results/omni/mas_init_mix.jsonl.gz \
  --group masinit_libri=results/omni/mas_init_libri.jsonl.gz \
  --group gptq=results/omni/gptq.jsonl.gz --group gptq_mix=results/omni/gptq_mix.jsonl.gz \
  --group uniform_mix=results/omni/uni_mix_r1.jsonl.gz,results/omni/uni_mix_r2.jsonl.gz,results/omni/uni_mix_r3.jsonl.gz \
  --group train_gptq=results/omni/gt_mas_mix_r1.jsonl.gz,results/omni/gt_mas_mix_r2.jsonl.gz,results/omni/gt_mas_mix_r3.jsonl.gz \
  --group gptq_clip_mix=results/omni/gt_clip_init_mix.jsonl.gz
```

**核心一行**：统一 scale 6.58 ± 0.03 vs 分模态 6.58 ± 0.38，差 −0.01（p=0.999）——
"分模态本身无增益"在音频上同样成立。

| 文件 | 配置 | WER |
|---|---|---|
| `fp16_full.jsonl.gz` | FP16 | **4.78** |
| `mas_init.jsonl.gz` | MAS 闭式解 · 原校准 | 7.38 |
| `mas_init_mix.jsonl.gz` | MAS 闭式解 · 混合校准 0.5 | 7.02 |
| `mas_init_libri.jsonl.gz` | MAS 闭式解 · 纯音频校准 | 6.97 |
| `mas_r{1,2,3}.jsonl.gz` | MAS 训练 2ep · 原校准 ×3 | 8.03 ± 1.61 |
| `mas_mix_r{1,2,3}.jsonl.gz` | MAS 训练 2ep · 混合校准 ×3 | **6.58 ± 0.38** |
| `mas_libri_r{1,2,3}.jsonl.gz` | MAS 训练 2ep · 纯音频校准 ×3 | 8.59 ± 3.84 |
| `uni_mix_r{1,2,3}.jsonl.gz` | **统一 scale + 相同训练 · 混合校准 ×3** | **6.58 ± 0.03** |
| `gptq.jsonl.gz` | 闭式解 + GPTQ · 原校准 | 6.12 |
| `gptq_mix.jsonl.gz` | 闭式解 + GPTQ · 混合校准 | 5.92 |
| `gt_mas_mix_r{1,2,3}.jsonl.gz` | MAS 训练 + GPTQ · 混合 ×3 | **5.49 ± 0.27** |
| `gt_clip_init_mix.jsonl.gz` | 闭式解 + GPTQ + 逐行 γ + 坐标下降 · 混合 | **5.17** |
| `sq_head40.jsonl.gz` | SmoothQuant（前 40 条，已崩，提前停止） | 301.2 |
| `rtn_head316.jsonl.gz` | RTN 不平滑（前 316 条，已崩，提前停止） | 114.4 |

说明、配对 bootstrap 表与后续计划见 `docs/Omni_音频实验.md`。
