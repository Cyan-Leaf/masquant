# results/omni —— Qwen2.5-Omni-3B · W4A8 · LibriSpeech test-other（初步）

逐条转写（`main.py --eval_omni_task` 的原始输出），数据路径已脱敏为 `<DATA_ROOT>`。复算：

```bash
MASQUANT_DATA_ROOT=/path/to/data python tools/omni_wer.py results/omni/*.jsonl
```

| 文件 | 配置 | n | WER |
|---|---|---|---|
| `fp16_full.jsonl` | FP16 | 2676（全量） | 4.78 |
| `mas_init_head600.jsonl` | MAS 闭式解 | 前 600 | 9.16 |
| `mas_train_head600.jsonl` | MAS 训练 2 epoch | 前 600 | 14.21 |
| `rtn_head316.jsonl` | RTN（不平滑） | 前 316（已崩，提前停止） | 114.4 |
| `sq_head40.jsonl` | SmoothQuant | 前 40（已崩，提前停止） | 301.2 |

配对比较只能在同一批前 N 条上做（FP16 前 600 条 = 6.30）。说明与后续计划见 `docs/Omni_音频实验.md`。
