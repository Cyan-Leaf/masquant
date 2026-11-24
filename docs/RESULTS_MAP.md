# 结果索引（RESULTS_MAP）

本文件回答三个问题：**报告里的每张表/图，数据从哪来、怎么重算、原始文件在哪。**

- 报告： [`复现与改进_实验报告.md`](复现与改进_实验报告.md)
- 原始结果： [`../results/`](../results/) —— 已提交进仓库，可直接重算，不必重跑训练
- 实验脚本： [`../experiments/`](../experiments/)

> 约定：`n` = 独立训练次数（不是样本数）；`±` = 样本标准差；
> 所有 benchmark 的 `±` 一律是 **95% bootstrap CI**，不是 std。

---

## 1. 报告 §0.1 实验设置

无数据，说明与官方脚本的差异。

---

## 2. 报告 §1 PPL 主表（表 1.1）

**数据文件**：[`results/ppl_runs.json`](../results/ppl_runs.json)（从训练日志自动抽取，含每次的日志路径）

| 配置 | JSON key | 脚本 | 说明 |
|---|---|---|---|
| RTN | `rtn` | `scripts/run_ppl_baselines.sh` | 不平滑 |
| SmoothQuant | `sq` | `scripts/run_ppl_baselines.sh` | 闭式解，不训练 |
| 统一 scale + 训练 | `uni_opt` | `scripts/run_ppl_baselines.sh` | merged_scales |
| MAS 仅初始化 | `mas_noopt` | `scripts/run_ppl_baselines.sh` | 不做训练 |
| MAS 训练（原校准） | `mas_origcal` | `scripts/run_ppl_baselines.sh` | 官方口径 |
| MAS + 混合校准 0.5 | `mas_mix05` | `experiments/sweep_ratio.sh` | **推荐配置** |
| 统一 scale + 混合校准 | `uniform_mix05` | `experiments/sweep_ratio_uni.sh` | 对照 |
| MAS + LWC | `mas_lwc` | — | PPL 低但下游差的反例 |

**重算**：

```bash
# 单独重算某个配置的 PPL（需已有 mas_parameters.pth）
MODE=split_scales bash experiments/sweep_ratio.sh RERUN "0:0.5:1"
```

**注意**：`mas_lwc` 与 `mas_mix_lwc` 是两个不同实验（原校准 / 混合校准），
两者 PPL 都低但下游都差，报告里只用前者做反例。

---

## 3. 报告 §2 多模态 Benchmark 主表与 Fig.2

**数据文件**：

| 文件 | 内容 |
|---|---|
| [`results/bench/bootstrap.json`](../results/bench/bootstrap.json) | 各配置准确率 + 95% CI + 配对差异（本表直接由它生成） |
| `results/bench/{fp16,sq,mas05,mas075,uniform05,uniform075,mas_lwc}/*.jsonl` | **逐样本结果**（doc_id / exact_match / 回答），可重算任何统计 |

**脚本**：`experiments/bench_full.sh`（7 个配置并行，每配置一张卡）

**重算**：

```bash
# 从已提交的逐样本结果重算 CI 与配对差异
python tools/paired_bootstrap.py \
  --runs fp16=results/bench/fp16 sq=results/bench/sq mas05=results/bench/mas05 \
         mas075=results/bench/mas075 uniform05=results/bench/uniform05 \
         uniform075=results/bench/uniform075 mas_lwc=results/bench/mas_lwc \
  --tasks scienceqa_img textvqa_val --ref mas075 --out /tmp/bootstrap.json
```

**Fig.2**：`python tools/make_figures.py --outdir docs/figs`（直接读 `results/bench/bootstrap.json`）

> ⚠️ `results/bench/bootstrap.json` 里的 `mas05`/`uniform05` 是全量 W4A8 主表的配置；
> 报告正文引用时写作 "MAS(r=0.5)" / "Uniform(r=0.5)"。

---

## 4. 报告 §2.0 按题型拆分与 Fig.3

**数据文件**：

| 文件 | 内容 |
|---|---|
| [`results/breakdown/sqa_by_question_type.json`](../results/breakdown/sqa_by_question_type.json) | 分组准确率 + MAS−SQ / FP16−MAS 的配对差值 |
| [`results/breakdown/flipped_cases_sqa.json`](../results/breakdown/flipped_cases_sqa.json) | 翻转样本的可读案例（题干、选项、各方回答） |

**脚本**：

```bash
python tools/sqa_breakdown.py \
  --runs fp16=results/bench/fp16 sq=results/bench/sq mas=results/bench/mas05 uni=results/bench/uniform05 \
  --parquet $SQA_PARQUET --out /tmp/breakdown.json

python tools/error_analysis.py \
  --runs fp16=results/bench/fp16 mas075=results/bench/mas075 sq=results/bench/sq \
  --task scienceqa_img --ref fp16 --out /tmp/cases.json
```

`$SQA_PARQUET` 指向 ScienceQA-IMG test（2017 条）的 parquet，需自行下载；
脚本按样本数匹配，不依赖文件名。

**Fig.3**：`tools/make_figures.py` 读 `results/breakdown/sqa_by_question_type.json`。

> 报告的"必须读图"分组 = `hint` 字段为空。该组 41% 是"州首府"地图题。

---

## 5. 报告 §2.1 校准比例扫描与 Fig.1

**数据文件**：`results/ppl_runs.json` 里的 `mix_ratio0125` / `mix_ratio025` / `mix_ratio0375` /
`mix_ratio0875`，以及 `mas_mix05`（=0.5）、`mas_mix075`（=0.75）、`mas_origcal`（=0）。

**跨领域（C4）**：同文件的 `_c4_out_of_domain` 段。

**脚本**：

```bash
# wikitext：扫比例（每个比例一张卡，各跑 3 次）
ACT=./act_scales/<model>-text-vision-mix_tr0.75-128.pt \
  bash experiments/sweep_ratio.sh S "0:0.75:1 1:0.75:2 2:0.75:3"

# C4 跨领域：同一批 checkpoint 换测试集
bash experiments/c4_ppl.sh mas_mix05 0 split_scales <ckpt> --let
```

**Fig.1**：`tools/make_figures.py`（比例曲线的数字写在该脚本 `fig1()` 里，方便核对）。

> ⚠️ 比例扫描必须用**与比例匹配的 act_scales**，否则结论会被混杂污染（报告 §2.1.4）。

---

## 6. 报告 §3 NVFP4（表 3.1 / §3.4）

| 内容 | 文件 | 脚本 |
|---|---|---|
| NVFP4 PPL | `results/ppl_runs.json` 的 `nvfp4_*` | `experiments/repeat_ppl.sh` |
| 尺度敏感性（合成激活） | [`results/nvfp4/scale_curve_synthetic.json`](../results/nvfp4/scale_curve_synthetic.json) | `tools/nvfp4_scale_curve.py` |
| 尺度敏感性（真实激活） | [`results/nvfp4/scale_curve_real_activations.json`](../results/nvfp4/scale_curve_real_activations.json) | `tools/nvfp4_scale_curve.py --real` |
| 尺度缩放后 PPL | `results/ppl_runs.json` 的 `nvfp4_scale_k*` | `tools/scale_ckpt.py` + 评测 |

```bash
# 复现尺度敏感曲线（真实激活）
python tools/nvfp4_scale_curve.py --scales <mas_parameters.pth> --real \
    --layers 0,18,35 --mods self_attn.q_proj,mlp.up_proj --out /tmp/curve.json
```

---

## 7. 报告 §4 CMC（n=500）

| 文件 | 内容 |
|---|---|
| [`results/bench/cmc_bootstrap_n500.json`](../results/bench/cmc_bootstrap_n500.json) | 无补偿 vs CMC 的准确率、CI、配对差异 |
| `results/bench/cmc_n500/{nocomp,cmc}_samples_*.jsonl` | 逐样本结果 |

```bash
python tools/paired_bootstrap.py \
  --runs nocomp=results/bench/cmc_n500 cmc=results/bench/cmc_n500 \
  --tasks scienceqa_img textvqa_val --ref nocomp
```
（`cmc_n500/` 里同一目录有两种前缀的文件，脚本按 `samples_*.jsonl` 匹配，
因此这里建议直接用报告里的两个 n=500 目录重跑，见报告 §4。）

---

## 8. 报告 §5 SQNR 与视频、Fig.4

| 内容 | 文件 | 脚本 |
|---|---|---|
| 分模态 SQNR（288 配对，Fig.4） | [`results/snr/modality_sqnr_w4a8_288pairs.json`](../results/snr/modality_sqnr_w4a8_288pairs.json) | `tools/snr_probe.py` |
| 修复前后对比 | [`results/snr/sqnr_before_vs_after_fix.json`](../results/snr/sqnr_before_vs_after_fix.json) | 同上 |
| 旧版（有 bug，**勿引用**） | `results/snr/video_buggy_old_DO_NOT_USE.json` | — |
| 视频尺度实验保真度 | `results/video/fidelity_V_vshare{0,1}_r*.json` | `experiments/video_eval.sh` |

```bash
# 正式规模：8 样本 × 每 6 层 = 288 配对
python tools/snr_probe.py --model $MODEL --scales <mas_parameters.pth> \
    --data-type text-vision-video --nsamples 8 --layer-step 6 \
    --modes rtn,mas,uniform --out /tmp/snr.json
```

**Fig.4**：`tools/make_figures.py` 读该 JSON；误差棒是非配对标准误，
配对 CI 见报告 §5.2 正文。

---

## 9. 报告 §6 保真度与阴性结果

| 内容 | 文件 |
|---|---|
| bug 修复前后 KL/一致率 | [`results/fidelity/after_fix_uniform_vs_mas.json`](../results/fidelity/after_fix_uniform_vs_mas.json) |
| uniform vs CMC 交叉验证 | [`results/fidelity/uniform_vs_cmc_crosscheck.json`](../results/fidelity/uniform_vs_cmc_crosscheck.json) |
| 旧版（有 bug，**勿引用**） | `results/fidelity/buggy_before_fix_DO_NOT_USE.json` |
| 平滑因子正值约束 / 按模态归一化 | `results/ppl_runs.json`（`mas_ratio075_posproject` / `pn_ratio*`） |

```bash
python tools/fidelity_eval.py --model $MODEL --mas <ckpt> \
    --modes uniform,mas --data-type text-vision-video --compare-type text-vision \
    --nsamples 8 --out /tmp/fid.json
```

---

## 10. 报告 §7 工程问题

无数据文件，全部是环境与踩坑记录。

---

## 11. 复现全流程（从零到这 750 行报告）

预计 8 卡 × 约 40 机器小时。最小可复现路径（约 6 小时，只复现主表与核心改进）：

```bash
# 0) 环境与数据
export MODEL=/path/to/Qwen2.5-VL-3B-Instruct
export MASQUANT_DATA_ROOT=/path/to/data      # coco/ wikitext/ hf_datasets/

# 1) 激活统计（官方校准集）
python generate_act_scale_shift.py --model $MODEL --dataset-type text-vision --nsamples 128

# 2) 三组基线 PPL（RTN / SQ / MAS 仅初始化）
bash scripts/run_ppl_baselines.sh

# 3) MAS 训练（原校准，3 次重复）
for r in 1 2 3; do
  inference_mode=split_scales CUDA_VISIBLE_DEVICES=$r python main.py --model $MODEL \
    --mode train --epochs 2 --wbits 4 --abits 8 --symmetric --group_size 0 --let \
    --loss_multi_modal_mae_alpha --dataset-type text-vision --nsamples 128 --eval_ppl \
    --output_dir outputs_par/mas_origcal_r$r
done

# 4) 混合校准（本工作主改进，3 次）
for r in 1 2 3; do
  MAS_CALI_TEXT_RATIO=0.5 bash experiments/sweep_ratio.sh MIX "0:0.5:$r"
done

# 5) 全量 benchmark（7 配置，约 2 小时）
bash experiments/bench_full.sh

# 6) 出图
python tools/make_figures.py --outdir docs/figs
```

---

## 12. B200 复跑与补充实验（2026-11-23 新增）

**这是一整套独立的结果**，与 §1–§11 的原机器结果并列，**不要混用**：绝对值不可跨机器直接比较，
详见 [`复现性诊断.md`](复现性诊断.md)。

| 内容 | 数据文件 | 文档 |
|---|---|---|
| 跨机器复现排查（训练噪声、确定性模式、act_scales 差异、逐层 loss） | [`results/repro_b200/logs/`](../results/repro_b200/logs/)（60+ 个原始日志，gzip） | [`复现性诊断.md`](复现性诊断.md) |
| B200 上的 benchmark（fp16 / sq / mas / uniform，SQA 2000 + TVQA 2000） | [`results/bench_b200/`](../results/bench_b200/)、`boot_*.json` | 复现性诊断 §3.2 |
| **NVFP4 W4A4 下游**（rtn / mas 闭式解） | `results/bench_b200/nvfp4_{rtn,mas}/`、`boot_nvfp4.json` | [`补充实验.md`](补充实验.md) §1 |
| **W4A6 下游**（sq / mas 混合校准） | `results/bench_b200/{sq,mas}_w4a6/`、`boot_w4a6_nvfp4.json` | 补充实验 §2 |
| 学习率稳健性（`--let_lr 1e-2` ×4） | `results/repro_b200/logs/lr01_*.log.gz` | 补充实验 §3 |
| **CMC 全量下游**（新 ckpt：不补偿 / r0.05 / r0.2 / 自适应；旧 ckpt：不补偿 / r0.05） | `results/cmc_b200/{new,old}_*/`、`boot_cmc*.json` | 补充实验 §4.1–4.2 |
| CMC 单层诊断（72 模块 × 校准/未见） | `results/cmc_b200/cmc_probe_{new,old}_r005.json`（`tools/cmc_probe.py`） | 补充实验 §4.3 |
| B200 上所有 PPL 汇总（n / mean / std / 每次的值） | `results/repro_b200/ppl_b200_summary.json` | 复现性诊断 §2.6、§3.1 |

**重算示例**：

```bash
python tools/paired_bootstrap.py \
    --runs fp16=results/bench_b200/fp16 nvfp4_mas=results/bench_b200/nvfp4_mas \
           mas_w4a8=results/bench_b200/mas_w4a8 --tasks scienceqa_img textvqa_val --ref fp16
```

**PPL 抽取**（wikitext2 与 C4 都在日志里）：

```bash
zgrep -hoE "(wikitext2|c4) : [0-9.]+" results/repro_b200/logs/*.log.gz | sort | uniq -c
```

---

## 12b. 改进 F：平滑后的 GPTQ（2026-11-23 新增，B200）

| 内容 | 数据文件 | 文档 |
|---|---|---|
| 全部 PPL 运行（~190 个：平滑方式 × 求解器 × checkpoint，wikitext2 + C4） | `results/gptq/ppl_summary.json`（`runs` 为每次的值，`paired` 为同一 ckpt 的配对差）、`ppl_summary.txt`、原始日志 `results/gptq/logs/gptq_<tag>.log.gz` | [`改进F_平滑后GPTQ.md`](改进F_平滑后GPTQ.md) §3.1–3.5 |
| 下游逐题预测（mas_gptq / mas_clip / single_clip / sq_clip / plain_gptq / plain_clip / masinit_clip / uni_clip） | `results/gptq/bench/<name>/samples_*.jsonl` | 改进 F §3.3 |
| 下游配对 bootstrap | `results/gptq/subset2000_vs_{plain,fp16,masrtn}.txt`（同一批前 2000 题）、`boot_solver_*.json` | 改进 F §3.3 |
| Fig.5 / Fig.6 | `tools/make_fig_gptq.py results/gptq/ppl_summary.json docs/figs` | 书面报告 §4.6 |

日志 tag 命名：`<组>_<ckpt>_<求解器>`。组：`mas`（混合校准训练的 MAS）、`uni`（统一 scale 训练）、`orig`（原校准训练的 MAS）、
`xcal`（原校准训练的尺度 + 混合校准的 Hessian）、`single`（只用文本尺度单权重）、`sq`/`masinit`/`plain`（闭式解/不平滑）、`nv_*`（NVFP4 W4A4）。
求解器：`rtn`、`gptq`、`clip`（GPTQ+clip）、`clipcd`、`rtnclip`（只 clip 不 GPTQ）、`actq`、`bal`、`hall`、`*32`（NVFP4 求解用 FP32 块缩放）。

```bash
python tools/gptq_summary.py <logs 目录> --json results/gptq/ppl_summary.json   # 重算 PPL 表与配对差
python tools/subset_boot.py 2000 fp16=results/bench_b200/fp16 mas_rtn=results/bench_b200/mas_w4a8 \
    mas_clip=results/gptq/bench/mas_clip plain_clip=results/gptq/bench/plain_clip --ref plain_clip
```

---

## 12.5 报告 §6.7 / §2b 音频表（Qwen2.5-Omni-3B）

**数据文件**：[`results/omni/`](../results/omni/)（逐条转写，`.jsonl.gz`，数据路径已脱敏为 `<DATA_ROOT>`）

| 配置 | 文件 | 重算 |
|---|---|---|
| FP16 | `fp16_full.jsonl.gz` | `tools/omni_paired.py --ref ...` |
| MAS 闭式解（原 / 混合 / 纯音频） | `mas_init*.jsonl.gz` | 同上 |
| MAS 训练（原 / 混合 / 纯音频）×3 | `mas_*_r{1,2,3}.jsonl.gz` | 同上，`--group` 给三次 |
| **统一 scale 训练 ×3** | `uni_mix_r{1,2,3}.jsonl.gz` | 同上 |
| 闭式解 + GPTQ（原 / 混合） | `gptq*.jsonl.gz` | 同上 |
| **训练 + GPTQ ×3** | `gt_mas_mix_r{1,2,3}.jsonl.gz` | 同上 |
| **GPTQ + clip + cd** | `gt_clip_init_mix.jsonl.gz` | 同上 |
| SQ / RTN（崩溃，提前停止） | `sq_head40 / rtn_head316.jsonl.gz` | 同上 |

一键复算全表见 [`Omni_音频实验.md`](Omni_音频实验.md) §7；WER 定义（语料级、小写去标点）见 `tools/omni_wer.py`。
**不需要 GPU**：参考转写从 LibriSpeech 的 `*.trans.txt` 回查，只需 `MASQUANT_DATA_ROOT`。

---

## 13. 数字自查清单

报告里每个数字都应能被 `results/` 复算。抽查方式：

```bash
# PPL 均值/标准差
python -c "import json;d=json.load(open('results/ppl_runs.json'));\
print({k:(v['n'],v['mean'],v['std']) for k,v in d.items() if 'n' in v})"

# benchmark 准确率与 CI
python -c "import json;d=json.load(open('results/bench/bootstrap.json'));\
print({t:{k:round(v['acc']*100,2) for k,v in d[t].items() if isinstance(v,dict)} for t in d})"
```

若发现不一致，**以 `results/` 为准**并在报告里标注更正 —— 本项目已多次出现
"小样本结论被全量数据推翻"的情况（见报告 §2、§5.2、§6.3）。
