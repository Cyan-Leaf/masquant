# MASQuant 复现与改进（多模态学习课程大作业）

本仓库基于官方 [MASQuant](https://github.com/alibaba/EfficientAI)（`masquant/` 子目录）完成了：
论文复现、**7 个实现 bug 的修复**、4 个方向的改进/拓展，以及一轮系统性的可复现性诊断。

- 第一个 commit 是上游源码原样导入，第二个 commit 起是本次的全部改动 —— `git diff` 可直接看出改了什么。

**文档（建议按这个顺序读）**

| 文档 | 用途 |
|---|---|
| [docs/汇报提纲.md](docs/汇报提纲.md) | **15 分钟讲稿**：每页讲什么、时间分配、可能被问的问题 |
| [docs/复现与改进_实验报告.md](docs/复现与改进_实验报告.md) | 完整实验数据：4 张图、配对 bootstrap、阴性结果、局限性 |
| [docs/RESULTS_MAP.md](docs/RESULTS_MAP.md) | **结果索引**：每张表/图 → 脚本 → 原始文件，可复算 |
| [docs/MASQuant_论文精读.md](docs/MASQuant_论文精读.md) | 论文推导、代码对照、与论文的分歧点 |
| [docs/复现与改进_使用说明.md](docs/复现与改进_使用说明.md) | 环境搭建与所有开关 |

**原始数据**：[`results/`](results/) 里是报告用到的全部结果文件（2 MB），
包括 7 个配置 × 两个 benchmark 的**逐样本结果**，可以不重跑训练直接复算任何统计量。

## 一句话结论

> **MAS 的可学习平滑确实有效（W4A8 下显著优于 SmoothQuant），但"分模态"这个设计本身没有带来增益**
> —— 在相同校准与相同初始化统计下，分模态与"统一 scale + 训练"在 PPL（p=0.97）和两个全量 benchmark
> （SQA 2017 / TextVQA 2000，配对 bootstrap）上都打平。真正带来收益的是**校准数据的模态平衡**：
> 原校准集文本 token 仅占 8.1%，调整后 PPL 从 18.46±1.43 降到 16.35±0.54（wikitext）/
> 20.21±0.23 → 18.53±0.11（C4，跨领域 p=0.0001）。

## 主要发现

**1. 复现成功，但增益来源需要澄清**

| 配置 | ScienceQA (n=2017) | TextVQA (n=2000) | PPL |
|---|---|---|---|
| FP16 | **80.61** | **75.78** | 11.11 |
| SmoothQuant | 75.51 | 62.58 | 22.63 |
| MAS + 混合校准 | 79.38 | 65.98 | 16.35 |
| 统一 scale + 同校准同训练 | 78.98 | 65.45 | 16.35 |

MAS 显著优于 SmoothQuant（+3.2 / +3.4，CI 不含 0）；但**分模态没有超过统一 scale**。

**2. 量化伤害的是视觉通路**（[Fig.3](docs/figs/fig3_breakdown.png)）
按 ScienceQA 的 `hint` 拆分：有文字提示的题上 MAS 已追平 FP16（+0.64，不显著），
**必须读图的题上仍差 2.24 点（显著）** —— 量化剩余损失集中在依赖视觉定位的题上。

**3. 改进经得起跨领域检验**（[Fig.1](docs/figs/fig1_calib_ratio.png)）
wikitext 既是补充的校准文本又是测试集，有同域作弊风险。换到 C4 重测，收益保留约 80%，p=0.0001。

**4. 修掉 7 个实现 bug**，其中 3 个会直接产出错误结论：

| bug | 影响 |
|---|---|
| `snr_probe.py` 构造了权重量化器却从未调用 | 只量化激活，量出来的不是 W4A8 误差 |
| `snr_probe.py` uniform 模式读错 key，静默回退成 text 尺度 | "统一 scale" 一列算错 |
| `fidelity_eval.py` 先拷贝尺度后加载，拷进去的是全 1 | 等价于裸 RTN，KL 虚高 3.6 倍 |
| `reuse_scale` 漏拷 `video_smooth_scale` | k/v/gate 视频尺度一直是全 1 |
| LWC 参数没进优化器 + dtype 崩溃 | `--lwc` 实际不可用 |
| 纯文本样本复用上个样本已 backward 的 `vision_diff` | "backward twice" |

## 代码结构

```
quantize/nvfp4.py      NVFP4（E2M1 + FP8 块缩放）伪量化            ← 新增
quantize/rank_alloc.py CMC 的按层自适应 rank 分配                  ← 新增
tools/snr_probe.py     按模态输出 SQNR（复现论文 Fig.2 的度量）      ← 新增
tools/fidelity_eval.py 无标签保真度（相对 FP16 的 KL / top-1 一致率）← 新增
tools/paired_bootstrap.py 配对 bootstrap 置信区间                   ← 新增
tools/sqa_breakdown.py 按题型拆分量化损失                           ← 新增
tools/error_analysis.py 导出翻转样本的可读案例                       ← 新增
tools/nvfp4_scale_curve.py 量化误差对平滑尺度的敏感性诊断             ← 新增
tools/make_figures.py  由落盘数字生成报告用图                        ← 新增
tools/scale_ckpt.py    按因子缩放 checkpoint 的平滑尺度              ← 新增
experiments/*.sh       全部实验的启动脚本（脱敏，改了路径即可跑）      ← 新增
```

主要改动：`quantize/masquant.py`（视频尺度、loss、bug 修复）、`quantize/int_linear.py`
（按模态 gather 的快速前向，prefill 3.84× / decode 9.24× 且逐位一致）、`custom_dataset.py`
（混合校准、比例可调）、`main.py`（NVFP4、确定性开关、多测试集 PPL）。

## 快速开始

```bash
export MODEL=/path/to/Qwen2.5-VL-3B-Instruct
export MASQUANT_DATA_ROOT=/path/to/data      # 含 coco/ wikitext/ hf_datasets/

# 1) 统计激活尺度
python generate_act_scale_shift.py --model $MODEL --dataset-type text-vision --nsamples 128
# 2) 训练 MAS 并测 PPL（官方口径 W4A8）
inference_mode=split_scales python main.py --model $MODEL --mode train --epochs 2 \
    --wbits 4 --abits 8 --symmetric --group_size 0 --let \
    --loss_multi_modal_mae_alpha --dataset-type text-vision --nsamples 128 --eval_ppl
# 3) 本工作的改进：混合校准（纯文本占比可调）
MAS_CALI_TEXT_RATIO=0.5 ... --dataset-type text-vision-mix
```

所有新增开关见 [`docs/复现与改进_使用说明.md`](docs/复现与改进_使用说明.md)。

## 局限性

只在 Qwen2.5-VL-3B 上验证；benchmark 仅 SQA/TextVQA；视频是合成数据；部分结论重复次数为 3 次。
详见报告 [§8 局限性](docs/复现与改进_实验报告.md)。

## 致谢

上游代码来自 [alibaba/EfficientAI](https://github.com/alibaba/EfficientAI) 的 `masquant/` 子目录，
论文为 *MAS-Quant: Modality-Aware Symmetric Quantization for Multimodal Large Language Models*。
本仓库的所有改动均为课程作业性质，沿用上游 LICENSE。
