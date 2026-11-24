# Omni / 音频实验：全量结果

> 目的：把"音频没测"这条局限补上。VL 上我们得到"分模态本身无增益"，而**论文最有力的证据恰好在音频**
> （SmoothQuant 让 Omni 的 LibriSpeech WER 从 3.9 崩到 77.4）。补这个实验，要么把结论推广到三模态，
> 要么找到它失效的边界——两种结果都比现在强。
>
> 状态（2026-11-24 晚）：**全量 2676 条跑完，训练配置各 3 次重复，GPTQ 两组，配对 bootstrap 已做。**
> 上一版（同日早）基于 n=600 的初步数**已被 §0 的数字取代**，差异见 §5。

---

## 0. 主表（Qwen2.5-Omni-3B · W4A8 · LibriSpeech test-other 全量 2676 条 / 47790 词）

WER 是**语料级**（总错误词数 / 总参考词数），全部在同一批 2676 条上算；配对 bootstrap 10000 次。
PPL 在同一个 checkpoint 上另测（FP16 的 PPL 是本次补的锚点）。

| 配置 | WER ↓ | wikitext2 PPL ↓ | C4 PPL ↓ |
|---|---|---|---|
| FP16 | **4.78** | **9.49** | **13.30** |
| MAS 闭式解（原校准） | 7.38 | 16.69 | 20.18 |
| MAS 闭式解（混合 0.5） | 7.02 | 16.02 | 20.08 |
| MAS 闭式解（纯音频） | 6.97 | 16.80 | 20.56 |
| MAS 训练 2ep（原校准） | 8.03 ± 1.61（3） | 13.90 ± 0.15 | 18.32 ± 0.08 |
| **MAS 训练 2ep（混合 0.5）** | **6.58 ± 0.38（3）** | **13.81 ± 0.17** | **18.07 ± 0.12** |
| MAS 训练 2ep（纯音频） | 8.59 ± 3.84（3） | 14.79 ± 0.20 | 19.11 ± 0.11 |
| MAS 闭式解 + GPTQ（原校准） | 6.12 | 14.83 | 18.63 |
| **MAS 闭式解 + GPTQ（混合 0.5）** | **5.92** | **12.47** | **17.44** |
| SmoothQuant W4A8 | 崩（n=40 时已 301） | — | — |
| RTN（不平滑） | 崩（n=316 时已 114） | — | — |

三次重复的逐次数：MAS 训练 9.85 / 6.78 / 7.46；混合 6.40 / 6.32 / 7.02；纯音频 13.03 / 6.22 / 6.53。

**配对差异（WER 点，95% CI，p）**

| 对比 | 差 | CI | p |
|---|---|---|---|
| FP16 − GPTQ(混合) | −1.14 | [−1.41, −0.90] | 0.0001 |
| FP16 − MAS 闭式解 | −2.59 | [−3.49, −1.89] | 0.0001 |
| 训练(混合) − 训练(原校准) | **−1.45** | [−2.18, −0.65] | **0.0002** |
| 训练(纯音频) − 训练(混合) | **+2.01** | [+1.23, +2.78] | **0.0001** |
| 闭式解 − 训练（原校准） | −0.66 | [−1.58, +0.39] | 0.19 |
| 闭式解 − 训练（混合） | −0.44 | [−1.34, +0.34] | 0.26 |
| 闭式解 − 训练（纯音频） | **−1.62** | [−2.45, −0.75] | 0.0008 |
| GPTQ(原) − 闭式解(原) | **−1.25** | [−2.28, −0.31] | 0.011 |
| GPTQ(混合) − 闭式解(混合) | **−1.10** | [−1.70, −0.63] | 0.0001 |
| GPTQ(混合) − 训练(混合) | **−0.66** | [−1.36, −0.10] | 0.017 |
| GPTQ(原) − GPTQ(混合) | +0.20 | [−0.31, +0.88] | 0.56 |

闭式解下三种校准两两全部不显著（差 0.05~0.40，p≥0.40）。

---

## 1. 结论（可以直接写进报告的五条）

1. **论文的核心主张在音频上复现**：W4A8 下 SmoothQuant 与 RTN 在 ASR 上直接崩
   （SQ 40 条就 301%，RTN 316 条 114%），而 MAS 闭式解 7.38、最好配置 5.92，相对 FP16 只差 1~1.7 点。
   这是"分模态平滑在音频这种长序列、高动态范围模态上是必需的"最强的一组证据。

2. **混合校准在音频上的收益比 VL 上更大**：训练路径 8.03 ± 1.61 → 6.58 ± 0.38
   （−1.45，p=0.0002），**方差同时降到 1/4**，与 VL 上"PPL 18.46 ± 1.43 → 16.2 ± 0.4、方差 1/4"完全同构。
   这是**改进 A 的跨模态复现**。注意 Omni 原校准集文本 token 已占 38.7%（VL 只有 8.1%），
   仍有这么大收益，说明问题不只是"文本比例"，还有**校准样本的模态多样性**。

3. **"音频监督不足"的假设被否证**。用纯音频校准（65.2% 音频 token、263 条、与评测集说话人不重叠）训练，
   **比混合校准差 2.01 点（p=0.0001），方差反而最大（±3.84，出现一次 13.03 的崩塌）**。
   纯音频校准的文本 token 占 34.8%，但全是同一句"转写指令 + 转写结果"的模板，分布单一。
   所以正确的表述是：**校准集需要的是"模态多样 + 文本多样"，不是"把音频补够"**。
   这条把改进 A 从"VL 上的 PPL 改进"升级成**跨模态、跨模型一致的校准配比问题**——价值高于"补个音频"本身。

4. **训练并不优于闭式解**：原校准 +0.66（ns）、混合 +0.44（ns）、纯音频 +1.62（显著更差）。
   VL 上 NVFP4 侧"训练 12.14 vs 仅初始化 11.75"的同一现象，在音频上同向、量级 ~1 点。
   **上一版"训练比闭式解差 5 点"作废**，那是 n=600 前段的噪声（前 600 条 FP16 就有 6.30，整体 4.78）。

5. **改进 F（平滑后 GPTQ）在音频上有效，且对校准失衡更鲁棒**：
   原校准 6.12（比同校准闭式解 −1.25，p=0.011）、混合校准 **5.92**（−1.10，p=0.0001），
   是全表最好的 W4A8 配置，把与 FP16 的差距从 2.59 点压到 1.14 点。
   **与 VL 侧不同的是，音频上 GPTQ 在原校准上就有效**（VL 上分模态 GPTQ 在文本 8.1% 的校准集上 9/9 变差）。
   机理自洽：GPTQ 需要在足够多、足够多样的文本 token 上统计 Hessian，Omni 原校准已有 38.7% 文本，
   VL 原校准只有 8.1%。**这是"改进 F 依赖改进 A"这条因果链的跨模型证据。**

---

## 2. 两个新现象（值得单独讲）

### 2.1 PPL 与 WER 再次背离，且方向相反

| 配置 | WER | wikitext2 PPL |
|---|---|---|
| MAS 闭式解（原校准） | **7.38** | 16.69 |
| MAS 训练（原校准） | 8.03 | **13.90** |

纯文本 PPL 说"训练好 2.8"，ASR 说"闭式解好 0.65"。这是报告 §2 那条"PPL 与下游会背离"的**音频版**，
而且比 VL 上的 LWC 例子更干净：这次是同一方法、同一校准集，**只换指标就换结论**。

### 2.2 失效样本几乎全是"复读/跑题"，而且训练会制造它们

| 配置 | 复读/跑题条数（错误数 > 参考词数） | 完全正确率 |
|---|---|---|
| FP16 | 1.0 | 58.1% |
| MAS 闭式解（原） | 7.0 | 48.7% |
| MAS 闭式解（混合） | 4.0 | 48.4% |
| MAS 训练（原） | 8.7 | 48.1% |
| MAS 训练（混合） | 5.0 | **52.0%** |
| MAS 训练（纯音频） | 11.0 | 49.1% |
| GPTQ（原） | 4.0 | 52.8% |
| GPTQ（混合） | 2.0 | 51.6% |

只有个位数的样本，但单条能贡献 100+ 个错误词，**足以左右千分位级的 WER**。
纯音频校准的 ±3.84 方差主要就来自那一次 13.03 里的崩塌样本。
教训与 §3.3 的"均值 SQNR 看不出 RTN 崩"同源：**尾部样本决定音频指标，均值类代理指标不可靠。**

---

## 3. 环境与数据（已就位，勿重复下载）

| 项 | 位置 | 说明 |
|---|---|---|
| 模型 | `models/Qwen2.5-Omni-3B`（11.9 GB，3 shards） | 从 ModelScope 拉的，脚本 `tools/msdl_omni.py` |
| LibriSpeech test-other | `data/librispeech/LibriSpeech/test-other/`（2939 条，33 说话人） | `data/librispeech/test-other.tar.gz` 可删 |
| 评测集 | `data/jsonls/libri_test_other_eval.jsonl` | **2676 条**，30 个说话人 |
| 音频校准池 | `data/jsonls/libri_test_other_calib.jsonl` | **263 条**，**另外 3 个说话人**（与评测集无重叠） |
| 三模态校准 | `data/jsonls/omnibench.jsonl` | 128 条，ASR / 图像描述 / 音频+图像 三类轮换 |
| 混合校准 | `data/jsonls/omnibench_mix0.5.jsonl` | 128 条，50% 换成 wikitext 纯文本 |
| 数据构造脚本 | `tools/build_omni_data.py` | 一条命令重建下面四个文件 |
| WER / 配对比较 | `tools/omni_wer.py`、`tools/omni_paired.py` | 后者按音频对齐、支持重复取均值、配对 bootstrap |
| 逐条转写 | `results/omni/*.jsonl.gz` | 已脱敏，可直接复算 §0 全表 |

**三套校准集的模态 token 占比**（在 dataloader 缓存上按 `audio_token_index` / `image_token_index` 统计）：

| 校准集 | 音频 | 图像 | 文本 |
|---|---|---|---|
| `omnibench.jsonl`（原） | 19.4% | 41.9% | 38.7% |
| `omnibench_mix0.5.jsonl` | 11.4% | 25.3% | 63.3% |
| `libri_test_other_calib.jsonl` | 65.2% | 0 | 34.8% |

重建数据：

```bash
export MASQUANT_DATA_ROOT=<顶层>/data
python tools/build_omni_data.py --text_ratio 0.5
```

---

## 4. 一键跑法

```bash
export PY=<顶层>/venv/bin/python
export MASQUANT_DATA_ROOT=<顶层>/data
export OMNI_MODEL=<顶层>/models/Qwen2.5-Omni-3B     # release/masquant 下必须传，REPO_ROOT 指不到顶层 models/

cd release/masquant
bash scripts/omni_run.sh <tag> <gpu> <method>
# method: fp16 | rtn | sq | mas_init | mas | uniform
```

环境变量：

| 变量 | 默认 | 用途 |
|---|---|---|
| `EVAL_N` | 空（全量 2676） | 只评前 N 条，冒烟用。**配对比较必须同一批** |
| `PPL_SETS` | `wikitext2,c4` | 设为空字符串则不测 PPL（评测会快很多） |
| `BS` | 2 | 评测 batch |
| `MASQUANT_OMNI_CALIB_JSON` | 空 → 用 `omnibench.jsonl` | 换成 `omnibench_mix0.5.jsonl` / `libri_test_other_calib.jsonl` |
| `W` / `A` | 4 / 8 | 位宽 |

产物：`outputs_omni/<tag>.jsonl`（逐条转写）、`outputs_omni/<tag>/`（checkpoint）、`logs/omni_<tag>.log`。

**换校准集时必须先重算 act_scales**，否则 `main.py` 直接硬报错（这是故意的）：

```bash
MASQUANT_OMNI_CALIB_JSON=$MASQUANT_DATA_ROOT/jsonls/omnibench_mix0.5.jsonl \
  $PY generate_act_scale_shift.py --model $OMNI_MODEL --dataset-type text-audio-vision --nsamples 128
```

已有的 act_scales：`act_scales/Qwen2.5-Omni-3B-text-audio-vision-128.pt`（原）、
`..._omnibench_mix0.5-128.pt`、`..._libri_test_other_calib-128.pt`。纯音频那份的 `vision_scale` 是 0
（校准集里没有图像），ASR 评测不经过视觉分支，不影响 WER，但**不能拿它去做带图像的评测**。

GPTQ：在 `mas_init` 后面加 `--gptq` 即可。`--gptq_hsrc own` 默认按模态分 Hessian，
Omni 的 `audio_scale` 会拿到自己的一份（日志里能看到 `tags=['audio'] tokens={'audio': 13077}`）。

时间参考（单卡 B200）：FP16 全量 2676 条约 40 分钟（BS=2）；MAS 训练 2 epoch 约 5 分钟 + 评测约 45 分钟；
GPTQ 每个 checkpoint 约 15 分钟（含求解）。本次 14 个全量任务在 4 卡上约 4.5 小时跑完。

---

## 5. 上一版（n=600 初步数）与全量数的差异 —— 别再引用旧数

| 配置 | 旧（n=600 前段） | **新（全量 2676）** | 说明 |
|---|---|---|---|
| FP16 | 4.78（全量）/ 6.30（前 600） | 4.78 / — | 前 600 条明显更难，**旧表的"9.16 / 14.21"是难样本子集上的数** |
| MAS 闭式解 | 9.16 | **7.38** | |
| MAS 训练 | 14.21 | **8.03 ± 1.61** | 旧表"训练比闭式解差 5 点"→ 实际 +0.66（不显著） |
| SQ | 崩（301，n=40） | 同左 | 结论不变 |
| RTN | 崩（114，n=316） | 同左 | 结论不变 |

教训与 HANDOFF §6 第 7 条（"报数必须给重复次数"）并列：**子集上的相对结论不可外推**，
前 600 条与整体的 WER 差 1.5 点，足够把"差 5 点"变成"差 0.7 点"。

---

## 6. 三个必须知道的坑（都踩过，已修）

### 6.1 评测集不能带 system prompt（**影响 >30 个点，最重要**）

Omni 的默认 system prompt 是"你是 Qwen 虚拟人……"。带着它做 ASR，约 15% 的短句会被当成闲聊来回答：

```
REF: the egyptian background of the bondage
HYP: well you know the egyptian background of bondage is really complex there were a lot
     of factors at play for one thing the egyptians had a very hierarch...
```

FP16 的 WER 因此从 **4.78 虚高到 42.7**。`build_omni_data.py` 生成的**评测集已去掉 system prompt**，
校准集**保留**（那是 Omni 的标准对话格式，训练/激活统计都该按真实用法来）。

### 6.2 批量评测会撞量化层的 2D mask（已修）

模型以 `flash_attention_2` 加载，但量化层 `int_qwen_omni_layer.py` 固定走 `forward_sdpa`，
批量生成时传进来的是 2D padding mask，原代码按 4D 索引 → `IndexError`。
已在 `forward_sdpa` 里加 2D→4D 转换（含 left-padding 与因果，且每行至少能看见自己以免全掩码出 NaN）。
**batch=1 不经过该分支**，所以冒烟阶段可以先 `BS=1` 绕过诊断。

### 6.3 `--epochs` 必须显式给

`main.py` 里 `assert args.lwc or args.let` 只在 `epochs>0` 时触发，但 FP16 路径也要 `--epochs 0`，
否则会走到训练分支。`omni_run.sh` 已按 method 自动设好。

---

## 7. 复现校验（接手先跑这几条，能过说明环境没坏）

```bash
# 1) 数据在不在
wc -l $MASQUANT_DATA_ROOT/jsonls/libri_test_other_eval.jsonl   # 2676
wc -l $MASQUANT_DATA_ROOT/jsonls/omnibench.jsonl               # 128

# 2) 小样本把 FP16 跑一遍（约 5 分钟），WER 应接近 5%（n=40 时约 14%，样本太小，别慌）
EVAL_N=60 PPL_SETS= bash scripts/omni_run.sh check_fp16 0 fp16

# 3) 复算 §0 全表（不需要 GPU）
MASQUANT_DATA_ROOT=$MASQUANT_DATA_ROOT python tools/omni_paired.py \
  --ref results/omni/fp16_full.jsonl.gz \
  --group mas=results/omni/mas_r1.jsonl.gz,results/omni/mas_r2.jsonl.gz,results/omni/mas_r3.jsonl.gz \
  --group mix=results/omni/mas_mix_r1.jsonl.gz,results/omni/mas_mix_r2.jsonl.gz,results/omni/mas_mix_r3.jsonl.gz \
  --group libri=results/omni/mas_libri_r1.jsonl.gz,results/omni/mas_libri_r2.jsonl.gz,results/omni/mas_libri_r3.jsonl.gz \
  --group masinit=results/omni/mas_init.jsonl.gz --group gptq=results/omni/gptq.jsonl.gz \
  --group gptq_mix=results/omni/gptq_mix.jsonl.gz
```

---

## 8. 还能做的（优先级从高到低）

1. **GPTQ 加在训练后的 checkpoint 上**：目前 GPTQ 只加在闭式解上（5.92），训练(混合) 是 6.58。
   若 GPTQ 在训练 ckpt 上同样有效，可能再降 0.5 点。约 4 个任务 × 20 分钟。
2. **`--gptq_clip`（逐行 γ）+ `--gptq_cd`**：VL 上 clip 在 NVFP4 上有用，音频上未测。
3. **混合比例扫描**：目前只测了 0.5（63.3% 文本）。0.25 / 0.75 各 3 次可确认是平台还是尖峰。
4. **W4A6 / NVFP4**：音频上 RTN 已经崩，W4A6 大概率也崩，除非上 MAS + GPTQ。
5. **局限（照实写）**：只用 thinker（`LMClass.py` 里 `enable_audio_output=False`），不测语音生成；
   校准 128 样本；LibriSpeech test-other 单一测试集，且**评测集与校准集按说话人切分**
   （比论文原设定"校准=测试同源"更严，这是优点，报告里要说明）。

---

## 9. 代码改动清单（相对 VL 版本）

| 文件 | 改动 |
|---|---|
| `custom_dataset.py` | `text-audio-vision` 的校准 jsonl 支持 `MASQUANT_OMNI_CALIB_JSON` 覆盖 |
| `main.py` | ① `_omni_calib_tag()`：换了校准文件就把文件名带进 `act_scales` / dataloader 缓存的名字，避免串用；② 换了文件但 act_scales 不存在时**硬报错**（回退会得到错误的初始化统计） |
| `generate_act_scale_shift.py` | 同上，缓存与 `act_scales` 文件名带 Omni 校准标签 |
| `models/int_qwen_omni_layer.py` | `forward_sdpa` 支持 2D padding mask（见 §6.2） |
| `scripts/omni_run.sh` | 新增：Omni 的量化 + 训练 + 评测一键入口 |
| `tools/build_omni_data.py` | 新增：构造校准 / 评测 jsonl |
| `tools/omni_wer.py` | 新增：WER（支持仓库内脱敏路径 `<DATA_ROOT>`、`.gz` 输入） |
| `tools/omni_paired.py` | 新增：按音频对齐的配对 bootstrap（多重复取均值、CI、p） |
| `results/omni/` | 新增：全量逐条转写（脱敏、gzip），可离线复算 §0 全表 |

`tools/msdl_omni.py`（下载器）在项目顶层 `tools/`，未进仓库。
