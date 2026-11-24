# Omni / 音频实验：链路已跑通，等大规模验证

> 目的：把"音频没测"这条局限补上。VL 上我们得到"分模态本身无增益"，而**论文最有力的证据恰好在音频**
> （SmoothQuant 让 Omni 的 LibriSpeech WER 从 3.9 崩到 77.4）。补这个实验，要么把结论推广到三模态，
> 要么找到它失效的边界——两种结果都比现在强。
>
> 状态（2026-11-24）：**链路跑通并已出一组 n=600 的初步数**，未做重复、未做 GPTQ、未做混合校准的正式对比。
> 最后更新：2026-11-24。

---

## 0. 一句话结论（初步，勿直接写进报告正文）

在 Qwen2.5-Omni-3B、W4A8、LibriSpeech test-other 上：

| 配置 | WER ↓ | 备注 |
|---|---|---|
| FP16 | **4.78（n=2676，全量）** | 官方口径，与 Qwen 报告的同量级 |
| FP16（前 600 条，用于配对） | 6.30 | 评测集前段比整体难一些，**配对比较只能用同一批** |
| MAS 闭式解（不训练） | **9.16（n=600）** | 未崩溃 |
| MAS 训练 2 epoch | **14.21（n=600）** | 比闭式解**差 5 点**，见 §4 |
| SmoothQuant W4A8 | **崩**（n=40 就已 301） | 与论文"WER 77.4"定性一致 |
| RTN（不平滑） | **崩**（114，n=316） | 与 VL 上的行为一致 |

**三条可说的话**：① SQ 在音频上确实崩，而 MAS 闭式解不崩——论文的核心主张在音频上复现了；
② 但这个"不崩"来自**分模态的闭式解**，不是来自训练；③ 训练反而更差，和 NVFP4 上"训练不如闭式解"是同一现象。

**还不能说的话**：不能说混合校准在音频上有效/无效——**没跑**。这正是最该补的一组。

---

## 1. 环境与数据（已就位，勿重复下载）

| 项 | 位置 | 说明 |
|---|---|---|
| 模型 | `models/Qwen2.5-Omni-3B`（11.9 GB，3 shards） | 从 ModelScope 拉的，脚本 `tools/msdl_omni.py` |
| LibriSpeech test-other | `data/librispeech/LibriSpeech/test-other/`（2939 条，33 说话人） | `data/librispeech/test-other.tar.gz` 可删 |
| 评测集 | `data/jsonls/libri_test_other_eval.jsonl` | **2676 条**，30 个说话人 |
| 音频校准池 | `data/jsonls/libri_test_other_calib.jsonl` | **263 条**，**另外 3 个说话人**（与评测集无重叠） |
| 三模态校准 | `data/jsonls/omnibench.jsonl` | 128 条，ASR / 图像描述 / 音频+图像 三类轮换 |
| 混合校准 | `data/jsonls/omnibench_mix0.5.jsonl` | 128 条，50% 换成 wikitext 纯文本 |
| 数据构造脚本 | `tools/build_omni_data.py` | 一条命令重建上面四个文件 |
| WER 脚本 | `tools/omni_wer.py` | 词级编辑距离，小写去标点 |

重建数据：

```bash
export MASQUANT_DATA_ROOT=<顶层>/data
python tools/build_omni_data.py --text_ratio 0.5
```

正常输出：`calib 263 (3 spk), eval 2676`、四个 jsonl。

---

## 2. 一键跑法

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
| `MASQUANT_OMNI_CALIB_JSON` | 空 → 用 `omnibench.jsonl` | 换成 `omnibench_mix0.5.jsonl` 即混合校准 |
| `W` / `A` | 4 / 8 | 位宽 |

产物：`outputs_omni/<tag>.jsonl`（逐条转写，可直接喂 `tools/omni_wer.py`）、`outputs_omni/<tag>/`（checkpoint）、`logs/omni_<tag>.log`。

**换校准集时必须先重算 act_scales**，否则 `main.py` 会直接报错（这是故意加的硬报错，不是 bug）：

```bash
MASQUANT_OMNI_CALIB_JSON=$MASQUANT_DATA_ROOT/jsonls/omnibench_mix0.5.jsonl \
  $PY generate_act_scale_shift.py --model $OMNI_MODEL --dataset-type text-audio-vision --nsamples 128
```

已有的 act_scales：`act_scales/Qwen2.5-Omni-3B-text-audio-vision-128.pt`（原校准）。
混合校准那份**还没生成**。

时间参考（单卡 B200）：FP16 全量 2676 条约 40 分钟（BS=2）；MAS 训练 2 epoch 约 5 分钟 + 评测 600 条约 10 分钟。

---

## 3. 三个必须知道的坑（都踩过，已修）

### 3.1 评测集不能带 system prompt（**影响 >30 个点，最重要**）

Omni 的默认 system prompt 是"你是 Qwen 虚拟人……"。带着它做 ASR，约 15% 的短句会被当成闲聊来回答：

```
REF: the egyptian background of the bondage
HYP: well you know the egyptian background of bondage is really complex there were a lot
     of factors at play for one thing the egyptians had a very hierarch...
```

FP16 的 WER 因此从 **4.78 虚高到 42.7**。`build_omni_data.py` 生成的**评测集已去掉 system prompt**，
校准集**保留**（那是 Omni 的标准对话格式，训练/激活统计都该按真实用法来）。
对照实验：`/tmp/probe_prompt.py` 的写法，A（带 system）/ B（不带）/ C（改成严格指令）三档，B 和 C 都正确。

### 3.2 批量评测会撞量化层的 2D mask（已修）

模型以 `flash_attention_2` 加载，但量化层 `int_qwen_omni_layer.py` 固定走 `forward_sdpa`，
批量生成时传进来的是 2D padding mask，原代码按 4D 索引 → `IndexError: too many indices for tensor of dimension 2`。
已在 `forward_sdpa` 里加 2D→4D 转换（含 left-padding 与因果，且每行至少能看见自己以免全掩码出 NaN）。
**batch=1 不经过该分支**，所以冒烟阶段可以先 `BS=1` 绕过诊断。

### 3.3 `--epochs` 必须显式给

`main.py` 里 `assert args.lwc or args.let` 只在 `epochs>0` 时触发，但 FP16 路径也要 `--epochs 0`，
否则会走到训练分支。`omni_run.sh` 已按 method 自动设好。

---

## 4. 初步结果里最值得追的一条：训练比闭式解差 5 点

MAS 闭式解 9.16 → 训练 2 epoch **14.21**（同一批 600 条）。这和 VL 侧 NVFP4 的观察
（"训练 12.14 vs 闭式解 11.75，机制未查清"）是同一现象，而音频上的差距大得多（5 点 vs 0.4）。
可能原因（都未验证）：

- 混合 loss 里音频/文本的权重 $\lambda_m$ 不匹配（代码默认 `--grad_info_path` 未传时用 0.5）；
- 128 个校准样本里音频样本只有约 1/3（`omnibench.jsonl` 三类轮换），音频那套尺度过拟合；
- 训练把音频尺度推向"校准集最优"而非"真实分布最优"——即 §6.2.3 那个"校准集模态失衡"问题的音频版。

**这一条如果做实，比单纯"补个音频"有价值得多**：它把改进 A（校准集模态平衡）的必要性
从"VL 上 PPL 差 1.5"升级成"音频上 WER 差 5 点"。

---

## 5. 下一步（按优先级，给新会话直接抄）

### P0：把主表跑全（约 4~6 小时机时）

```bash
# 1) 先补混合校准的 act_scales（约 5 分钟）
MASQUANT_OMNI_CALIB_JSON=$MASQUANT_DATA_ROOT/jsonls/omnibench_mix0.5.jsonl \
  $PY generate_act_scale_shift.py --model $OMNI_MODEL --dataset-type text-audio-vision --nsamples 128

# 2) 六个配置 × 全量 2676（每卡一个，4 卡并行）
bash scripts/omni_run.sh omni_sq       0 sq        &
bash scripts/omni_run.sh omni_rtn      1 rtn       &
bash scripts/omni_run.sh omni_masinit  2 mas_init  &
bash scripts/omni_run.sh omni_mas      3 mas       &
# 混合校准（同一批题、同一套训练超参）
MASQUANT_OMNI_CALIB_JSON=$MASQUANT_DATA_ROOT/jsonls/omnibench_mix0.5.jsonl \
  bash scripts/omni_run.sh omni_mas_mix 0 mas &
MASQUANT_OMNI_CALIB_JSON=$MASQUANT_DATA_ROOT/jsonls/omnibench_mix0.5.jsonl \
  bash scripts/omni_run.sh omni_masinit_mix 0 mas_init &
```

> ⚠️ SQ / RTN 已经崩到不可用，全量跑它们是**浪费机时**；它们只需要报一个"崩"（现有 SQ n=40、RTN n=316 已足够，见 `results/omni/`）。
> 真正要全量的是 FP16（已有）、MAS 闭式解、MAS 训练、**MAS + 混合校准**。

### P1：重复次数（结论是否成立的关键）

训练路径有非确定性（与 VL 侧相同，同卡同种子 PPL 可差 1.6）。**每个训练配置至少 3 次**，
报 mean ± std。评测是确定性的，不需要重复。

### P2：解释"训练比闭式解差"

- 统计混合校准下音频 token 的占比（改 `build_omni_data.py` 时顺手打印），验证"音频监督不足"这个假设；
- 用 `MASQUANT_OMNI_CALIB_JSON=libri_test_other_calib.jsonl`（**纯音频校准**，263 条）训练一组，
  看是否比三类轮换更好——这是最直接的因果检验；
- 若坐实，就把改进 A 从"VL 上的 PPL 改进"扩展成"跨模态的校准配比问题"。

### P3：GPTQ（改进 F）在音频上的作用

机理上应该有效：`quantize/gptq.py` 的 `--gptq_hsrc own` 按模态分 Hessian，Omni 有独立的 `audio_scale`，
听起来正好能用。**未验证**。若有效，就是"权重侧才是瓶颈"这一结论的跨模态证据。

### P4：实现限制（照实写进局限）

- Omni 只用了 **thinker**（`LMClass.py` 里 `enable_audio_output=False`），不测语音生成；
- 校准 128 样本、W4A8，未测 W4A6 / NVFP4；
- LibriSpeech test-other 只有一个测试集，且评测按说话人切分，与论文原设定（校准=测试同源）不同——
  **我们的切分更严**，这是优点，报告里要说明。

---

## 6. 代码改动清单（相对上一版，都已进 `gh_work`）

| 文件 | 改动 |
|---|---|
| `custom_dataset.py` | `text-audio-vision` 的校准 jsonl 支持 `MASQUANT_OMNI_CALIB_JSON` 覆盖 |
| `main.py` | ① `_omni_calib_tag()`：换了校准文件就把文件名带进 `act_scales` / dataloader 缓存的名字，避免串用；② 换了文件但 act_scales 不存在时**硬报错**（不做静默回退，回退会得到错误的初始化统计） |
| `generate_act_scale_shift.py` | 同上，缓存与 `act_scales` 文件名带 Omni 校准标签 |
| `models/int_qwen_omni_layer.py` | `forward_sdpa` 支持 2D padding mask（见 §3.2） |
| `scripts/omni_run.sh` | 新增：Omni 的量化 + 训练 + 评测一键入口 |
| `tools/build_omni_data.py` | 新增：构造校准 / 评测 jsonl |
| `tools/omni_wer.py` | 新增：WER（支持仓库内脱敏路径 `<DATA_ROOT>`） |
| `results/omni/` | 新增：初步结果的逐条转写（脱敏），可直接复算上表 |

`tools/msdl_omni.py`（下载器）在项目顶层 `tools/`，未进仓库。

---

## 7. 复现校验（接手先跑这两条，能过说明环境没坏）

```bash
# 1) 数据在不在
wc -l $MASQUANT_DATA_ROOT/jsonls/libri_test_other_eval.jsonl   # 2676
wc -l $MASQUANT_DATA_ROOT/jsonls/omnibench.jsonl               # 128

# 2) 小样本把 FP16 跑一遍（约 5 分钟），WER 应接近 5%（n=40 时约 14%，样本太小，别慌）
EVAL_N=60 PPL_SETS= bash scripts/omni_run.sh check_fp16 0 fp16
```
