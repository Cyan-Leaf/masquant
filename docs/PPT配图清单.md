# PPT 配图清单（8 分钟汇报用）

> 给做 PPT 的人看：**哪张图能上台、哪张不能、每张图讲什么**。
> 配套文档：[`汇报文稿_8分钟.md`](汇报文稿_8分钟.md) §2.0 视觉清单、[`书面报告.md`](书面报告.md)。
> 最后更新：2026-11-26。

---

## 0. 一句话总则

**P4 / P5 / P7 只念 4×B200 的数字；音频只说"另一个模型上也成立"，不报跨模型的差。**

汇报只用一套口径（简称**口径 A**）：**4×B200，ScienceQA 2000 题 + TextVQA 2000 题（`doc_id<2000`）**。
仓库里还存着早期 8×RTX 机器、SQA 2017 题的口径（**口径 B**，书面报告 §3 用的就是它），
两套的绝对数差 0.2~0.5 点。**同一页里混用会被当场发现**（对着口播的 80.8 看到图上 80.6）。

---

## 1. 上台用的图（9 页里 6 页有图）

| 页 | 图 | 口径 | 图上写的字 | 组员要注意 |
|---|---|---|---|---|
| P1 | 无 | — | `瓶颈不在「分模态」` | 纯文字，结论前置 |
| **P2** | `fig7_why_one_scale.png` | 模型激活统计（跨机器差 5e-5，可视为与机器无关） | `离群通道不是同一批 → 一套缩放迁就不了两边` | 左栏两模态的峰不重合（本层 top-32 只重叠 19 个）；右栏逐通道 0.35×~4.2×，中位 1.53× |
| P3 | 无（建议手绘两栏：MAS / CMC） | — | `两招：分模态平滑 · 低秩补偿` | 右下角小红框写"定理 1 分子分母写反" |
| **P4** | **`fig13_bench_b200.png`**（新） | 口径 A | `复现成功，但没追平 FP16` | **不要用 `fig2_main_bench.png`**，见 §3 |
| **P5** | `fig8_three_ties.png` | a、c 面板 B200 / Omni；b 面板 B200 C4 PPL | `三处测量，同一个结论` | 三栏都打平：下游 63.5 vs 64.4、C4 PPL 18.9 vs 19.2（不显著）、音频 6.58 vs 6.58 |
| **P6** | `fig12_calib_b200.png`（主）+ `fig10_audio_calib.png`（右下小图） | 口径 A / Omni | `要的是多样性，不是某一种模态的比例` | 大字只留 `8% → 一半`；**不要用 `fig1`**，见 §3 |
| **P7** | `fig11_two_knobs.png` → **`fig14_solver_textvqa.png`**（新）→ `fig6_calib_gptq.png` | 示意图 / 口径 A / 口径 A | `两个旋钮，+7.5 点` | 见 §2.2 的归因规矩 |
| P8 | 无（三行清单，每行一个数字） | — | `其余三条` | 一秒钟都不展开 |
| P9 | 无（大字 + 一句部署建议） | — | `一份权重、零旁路` | 局限只说一句：只有 3B 这一档 |

---

## 2. 本轮新增 / 改动的图

### 2.1 `fig13_bench_b200.png`（新，P4 用）

复现结果，口径 A，带 95% bootstrap CI：

| | ScienceQA (2000) | TextVQA (2000) |
|---|---|---|
| FP16 | 80.8 | 75.8 |
| SmoothQuant + RTN | 75.2 | 61.1 |
| MAS（每个模态一套） | 78.6 | 63.5 |
| 统一 scale（一套） | 78.7 | 64.4 |

图上直接标两句：**MAS 比 SQ 高 +3.4 / +2.4（都显著）**、**但比 FP16 低 2.2 / 12.4（都显著）**。

### 2.2 `fig14_solver_textvqa.png`（新，P7 主结果图）

**同一个 MAS checkpoint，只换权重求解器**（TextVQA，`doc_id<2000`）：

| 配置 | TextVQA |
|---|---|
| MAS + RTN | 63.5 |
| MAS + GPTQ | 70.9（**+7.4**） |
| MAS + GPTQ + clip | 71.0（clip 只 **+0.1**） |
| 统一 scale + RTN | 64.4 |
| 统一 scale + GPTQ + clip | 73.5 |
| FP16（参考线） | 75.8 |

图上的归因箭头写着 `GPTQ +7.4` / `clip +0.1`。**这是有意为之**：讲的时候必须明说
"+7.5 里 GPTQ 占 7.4、clip 占 0.1，坐标下降（cd）默认关闭、下游无一致收益"，
否则听众会顺着"RTN → GPTQ → +cd"的台阶把功劳算给 cd。
（5000 题口径做同一实验是 +7.2 / +0.5，结论一样；**别把两个口径的数混着念**。）

### 2.3 `fig8_three_ties.png` 面板 (b) 换口径

原来面板 (b) 用的是另一台机器上 0.75 比例的配对（16.19 vs 16.20，p=0.97）。已换成 **B200 的 C4 PPL**：
MAS 18.9 ± 0.5（14 个 checkpoint）vs 统一 scale 19.2 ± 0.7（12 个），**不显著（p=0.20）**。

### 2.4 上一轮已加的图（仍然有效）

| 图 | 用途 | 一句话 |
|---|---|---|
| `fig7_why_one_scale.png` | P2 | 真实激活统计：一套缩放迁就不了两个模态 |
| `fig11_two_knobs.png` | P7 骨架 | 量化只有两个旋钮（步长 / 码字），替代原稿里的 ASCII 图 |
| `fig12_calib_b200.png` | P6 主图 | 原校准 → 混合校准：19.1→17.4、20.4→18.9（B200 口径） |
| `fig10_audio_calib.png` | P6 小图 | 纯音频校准（65% 音频 token）反而更差更抖 |
| `fig9_audio_ladder.png` | 备用 | 音频全表阶梯，SQ 301% / RTN 114% 崩在画外 |

---

## 3. 不能上台的图

| 图 | 为什么 | 用在哪 |
|---|---|---|
| ~~`fig2_main_bench.png`~~ | **8×RTX 机器、SQA 2017 题**（口径 B），与 P4 口播的 80.8 对不上 | 书面报告 §3 正文 |
| ~~`fig1_calib_ratio.png`~~ | 同上机器，且画的是 7 个比例的扫描曲线（wikitext 16.2 那个平台） | 报告插图；被问"比例怎么选"时翻出来 |
| ~~`fig3_breakdown.png`~~ | SQA 2017 口径的题型拆分 | 备用（问"损失集中在哪类题"） |
| ~~`fig4_sqnr.png`~~ | 单层 SQNR 代理指标 | 备用（问"为什么单层有优势端到端没有"） |
| `fig5_solver_ppl.png` | B200 没问题，但它讲的是"换求解器不改变平滑方式的排序"，是收口旁证不是主图 | P7 备用（问"是不是只在 TextVQA 上有效"） |

---

## 4. 怎么重新生成

全部来自已落盘的数字，**不重跑实验**：

```bash
# 口径 A 的两张（P4、P7）
python tools/make_slides_figs_b200.py --outdir docs/figs

# 其余（P2、P5、P6、P7 骨架、音频）
python tools/make_slides_figs.py --src results/figs_src --outdir docs/figs
```

数据来源：

| 图 | 数据文件 |
|---|---|
| fig13 | `results/bench_b200/boot_b200_4.json` |
| fig14 | `results/gptq/boot_solver_2000.json` + `results/gptq/bench/mas_gptq/samples_textvqa_val.jsonl` |
| fig8 | 上 + `results/repro_b200/ppl_b200_summary.json`（C4）+ `results/omni/` |
| fig12 | `results/repro_b200/ppl_b200_summary.json` |
| fig7 | `results/figs_src/act_scale_stats.npz`（由 `tools/extract_fig_src.py` 从 act_scales 抽取） |
| fig9 / fig10 | `results/omni/*.jsonl.gz` |

服务器没有中文字体，**所有图上都是英文**，中文标题请在 PPT 里另外加。

---

## 5. 上台前自查（配图部分）

- [ ] P4 用的是 `fig13`，不是 `fig2`；台上没有表格
- [ ] 每页只有一张主图 + 一行 ≤15 字的字
- [ ] P5 三个面板都讲"打平"，不念 p 值（p 值只在被追问时给）
- [ ] P7 明说 `+7.5 = GPTQ 7.4 + clip 0.1`，cd 默认关
- [ ] 音频的数字只在"另一个模型也一样"的语境里出现，不与 VL 的数字并排比较
- [ ] 图在投影分辨率下能看清（都是 160 dpi 导出，最小字号 8 pt）
