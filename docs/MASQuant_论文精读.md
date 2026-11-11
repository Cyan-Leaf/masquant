# MASQuant 论文精读与代码对照

> 面向《多模态学习》课程汇报。目标：讲清楚 **为什么传统 LLM 量化方法到了多模态模型上会崩**，以及 MASQuant 用 **MAS + CMC** 两招怎么修。
>
> 论文：*MASQuant: Modality-Aware Smoothing Quantization for Multimodal Large Language Models*（阿里云，arXiv 2603.04800）

---

## 0. 一页速览（讲给同学听的版本）

| 问题 | 传统做法 | 在多模态上为什么不行 |
|---|---|---|
| 激活有离群值（outlier） | SmoothQuant：用一个对角矩阵 $S$ 把激活的离群值搬到权重上 | 多模态的激活量级差 10–100 倍，一个 $S$ 只能迁就"最强势"的那个模态 |
| 想给每个模态各用一个 $S_m$ | 数学上完全合法（逐通道缩放是恒等变换） | 但推理时要存 **3 份**量化权重 $Q(S_mW)$，显存直接 ×3，量化的意义就没了 |

MASQuant 的两个贡献正好对应这两行：

1. **MAS（Modality-Aware Smoothing）**：给每个模态学一个独立的平滑因子 $S_m$ → 消除"平滑错位"，把逐通道平滑推到它理论上限。
2. **CMC（Cross-Modal Compensation）**：只存**一份**量化权重 $Q(S_tW)$（用 text 当基准），其它模态用**低秩修正** $\Delta W \approx L_1L_2$ 补回来。关键洞察是：**白化之后**模态间权重差是低秩的。

一句话总结：**"分模态平滑拿精度，低秩补偿还我单份权重。"**

---

## 1. 背景：PTQ 与"计算不变性"

### 1.1 均匀仿射量化

把浮点张量 $x$ 映射到 $N$ 位整数：

$$
\hat{x}_N = Q(x) = \left(\text{clamp}\left(\left\lfloor \frac{x}{\Delta} \right\rceil + z,\ q_{\min},\ q_{\max}\right) - z\right)\cdot \Delta
$$

- $\Delta$：量化步长（scale），$z$：零点（zero-point），$\lfloor\cdot\rceil$：四舍五入
- 对称量化时 $z=0$，$q \in [-2^{N-1}, 2^{N-1}-1]$

**为什么 LLM 的激活难量化？** 激活里有极少数"离群通道"，数值比其它通道大几十倍。per-tensor 量化时 $\Delta$ 被离群值撑大，正常通道全被压成 0；per-token 量化能缓解，但通道内的离群仍然存在，权重侧没有对应的离群时误差主要来自激活。

### 1.2 SmoothQuant：用等价变换把激活的锅甩给权重

对线性层 $Y = XW$（$X\in\mathbb{R}^{T\times D_{in}}$，$W\in\mathbb{R}^{D_{in}\times D_{out}}$），插入对角矩阵 $S=\text{diag}(s)$：

$$
Y = (XS^{-1})(SW)
$$

数学上恒等 → **计算不变性（computational invariance）**。但量化后不等价了：

$$
\mathcal{L} = \left\| Q(XS^{-1})\cdot Q(SW) - XW \right\|^2
$$

$s_i$ 大 → 第 $i$ 个输入通道被压小（激活好量化），但权重第 $i$ 行被放大（权重难量化）。$s_i$ 就是在两者之间做**重新分配**。经典闭式解（SmoothQuant，$\beta\in[0,1]$ 控制迁移强度）：

$$
s_i = \frac{\max_t |x_{t,i}|^{\beta}}{\max_j |w_{j,i}|^{1-\beta}}
$$

这里的两个 `max` 是**关键伏笔**：
- $\max_t |x_{t,i}|$ 是在**所有 token** 上取的最大值 —— 混模态校准数据里，这个 max 会被**量级最大的模态**支配。

> 💡 顺带说明：代码里用的是 $\beta=0.5$（见 §4.2 的初始化公式），即"两边各承担一半"的几何平均。

---

## 2. 核心问题一：Smoothing Misalignment（平滑错位）

### 2.1 现象

设模态集合 $\mathcal{M}=\{\text{text}, \text{vision}, \text{audio}\}$，逐通道激活范围

$$
R_i^m = \max_t |x^m_{t,i}|
$$

视觉 token 的 $R_i^{\text{vision}}$ 通常比文本大 10–100 倍（多模态模型里视觉 token 数量多、经过视觉编码器后幅值大）。统一校准时：

$$
s_i^{\text{uni}} = \frac{\left(\max_{m,t}|x^m_{t,i}|\right)^\beta}{\left(\max_j|w_{j,i}|\right)^{1-\beta}} \propto R_i^{m^*},\qquad m^*=\arg\max_m R_i^m
$$

也就是 $s^{\text{uni}}$ 完全跟着"强势模态" $m^*$ 走。对弱势模态 $m'$ 来说，它被一个**为别人量身定做的尺度**除，信号被过度压缩 —— 论文称 **smoothing misalignment**。

> 类比：给三个身高差 100 倍的人买同一件衣服。衣服按最高的人裁，矮的人穿上就只剩一坨布。

### 2.2 量化：错位到底损失多少 dB？（Theorem 1 的推导）

**Step 1：单 token 的 SQNR。**
per-token 对称量化，步长

$$
\Delta_t = \frac{\max_i|x_{t,i}/s_i|}{2^{b-1}-1}
$$

舍入误差近似在 $[-\Delta_t/2, \Delta_t/2]$ 均匀分布，$d$ 个通道的总均方误差 $= d\cdot \Delta_t^2/12$。信号功率 $=\sum_i (x_{t,i}/s_i)^2$。于是

$$
\text{SQNR} = 10\log_{10}\frac{\sum_i (x_{t,i}/s_i)^2}{d\Delta_t^2/12}
\;\propto\; 10\log_{10}\frac{\sum_i (x_{t,i}/s_i)^2}{\left(\max_i |x_{t,i}/s_i|\right)^2}
$$

这就是论文 Eq.(16)。

**Step 2：最优平滑的上限。**
取 $s_i = R_i$（每通道都刚好把范围归一化到 1），若 token 的激活分布形状与范围形状一致（$|x_{t,i}|/R_i \approx c$），则

$$
\text{SQNR}(s^{\text{opt}}, x_t) = 10\log_{10} d
$$

**这是一个非常漂亮的结论：SQNR 上限只取决于维度 $d$，与 bit 数无关。**（bit 数进的是被约掉的常数项，影响所有方案相同，所以比较相对值时无所谓。）

**Step 3：错位的损失。**
令 $\alpha_i = R_i^m / R_i^{m'}$（强势/弱势的范围比）。弱势模态用 $s^{\text{uni}}\propto R^m$ 时，$x_{t,i}/s_i^{\text{uni}} \propto (x_i/R_i)/\alpha_i$，因此

$$
\text{SQNR}(s^{\text{uni}}, x_t) = 10\log_{10}\frac{\sum_i \left(\tfrac{1}{\alpha_i}\right)^2}{\left(\max_i \tfrac{1}{\alpha_i}\right)^2}
= 10\log_{10}\frac{d}{\left(\min_i \alpha_i\right)^2}\cdot\frac{1}{d}\sum_i\frac{1}{\alpha_i^2}
$$

两者相减：

$$
\boxed{\;\Delta = \text{SQNR}(s^{\text{uni}}) - \text{SQNR}(s^{\text{opt}})
= 10\log_{10}\left[\frac{\left(\min_i \alpha_i\right)^2}{d}\sum_{i=1}^{d}\frac{1}{\alpha_i^2}\right] \le 0\;}
$$

等号成立 **当且仅当** $\alpha_1=\alpha_2=\cdots=\alpha_d$。

> ⚠️ **论文 Eq.(17) 的分子分母写反了**（把 $\min_i \alpha_i^2$ 放在分母）。检验一下：若所有 $\alpha_i$ 都相等（比如全等于 2），错位损失应该为 0，但按 Eq.(17) 会算出 $d\cdot\alpha^2\cdot d/\alpha^2$ 这种荒谬值。Eq.(20) 的推导过程是对的，结论就是上面这个方框。汇报时点出这个 typo 会很加分。

**Step 4：算几个数感受一下**（$d$ 很大时）

| $\alpha_i$ 分布 | $\Delta$ (dB) |
|---|---|
| 全部相等 $\alpha\equiv 2$ | 0（无损失） |
| $d=4$: $[1,1,1,100]$ | $-1.25$ |
| $d=4$: $[1,1,100,100]$ | $-3.0$ |
| 只有 1 个通道 $\alpha=1$，其余 $\alpha=100$（$d=2048$） | $\approx -33$ |

**结论：损失由 $\alpha$ 的*非均匀性*决定。** 只要各通道的模态间比例不一样，就一定掉点；而当某些通道视觉比文本大 100 倍、另一些通道几乎一样大时（论文 Fig.4c/4d 实测正是这样），损失会非常可观。

### 2.3 与代码的对应

- 激活范围统计：`generate_act_scale_shift.py:87-123`，`stat_tensor` 里对每个模态分别取 `torch.max`，存成 `{name}.text_scale / .vision_scale / .audio_scale / .all_in_one_scale`
- 论文 Fig.2（unified vs optimal 的逐层 SQNR）：`data_analysis/show_sqnr_diff.py`
- 论文 Fig.4c/4d（$\alpha_i$ 分布）：`data_analysis/show_act_scale.py`、`show_grid_info.py`

---

## 3. 核心问题二：Cross-Modal Computational Invariance（跨模态计算不变性）

### 3.1 两难

想让每个模态用自己的 $S_m$，推理时就得算

$$
Y = Q(X_mS_m^{-1})\cdot Q(S_mW)
$$

注意 $Q(S_mW)$ **随 $m$ 变化** —— 需要为每个模态存一份量化权重。三模态就 3 份，4-bit 的省显存意义全没了。

**论文的问题陈述：能否既拿分模态平滑的精度，又只存一份权重？**

### 3.2 CMC 的思路

挑 **text** 当基准模态，只存 $Q(S_tW)$。对视觉输入，理想计算是 $X_vS_v^{-1}\cdot(S_vW)$，实际只能用共享权重，于是产生残差：

$$
\Delta Y = X_vS_v^{-1}\cdot\underbrace{\left(S_vW - Q(S_tW)\right)}_{\Delta W}
$$

**朴素做法**：对 $\Delta W$ 直接做截断 SVD，取 $L_1L_2\approx\Delta W$。
**为什么不 work**：SVD 最小化的是 $\|\Delta W - L\|_F$（权重空间的欧氏误差），而我们真正关心的是**输出空间**的误差 $\|X_vS_v^{-1}(\Delta W-L)\|_F$。而且 $\Delta W$ 本身没有低秩结构（它是两个对角矩阵的差，即 $W$ 的行被不同尺度缩放，理论上可以是满秩的）。

### 3.3 白化把"满秩"变成"低秩"

**关键洞察**：先把输入"白化"（whiten），$\Delta W$ 在白化坐标下才显现低秩。

设视觉激活协方差

$$
\left(X_vS_v^{-1}\right)^\top\left(X_vS_v^{-1}\right) = P\Lambda P^\top,\qquad T = \left(P\Lambda^{1/2}\right)^\top
$$

则 $X_vS_v^{-1}T^{-1}$ 是**正交**的（各向同性）。在 $T$ 坐标下做截断 SVD：

$$
\text{SVD}(T\Delta W) = U\Sigma V^\top \approx U_r\Sigma_r V_r^\top
$$

反变换回去：

$$
\Delta W \approx L_1L_2,\qquad L_1 = T^{-1}U_r,\quad L_2 = \Sigma_r V_r^\top
$$

论文 Fig.5 用有效秩（effective rank）实测：加白化后 $\Delta W$ 的有效秩明显下降。**直觉**：白化把"激活实际能打到的方向"压缩到少数几个主成分上，输出的残差自然只在这几个方向上非零。

### 3.4 Theorem 2：白化+截断 SVD 是最优的

**命题**：$L^* = T^{-1}\text{Trunc}_r(T\Delta W)$ 最小化

$$
\mathcal{L}(L) = \left\|X_vS_v^{-1}(\Delta W - L)\right\|_F^2
$$

**证明（一行看懂）**：由 $X_vS_v^{-1} = U\Lambda^{1/2}P^\top = UT$（$U$ 正交），

$$
\left\|X_vS_v^{-1}(\Delta W-L)\right\|_F^2 = \left\|UT(\Delta W-L)\right\|_F^2 = \left\|T\Delta W - TL\right\|_F^2
$$

正交变换不改变 Frobenius 范数，所以问题等价于"在 $T$ 坐标下做最优低秩近似"——Eckart–Yuung 定理告诉我们截断 SVD 就是最优解。∎

> 这是这篇论文最漂亮的一步：**把一个加权的最小二乘低秩逼近问题，用白化变成了标准的低秩逼近问题。**

### 3.5 最终推理公式

$$
Y=\begin{cases}
Q(X_mS_m^{-1})\cdot Q(S_tW), & m=\text{text}\\[4pt]
Q(X_mS_m^{-1})\cdot Q(S_tW) + X_mS_m^{-1}\cdot L_1^mL_2^m, & m\neq\text{text}
\end{cases}
$$

**代价分析**（论文 Table 6）：text 走纯量化 GEMM；其它模态多一个低秩旁路，计算 $d+2rd$、显存 $2mrd$。因为 **decode 阶段只有文本 token**（自回归生成），所以低秩分支只在 prefill 触发 —— 这就是论文说的 "仅影响 prefill，额外 FLOPs ~2%"，也是**为什么选 text 当基准模态**。

---

## 4. MAS：把平滑因子当成可学习参数

### 4.1 从"搜 $\beta$"到"学 $S$"

SmoothQuant / AWQ / MBQ 都只调一个标量 $\beta$：

$$
s_i = \frac{\max_t|x_{t,i}|^\beta}{\max_j|w_{j,i}|^{1-\beta}}
\quad\text{或}\quad
s_i = \left(\frac{1}{n}\sum_t |x_{t,i}|\right)^{\beta}
$$

$S$ 的**形状**被这个公式锁死了。OmniQuant 指出：直接把 $S$ 当自由参数优化，能突破 $\beta$ 参数族的表达能力上限。MASQuant 把这套搬到 MLLM 上，并按模态拆开：

$$
\{S_m^*\} = \arg\min_{\{S_m\}}\ \sum_{m\in\mathcal{M}} \lambda_m\cdot\mathcal{L}_{\text{MAE}}(S_m, X^m, W)
$$

$$
\mathcal{L}_{\text{MAE}} = \left\| Q(X^mS_m^{-1})\cdot Q(S_mW) - X^mW \right\|_1
$$

用 L1（MAE）而不是 MSE：**对离群值不敏感**，符合"我们本来就要治离群"的动机。

### 4.2 初始化（代码实测细节）

$$
s_i^m = \sqrt{\frac{\max_t|x^m_{t,i}|}{\max_j|w_{j,i}|}}
$$

即 $\beta=0.5$ 的 SmoothQuant 解，但 **max 只在模态 $m$ 自己的 token 上取**。这是"消除支配"的第一步。

对应代码 `quantize/masquant.py:411-427`（`alpha` 默认 0.5，`--alpha` 可改）。视觉还有个可选的 $\alpha$ 网格搜索（`masquant.py:431-472`）：$\alpha_v \in [\alpha, 0.8]$ 步长 0.01，用 SQNR 选最优 —— 因为视觉激活范围大，需要比 text 更小的 $\alpha$ 来加强平滑。

### 4.3 训练目标（论文没写清、代码才是真相）

论文说 $\lambda_m$ 是"模态损失权重，本文取 $\lambda_t=\lambda_v=1.0$"。但**代码里其实是 MBQ 式的梯度均衡**（`masquant.py:572-604`）：

```python
loss = (text_diff.sum() + audio_grad*audio_diff.sum() + vision_grad*vision_diff.sum()) / total_mask
```

其中 `vision_grad = 0.126`（Qwen2.5-VL-3B 上通过 MBQ 方法采到的 `vis_avg_grad/cap_avg_grad`），audio 用 `grad_info` 里的实际梯度比。注释里写得很直白：*"针对 qwen-omni 模型，这个 0.5 是瞎猜的"*。

**汇报时可以点出：论文的 Table 4（$\lambda$ 消融）和代码里实际跑的配置不完全一致** —— 这是复现工作最容易踩的坑，也是很好的"批判性阅读"素材。

### 4.4 其它实现细节（读代码才知道的）

| 细节 | 代码位置 | 说明 |
|---|---|---|
| k/v 复用 q 的 scale，gate 复用 up 的 scale | `quantize/utils.py:123-155` | `merged_scales` 模式下直接共享参数对象；`split_scales` 模式下初始化时复制但独立更新 |
| 平滑因子截断 | `utils.py:111-121`，阈值 `1e-2` | 避免 AMP 下除零溢出 |
| 三个模态各跑一次 GEMM 再相加 | `int_linear.py:213-221` | `out = out_text + out_vision + out_audio`，配合 mask 实现（**注意：这在真实部署里是 3× 计算量，是论文没讨论的实现代价**） |
| decode 阶段退化 | `int_linear.py:126-131` | `input.shape[1]==1` 时全部当 text，因为 KV cache 里只有文本 token |
| `down_proj` 不训练 | `svd_utils.py:61-63` | scale 固定为单位阵（OmniQuant 传统：down 的输入是 SwiGLU 输出，不可控） |

---

## 5. 实验解读

### 5.1 主结果（Qwen2.5-VL，Table 1）

最值得讲的三个数：

| 配置 | RTN | SmoothQuant | MBQ | MASQuant | FP16 |
|---|---|---|---|---|---|
| W8A8 Avg | 69.8 | 69.9 | 70.1 | **70.9** | 70.1 |
| W4A8 Avg | **5.1** | 57.4 | 64.6 | 64.7 | 70.1 |
| W4A6 Avg | 53.6 | 59.7 | **59.7** | **59.9** | 70.1 |

- **W8A8 基本无损**（70.9 vs FP16 70.1，甚至略高，属于噪声）
- **W4A8 下 RTN 完全崩溃**（Avg 5.1，OCR/Viz/S-QA/T-VQA 全 0）—— 说明 4-bit 激活时离群值必须处理
- **W4A8 下 SmoothQuant 也崩**（57.4），而 MASQuant 64.7，只比 W4A16 的 MBQ(69.3) 低一点

### 5.2 三模态更惨烈（Qwen2.5-Omni，Table 2）

W4A8 上 SmoothQuant 的 ASR 性能**灾难性崩溃**：

| | Librispeech WER↓ | Wenetspeech WER↓ | Omnibench↑ |
|---|---|---|---|
| FP16 | 3.9 | 7.5 | 43.8 |
| SQ W4A8 | **77.4** | **94.2** | 27.3 |
| MBQ W4A8 | 9.5 | 8.5 | 36.7 |
| MASQuant W4A8 | **3.6** | 8.7 | **41.4** |

WER 从 3.9 到 77.4 是 20 倍劣化 —— **语音几乎被完全摧毁**。原因很直观：音频的激活量级在三模态里最小，是"平滑资源竞争"里第一个牺牲品。

> 这是论文最有说服力的一组实验：它证明"平滑错位"不是精度掉几个点的小问题，而是**弱势模态整体失效**的结构性问题。

### 5.3 消融（Table 1+2 汇总解读）

**Table 3（MAS 的必要性，W4A8，Omni-3B）**

| Method | Opt. | Libri | OmniB | T-VQA | MMMU |
|---|---|---|---|---|---|
| Uniform | | 77.4 | 27.3 | 51.3 | 25.6 |
| MAS | | 3.8 | 36.7 | 54.7 | 28.9 |
| Uniform | ✓ | 6.0 | 33.3 | 65.0 | 33.3 |
| MAS | ✓ | **3.6** | **47.7** | **68.2** | **46.7** |

两个洞察：
1. **MAS 单独就救回了 ASR**（77.4 → 3.8 WER），说明错位是主因
2. **可学习优化会*放大*而不是缩小差距**（33.3 → 47.7 vs 6.0 → 3.6）—— 因为统一 $S$ 的优化只能迁就一个模态，而分模态 $S_m$ 每个都能优化到自己的最优。这其实印证了 Theorem 1 的"非均匀 $\alpha$"结论。

**Table 4（$\lambda$ 消融）**：$\lambda_t=\lambda_v=1.0$ 最好（PPL 17.2，Avg 56.9）。降 $\lambda_v$ 到 0.1 时 MMMU 反而降、T-VQA 反而升 —— 说明存在微妙的 trade-off，但整体最优是"平等对待"。

**Table 5（epoch 消融）**：2 epoch 是甜点（Avg 61.2），再多 PPL 更好但下游反而降（过拟合到校准集）。代码默认 `--epochs 2` 与此一致。

**Fig.6（CMC vs 无白化）**：低秩比例 0.08 时 CMC 就超过 MAS 单模（3.5 SQNR 饱和），而无白化基线要 0.4 才追上 MBQ —— **白化让所需秩降低约 4×**，这正是 Theorem 2 的直接体现。

### 5.4 推理加速（Table 7）

RTX 4090，Qwen2.5-VL-7B，W4A4，seq=2048：

| BS | FP16 | MBQ | MAS(r=0.01) | MAS(r=0.05) |
|---|---|---|---|---|
| 1 | 191.8ms | 68.7ms (2.79×) | 71.6ms (2.67×) | 77.1ms (2.49×) |
| 8 | 2146ms | 643.7ms (3.33×) | 649.3ms (3.30×) | 696.4ms (3.07×) |

低秩分支带来 5–10% 的额外 prefill 延迟，decode 完全相同。用 Nunchaku 风格的融合 kernel 实现。

---

## 6. 局限与可改进方向（我自己的工作）

### 6.1 论文的漏洞 / 可以做文章的地方

| # | 问题 | 可能的改进 |
|---|---|---|
| 1 | **三模态 GEMM 三倍算力**：`int_linear.py` 里 text/vision/audio 各算一遍再相加，实际部署时 prefill 是 3× GEMM | 用 gather + 单 GEMM 实现（按 token 索引取对应 scale），或按 token block 重排 |
| 2 | **只治激活，权重侧没动**：$S_m$ 只优化激活，权重 4-bit 的误差没专门处理 | 加 weight clipping（LWC）或 GPTQ 式权重重建 |
| 3 | **$\lambda_m$ 手调/借 MBQ 梯度**：代码里 `vision_grad=0.126` 是硬编码常量，论文 Table 4 说"取 1.0" | 把 $\lambda_m$ 也做成可学习（梯度范数归一化的拉格朗日乘子） |
| 4 | **rank 是全局超参**：`--rank 0.05` 对所有层一样 | 按层自适应分配（用有效秩或 SQNR 增益做贪心分配），给定总预算下最优 |
| 5 | **基准模态固定为 text**：若某任务视觉/语音占主导，text 基准会吃亏 | 动态选择或加权组合基准：$Q(\bar S W)$，$\bar S$ 为各模态的几何平均 |
| 6 | **CMC 补偿的是浮点权重**：`quant_cmc=0` 时 $Q(S_vW)$ 用的是未量化的 $S_vW$，隐含假设理想权重可及 | 对比 `quant_cmc=1`（补偿到量化权重）的差异，分析哪种更一致 |
| 7 | **没有视频模态**：Omni 支持视频但论文只评了 image+audio+text | 加 video token mask，做 4 模态实验（见 §6.2） |
| 8 | **只有 INT 量化**：symmetric INT4/INT6/INT8 | 拓展到 **NVFP4**（E2M1 + FP8 块缩放），见 §6.3 |

### 6.2 拓展 A：视频模态（4 模态 MAS）

**动机**：视频 = 大量帧 × 每帧视觉 token，激活量级与图像不同（时间冗余导致相邻帧相似），且 Qwen2.5-VL/Omni 都有 `video_token_id`。

**做法**：
1. `generate_act_scale_shift.py` 的 `stat_tensor` 增加 `video_mask` → 存 `video_scale`
2. `QuantLinear` 增加 `video_smooth_scale` 参数、`forward_mas_*` 增加第 4 路
3. CMC 增加 video 的低秩分支（白化矩阵用视频帧激活统计）
4. 评测：Video-MME / MVBench + 原有一切任务

**预期发现**：视频 token 数量极大（可能上千 token），会进一步加剧"多数模态支配"，MAS 的收益应比 image 更明显。

### 6.3 拓展 B：NVFP4

**背景**：NVFP4 = E2M1（4-bit 浮点：1 sign + 2 exp + 1 mantissa）+ 每 16 个值共享一个 FP8(E4M3) 缩放 + 每张量一个 FP32 全局缩放。相比 INT4：
- 动态范围大得多（$2^{-6}$ 到 $6$），天然抗离群
- 但**只有 1 位尾数**，有效精度只有 $\{0, \pm0.5, \pm1, \pm1.5, \pm2, \pm3, \pm4, \pm6\}$ 这几档

**关键研究问题**：MAS 的收益在 FP4 下还成立吗？
- 假设 1：FP4 的动态范围已经吸收了部分离群 → 平滑带来的收益被稀释
- 假设 2：FP4 的**尾数精度极低**，SmoothQuant 式的"重新分配"仍然重要，因为 $\Delta$（block scale）的选择对 1-bit 尾数的影响被放大
- **这是很好的 neg/pos result 双赢选题**：无论哪个方向都是结论

**做法**：写一个 `NVFP4Quantizer`（复用 `UniformAffineQuantizer` 接口），把 `n_bits` 换成分块浮点量化，跑同样的 MAS/CMC 流程对比。

### 6.4 拓展 C：方法改进（我最想做的两个）

**(a) 自适应 rank 分配**
现在 `_low_rank` 里 `rank = int(sigma.size(0) * rank_ratio)`，所有层同一个比例，且 `if rank < 32: rank = 32` 硬下限。改成：在总参数预算 $\sum_l r_l \le B$ 下，用贪心 / 拉格朗日分配秩 —— 每层增加 1 个秩带来的 SQNR 增益 $\partial\text{SQNR}/\partial r$ 不同（由奇异值衰减速率决定），应按增益排序分配。

**(b) 用真实量化权重做基准（量化感知的基准模态选择）**
目前 $Q(S_tW)$ 是固定 text。可以定义基准模态的平滑因子为 $\bar s_i$，最小化所有模态的**加权**重建误差。这会变成一个小的优化问题，可能比"选 text"更好。

---

## 7. 复现要点（踩坑清单）

| 坑 | 说明 |
|---|---|
| `custom_dataset.py` 硬编码 `/nas/yuehu/...` | 音频 `libri_test_other.jsonl` 和图片路径都是作者的绝对路径，必须改写 |
| `data/jsonls/omnibench.jsonl` 不在仓库里 | `text-audio-vision` 校准集需要自己准备 |
| `act_scales` 路径 | 默认 `./act_scales/{net}-{dataset_type}-{nsamples}.pt`，`net` 来自模型路径最后一段，名字要对上 |
| `inference_mode` 环境变量 | 训练必须 `export inference_mode=split_scales`，否则走 `merged_scales` 分支，MAS 白训 |
| `net` 参数 | `infer_mas.py` 里 `net_choices` 只认 `qwen2.5-omni-3b/7b`、`qwen2.5-vl-3b/7b` 四个字符串 |
| transformers 版本 | 仓库基于 4.52，新版（4.57）的 `check_torch_load_is_safe` 等 API 有变化 |
| `--limit_multimodal` | 默认 1.0 跑全量，调试时设 0.05 |
| PPL 评测走 `llm.model.model(batch)` | 对 VL 模型是 `language_model`，对 Omni 是 `model`，分支写死在 `main.py:397` |

---

## 8. 汇报时间分配建议（15 分钟）

| 时间 | 内容 |
|---|---|
| 2 min | 背景：PTQ 一句话 + SmoothQuant 的 $Y=(XS^{-1})(SW)$ |
| 3 min | 问题：多模态激活量级差 10–100×，一个 $S$ 只能迁就强势模态 → SmoothQuant 在 Omni 上 WER 3.9→77.4（这个数字最抓人） |
| 2 min | Theorem 1：一个 token 的 SQNR 上限是 $10\log d$，错位的损失由 $\alpha_i$ 的非均匀性决定（讲结论 + 一个数例即可，不必推完） |
| 3 min | MAS：初始化 + 可学习优化；CMA：为什么必须低秩、为什么白化能带来低秩、Theorem 2 的一行证明 |
| 2 min | 结果：W4A8 下 MASQuant vs SQ vs MBQ；Table 3 的"优化反而拉大差距" |
| 3 min | 我们的改进实验（视频模态 / NVFP4 / 自适应 rank） |

---

## 9. 我们的复现与改进结论（汇报末尾 3–5 分钟）

### 9.1 复现是否成功？——是，且均值几乎一致

| 指标 | 论文 | 我们（3 次重复） |
|---|---|---|
| W4A8 PPL (epoch=2) | 18.6 | **18.46 ± 1.43** |
| W4A8 下 SQ vs MAS 排序 | MAS 优 | MAS 优（22.63 vs 19.43 仅初始化；训练后 18.46） |
| RTN W4A8 崩溃 | 是 | 是（PPL 6076） |
| TextVQA 上 MAS vs SQ（n=1000） | 优 | **62.69 vs 61.62，配对差异显著** |
| ScienceQA 上 MAS vs SQ | 优 | **79.10 vs 75.70，显著（+3.4）** |

### 9.2 最重要的发现：**指标决定结论**

| 指标 | MAS（分模态） | 统一 scale + 训练 | 谁赢 |
|---|---|---|---|
| wikitext2 PPL | 18.46 ± 1.43 | **15.02 ± 0.15** | 统一 |
| ScienceQA（1000 题） | **79.10** | — | — |
| TextVQA（1000 题） | 62.69 | — | — |
| **同上，都用混合校准（ratio=0.5）** | 79.90 / 64.49 | 79.60 / 65.12 | **打平（CI 重叠）** |
| 同上，ratio=0.75（匹配统计）：PPL | 16.19 ± 0.34 | 16.20 ± 0.33 | **完全打平（p=0.97）** |
| 同上，ratio=0.75：SQA / TVQA | 80.10 / 64.43 | 78.70 / 64.43 | 打平（p=0.17 / 1.00） |
| MAS + LWC 的极端反例 | PPL **15.45**（最好） | — | 但 SQA 76.5、TVQA 56.5（显著最差） |

**汇报口径**：
> "只报 PPL 会得出论文方法不如基线的结论；换成多模态 benchmark 结论又反过来。
> 而且**给了统一 scale 同样的训练和校准后，两者打平** —— 论文的提升主要来自'可学习平滑'这一步，
> 而不是'分模态'本身。"

**更细的一点（本轮新发现）：PPL 上谁赢取决于校准比例。**

| 纯文本样本占比 | MAS PPL | 统一 scale PPL | PPL 上谁赢 |
|---|---|---|---|
| 0%（原校准） | 18.46 ± 1.43 | **15.02 ± 0.15** | 统一（显著） |
| 50% | 16.35 ± 0.54 | 17.20（1 次） | 接近 |
| 75%（匹配统计） | 16.19 ± 0.34 | 16.20 ± 0.33 | 打平 |

解释：文本监督不足时，MAS 的 text 尺度欠拟合（方差大）；统一 scale 用全部 token 拟合一个尺度，
反而稳。文本充足后，MAS 的 text 尺度拟合到位，而统一 scale 被文本主导、对图像 token 不公平。
**这说明"分模态 vs 统一"的结论对校准配置非常敏感**，论文没有报告校准配置的消融，这是一个实质性的遗漏。
但无论哪种比例，**两个多模态 benchmark 都打平** —— 这一点是稳的。

### 9.3 四个改进/拓展

1. **混合校准（最有效的改进）**
   统计发现原校准集**文本 token 只占 8.1%**（平均每样本 30 个，还是重复的对话模板），
   MAS 的 text 尺度几乎无监督。扫过纯文本占比（0~0.875）后发现：**占比 ≥40% 之后 PPL 就进入平台**，
   从 18.46 ± 1.43 降到 **16.2 ± 0.4**，方差降到 ~1/4。曲线是宽平台不是尖峰，不必精调。
   **而且换到 C4 领域重测，收益仍保留约 80%**（20.21 → 18.53，p=0.0001），
   说明这不是"校准文本与测试集同域"造成的假提升。
   （本轮还踩了一个坑：早期报过"0.75 时 15.52，且分模态显著优于统一，p=0.003"，那是**初始化用的
   `act_scales` 与训练数据比例不匹配**导致的混杂。用匹配统计重做后两者完全打平，p=0.97。）

2. **NVFP4（W4A4）**
   论文只做 INT。我们实现 E2M1 + FP8 块缩放的 NVFP4：INT4A4 直接崩（PPL 96377），
   NVFP4 裸跑 12.25（FP16 11.11），加 MAS 训练到 12.11。**论文的核心论断在浮点 4-bit 下依然成立。**

3. **SQNR 探针的三个 bug + 视频模态**
   论文没做视频。我们写了 SQNR 探针，第一版结论是"统一平滑让视频 token 掉 7 dB"。
   复查发现探针有三个 bug —— **最致命的是构造了权重量化器却没有调用，只量化了激活**，
   所以量出来的根本不是 W4A8 的误差。修复后（并改为输出口径）结论变成：
   **分模态对视觉 token 的优势是 ~1.3 dB，不是 7 dB**，对文本只有 +0.4 dB。
   这反而与下游打平的结果自洽了。
   **教训**：拿有 bug 的代理指标去质疑一个正确的端到端结论，是本工作最该记录的错误；
   而"均值 SQNR 与端到端质量不单调"这一点，任何只用 SQNR 作论据的量化论文都该注意。

4. **可复现性诊断 + 推理加速**
   - 训练不确定性的来源是 SDPA 反向的原子累加；加了 `MAS_DETERMINISTIC` 后两次逐位相同。
     但确定性模式精度更低，**只用于排查不能报数**。
   - 按模态 gather 的单遍前向 + 量化权重缓存：单层 prefill **3.84×**、decode **9.24×**，
     文本行输出逐位一致，全模型 PPL 完全相同。

### 9.3.1 CMC（低秩补偿）的结果

n=150/任务：CMC 与"不补偿"持平（SQA 82.67 vs 82.67，TVQA 63.53 vs 64.00，差异不显著）；
自适应 rank 略差（不显著）。**在我们的规模下没能测到 CMC 的收益。**

### 9.4 发现的原仓库 bug（4 个）

1. **LWC 参数从未被优化**（优化器只注册了平滑参数），且一开就 dtype 崩溃 → 修复后 PPL 15.45，但下游变差。
2. **纯文本校准样本会触发 "backward twice"**：loss 复用了上一个样本已 backward 的 `vision_diff`。
3. **视频 mask 被丢弃**：`modeling_qwen2_5_vl.py` 只传 `(None, image_mask, text_mask)`。
4. 权重缓存泄漏显存（我们引入快路径后自己踩到并修掉）。

### 9.5 论文本身的问题（读代码发现）

1. **Eq.(17) 分子分母写反**（检验：α 全相等时损失应为 0）。
2. **λ 的说法与代码不一致**：论文 Table 4 称 λ=1.0；代码里未传 `--grad_info_path` 时实际用 0.5，
   `0.126` 只在传了 grad_info 时生效（README 的 VL 例子都没传）。
3. **未报告方差**：MAS 单次结果可在 17.0~19.9 之间波动。
4. **未讨论实现代价**：原实现在 prefill 阶段对每个模态各跑一遍完整 GEMM 再相加（3–4× 计算量）。

### 9.6 一句话总结

> **MASQuant 的动机有依据**（分模态对视觉 token 的输出 SQNR 比统一平滑高 ~1.3 dB，方向与论文一致），
> **但"分模态"这个设计在 VL-3B 上没有超过"统一尺度 + 同样训练"**——
> PPL（p=0.97）、两个多模态 benchmark（n=1000）、端到端 KL 保真度，三组独立证据都是打平。
> **真正重要的是校准数据的模态平衡**（纯文本占比 8%→≥40%，wikitext PPL 收益 2.1、
> C4 上收益 1.7 且高度显著，方差降到 1/4，远大于任何方法层面的改动）。
