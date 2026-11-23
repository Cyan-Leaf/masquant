"""平滑之后的权重量化求解器：把 MAS/SmoothQuant 默认的 RTN 换成 GPTQ（及其改进）。

MAS 的每个 QuantLinear 在前向里做 Q_a(X/s_m) · Q_w(W·s_m)ᵀ，其中 Q_w 是 per-channel RTN。
平滑只改变了"量化什么"，没有改变"怎么量化"。本模块在平滑尺度确定之后，对每一份
平滑后的权重 W·s_m 求解

    min_Q  tr[(W·s_m − Q) H (W·s_m − Q)ᵀ],   H = Σ_t x̃_t x̃_tᵀ,  x̃ = X/s_m

（Q 落在与原 RTN 相同的格点上，部署格式不变），并把结果作为该模态的量化权重。

多模态相关的设计：
  * 分模态 Hessian：H 在"该份权重实际服务的 token"上统计，且在该份权重的平滑空间里统计
    （x̃ = X/s_m）。分模态平滑下文本权重只看文本 token，视觉权重只看视觉 token。
  * 模态均衡 Hessian（--gptq_hmode balanced）：一份权重同时服务多个模态时
    （统一 scale、down_proj、只用文本尺度的单权重部署），原始 GPTQ 的 H 按 token 数加权，
    视觉 token 占校准集的大头会主导求解；balanced 先按模态各自归一化再平均。
求解器改进（来自 NVFP4 GPTQ 的经验，这里推广到 INT）：
  * clip：每行在若干 γ 下缩小量化步长 s·γ，按 H 加权误差逐行选优（行之间在该目标下独立，
    所以逐行选择是精确的）；同时把 RTN 也放进候选，保证逐行不劣于 RTN。
  * cd：GPTQ 之后做逐元素坐标下降（固定其余元素时目标对单元素是二次函数，枚举全部格点），
    严格单调下降，收敛到 1-opt 局部极小。
"""
import os
import torch

from quantize.int_linear import QuantLinear
from quantize.nvfp4 import E2M1_MAX, E4M3_MAX, _round_to_e4m3

GROUPS = [
    ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"],
    ["self_attn.o_proj"],
    ["mlp.gate_proj", "mlp.up_proj"],
    ["mlp.down_proj"],
]
E2M1_CODES = torch.tensor([-6., -4., -3., -2., -1.5, -1., -.5, 0., .5, 1., 1.5, 2., 3., 4., 6.])


def _e2m1(x):
    a = x.abs().clamp(max=E2M1_MAX)
    q = torch.where(a < 2.0, torch.round(a * 2.0) / 2.0,
                    torch.where(a < 4.0, torch.round(a), torch.round(a / 2.0) * 2.0))
    q = torch.where(a < 0.25, torch.zeros_like(q), q)
    return torch.sign(x) * q


class _IntFmt:
    """与 UniformAffineQuantizer(symmetric, per_channel) 完全相同的格点：q = clamp(round(w/s), lo, hi)·s。"""

    def __init__(self, quantizer):
        assert quantizer.symmetric, "GPTQ 目前只支持对称量化（主配置就是 --symmetric）"
        nb = quantizer.n_bits
        if quantizer.disable_zero_point:
            self.lo, self.hi = -(2 ** (nb - 1)), 2 ** (nb - 1) - 1
        else:
            zp = 2 ** (nb - 1) - 1
            self.lo, self.hi = quantizer.qmin - zp, quantizer.qmax - zp
        self.nb = nb
        self.codes = None

    def scales(self, W, gamma):
        # 与量化器同样在 bf16 下算 absmax/(2^(b-1)-1)，保证 γ=1 时与 RTN 逐位同格
        wb = W.to(torch.bfloat16)
        s = (wb.abs().amax(-1, keepdim=True) / (2 ** (self.nb - 1) - 1)).clamp(min=1e-5, max=1e4)
        s = s.float() * gamma
        return s.expand_as(W).contiguous()

    def rnd(self, x):
        return torch.clamp(torch.round(x), self.lo, self.hi)

    def cand(self, device):
        if self.codes is None or self.codes.device != device:
            self.codes = torch.arange(self.lo, self.hi + 1, dtype=torch.float32, device=device)
        return self.codes


class _NVFP4Fmt:
    """与 NVFP4Quantizer 相同：per-tensor FP32 全局缩放 + 每 16 元素一个 E4M3 块缩放 + E2M1 码字。"""

    def __init__(self, quantizer, solve_fp32_scale=False):
        self.bs = quantizer.block_size
        self.fp8 = quantizer.fp8_block_scale
        # 求解时是否跳过块缩放的 E4M3 量化（只影响求解，部署读格点时再按 FP8 落盘）
        self.solve_fp32_scale = solve_fp32_scale
        self.codes = None

    def scales(self, W, gamma):
        N, K = W.shape
        assert K % self.bs == 0
        wb = W.to(torch.bfloat16).float()
        amax_t = wb.abs().amax().clamp(min=1e-8)
        g = amax_t / (E2M1_MAX * E4M3_MAX)
        blk = wb.reshape(N, K // self.bs, self.bs).abs().amax(-1) / E2M1_MAX / g * gamma
        if self.fp8 and not self.solve_fp32_scale:
            blk = _round_to_e4m3(blk)
        s = (blk * g).clamp(min=1e-12)
        return s.repeat_interleave(self.bs, dim=1).contiguous()

    def rnd(self, x):
        return _e2m1(x)

    def cand(self, device):
        if self.codes is None or self.codes.device != device:
            self.codes = E2M1_CODES.to(device)
        return self.codes


def make_fmt(module, solve_fp32_scale=False):
    q = module.weight_quantizer
    if q.__class__.__name__ == "NVFP4Quantizer":
        return _NVFP4Fmt(q, solve_fp32_scale)
    return _IntFmt(q)


def _row_loss(D, H):
    return ((D @ H) * D).sum(-1)


def gptq_solve(W, Hinv, S, fmt, blocksize=128):
    W = W.clone()
    R, K = W.shape
    Q = torch.zeros_like(W)
    for i1 in range(0, K, blocksize):
        i2 = min(i1 + blocksize, K)
        W1 = W[:, i1:i2].clone()
        Err1 = torch.zeros_like(W1)
        Hinv1 = Hinv[i1:i2, i1:i2]
        for j in range(i2 - i1):
            w = W1[:, j]
            s = S[:, i1 + j]
            q = fmt.rnd(w / s) * s
            Q[:, i1 + j] = q
            err = (w - q) / Hinv1[j, j]
            W1[:, j + 1:] -= err.unsqueeze(1) * Hinv1[j, j + 1:].unsqueeze(0)
            Err1[:, j] = err
        W[:, i2:] -= Err1 @ Hinv[i1:i2, i2:]
    return Q


def cd_polish(W, Q, H, S, fmt, passes):
    """逐元素精确坐标下降（列顺序 Gauss-Seidel），只接受降 loss 的改动，单调。"""
    R, K = W.shape
    D = W - Q
    G = D @ H
    hd = torch.diag(H)
    codes = fmt.cand(W.device)
    rows = torch.arange(R, device=W.device)
    changed_hist = []
    for _ in range(passes):
        changed = 0
        for j in range(K):
            cand = codes.view(1, -1) * S[:, j].view(-1, 1)
            delta = (W[:, j].view(-1, 1) - cand) - D[:, j].view(-1, 1)
            dl = 2 * delta * G[:, j].view(-1, 1) + delta * delta * hd[j]
            b = dl.argmin(1)
            best = dl[rows, b]
            bd = torch.where(best < -1e-12 * hd[j].abs().clamp(min=1e-30), delta[rows, b], torch.zeros_like(best))
            nz = bd != 0
            if nz.any():
                changed += int(nz.sum())
                D[:, j] += bd
                G += bd.view(-1, 1) * H[j].view(1, -1)
        changed_hist.append(changed)
        if changed == 0:
            break
    return W - D, changed_hist


@torch.no_grad()
def quantize_weight(W, H, fmt, gammas=(1.0,), cd_passes=0, percdamp=0.01, actorder=False, rtn_only=False):
    """返回 (Q, stats)。W: [N,K] fp32（已乘平滑尺度），H: [K,K] fp32。"""
    W = W.float()
    N, K = W.shape
    H = H.clone()
    dead = torch.diag(H) <= 0
    H[dead, dead] = 1.0
    W = W.clone()
    W[:, dead] = 0
    H.diagonal().add_(percdamp * torch.diag(H).mean())

    S_rtn = fmt.scales(W, 1.0)
    Q_rtn = fmt.rnd(W / S_rtn) * S_rtn
    l_rtn = _row_loss(W - Q_rtn, H)

    perm = torch.argsort(torch.diag(H), descending=True) if actorder else None
    Hp = H[perm][:, perm] if actorder else H
    L = torch.linalg.cholesky(Hp)
    Hinv = torch.cholesky_inverse(L)
    Hinv = torch.linalg.cholesky(Hinv, upper=True)

    G = len(gammas)
    S_all = torch.cat([fmt.scales(W, g) for g in gammas], 0)
    W_all = W.repeat(G, 1)
    if rtn_only:
        Q_all = fmt.rnd(W_all / S_all) * S_all
    elif actorder:
        Q_all = gptq_solve(W_all[:, perm], Hinv, S_all[:, perm], fmt)
        Qo = torch.empty_like(Q_all)
        Qo[:, perm] = Q_all
        Q_all = Qo
    else:
        Q_all = gptq_solve(W_all, Hinv, S_all, fmt)
    l_all = _row_loss(W_all - Q_all, H).view(G, N)
    l_cands = torch.cat([l_all, l_rtn.view(1, N)], 0)
    pick = l_cands.argmin(0)
    Q = Q_rtn.clone()
    S = S_rtn.clone()
    for gi in range(G):
        m = pick == gi
        if m.any():
            Q[m] = Q_all.view(G, N, K)[gi][m]
            S[m] = S_all.view(G, N, K)[gi][m]
    l_gptq = l_cands.gather(0, pick.view(1, N)).squeeze(0)
    cd_hist = []
    if cd_passes > 0:
        Q, cd_hist = cd_polish(W, Q, H, S, fmt, cd_passes)
    l_final = _row_loss(W - Q, H)
    tot_rtn = l_rtn.sum().clamp(min=1e-30)
    stats = {
        "rec_gptq": 1 - (l_gptq.sum() / tot_rtn).item(),
        "rec_final": 1 - (l_final.sum() / tot_rtn).item(),
        "pick": torch.bincount(pick, minlength=G + 1).tolist(),
        "cd_changed": cd_hist,
    }
    return Q, stats


class _HessianCollector:
    def __init__(self, ctx, act_quant, hsrc="own"):
        self.ctx = ctx
        self.act_quant = act_quant
        self.hsrc = hsrc
        self.H = {}
        self.n = {}

    def hook(self, name):
        def fn(module, args, kwargs, out):
            x = args[0]
            own_mm = args[1] if len(args) > 1 else kwargs.get("multi_modal_mask")
            K = x.shape[-1]
            x2 = x.reshape(-1, K).to(torch.bfloat16)
            lab_groups, _ = module._token_groups(x, self.ctx["mm"])
            n = x2.shape[0]
            lab = torch.zeros(n, dtype=torch.long, device=x2.device)
            mods = []
            for mi, (m, idx) in enumerate(lab_groups.items()):
                mods.append(m)
                if idx is None:
                    lab[:] = mi
                else:
                    lab[idx] = mi
            if module.inference_mode == "merged_scales" and module.mode != "infer":
                tag_groups = {"merged": None}
            else:
                tag_groups, _ = module._token_groups(x, own_mm)
            eps = 0.0 if "merged" in tag_groups else 1e-5
            if self.hsrc == "all" and len(tag_groups) > 1:
                # 跨模态共享 Hessian：每份权重都在全部 token 上统计（各自的平滑空间里），
                # 用于某个模态 token 太少时（原校准文本只占 8%）给它的权重补统计量
                tag_groups = {t: None for t in tag_groups}
            for tag, idx in tag_groups.items():
                s = module.all_in_one_smooth_scale if tag == "merged" else module._scale_of(tag)
                rows = x2 if idx is None else x2.index_select(0, idx)
                rl = lab if idx is None else lab.index_select(0, idx)
                xs = rows / (s + eps) if eps else rows / s
                if self.act_quant and module.act_quantizer is not None:
                    xs = module.act_quantizer(xs)
                xs = xs.float()
                for mi, m in enumerate(mods):
                    sel = rl == mi
                    cnt = int(sel.sum())
                    if cnt == 0:
                        continue
                    xm = xs[sel]
                    key = (name, tag, m)
                    if key not in self.H:
                        self.H[key] = torch.zeros(K, K, dtype=torch.float32, device=x2.device)
                        self.n[key] = 0
                    self.H[key].addmm_(xm.t(), xm)
                    self.n[key] += cnt
        return fn


def _parse_modw(spec):
    out = {}
    for kv in (spec or "").split(","):
        if ":" in kv:
            k, v = kv.split(":")
            out[k.strip()] = float(v)
    return out


def _combine(Hs, ns, hmode, modw):
    """Hs/ns: {modality: H_sum}/{modality: count}。"""
    if hmode == "raw" and not modw:
        H = sum(Hs.values())
        return H / max(sum(ns.values()), 1)
    tot = 0.0
    H = None
    for m, Hm in Hs.items():
        w = modw.get(m, 1.0)
        term = Hm * (w / max(ns[m], 1))
        H = term if H is None else H + term
        tot += w
    return H / max(tot, 1e-12)


def _scale_tensor(module, tag):
    return module.all_in_one_smooth_scale if tag == "merged" else module._scale_of(tag)


@torch.no_grad()
def gptq_qlayer(qlayer, run_sample, nsamples, ctx, args, logger, layer_idx, cache=None):
    """对一个 decoder 层按 GROUPS 顺序做 GPTQ（每组的输入来自前面已 GPTQ 的组）。

    run_sample(j): 用 ctx['mm'] = 第 j 个样本的 multi_modal_mask 跑一遍 qlayer。
    结果作为 buffer `gptq_w_<tag>` 挂在 QuantLinear 上，_smoothed_weight 会优先使用它。
    """
    gammas = [float(g) for g in str(getattr(args, "gptq_clip", "1.0")).split(",") if g]
    hmode = getattr(args, "gptq_hmode", "raw")
    modw = _parse_modw(getattr(args, "gptq_modw", ""))
    mods = dict(qlayer.named_modules())
    layer_out = {}
    for group in GROUPS:
        group = [g for g in group if g in mods and isinstance(mods[g], QuantLinear)]
        if not group:
            continue
        if cache is not None and all(g in cache for g in group):
            for g in group:
                for tag, Q in cache[g].items():
                    _set_override(mods[g], tag, Q.to(mods[g].weight.device))
                layer_out[g] = cache[g]
            continue
        col = _HessianCollector(ctx, getattr(args, "gptq_actq", False), getattr(args, "gptq_hsrc", "own"))
        hs = [mods[g].register_forward_hook(col.hook(g), with_kwargs=True) for g in group]
        try:
            for j in range(nsamples):
                run_sample(j)
        finally:
            for h in hs:
                h.remove()
        for g in group:
            m = mods[g]
            tags = sorted({t for (n, t, _) in col.H if n == g})
            # 平滑尺度相同的 tag 共用一份权重（例如 down_proj、单尺度部署），Hessian 合并
            buckets = []
            for t in tags:
                s = _scale_tensor(m, t)
                for b in buckets:
                    s0 = _scale_tensor(m, b[0])
                    if s0 is s or (s0.shape == s.shape and torch.equal(s0, s)):
                        b.append(t)
                        break
                else:
                    buckets.append([t])
            fmt = make_fmt(m, getattr(args, "gptq_fp32_bs", False))
            layer_out[g] = {}
            for b in buckets:
                Hs, ns = {}, {}
                for (n, t, mm_), H in col.H.items():
                    if n == g and t in b:
                        Hs[mm_] = Hs.get(mm_, 0) + H
                        ns[mm_] = ns.get(mm_, 0) + col.n[(n, t, mm_)]
                H = _combine(Hs, ns, hmode, modw)
                s = _scale_tensor(m, b[0])
                W = (m.weight.to(torch.bfloat16) * s).float()
                Q, st = quantize_weight(W, H, fmt, gammas=gammas,
                                        cd_passes=getattr(args, "gptq_cd", 0),
                                        percdamp=getattr(args, "gptq_percdamp", 0.01),
                                        actorder=getattr(args, "gptq_actorder", False),
                                        rtn_only=getattr(args, "gptq_rtn_only", False))
                Qb = Q.to(torch.bfloat16)
                for t in b:
                    _set_override(m, t, Qb)
                    layer_out[g][t] = Qb.cpu()
                if logger is not None:
                    logger.info(f"[gptq] L{layer_idx} {g} tags={b} tokens={ns} "
                                f"rec_gptq={st['rec_gptq']*100:.1f}% rec_final={st['rec_final']*100:.1f}% "
                                f"pick={st['pick']} cd={st['cd_changed']}")
        del col
        torch.cuda.empty_cache()
    return layer_out


def _set_override(module, tag, Q):
    name = f"gptq_w_{tag}"
    if name in module._buffers:
        module._buffers[name] = Q
    else:
        module.register_buffer(name, Q)
    module.clear_weight_cache()
