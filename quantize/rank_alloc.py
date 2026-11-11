import json
import math

import torch


def _energy_curve(sigma, max_rank=None):
    """返回每个秩 r 下的能量占比（Frobenius 意义下被保留的比例）。"""
    s2 = sigma.float().pow(2)
    total = s2.sum().clamp(min=1e-20)
    cum = torch.cumsum(s2, dim=0) / total
    if max_rank is not None:
        cum = cum[:max_rank]
    return cum


def _marginal_gain(sigma, r):
    """加到第 r 个奇异值带来的能量增益占比。"""
    total = sigma.float().pow(2).sum().clamp(min=1e-20)
    return (sigma.float()[r] ** 2 / total).item()


def allocate_ranks(whitened_sigmas, budget_ratio, min_rank=32, rank_ratio_cap=None):
    """
    在总预算 sum(r_l) <= budget_ratio * sum(rank_max_l) 下分配每层秩。

    贪心策略：每步把 1 个秩给"单位秩增益最大"的那一层，
    增益用白化后奇异值的边际能量贡献衡量（Eckart-Young 下即重构误差的下降量）。

    whitened_sigmas: {layer_name: sigma (1D tensor, 白化后奇异值，降序)}
    budget_ratio:    总秩预算占满秩的比例
    min_rank:        每层秩下限
    rank_ratio_cap:  每层秩上限占满秩的比例（None 表示不设限）
    返回 {layer_name: rank}
    """
    names = list(whitened_sigmas.keys())
    caps = {}
    for n in names:
        d = whitened_sigmas[n].numel()
        cap = d if rank_ratio_cap is None else max(min_rank, int(d * rank_ratio_cap))
        caps[n] = cap

    total_budget = int(budget_ratio * sum(whitened_sigmas[n].numel() for n in names))
    ranks = {n: min(min_rank, caps[n]) for n in names}
    spent = sum(ranks.values())
    if spent >= total_budget:
        return ranks

    gains = {n: _marginal_gain(whitened_sigmas[n], ranks[n]) if ranks[n] < caps[n] else -1.0
             for n in names}

    while spent < total_budget:
        candidates = [n for n in names if ranks[n] < caps[n]]
        if not candidates:
            break
        best = max(candidates, key=lambda n: gains[n])
        ranks[best] += 1
        spent += 1
        r = ranks[best]
        gains[best] = _marginal_gain(whitened_sigmas[best], r) if r < caps[best] else -1.0

    return ranks


def allocate_ranks_by_energy(whitened_sigmas, target_energy=0.99, min_rank=32, max_rank_ratio=None):
    """
    按能量阈值逐层分配：取最小的 r 使累计能量 >= target_energy。
    更激进地利用奇异值衰减 —— 衰减快的层给很小的秩。
    """
    out = {}
    for n, sigma in whitened_sigmas.items():
        d = sigma.numel()
        cap = d if max_rank_ratio is None else max(min_rank, int(d * max_rank_ratio))
        cum = _energy_curve(sigma, max_rank=cap)
        idx = torch.searchsorted(cum, torch.tensor(target_energy, device=cum.device))
        r = int(idx.item()) + 1
        out[n] = max(min_rank, min(r, cap))
    return out


def summarize_ranks(ranks):
    vals = sorted(ranks.values())
    n = len(vals)
    if n == 0:
        return {}
    return {
        "n_layers": n,
        "min": vals[0],
        "median": vals[n // 2],
        "max": vals[-1],
        "mean": sum(vals) / n,
        "total": sum(vals),
    }


def dump_ranks(ranks, path):
    with open(path, "w") as f:
        json.dump(ranks, f, indent=2)
