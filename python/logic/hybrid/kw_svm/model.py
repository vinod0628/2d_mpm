"""M2 KW-SVM — knowledge-weighted (fuzzy) support vector machine (user's "Improved hybrid MPM methods", §2.2).

The kernel works on the expert memberships, each scaled by the expert importance of its layer ω_j = w_j v_k(j)
(mean 1):

    k(x_i, x_l) = exp(−γ Σ_j ω_j (μ_ij − μ_lj)²),  γ = 1/m       (an RBF kernel on Z = μ · √ω)

and non-deposit samples get fuzzy-SVM weights s_i ∝ (1 − K_i)^a that sum to N (cells the expert calls favourable
are weak, possibly mislabelled negatives): min ½‖w‖² + C Σ s_i ξ_i. Output: mean over bags of the Platt-calibrated
decision value.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.special import expit

from logic.hybrid.common.knowledge import KnowledgeNetwork, draw_negatives, membership_matrix, platt

CHUNK = 100_000
ProgressFn = Optional[Callable[[float, str], None]]


@dataclass
class KWSVMResult:
    prob: np.ndarray
    prob_sd: np.ndarray
    expert: np.ndarray
    F: np.ndarray
    kernel_weights: pd.DataFrame
    bags: pd.DataFrame
    summary: pd.DataFrame
    stats: Dict[str, float]
    messages: List[str] = field(default_factory=list)


def kw_svm(net: KnowledgeNetwork, pos: np.ndarray, neg_cand: np.ndarray, n_bags: int = 10, ratio: float = 3.0,
           a: float = 1.0, C: float = 1.0, gamma: Optional[float] = None, seed: int = 0,
           progress: ProgressFn = None) -> KWSVMResult:
    from sklearn.svm import SVC
    say = progress or (lambda f, m: None)
    N = int(pos.size)
    if N < 3:
        raise ValueError(f"Only {N} training deposit cell(s) — need at least 3.")
    msgs = []
    _, F, K = net.expert()
    mu = membership_matrix(net)
    om = net.table.layer_weights()
    Z = mu * np.sqrt(om)
    m = Z.shape[1]
    g = 1.0 / m if gamma is None or gamma <= 0 else float(gamma)
    n_neg = int(round(ratio * N))
    n = net.cells.size
    s1, s2 = np.zeros(n), np.zeros(n)
    rows = []
    for b in range(n_bags):
        say(0.05 + 0.9 * b / n_bags, f"SVM {b + 1} / {n_bags}")
        rng = np.random.default_rng(seed + 1000 * b)
        neg = draw_negatives(rng, neg_cand, n_neg)                     # uniform outside the buffer
        s_neg = (1.0 - K[neg]) ** a + 1e-3
        s_neg *= N / s_neg.sum()
        idx = np.r_[pos, neg]
        y = np.r_[np.ones(N, int), np.zeros(neg.size, int)]
        sw = np.r_[np.ones(N), s_neg]
        svm = SVC(C=C, kernel="rbf", gamma=g).fit(Z[idx], y, sample_weight=sw)
        f = svm.decision_function(Z[idx])
        pa, pb = platt(f, y, sw)
        if pa <= 0:
            pa = 1e-6
            msgs.append(f"bag {b + 1}: decision values do not separate the samples — calibration kept monotone")
        for s in range(0, n, CHUNK):
            ch = slice(s, min(n, s + CHUNK))
            p = expit(pa * svm.decision_function(Z[ch]) + pb)
            s1[ch] += p
            s2[ch] += p * p
        rows.append(dict(bag=b + 1, support_vectors=int(svm.n_support_.sum()), platt_a=pa, platt_b=pb,
                         training_auc=_auc(f[y == 1], f[y == 0]), mean_weight_negatives=float(s_neg.mean()),
                         min_weight_negatives=float(s_neg.min())))
    prob = s1 / n_bags
    sd = np.sqrt(np.maximum(s2 / n_bags - prob ** 2, 0.0))
    kw = pd.DataFrame(dict(layer=[l.name for l in net.table.layers], process=[l.process for l in net.table.layers],
                           expert_weight_in_process=net.w0, process_weight=net.v0[net.k], kernel_weight=om))
    bags = pd.DataFrame(rows)
    st = dict(deposits=N, non_deposits=n_neg, gamma=g, C=C, bags=n_bags,
              support_vectors=float(bags.support_vectors.mean()), training_auc=float(bags.training_auc.mean()))
    summary = pd.DataFrame([
        ("Training deposits (cells)", str(N)), ("Non-deposit samples per SVM", str(n_neg)),
        ("SVMs (bags)", str(n_bags)), ("Kernel", f"RBF on μ·√ω, γ = {g:.4g}" + (" (1/m)" if gamma in (None, 0) else "")),
        ("C (soft margin)", f"{C:g}"), ("Weight of non-deposits", f"∝ (1 − K)^{a:g}, scaled to sum N"),
        ("Support vectors (mean)", f"{st['support_vectors']:.0f}"),
        ("Training AUC (samples, mean; Platt fitted in-sample)", f"{st['training_auc']:.4f}"),
    ], columns=["item", "value"])
    say(1.0, "done")
    return KWSVMResult(prob, sd, K, F, kw, bags, summary, st, msgs)


def _auc(pos, neg) -> float:
    from scipy.stats import rankdata
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    r = rankdata(np.r_[pos, neg])
    return float((r[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))
