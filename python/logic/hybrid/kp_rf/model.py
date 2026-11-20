"""M1 KP-RF — knowledge-pooled random forest (user's "Improved hybrid MPM methods", §2.1).

The expert network K (fixed, built without labels) and a random forest trained on [evidence, process scores F] are
pooled logarithmically (Genest & Zidek 1986):

    logit P = (1 − λ) logit P_RF + λ η^K,     η^K = a_K logit K + b_K (Platt on uniform background),
    λ = κ / (κ + N)                            (the expert is worth κ pseudo-deposits)

so the expert dominates with few deposits and the forest takes over as N grows. Non-deposit samples of every bag are
drawn outside a buffer with probability ∝ (1 − K)^a (knowledge-guided negatives; a = 0 = uniform).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.special import expit

from logic.hybrid.common.knowledge import KnowledgeNetwork, clogit, draw_negatives, platt

CHUNK = 200_000
ProgressFn = Optional[Callable[[float, str], None]]


@dataclass
class KPRFResult:
    prob: np.ndarray
    prob_rf: np.ndarray
    prob_rf_sd: np.ndarray
    prob_k: np.ndarray
    expert: np.ndarray
    F: np.ndarray
    lam: float
    calib: tuple
    importance: pd.DataFrame
    bags: pd.DataFrame
    summary: pd.DataFrame
    stats: Dict[str, float]
    messages: List[str] = field(default_factory=list)


def kp_rf(net: KnowledgeNetwork, pos: np.ndarray, neg_cand: np.ndarray, n_bags: int = 10, ratio: float = 3.0,
          a: float = 1.0, lam: Optional[float] = None, kappa: float = 20.0, n_trees: int = 200, min_leaf: int = 2,
          region: Optional[np.ndarray] = None, use_process_scores: bool = True, seed: int = 0,
          progress: ProgressFn = None) -> KPRFResult:
    """region: optional boolean mask over the study cells — negatives (and the K calibration) only inside it."""
    from sklearn.ensemble import RandomForestClassifier
    say = progress or (lambda f, m: None)
    N = int(pos.size)
    if N < 3:
        raise ValueError(f"Only {N} training deposit cell(s) — need at least 3.")
    msgs = []
    mu, F, K = net.expert()
    names = [l.name for l in net.table.layers]
    Xa = net.features()
    if use_process_scores:
        Xa = np.column_stack([Xa, F])
        names += [f"F[{p}]" for p in net.table.processes]
    cand = neg_cand if region is None else neg_cand[region[neg_cand]]
    if cand.size < 10:
        raise ValueError("Too few candidate cells for non-deposit samples.")
    n_neg = int(round(ratio * N))
    n = net.cells.size
    s1, s2 = np.zeros(n), np.zeros(n)
    imp = np.zeros(len(names))
    rows = []
    for b in range(n_bags):
        say(0.05 + 0.85 * b / n_bags, f"forest {b + 1} / {n_bags}")
        rng = np.random.default_rng(seed + 1000 * b)
        neg = draw_negatives(rng, cand, n_neg, K, a)
        idx = np.r_[pos, neg]
        y = np.r_[np.ones(N, int), np.zeros(neg.size, int)]
        rf = RandomForestClassifier(n_estimators=n_trees, min_samples_leaf=min_leaf, n_jobs=-1, oob_score=True,
                                    random_state=seed + b).fit(Xa[idx], y)
        for s in range(0, n, CHUNK):
            ch = slice(s, min(n, s + CHUNK))
            p = rf.predict_proba(Xa[ch])[:, 1]
            s1[ch] += p
            s2[ch] += p * p
        imp += rf.feature_importances_
        oob = rf.oob_decision_function_[:, 1] if hasattr(rf, "oob_decision_function_") else np.full(y.size, np.nan)
        rows.append(dict(bag=b + 1, non_deposits=int(neg.size), oob_auc=_auc(oob[y == 1], oob[y == 0]),
                         mean_K_negatives=float(K[neg].mean())))
    p_rf = s1 / n_bags
    p_rf_sd = np.sqrt(np.maximum(s2 / n_bags - p_rf ** 2, 0.0))
    # calibration of the expert map on uniform background (10 × more non-deposits, weight 0.1)
    rng = np.random.default_rng(seed + 7)
    unif = rng.choice(cand, min(cand.size, int(10 * ratio * N)), replace=False)
    LK = clogit(K)
    aK, bK = platt(LK[np.r_[pos, unif]], np.r_[np.ones(N), np.zeros(unif.size)],
                   np.r_[np.ones(N), np.full(unif.size, 0.1)])
    if aK <= 0:
        msgs.append("the expert map does not rank the training deposits above the background (Platt slope ≤ 0) — "
                    "its log-odds were set constant; check the directions in the knowledge table")
        aK = 0.0
        bK = float(clogit(np.average(np.r_[np.ones(N), np.zeros(unif.size)],
                                     weights=np.r_[np.ones(N), np.full(unif.size, 0.1)])))
    etaK = aK * LK + bK
    lam_v = kappa / (kappa + N) if lam is None else float(lam)
    eta = (1 - lam_v) * clogit(p_rf) + lam_v * etaK
    imp_t = pd.DataFrame(dict(feature=names, importance=imp / n_bags)).sort_values("importance", ascending=False)
    imp_t.insert(0, "rank", np.arange(1, len(imp_t) + 1))
    imp_t = imp_t.rename(columns={"feature": "layer"}).reset_index(drop=True)
    bags = pd.DataFrame(rows)
    st = dict(deposits=N, non_deposits=n_neg, lam=lam_v, a_K=aK, b_K=bK, oob_auc=float(bags.oob_auc.mean()),
              bags=n_bags, mean_prob=float(expit(eta).mean()))
    summary = pd.DataFrame([
        ("Training deposits (cells)", str(N)), ("Non-deposit samples per forest", str(n_neg)),
        ("Forests (bags) × trees", f"{n_bags} × {n_trees} (min. leaf {min_leaf})"),
        ("Features", f"{len(names)} ({'layers + process scores F' if use_process_scores else 'layers'})"),
        ("Non-deposit sampling", "uniform" if a == 0 else f"knowledge-guided ∝ (1 − K)^{a:g}"),
        ("Pooling weight λ of the expert", f"{lam_v:.3f}" + (f" = κ/(κ + N), κ = {kappa:g}" if lam is None else
                                                            " (fixed)")),
        ("Expert calibration a_K / b_K", f"{aK:.3f} / {bK:.3f}"),
        ("Out-of-bag AUC (samples, mean over forests)", f"{st['oob_auc']:.4f}"),
        ("Negatives restricted to a region", "yes" if region is not None else "no"),
    ], columns=["item", "value"])
    say(1.0, "done")
    return KPRFResult(expit(eta), p_rf, p_rf_sd, expit(etaK), K, F, lam_v, (aK, bK), imp_t, bags, summary, st, msgs)


def _auc(pos, neg) -> float:
    from scipy.stats import rankdata
    pos, neg = np.asarray(pos), np.asarray(neg)
    ok_p, ok_n = pos[np.isfinite(pos)], neg[np.isfinite(neg)]
    if ok_p.size == 0 or ok_n.size == 0:
        return float("nan")
    r = rankdata(np.r_[ok_p, ok_n])
    return float((r[: ok_p.size].sum() - ok_p.size * (ok_p.size + 1) / 2) / (ok_p.size * ok_n.size))
