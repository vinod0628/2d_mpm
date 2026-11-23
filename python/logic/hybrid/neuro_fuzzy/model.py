"""Neuro-fuzzy inference (ANFIS-type) on the mineral-system knowledge network.

A first-order Takagi–Sugeno fuzzy inference system (Takagi & Sugeno 1985) whose premise membership functions start
from the expert knowledge and are tuned on deposit / non-deposit samples with the hybrid learning rule of ANFIS
(Jang 1993): least squares for the consequents, gradient descent for the premises — the hybrid neuro-fuzzy approach
of Porwal, Carranza & Hale (2004) to mineral potential mapping.

Inputs x_i ∈ [0, 1] (directed: 1 = favourable) are either the expert process scores F_k of the knowledge network
(one input per mineral-system process) or the expert memberships μ_j of the evidence layers (one input per layer;
missing = 0.5). Every input has M Gaussian membership functions ("low", ("medium",) "high")

    A_im(x) = exp(−(x − c_im)² / (2 σ_im²)),   c⁰ = 0, (0.5,) 1,   σ⁰ = spacing / (2 √(2 ln 2))  (0.5 at crossings)

and the rule base is the grid partition (M^n rules) R_r: IF x_1 is A_1r AND … THEN f_r = p_r·x + q_r, with

    w_r = Π_i A_i,r_i(x_i),   w̄_r = w_r / Σ w,   ŷ = Σ_r w̄_r f_r.

Knowledge prior on the consequents: q_r is shrunk towards q⁰_r, the expert favourability K at the rule prototype
(inputs at the initial MF centres), calibrated to the sample by a Platt fit of logit K, and the slopes towards 0:

    θ = argmin Σ_i s_i (y_i − ŷ_i)² + κ Σ_r [(q_r − q⁰_r)² + ‖p_r‖²]        (ridge least squares per epoch)

so rules with little data keep the expert's answer. Premises (c, ln σ) follow Adam on the squared error; the epoch
with the smallest error on a held-out checking set is kept (Jang 1993). Bagging over non-deposit draws; every
model's output is Platt-calibrated on its training sample; the map is the mean over the bags.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.special import expit, softmax

from logic.hybrid.common.knowledge import KnowledgeNetwork, clogit, draw_negatives, platt

INPUT_MODES = ("auto", "processes", "layers")
CONSEQUENTS = ("linear", "constant")
MF_NAMES = {2: ("low", "high"), 3: ("low", "medium", "high")}
MAX_RULES = 256
AUTO_LAYER_RULES = 64             # auto: one input per layer while M^m ≤ 64, else one per process
CHUNK = 20_000
SIGMA_MIN, SIGMA_MAX = 0.03, 2.0
C_MIN, C_MAX = -0.5, 1.5
ProgressFn = Optional[Callable[[float, str], None]]


@dataclass
class NeuroFuzzyResult:
    prob: np.ndarray
    prob_sd: np.ndarray
    expert: np.ndarray
    F: np.ndarray
    inputs: List[str]
    input_mode: str
    mf: pd.DataFrame
    rules: pd.DataFrame
    importance: pd.DataFrame
    bags: pd.DataFrame
    summary: pd.DataFrame
    stats: Dict[str, float]
    messages: List[str] = field(default_factory=list)


# ============================================================================== Takagi–Sugeno system
class TSKSystem:
    """First- (or zero-) order Takagi–Sugeno system with Gaussian MFs and a grid-partition rule base."""

    def __init__(self, n_in: int, n_mf: int = 2, consequent: str = "linear"):
        if n_mf not in MF_NAMES:
            raise ValueError("2 or 3 membership functions per input")
        if consequent not in CONSEQUENTS:
            raise ValueError(f"unknown consequent {consequent!r}")
        self.n_in, self.n_mf, self.consequent = int(n_in), int(n_mf), consequent
        self.rule_idx = np.array(list(itertools.product(range(n_mf), repeat=n_in)), dtype=int)   # R × n
        self.R = self.rule_idx.shape[0]
        centres = np.linspace(0.0, 1.0, n_mf)
        self.c = np.tile(centres, (n_in, 1))
        self.sigma = np.full((n_in, n_mf), (1.0 / (n_mf - 1)) / (2.0 * np.sqrt(2.0 * np.log(2.0))))
        self.theta = np.zeros(self.R * (n_in + 1) if consequent == "linear" else self.R)

    # ------------------------------------------------------------------ forward pieces
    def prototypes(self, c: Optional[np.ndarray] = None) -> np.ndarray:
        """Input vector of every rule at the MF centres (R × n)."""
        c = self.c if c is None else c
        return c[np.arange(self.n_in)[None, :], self.rule_idx]

    def log_firing(self, X: np.ndarray) -> np.ndarray:
        """ln w (n × R), accumulated input by input (memory n × R)."""
        lw = np.zeros((X.shape[0], self.R))
        for i in range(self.n_in):
            lm = -(X[:, i, None] - self.c[i]) ** 2 / (2.0 * self.sigma[i] ** 2)     # n × M
            lw += lm[:, self.rule_idx[:, i]]
        return lw

    def normalised(self, X: np.ndarray) -> np.ndarray:
        return softmax(self.log_firing(X), axis=1)

    def design(self, X: np.ndarray, wbar: np.ndarray) -> np.ndarray:
        """Least-squares design matrix: [w̄_r x, w̄_r] (linear) or w̄ (constant)."""
        if self.consequent == "constant":
            return wbar
        n = X.shape[0]
        Xa = np.c_[X, np.ones(n)]                                                  # n × (n_in + 1)
        return (wbar[:, :, None] * Xa[:, None, :]).reshape(n, -1)

    def rule_outputs(self, X: np.ndarray) -> np.ndarray:
        """f_r(x) (n × R)."""
        if self.consequent == "constant":
            return np.broadcast_to(self.theta, (X.shape[0], self.R))
        P = self.theta.reshape(self.R, self.n_in + 1)
        return X @ P[:, :-1].T + P[:, -1]

    def predict(self, X: np.ndarray, chunk: int = CHUNK) -> np.ndarray:
        out = np.empty(X.shape[0])
        for s in range(0, X.shape[0], chunk):
            Xc = X[s:s + chunk]
            out[s:s + chunk] = (self.normalised(Xc) * self.rule_outputs(Xc)).sum(1)
        return out

    def constant_index(self) -> np.ndarray:
        """Positions of the rule constants q_r in θ."""
        if self.consequent == "constant":
            return np.arange(self.R)
        return np.arange(self.R) * (self.n_in + 1) + self.n_in

    # ------------------------------------------------------------------ learning
    def solve_consequents(self, X, y, sw, q0, kappa) -> None:
        """Ridge least squares with the knowledge prior (q → q⁰, p → 0)."""
        wbar = self.normalised(X)
        Phi = self.design(X, wbar)
        prior = np.zeros(Phi.shape[1])
        prior[self.constant_index()] = q0
        A = (Phi * sw[:, None]).T @ Phi + kappa * np.eye(Phi.shape[1])
        b = (Phi * sw[:, None]).T @ y + kappa * prior
        try:
            self.theta = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            self.theta = np.linalg.lstsq(A, b, rcond=None)[0]

    def error_and_grad(self, X, y, sw):
        """Weighted mean squared error and its gradient w.r.t. (c, ln σ) at the current consequents."""
        wbar = self.normalised(X)
        f = self.rule_outputs(X)
        yhat = (wbar * f).sum(1)
        e = yhat - y
        tot = sw.sum()
        mse = float((sw * e * e).sum() / tot)
        H = (2.0 / tot) * (sw * e)[:, None] * wbar * (f - yhat[:, None])           # ∂E/∂ ln w_r
        gc, gs = np.zeros_like(self.c), np.zeros_like(self.sigma)
        for i in range(self.n_in):
            onehot = np.eye(self.n_mf)[self.rule_idx[:, i]]                         # R × M
            Mi = H @ onehot                                                         # n × M
            d = X[:, i, None] - self.c[i]
            gc[i] = (Mi * d).sum(0) / self.sigma[i] ** 2
            gs[i] = (Mi * d * d).sum(0) / self.sigma[i] ** 2                       # ∂E/∂ ln σ
        return mse, yhat, gc, gs

    def clip(self):
        self.c = np.clip(self.c, C_MIN, C_MAX)
        self.sigma = np.clip(self.sigma, SIGMA_MIN, SIGMA_MAX)

    def state(self):
        return self.c.copy(), self.sigma.copy(), self.theta.copy()

    def set_state(self, st):
        self.c, self.sigma, self.theta = (a.copy() for a in st)


def train_tsk(sys: TSKSystem, X, y, sw, q0, kappa=1.0, epochs=100, lr=0.02, check=None, tune_premises=True):
    """Hybrid learning: per epoch LSE for the consequents, one Adam step for the premises. ``check`` = (X, y, sw)
    of the checking set: the parameters of the epoch with the smallest checking error are kept.
    Returns the history list of (training mse, checking mse)."""
    m_c = v_c = m_s = v_s = 0.0
    b1, b2 = 0.9, 0.999
    best, best_err, hist = None, np.inf, []
    for t in range(1, max(1, int(epochs)) + 1):
        sys.solve_consequents(X, y, sw, q0, kappa)
        mse, _, gc, gs = sys.error_and_grad(X, y, sw)
        if check is not None:
            Xc, yc, swc = check
            ec = sys.predict(Xc) - yc
            cerr = float((swc * ec * ec).sum() / swc.sum())
        else:
            cerr = mse
        hist.append((mse, cerr))
        if cerr < best_err - 1e-12:
            best, best_err = sys.state(), cerr
        if not tune_premises or t == epochs:
            continue
        m_c = b1 * m_c + (1 - b1) * gc
        v_c = b2 * v_c + (1 - b2) * gc * gc
        m_s = b1 * m_s + (1 - b1) * gs
        v_s = b2 * v_s + (1 - b2) * gs * gs
        corr1, corr2 = 1 - b1 ** t, 1 - b2 ** t
        sys.c = sys.c - lr * (m_c / corr1) / (np.sqrt(v_c / corr2) + 1e-8)
        sys.sigma = np.exp(np.log(sys.sigma) - lr * (m_s / corr1) / (np.sqrt(v_s / corr2) + 1e-8))
        sys.clip()
    if best is not None:
        sys.set_state(best)
    return hist


# ============================================================================== knowledge → inputs
def choose_inputs(net: KnowledgeNetwork, mode: str, n_mf: int):
    """(mode actually used, input names)."""
    if mode not in INPUT_MODES:
        raise ValueError(f"unknown input mode {mode!r}")
    m = len(net.table.layers)
    if mode == "auto":
        mode = "layers" if n_mf ** m <= AUTO_LAYER_RULES or net.K < 2 else "processes"
    names = [l.name for l in net.table.layers] if mode == "layers" else list(net.table.processes)
    if n_mf ** len(names) > MAX_RULES:
        raise ValueError(f"{n_mf}^{len(names)} = {n_mf ** len(names)} rules — more than {MAX_RULES}. Use one input "
                         "per process, 2 membership functions per input, or fewer layers.")
    return mode, names


def expert_at(net: KnowledgeNetwork, mode: str, P: np.ndarray) -> np.ndarray:
    """Expert favourability K of input vectors P (rows) — process scores or layer memberships."""
    P = np.clip(np.asarray(P, float), 0.01, 0.99)
    F = P if mode == "processes" else net.process_scores(P)
    return net.favourability(F)


def _auc(pos, neg) -> float:
    from scipy.stats import rankdata
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    r = rankdata(np.r_[pos, neg])
    return float((r[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def _split_check(rng, y, frac):
    """Stratified checking-set mask (both classes keep ≥ 2 training samples)."""
    chk = np.zeros(y.size, bool)
    if frac <= 0:
        return chk
    for cls in (0, 1):
        idx = np.flatnonzero(y == cls)
        k = int(round(frac * idx.size))
        k = min(k, max(0, idx.size - 2))
        if k > 0:
            chk[rng.choice(idx, k, replace=False)] = True
    return chk


# ============================================================================== the method
def neuro_fuzzy(net: KnowledgeNetwork, pos: np.ndarray, neg_cand: np.ndarray, input_mode: str = "auto",
                n_mf: int = 2, consequent: str = "linear", kappa: float = 1.0, epochs: int = 100, lr: float = 0.02,
                check_fraction: float = 0.25, n_bags: int = 10, ratio: float = 3.0, seed: int = 0,
                progress: ProgressFn = None) -> NeuroFuzzyResult:
    say = progress or (lambda f, m: None)
    N = int(pos.size)
    if N < 3:
        raise ValueError(f"Only {N} training deposit cell(s) — need at least 3.")
    msgs: List[str] = []
    mu_raw, F, K = net.expert()
    mode, names = choose_inputs(net, input_mode, int(n_mf))
    Xall = F.copy() if mode == "processes" else np.nan_to_num(mu_raw, nan=0.5)
    Xall = np.clip(Xall, 0.0, 1.0)
    if mode == "processes" and net.K < 2:
        msgs.append("Only one mineral-system process: the neuro-fuzzy model has one input (a calibration of the "
                    "expert score) — use one input per layer to let the data shape every layer.")
    proto_sys = TSKSystem(len(names), int(n_mf), consequent)
    K_proto = expert_at(net, mode, proto_sys.prototypes())
    n_neg = int(round(ratio * N))
    n = net.cells.size
    s1, s2 = np.zeros(n), np.zeros(n)
    c_list, s_list, q_list, out_list, fire_list, dep_list, imp_list, rows = [], [], [], [], [], [], [], []
    for b in range(int(n_bags)):
        say(0.05 + 0.85 * b / n_bags, f"neuro-fuzzy model {b + 1} / {n_bags}")
        rng = np.random.default_rng(seed + 1000 * b)
        neg = draw_negatives(rng, neg_cand, n_neg)
        idx = np.r_[pos, neg]
        y = np.r_[np.ones(N), np.zeros(neg.size)]
        sw = np.ones(idx.size)
        X = Xall[idx]
        chk = _split_check(rng, y, check_fraction)
        tr_, ck = ~chk, chk
        # knowledge prior of the rule constants: expert K at the prototypes, calibrated on this sample
        a0, b0 = platt(clogit(K[idx[tr_]]), y[tr_].astype(int), sw[tr_])
        q0 = expit(max(a0, 1e-6) * clogit(K_proto) + b0)
        sys = TSKSystem(len(names), int(n_mf), consequent)
        hist = train_tsk(sys, X[tr_], y[tr_], sw[tr_], q0, kappa, epochs, lr,
                         check=(X[ck], y[ck], sw[ck]) if ck.any() else None)
        best_epoch = int(np.argmin([h[1] for h in hist])) + 1
        yhat = sys.predict(X)
        pa, pb = platt(yhat, y.astype(int), sw)
        if pa <= 0:
            pa = 1e-6
            msgs.append(f"model {b + 1}: outputs do not separate the samples — calibration kept monotone")
        fire = np.zeros(sys.R)
        for s in range(0, n, CHUNK):
            ch = slice(s, min(n, s + CHUNK))
            wb = sys.normalised(Xall[ch])
            out = (wb * sys.rule_outputs(Xall[ch])).sum(1)
            p = expit(pa * out + pb)
            s1[ch] += p
            s2[ch] += p * p
            fire += wb.sum(0)
        fire /= n
        dep_fire = sys.normalised(Xall[pos]).mean(0)
        # permutation importance on this model's sample (AUC drop)
        base = _auc(yhat[y == 1], yhat[y == 0])
        imp = []
        for i in range(len(names)):
            Xp = X.copy()
            Xp[:, i] = rng.permutation(Xp[:, i])
            yp = sys.predict(Xp)
            imp.append(base - _auc(yp[y == 1], yp[y == 0]))
        c_list.append(sys.c.copy())
        s_list.append(sys.sigma.copy())
        protos = sys.prototypes()
        q_list.append(sys.theta[sys.constant_index()].copy())
        out_list.append((sys.rule_outputs(protos)[np.arange(sys.R), np.arange(sys.R)]))
        fire_list.append(fire)
        dep_list.append(dep_fire)
        imp_list.append(imp)
        rows.append(dict(bag=b + 1, training_samples=int(tr_.sum()), checking_samples=int(ck.sum()),
                         best_epoch=best_epoch, training_rmse=float(np.sqrt(hist[best_epoch - 1][0])),
                         checking_rmse=float(np.sqrt(hist[best_epoch - 1][1])) if ck.any() else np.nan,
                         sample_auc=base, platt_a=pa, platt_b=pb))
    prob = s1 / n_bags
    sd = np.sqrt(np.maximum(s2 / n_bags - prob ** 2, 0.0))

    labels = MF_NAMES[int(n_mf)]
    C, S = np.array(c_list), np.array(s_list)
    mf_rows = []
    for i, nm in enumerate(names):
        for m_, lab in enumerate(labels):
            mf_rows.append(dict(input=nm, mf=lab, centre_expert=proto_sys.c[i, m_], centre=C[:, i, m_].mean(),
                                centre_sd=C[:, i, m_].std(), sigma_expert=proto_sys.sigma[i, m_],
                                sigma=S[:, i, m_].mean()))
    mf = pd.DataFrame(mf_rows)
    fire = np.mean(fire_list, 0)
    dep = np.mean(dep_list, 0)
    rule_rows = []
    for r in range(proto_sys.R):
        ante = " AND ".join(f"{nm} is {labels[proto_sys.rule_idx[r, i]]}" for i, nm in enumerate(names))
        rule_rows.append(dict(rule=r + 1, antecedent=ante, expert_K=float(K_proto[r]),
                              output=float(np.mean([o[r] for o in out_list])),
                              constant_q=float(np.mean([q[r] for q in q_list])),
                              firing_area_pct=100 * fire[r], firing_deposits_pct=100 * dep[r],
                              lift=float(dep[r] / fire[r]) if fire[r] > 1e-12 else np.nan))
    rules = pd.DataFrame(rule_rows).sort_values("output", ascending=False).reset_index(drop=True)
    imp = np.array(imp_list)
    importance = pd.DataFrame(dict(input=names, auc_drop=imp.mean(0), auc_drop_sd=imp.std(0))) \
        .sort_values("auc_drop", ascending=False).reset_index(drop=True)
    bags = pd.DataFrame(rows)
    st = dict(deposits=N, non_deposits=n_neg, rules=proto_sys.R, inputs=len(names), bags=int(n_bags),
              sample_auc=float(bags.sample_auc.mean()), best_epoch=float(bags.best_epoch.mean()))
    in_text = "one per mineral-system process (expert process scores F)" if mode == "processes" else \
        "one per evidence layer (expert memberships μ, missing = 0.5)"
    summary = pd.DataFrame([
        ("Inputs", f"{len(names)} — {in_text}"),
        ("Membership functions per input", f"{n_mf} Gaussian ({', '.join(labels)}), expert start c = "
                                           f"{', '.join(f'{v:g}' for v in np.linspace(0, 1, int(n_mf)))}"),
        ("Rules (grid partition)", f"{proto_sys.R}"),
        ("Consequents", "first order f = p·x + q" if consequent == "linear" else "zero order f = q"),
        ("Knowledge prior κ", f"{kappa:g} (q → calibrated expert K at the rule prototype, p → 0)"),
        ("Hybrid learning", f"LSE consequents + Adam premises, ≤ {int(epochs)} epochs, learning rate {lr:g}"),
        ("Checking set (early stopping)", f"{100 * check_fraction:.0f} % of the samples" if check_fraction > 0
         else "off (last epoch)"),
        ("Best epoch (mean)", f"{st['best_epoch']:.1f}"),
        ("Training deposits (cells)", str(N)), ("Non-deposit samples per model", str(n_neg)),
        ("Models (bags)", str(int(n_bags))),
        ("Sample AUC (mean, in-sample)", f"{st['sample_auc']:.4f}"),
    ], columns=["item", "value"])
    say(1.0, "done")
    return NeuroFuzzyResult(prob, sd, K, F, names, mode, mf, rules, importance, bags, summary, st, msgs)


