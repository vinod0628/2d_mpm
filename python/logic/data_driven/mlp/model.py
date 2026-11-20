"""Artificial neural network (multilayer perceptron, MLP) for mineral prospectivity mapping — scikit-learn, no GUI.

A feed-forward network with ``hidden`` fully connected layers (Rumelhart et al. 1986): h_l = g(W_l h_{l−1} + b_l),
output P(x) = σ(w_oᵀ h_L + b_o), trained by back-propagation with the Adam optimiser (Kingma & Ba 2015) on the binary
cross-entropy plus an L2 penalty α‖W‖²/2. The architecture and defaults follow eis_toolkit's ``train_MLP_classifier``
(dense layers, ReLU, Adam, learning rate 0.001, batch size 32, binary cross-entropy, early stopping), implemented
with scikit-learn's ``MLPClassifier`` instead of TensorFlow. Training data as for the other machine-learning methods:
deposit cells (1) and random non-deposit cells (0) away from known deposits (Carranza et al. 2008); continuous
evidence is z-scored, categorical evidence one-hot encoded. In MPM: Brown et al. (2000), Porwal et al. (2003).

Number of epochs: the samples are split into k stratified folds; one network per fold is trained epoch by epoch
(``partial_fit``) in lockstep and the mean held-out AUC is recorded after every epoch; training stops when it has not
improved for ``patience`` epochs and E* = best epoch (scikit-learn's own early stopping uses accuracy, which is
uninformative with imbalanced classes). The final networks are trained for E* epochs on all samples.

Ensemble: ``networks`` networks with different initial weights per random non-deposit sample, R samples →
score = mean output of the R·n networks (a favourability in [0, 1] that depends on the deposit / non-deposit ratio —
not a probability per unit cell), score_std = s.d. of the network outputs (model uncertainty).
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from i18n import N_
from logic.common.features import Feature, FeatureStack, study_area
from logic.common.training import auto_spacing, background_candidates, sample_positions

ACTIVATIONS = ("relu", "tanh", "logistic")
ACTIVATION_LABELS = {"relu": N_("ReLU"), "tanh": N_("tanh"), "logistic": N_("sigmoid (logistic)")}
CHUNK = 250_000
ProgressFn = Optional[Callable[[float, str], None]]


def parse_hidden(text) -> Tuple[int, ...]:
    """'32, 16' -> (32, 16)."""
    if isinstance(text, (tuple, list)):
        vals = tuple(int(v) for v in text)
    else:
        vals = tuple(int(v) for v in str(text).replace(";", ",").split(",") if v.strip())
    if not vals or any(v < 1 for v in vals):
        raise ValueError("hidden layers: give one or more positive neuron counts, e.g. '32, 16'")
    return vals


def _auc(pos: np.ndarray, neg: np.ndarray) -> float:
    from scipy.stats import rankdata
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    r = rankdata(np.r_[pos, neg])
    return float((r[: pos.size].sum() - pos.size * (pos.size + 1) / 2) / (pos.size * neg.size))


@dataclass
class MLPResult:
    score: np.ma.MaskedArray
    score_std: np.ma.MaskedArray          # s.d. of the outputs of all networks
    importance: pd.DataFrame
    curve: pd.DataFrame                   # mean training / held-out AUC per epoch (sample 1)
    sample_validation: pd.DataFrame
    summary: pd.DataFrame
    repeats: pd.DataFrame
    study_mask: np.ndarray
    n_deposits: int
    n_negatives: int
    stats: Dict[str, float]
    params: Dict[str, object]
    messages: List[str] = field(default_factory=list)


class MLPModel:
    """MLP prospectivity model (see module doc)."""

    def __init__(self, hidden=(32, 16), activation: str = "relu", learning_rate: float = 0.001, alpha: float = 1e-4,
                 batch_size: int = 32, max_epochs: int = 500, patience: int = 20, networks: int = 5,
                 neg_ratio: float = 1.0, min_distance: float = 0.0, min_spacing: float = -1.0,
                 res: Tuple[float, float] = (1.0, 1.0), repeats: int = 1, cv_folds: int = 5, missing: str = "exclude",
                 seed: int = 0):
        if activation not in ACTIVATIONS:
            raise ValueError(f"activation must be one of {ACTIVATIONS}")
        if cv_folds < 2:
            raise ValueError("cv_folds must be ≥ 2")
        self.hidden, self.activation = parse_hidden(hidden), activation
        self.learning_rate, self.alpha = float(learning_rate), float(alpha)
        self.batch_size, self.max_epochs = max(1, int(batch_size)), max(1, int(max_epochs))
        self.patience, self.networks = max(0, int(patience)), max(1, int(networks))
        self.neg_ratio, self.min_distance, self.min_spacing = float(neg_ratio), float(min_distance), float(min_spacing)
        self.res, self.repeats, self.cv_folds = tuple(res), max(1, int(repeats)), int(cv_folds)
        self.missing, self.seed = missing, int(seed)

    def net(self, seed: int):
        from sklearn.neural_network import MLPClassifier
        return MLPClassifier(hidden_layer_sizes=self.hidden, activation=self.activation, solver="adam",
                             alpha=self.alpha, batch_size=self.batch_size, learning_rate_init=self.learning_rate,
                             max_iter=1, shuffle=True, random_state=seed)

    @staticmethod
    def _p1(net, X):
        return net.predict_proba(X)[:, list(net.classes_).index(1)]

    def train(self, X, y, epochs: int, seed: int):
        net = self.net(seed)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for _ in range(epochs):
                net.partial_fit(X, y, classes=np.array([0, 1]))
        return net

    def folds(self, y, seed):
        k = min(self.cv_folds, int(min((y == 1).sum(), (y == 0).sum())))
        fold = np.empty(y.size, int)
        rng = np.random.default_rng(seed)
        for cls in (1, 0):
            ids = rng.permutation(np.flatnonzero(y == cls))
            fold[ids] = np.arange(ids.size) % k
        return fold

    def cross_validate(self, X, y, seed):
        """Fold networks trained epoch by epoch in lockstep; mean held-out AUC curve; E* by early stopping;
        held-out predictions of the networks at E*."""
        fold = self.folds(y, seed)
        ids = np.unique(fold)
        nets = [self.net(seed + 31 * int(f)) for f in ids]
        tr_c, te_c = [], []
        best, best_e, best_pred = -np.inf, 0, None
        snap = None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for e in range(1, self.max_epochs + 1):
                held = np.empty(y.size)
                tr_auc = []
                for net, f in zip(nets, ids):
                    te = fold == f
                    net.partial_fit(X[~te], y[~te], classes=np.array([0, 1]))
                    held[te] = self._p1(net, X[te])
                    p = self._p1(net, X[~te])
                    tr_auc.append(_auc(p[y[~te] == 1], p[y[~te] == 0]))
                a = _auc(held[y == 1], held[y == 0])
                tr_c.append(float(np.mean(tr_auc)))
                te_c.append(a)
                if a > best + 1e-6:
                    import copy
                    best, best_e, best_pred = a, e, held.copy()
                    snap = [copy.deepcopy(n) for n in nets]
                elif self.patience and e - best_e >= self.patience:
                    break
        curve = pd.DataFrame(dict(epoch=np.arange(1, len(te_c) + 1), train_auc=tr_c, cv_auc=te_c))
        models = [(fold == f, n) for f, n in zip(ids, snap)]
        return best_e, best_pred, curve, models, fold

    # -------------------------------------------------------------- full model
    def fit_predict(self, features: Sequence[Feature], dep_rows, dep_cols, study_mask: Optional[np.ndarray] = None,
                    known_deposits: Optional[Tuple[np.ndarray, np.ndarray]] = None,
                    progress: ProgressFn = None) -> MLPResult:
        say = progress or (lambda f, m: None)
        say(0.02, N_("encoding evidence"))
        base = study_area(features, study_mask, self.missing)
        stack = FeatureStack(features, base, self.missing)
        msgs = list(stack.notes)
        dep_pos = np.unique(stack.positions(dep_rows, dep_cols))
        n_d = int(dep_pos.size)
        if n_d < 3:
            raise ValueError(f"Only {n_d} training deposit cell(s) inside the study area — need at least 3.")
        m = int(round(self.neg_ratio * n_d))
        if m < 2:
            raise ValueError("The number of non-deposit samples must be at least 2.")
        kr, kc = known_deposits if known_deposits is not None else (dep_rows, dep_cols)
        cand = background_candidates(base, kr, kc, self.min_distance, self.res)
        cand_pos = np.searchsorted(stack.cells, np.flatnonzero(cand))
        spacing = auto_spacing(cand_pos.size, m, self.res) if self.min_spacing < 0 else self.min_spacing
        cand_cells = stack.cells[cand_pos]
        width = base.shape[1]
        p = stack.n_columns
        shift = np.array([float(stack.column(j).mean()) for j in range(p)])
        scale = np.array([float(stack.column(j).std()) or 1.0 for j in range(p)])
        n_weights = sum(a * b + b for a, b in zip((p,) + self.hidden, self.hidden + (1,)))
        if n_weights > 20 * (n_d + m):
            msgs.append(f"{n_weights} network weights for {n_d + m} training samples: the network is large for the "
                        "data — consider fewer neurons or a larger L2 penalty")

        n = stack.n_cells
        s1, s2, cnt = np.zeros(n), np.zeros(n), 0
        rep_rows, first = [], None
        for r in range(self.repeats):
            frac = r / self.repeats
            step = 0.85 / self.repeats
            rng = np.random.default_rng(self.seed + 1000 * r)
            neg = sample_positions(cand_pos, cand_cells // width, cand_cells % width, m, rng, spacing, self.res)
            idx = np.r_[dep_pos, neg]
            X = (stack.matrix(idx) - shift) / scale
            y = np.r_[np.ones(n_d, int), np.zeros(neg.size, int)]
            say(0.05 + 0.85 * frac, N_("cross-validating the number of epochs"))
            e_best, held, curve, fold_models, fold = self.cross_validate(X, y, self.seed + 7 + r)
            say(0.05 + 0.85 * frac + 0.4 * step, N_("training the networks"))
            nets = [self.train(X, y, e_best, self.seed + 1000 * r + 101 * k) for k in range(self.networks)]
            say(0.05 + 0.85 * frac + 0.7 * step, N_("predicting the study area"))
            for s in range(0, n, CHUNK):
                ch = np.arange(s, min(n, s + CHUNK))
                Xc = (stack.matrix(ch) - shift) / scale
                for net in nets:
                    pr = self._p1(net, Xc)
                    s1[ch] += pr
                    s2[ch] += pr * pr
            cnt += len(nets)
            p_tr = np.mean([self._p1(net, X) for net in nets], 0)
            rep_rows.append(dict(repeat=r + 1, seed=self.seed + 1000 * r, deposits=n_d, non_deposits=int(neg.size),
                                 epochs=e_best, networks=len(nets), training_auc=_auc(p_tr[y == 1], p_tr[y == 0]),
                                 cv_auc=_auc(held[y == 1], held[y == 0])))
            if first is None:
                first = dict(X=X, y=y, held=held, p_tr=p_tr, curve=curve, fold_models=fold_models, fold=fold,
                             e_best=e_best)
        if spacing > 0:
            msgs.append(f"non-deposit samples are at least {spacing:,.0f} map units apart"
                        + (" (auto: ½·√(area / n))" if self.min_spacing < 0 else ""))
        R = self.repeats
        if R > 1:
            msgs.append(f"{R} random non-deposit samples × {self.networks} networks averaged; importance, learning "
                        "curve and validation refer to sample 1")
        score = s1 / cnt
        std = np.sqrt(np.maximum(s2 / cnt - score ** 2, 0.0))

        say(0.92, N_("variable importance"))
        imp = self._importance(first, stack)
        val = self._validation(first)

        def grid(v):
            out = np.zeros(base.shape)
            out.ravel()[stack.cells] = v
            return np.ma.array(out, mask=~base)

        rep = pd.DataFrame(rep_rows)
        st = dict(cv_auc=float(rep.cv_auc.iloc[0]), training_auc=float(rep.training_auc.iloc[0]),
                  epochs=int(first["e_best"]), epochs_mean=float(rep.epochs.mean()), weights=int(n_weights),
                  networks=int(cnt), spacing=spacing)
        summary = pd.DataFrame([
            ("Training deposits (cells)", str(n_d)), ("Non-deposit samples per draw", str(m)),
            ("Random draws × networks", f"{R} × {self.networks} = {cnt}"),
            ("Architecture", f"{p} inputs → " + " → ".join(str(h) for h in self.hidden) + " → 1 (sigmoid)"),
            ("Weights per network", str(n_weights)), ("Activation", self.activation),
            ("Optimiser", f"Adam, learning rate {self.learning_rate:g}, batch {self.batch_size}"),
            ("L2 penalty α", f"{self.alpha:g}"),
            ("Epochs E* (sample 1 / mean)", f"{st['epochs']} / {st['epochs_mean']:.0f} of max. {self.max_epochs}"
             + (f" (patience {self.patience})" if self.patience else "")),
            ("Evidence columns", ", ".join(stack.names)[:300]), ("Study area (cells)", f"{n:,}"),
            ("Min. distance of non-deposits from deposits", f"{self.min_distance:g}"),
            ("Min. spacing between non-deposits", f"{spacing:,.0f}"),
            ("Training AUC (samples)", f"{st['training_auc']:.4f}"),
            ("Cross-validated AUC (samples)", f"{st['cv_auc']:.4f}"),
            ("Mean score in the study area", f"{score.mean():.4f}"),
        ], columns=["item", "value"])
        params = dict(hidden=list(self.hidden), activation=self.activation, learning_rate=self.learning_rate,
                      alpha=self.alpha, batch_size=self.batch_size, max_epochs=self.max_epochs, patience=self.patience,
                      networks=self.networks, neg_ratio=self.neg_ratio, min_distance=self.min_distance,
                      min_spacing=spacing, repeats=R, cv_folds=self.cv_folds, missing=self.missing, seed=self.seed)
        say(1.0, N_("done"))
        return MLPResult(grid(score), grid(std), imp, first["curve"], val, summary, rep, base, n_d, m, st, params, msgs)

    # -------------------------------------------------------------- importance / validation
    def _importance(self, first, stack: FeatureStack) -> pd.DataFrame:
        X, y, held = first["X"], first["y"], first["held"]
        base_auc = _auc(held[y == 1], held[y == 0])
        rng = np.random.default_rng(self.seed + 17)
        rows = []
        for layer, cols in stack.groups.items():
            drops = []
            for _ in range(5):
                h = np.empty(y.size)
                for te, net in first["fold_models"]:
                    Xt = X[te].copy()
                    Xt[:, cols] = Xt[rng.permutation(Xt.shape[0])][:, cols]
                    h[te] = self._p1(net, Xt)
                drops.append(base_auc - _auc(h[y == 1], h[y == 0]))
            rows.append(dict(layer=layer, columns=len(cols), permutation_auc_drop=float(np.mean(drops)),
                             permutation_sd=float(np.std(drops))))
        tab = pd.DataFrame(rows).sort_values("permutation_auc_drop", ascending=False).reset_index(drop=True)
        tab.insert(0, "rank", np.arange(1, len(tab) + 1))
        return tab

    @staticmethod
    def _validation(first) -> pd.DataFrame:
        y, held, p_tr = first["y"], first["held"], first["p_tr"]
        pos, neg = y == 1, y == 0

        def conf(p):
            pr = p >= 0.5
            return (int((neg & ~pr).sum()), int((neg & pr).sum()), int((pos & ~pr).sum()), int((pos & pr).sum()))
        tn, fp, fn, tp = conf(p_tr)
        ctn, cfp, cfn, ctp = conf(held)
        fold = first["fold"]
        accs = [float(((held[fold == f] >= 0.5) == pos[fold == f]).mean()) for f in np.unique(fold)]
        items = [("Samples (deposits / non-deposits)", f"{int(pos.sum())} / {int(neg.sum())}"),
                 ("Training confusion: true non-deposit → predicted non-deposit / deposit", f"{tn} / {fp}"),
                 ("Training confusion: true deposit → predicted non-deposit / deposit", f"{fn} / {tp}"),
                 ("Training accuracy", f"{(tn + tp) / y.size:.4f}"),
                 ("Training AUC", f"{_auc(p_tr[pos], p_tr[neg]):.4f}"),
                 ("Cross-validation", f"{int(np.unique(fold).size)}-fold (stratified)"),
                 ("Epochs E* (best mean held-out AUC)", str(first["e_best"])),
                 ("CV confusion: true non-deposit → predicted non-deposit / deposit", f"{ctn} / {cfp}"),
                 ("CV confusion: true deposit → predicted non-deposit / deposit", f"{cfn} / {ctp}"),
                 ("Cross-validated accuracy (mean of folds)", f"{np.mean(accs):.4f}"),
                 ("Cross-validated AUC", f"{_auc(held[pos], held[neg]):.4f}")]
        return pd.DataFrame(items, columns=["item", "value"])
