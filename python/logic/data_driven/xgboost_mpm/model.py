"""XGBoost (extreme gradient boosting) for mineral prospectivity mapping — xgboost library, no GUI.

Theory (Friedman 2001; Chen & Guestrin 2016): an additive model of K regression trees on the log-odds scale,
F(x) = Σ_k η f_k(x), grown one after another; tree k fits the gradient g_i = p_i − y_i and Hessian h_i = p_i(1 − p_i) of
the logistic loss of the current model (second-order boosting) and its structure maximises the regularised gain

    gain = ½ [ G_L²/(H_L+λ) + G_R²/(H_R+λ) − (G_L+G_R)²/(H_L+H_R+λ) ] − γ,   leaf weight w = −G/(H+λ)

(G, H = sums of g, h in a node). Shrinkage η, row / column subsampling, λ, α and γ regularise the ensemble. Missing
values get a learned default direction at every split (sparsity-aware split finding). In MPM the training data are
deposit cells and random non-deposit cells away from known deposits, as for the random forest (Parsa 2021).

Workflow of :meth:`XGBoostModel.fit_predict` (per random non-deposit sample r = 1 … R):
    1. stratified k-fold cross-validation, every fold boosted up to ``n_rounds`` with the held-out AUC recorded per
       round; the number of trees K* = best round of the mean held-out AUC curve with early stopping (patience
       ``early_stopping`` rounds); held-out predictions at K* → CV AUC / confusion, permutation importance (sample 1);
    2. final booster on all samples with K* trees; probability for every study cell; optional TreeSHAP contributions
       (Lundberg et al. 2020) of every evidence layer to the log-odds.
Outputs: score = mean predicted probability over the samples (a favourability in [0, 1] that depends on the deposit /
non-deposit ratio — not a probability per unit cell), score_std over the samples (R ≥ 2), SHAP maps per layer.
Device: "auto" uses CUDA when an NVIDIA GPU is visible (probed once), otherwise the CPU.
"""
from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from i18n import N_
from logic.common.features import Feature, FeatureStack, study_area
from logic.common.training import auto_spacing, background_candidates, sample_positions

CLASS_WEIGHTS = ("none", "balanced")
CLASS_WEIGHT_LABELS = {"none": N_("none (sample ratio)"), "balanced": N_("balanced (deposits weighted up)")}
MISSING_LABELS = {"exclude": N_("exclude cells from study area"),
                  "impute": N_("median (continuous) / most frequent class (categorical)"),
                  "keep": N_("keep as missing — learned default direction")}
DEVICES = ("auto", "cpu", "cuda")
DEVICE_LABELS = {"auto": N_("auto (GPU if available)"), "cpu": N_("CPU"), "cuda": N_("GPU (CUDA)")}
MONOTONE = (0, 1, -1)
MONOTONE_LABELS = {0: N_("none"), 1: N_("increasing"), -1: N_("decreasing")}
CHUNK = 250_000
ProgressFn = Optional[Callable[[float, str], None]]
_CUDA: Optional[bool] = None


def cuda_available() -> bool:
    """True if xgboost can train on an NVIDIA GPU (probed once with a tiny booster)."""
    global _CUDA
    if _CUDA is None:
        try:
            import xgboost as xgb
            if not xgb.build_info().get("USE_CUDA"):
                _CUDA = False
            else:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    x = np.array([[0.0], [1.0], [0.0], [1.0]])
                    b = xgb.train({"device": "cuda", "tree_method": "hist", "objective": "binary:logistic",
                                   "verbosity": 0}, xgb.DMatrix(x, label=[0, 1, 0, 1]), 1)
                    dev = json.loads(b.save_config())["learner"]["generic_param"]["device"]
                _CUDA = dev.startswith("cuda")
        except Exception:  # noqa: BLE001 — no GPU build / driver: CPU
            _CUDA = False
    return _CUDA


@dataclass
class XGBResult:
    score: np.ma.MaskedArray                 # mean predicted probability over the samples
    score_std: Optional[np.ma.MaskedArray]   # s.d. over the samples (repeats ≥ 2)
    shap_maps: Dict[str, np.ma.MaskedArray]  # per evidence layer: mean SHAP contribution to the log-odds
    importance: pd.DataFrame                 # per evidence layer (SHAP, gain, permutation, direction)
    curve: pd.DataFrame                      # mean training / held-out AUC per boosting round (sample 1)
    sample_validation: pd.DataFrame
    summary: pd.DataFrame
    repeats: pd.DataFrame
    study_mask: np.ndarray
    n_deposits: int
    n_negatives: int
    stats: Dict[str, float]
    params: Dict[str, object]
    messages: List[str] = field(default_factory=list)


def _auc(pos: np.ndarray, neg: np.ndarray) -> float:
    from scipy.stats import rankdata
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    r = rankdata(np.r_[pos, neg])
    return float((r[: pos.size].sum() - pos.size * (pos.size + 1) / 2) / (pos.size * neg.size))


def best_round(curve: np.ndarray, patience: int) -> int:
    """Index of the best value of a (mean held-out AUC) curve with early stopping: stop when the curve has not
    improved for ``patience`` rounds (patience ≤ 0 = no early stopping, plain arg-max)."""
    best, best_i = -np.inf, 0
    for i, v in enumerate(curve):
        if v > best + 1e-12:
            best, best_i = v, i
        elif patience > 0 and i - best_i >= patience:
            break
    return best_i


class XGBoostModel:
    """XGBoost prospectivity model (see module doc).

    Booster: ``n_rounds`` (maximum trees), ``learning_rate`` η, ``max_depth``, ``min_child_weight``, ``subsample``,
    ``colsample`` (per tree), ``reg_lambda`` λ, ``reg_alpha`` α, ``min_split_loss`` γ, ``early_stopping`` (patience in
    rounds, 0 = use all rounds), ``class_weight`` ("none" | "balanced" = scale_pos_weight n₀/n₁), ``monotone``
    {layer: +1 | −1 | 0} for continuous layers; training samples as in the random forest; ``cv_folds`` ≥ 2
    (stratified); ``missing`` "exclude" | "impute" | "keep" (NaN, learned default direction); ``device`` "auto" |
    "cpu" | "cuda"; ``shap`` = compute SHAP maps of the study area.
    """

    def __init__(self, n_rounds: int = 500, learning_rate: float = 0.05, max_depth: int = 3,
                 min_child_weight: float = 1.0, subsample: float = 0.8, colsample: float = 0.8,
                 reg_lambda: float = 1.0, reg_alpha: float = 0.0, min_split_loss: float = 0.0,
                 early_stopping: int = 50, class_weight: str = "none", monotone: Optional[Dict[str, int]] = None,
                 neg_ratio: float = 1.0, min_distance: float = 0.0, min_spacing: float = -1.0,
                 res: Tuple[float, float] = (1.0, 1.0), repeats: int = 1, cv_folds: int = 5, missing: str = "exclude",
                 device: str = "auto", shap: bool = False, seed: int = 0, n_jobs: int = -1):
        if class_weight not in CLASS_WEIGHTS:
            raise ValueError(f"class_weight must be one of {CLASS_WEIGHTS}")
        if device not in DEVICES:
            raise ValueError(f"device must be one of {DEVICES}")
        if cv_folds < 2:
            raise ValueError("cv_folds must be ≥ 2")
        if not 0 < subsample <= 1 or not 0 < colsample <= 1:
            raise ValueError("subsample and colsample must be in (0, 1]")
        self.n_rounds, self.learning_rate = max(1, int(n_rounds)), float(learning_rate)
        self.max_depth, self.min_child_weight = max(1, int(max_depth)), float(min_child_weight)
        self.subsample, self.colsample = float(subsample), float(colsample)
        self.reg_lambda, self.reg_alpha, self.min_split_loss = float(reg_lambda), float(reg_alpha), float(min_split_loss)
        self.early_stopping, self.class_weight = max(0, int(early_stopping)), class_weight
        self.monotone = {k: int(np.sign(v)) for k, v in (monotone or {}).items()}
        self.neg_ratio, self.min_distance, self.min_spacing = float(neg_ratio), float(min_distance), float(min_spacing)
        self.res, self.repeats, self.cv_folds = tuple(res), max(1, int(repeats)), int(cv_folds)
        self.missing, self.device, self.shap = missing, device, bool(shap)
        self.seed, self.n_jobs = int(seed), int(n_jobs)

    # -------------------------------------------------------------- building blocks
    def resolved_device(self) -> Tuple[str, Optional[str]]:
        if self.device == "cpu":
            return "cpu", None
        if cuda_available():
            return "cuda", None
        if self.device == "cuda":
            return "cpu", "no usable NVIDIA GPU / CUDA driver found — XGBoost runs on the CPU"
        return "cpu", None

    def booster_params(self, stack: FeatureStack, y: np.ndarray, seed: int, device: str) -> dict:
        mono = [self.monotone.get(stack.layer_of[j], 0) if stack.groups[stack.layer_of[j]] == [j] else 0
                for j in range(stack.n_columns)]
        p = dict(objective="binary:logistic", eval_metric="auc", tree_method="hist", device=device,
                 eta=self.learning_rate, max_depth=self.max_depth, min_child_weight=self.min_child_weight,
                 subsample=self.subsample, colsample_bytree=self.colsample, reg_lambda=self.reg_lambda,
                 reg_alpha=self.reg_alpha, gamma=self.min_split_loss, seed=seed, verbosity=0,
                 nthread=self.n_jobs if self.n_jobs > 0 else 0)
        if any(mono):
            p["monotone_constraints"] = "(" + ",".join(str(m) for m in mono) + ")"
        if self.class_weight == "balanced":
            p["scale_pos_weight"] = float((y == 0).sum()) / max(1.0, float((y == 1).sum()))
        return p

    def folds(self, y: np.ndarray, seed: int) -> np.ndarray:
        k = min(self.cv_folds, int(min((y == 1).sum(), (y == 0).sum())))
        fold = np.empty(y.size, int)
        rng = np.random.default_rng(seed)
        for cls in (1, 0):
            ids = rng.permutation(np.flatnonzero(y == cls))
            fold[ids] = np.arange(ids.size) % k
        return fold

    def cross_validate(self, X, y, params, seed):
        """Boost every fold up to n_rounds; mean training / held-out AUC per round; K* by early stopping;
        held-out predictions at K* and the fold models."""
        import xgboost as xgb
        fold = self.folds(y, seed)
        ids = np.unique(fold)
        data = [(fold == f, xgb.DMatrix(X[fold != f], label=y[fold != f]), xgb.DMatrix(X[fold == f], label=y[fold == f]))
                for f in ids]
        boosters = [None] * len(ids)
        tr_curves = [[] for _ in ids]
        te_curves = [[] for _ in ids]
        done, chunk = 0, 25
        while done < self.n_rounds:          # all folds in lockstep, stop when the mean held-out curve stops improving
            k = min(chunk, self.n_rounds - done)
            for i, (te, dtr, dte) in enumerate(data):
                ev = {}
                boosters[i] = xgb.train(params, dtr, k, evals=[(dtr, "train"), (dte, "test")], evals_result=ev,
                                        verbose_eval=False, xgb_model=boosters[i])
                tr_curves[i] += ev["train"]["auc"]
                te_curves[i] += ev["test"]["auc"]
            done += k
            mean_te = np.nanmean(np.array(te_curves, float), 0)
            if self.early_stopping and done - 1 - best_round(mean_te, self.early_stopping) >= self.early_stopping:
                break
        models = [(te, b) for (te, _, _), b in zip(data, boosters)]
        tr, tes = np.nanmean(np.array(tr_curves, float), 0), np.nanmean(np.array(te_curves, float), 0)
        k_best = best_round(tes, self.early_stopping) + 1
        held = np.empty(y.size)
        for te, b in models:
            held[te] = b.predict(xgb.DMatrix(X[te]), iteration_range=(0, k_best))
        curve = pd.DataFrame(dict(round=np.arange(1, tes.size + 1), train_auc=tr, cv_auc=tes))
        return k_best, held, curve, models, fold

    # -------------------------------------------------------------- full model
    def fit_predict(self, features: Sequence[Feature], dep_rows, dep_cols, study_mask: Optional[np.ndarray] = None,
                    known_deposits: Optional[Tuple[np.ndarray, np.ndarray]] = None,
                    progress: ProgressFn = None) -> XGBResult:
        import xgboost as xgb
        say = progress or (lambda f, m: None)
        say(0.02, N_("encoding evidence"))
        base = study_area(features, study_mask, self.missing)
        stack = FeatureStack(features, base, self.missing)
        msgs = list(stack.notes)
        device, dev_msg = self.resolved_device()
        if dev_msg:
            msgs.append(dev_msg)
        dep_pos = np.unique(stack.positions(dep_rows, dep_cols))
        n_d = int(dep_pos.size)
        if n_d < 3:
            raise ValueError(f"Only {n_d} training deposit cell(s) inside the study area — need at least 3.")
        m = int(round(self.neg_ratio * n_d))
        if m < 2:
            raise ValueError("The number of non-deposit samples must be at least 2.")
        bad = [k for k, v in self.monotone.items() if v and k in stack.groups and len(stack.groups[k]) > 1]
        if bad:
            msgs.append("monotone constraints apply to continuous layers only — ignored for " + ", ".join(bad))
        kr, kc = known_deposits if known_deposits is not None else (dep_rows, dep_cols)
        cand = background_candidates(base, kr, kc, self.min_distance, self.res)
        cand_pos = np.searchsorted(stack.cells, np.flatnonzero(cand))
        spacing = auto_spacing(cand_pos.size, m, self.res) if self.min_spacing < 0 else self.min_spacing
        cand_cells = stack.cells[cand_pos]
        width = base.shape[1]
        if n_d < 20:
            msgs.append(f"only {n_d} training deposits: the number of trees and the importances are uncertain — "
                        "keep the trees shallow and use several repeats")

        n = stack.n_cells
        p_sum, p_sq = np.zeros(n), np.zeros(n)
        shap_sum = np.zeros((n, len(stack.groups))) if self.shap else None
        rep_rows, first = [], None
        for r in range(self.repeats):
            frac = r / self.repeats
            step = 0.85 / self.repeats
            rng = np.random.default_rng(self.seed + 1000 * r)
            neg = sample_positions(cand_pos, cand_cells // width, cand_cells % width, m, rng, spacing, self.res)
            idx = np.r_[dep_pos, neg]
            X = stack.matrix(idx)
            y = np.r_[np.ones(n_d, int), np.zeros(neg.size, int)]
            params = self.booster_params(stack, y, self.seed + r, device)
            say(0.05 + 0.85 * frac, N_("cross-validating the number of trees"))
            k_best, held, curve, fold_models, fold = self.cross_validate(X, y, params, self.seed + 7 + r)
            say(0.05 + 0.85 * frac + 0.5 * step, N_("training the booster"))
            bst = xgb.train(params, xgb.DMatrix(X, label=y), k_best)
            say(0.05 + 0.85 * frac + 0.6 * step, N_("predicting the study area"))
            for s in range(0, n, CHUNK):
                ch = np.arange(s, min(n, s + CHUNK))
                dm = xgb.DMatrix(stack.matrix(ch))
                prob = bst.predict(dm)
                p_sum[ch] += prob
                p_sq[ch] += prob * prob
                if self.shap:
                    c = bst.predict(dm, pred_contribs=True)[:, :-1]
                    for li, cols in enumerate(stack.groups.values()):
                        shap_sum[ch, li] += c[:, cols].sum(1)
            p_tr = bst.predict(xgb.DMatrix(X))
            rep_rows.append(dict(repeat=r + 1, seed=self.seed + 1000 * r, deposits=n_d, non_deposits=int(neg.size),
                                 trees=k_best, training_auc=_auc(p_tr[y == 1], p_tr[y == 0]),
                                 cv_auc=_auc(held[y == 1], held[y == 0])))
            if first is None:
                first = dict(bst=bst, X=X, y=y, held=held, p_tr=p_tr, curve=curve, fold_models=fold_models,
                             fold=fold, k_best=k_best)
        if spacing > 0:
            msgs.append(f"non-deposit samples are at least {spacing:,.0f} map units apart"
                        + (" (auto: ½·√(area / n))" if self.min_spacing < 0 else ""))
        R = self.repeats
        if R > 1:
            msgs.append(f"{R} random non-deposit samples: one booster each, scores averaged; importance, boosting "
                        "curve and validation refer to sample 1")
        score = p_sum / R
        std = np.sqrt(np.maximum(p_sq / R - score ** 2, 0.0)) if R > 1 else None

        say(0.92, N_("variable importance"))
        imp = self._importance(first, stack)
        val = self._validation(first)

        def grid(v):
            out = np.full(base.shape, 0.0)
            out.ravel()[stack.cells] = v
            return np.ma.array(out, mask=~base)

        shap_maps = {} if shap_sum is None else {k: grid(shap_sum[:, i] / R) for i, k in enumerate(stack.groups)}
        rep = pd.DataFrame(rep_rows)
        st = dict(cv_auc=float(rep.cv_auc.iloc[0]), training_auc=float(rep.training_auc.iloc[0]),
                  trees=int(first["k_best"]), trees_mean=float(rep.trees.mean()), spacing=spacing, device=device)
        mono_txt = ", ".join(f"{k} {'↑' if v > 0 else '↓'}" for k, v in self.monotone.items() if v) or "none"
        summary = pd.DataFrame([
            ("Training deposits (cells)", str(n_d)), ("Non-deposit samples per draw", str(m)),
            ("Random draws (repeats)", str(R)),
            ("Trees K* (sample 1 / mean)", f"{st['trees']} / {st['trees_mean']:.0f} of max. {self.n_rounds}"
             + (f" (early stopping, patience {self.early_stopping})" if self.early_stopping else "")),
            ("Learning rate η / max. depth", f"{self.learning_rate:g} / {self.max_depth}"),
            ("Min. child weight / min. split loss γ", f"{self.min_child_weight:g} / {self.min_split_loss:g}"),
            ("Row / column subsample", f"{self.subsample:g} / {self.colsample:g}"),
            ("L2 λ / L1 α", f"{self.reg_lambda:g} / {self.reg_alpha:g}"),
            ("Class weight", self.class_weight), ("Monotone constraints", mono_txt),
            ("Missing evidence", self.missing), ("Device", device),
            ("Evidence columns", ", ".join(stack.names)[:300]), ("Study area (cells)", f"{n:,}"),
            ("Min. distance of non-deposits from deposits", f"{self.min_distance:g}"),
            ("Min. spacing between non-deposits", f"{spacing:,.0f}"),
            ("Training AUC (samples)", f"{st['training_auc']:.4f}"),
            ("Cross-validated AUC (samples)", f"{st['cv_auc']:.4f}"),
            ("Mean score in the study area", f"{score.mean():.4f}"),
        ], columns=["item", "value"])
        params = dict(n_rounds=self.n_rounds, learning_rate=self.learning_rate, max_depth=self.max_depth,
                      min_child_weight=self.min_child_weight, subsample=self.subsample, colsample=self.colsample,
                      reg_lambda=self.reg_lambda, reg_alpha=self.reg_alpha, min_split_loss=self.min_split_loss,
                      early_stopping=self.early_stopping, class_weight=self.class_weight, monotone=self.monotone,
                      neg_ratio=self.neg_ratio, min_distance=self.min_distance, min_spacing=spacing, repeats=R,
                      cv_folds=self.cv_folds, missing=self.missing, device=device, seed=self.seed)
        say(1.0, N_("done"))
        return XGBResult(grid(score), None if std is None else grid(std), shap_maps, imp, first["curve"], val, summary,
                         rep, base, n_d, m, st, params, msgs)

    # -------------------------------------------------------------- importance / validation
    def _importance(self, first, stack: FeatureStack) -> pd.DataFrame:
        """Per evidence layer: mean |SHAP| of the training samples (share %), effect direction (Spearman correlation
        of value and SHAP, continuous layers), total gain (share %) and permutation importance (drop of the CV AUC
        when the layer is shuffled in the held-out folds; 5 shuffles)."""
        import xgboost as xgb
        from scipy.stats import spearmanr
        bst, X, y = first["bst"], first["X"], first["y"]
        contrib = bst.predict(xgb.DMatrix(X), pred_contribs=True)[:, :-1]
        gain = bst.get_score(importance_type="total_gain")            # keys f0, f1, … (unused columns absent)
        gcol = np.array([gain.get(f"f{j}", 0.0) for j in range(stack.n_columns)])
        k_best, models = first["k_best"], first["fold_models"]
        held = first["held"]
        base_auc = _auc(held[y == 1], held[y == 0])
        rng = np.random.default_rng(self.seed + 17)
        rows = []
        for layer, cols in stack.groups.items():
            phi = contrib[:, cols].sum(1)
            drops = []
            for _ in range(5):
                h = np.empty(y.size)
                for te, b in models:
                    Xt = X[te].copy()
                    Xt[:, cols] = Xt[rng.permutation(Xt.shape[0])][:, cols]
                    h[te] = b.predict(xgb.DMatrix(Xt), iteration_range=(0, k_best))
                drops.append(base_auc - _auc(h[y == 1], h[y == 0]))
            direction = None
            if len(cols) == 1:
                v = X[:, cols[0]]
                ok = np.isfinite(v)
                if ok.sum() > 2 and np.ptp(phi[ok]) > 0:
                    direction = float(spearmanr(v[ok], phi[ok]).correlation)
            rows.append(dict(layer=layer, columns=len(cols), mean_abs_shap=float(np.abs(phi).mean()),
                             shap_deposits=float(phi[y == 1].mean()), direction=direction,
                             gain=float(gcol[cols].sum()), permutation_auc_drop=float(np.mean(drops)),
                             permutation_sd=float(np.std(drops))))
        tab = pd.DataFrame(rows)
        tab["shap_pct"] = 100 * tab.mean_abs_shap / max(tab.mean_abs_shap.sum(), 1e-300)
        tab["gain_pct"] = 100 * tab.gain / max(tab.gain.sum(), 1e-300)
        tab = tab.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
        tab.insert(0, "rank", np.arange(1, len(tab) + 1))
        return tab[["rank", "layer", "columns", "mean_abs_shap", "shap_pct", "shap_deposits", "direction", "gain_pct",
                    "permutation_auc_drop", "permutation_sd"]]

    @staticmethod
    def _validation(first) -> pd.DataFrame:
        y, held, p_tr = first["y"], first["held"], first["p_tr"]
        pos, neg = y == 1, y == 0

        def conf(p):
            pr = p >= 0.5
            return (int((neg & ~pr).sum()), int((neg & pr).sum()), int((pos & ~pr).sum()), int((pos & pr).sum()))
        tn, fp, fn, tp = conf(p_tr)
        ctn, cfp, cfn, ctp = conf(held)
        k = int(np.unique(first["fold"]).size)
        accs = [float(((held[first["fold"] == f] >= 0.5) == pos[first["fold"] == f]).mean())
                for f in np.unique(first["fold"])]
        items = [("Samples (deposits / non-deposits)", f"{int(pos.sum())} / {int(neg.sum())}"),
                 ("Training confusion: true non-deposit → predicted non-deposit / deposit", f"{tn} / {fp}"),
                 ("Training confusion: true deposit → predicted non-deposit / deposit", f"{fn} / {tp}"),
                 ("Training accuracy", f"{(tn + tp) / y.size:.4f}"),
                 ("Training AUC", f"{_auc(p_tr[pos], p_tr[neg]):.4f}"),
                 ("Cross-validation", f"{k}-fold (stratified)"),
                 ("Trees K* (best mean held-out AUC)", str(first["k_best"])),
                 ("CV confusion: true non-deposit → predicted non-deposit / deposit", f"{ctn} / {cfp}"),
                 ("CV confusion: true deposit → predicted non-deposit / deposit", f"{cfn} / {ctp}"),
                 ("Cross-validated accuracy (mean of folds)", f"{np.mean(accs):.4f}"),
                 ("Cross-validated AUC", f"{_auc(held[pos], held[neg]):.4f}")]
        return pd.DataFrame(items, columns=["item", "value"])
