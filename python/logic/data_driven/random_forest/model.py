"""Random forest (RF) for mineral prospectivity mapping — scikit-learn RandomForestClassifier, no GUI.

Breiman (2001): an ensemble of classification trees, each grown on a bootstrap sample of the training data, with a
random subset of ``max_features`` evidence columns tried at every split; the prediction is the share of trees that
vote "deposit". In MPM (Carranza & Laborte 2015; Rodriguez-Galiano et al. 2015; McKay & Harris 2016) the training data
are the deposit cells (class 1) and randomly drawn non-deposit cells (class 0) away from known deposits
(Carranza et al. 2008); continuous evidence is used as values, categorical evidence as one-hot columns.

Outputs
    score   mean tree vote P̄(x) = (1/T) Σ_t p_t(x) ∈ [0, 1] — a prospectivity (favourability) score; it depends on the
            deposit / non-deposit ratio of the sample and is not a probability per unit cell
    std     standard deviation of the tree votes s(x) = √((1/T) Σ_t (p_t − P̄)²) — model uncertainty
    With R random non-deposit samples (Zuo & Wang 2020) the R·T trees are pooled.

Validation (training samples; the study-area ROC/AUC is computed by the shared evaluation)
    out-of-bag (OOB): every sample is predicted only by the ~37 % of trees whose bootstrap did not contain it —
    an internal test set (Breiman 2001): OOB AUC, OOB confusion matrix (score ≥ 0.5) and accuracy;
    optional stratified k-fold / leave-one-out cross-validation (as in ArcSDM's RF Train tool).
Variable importance
    impurity (MDI): mean decrease of Gini impurity, normalised to 100 % (Breiman 2001) — biased towards layers
    with many distinct values (Strobl et al. 2007); permutation: decrease of the OOB AUC when the values of one
    evidence layer are shuffled (all its one-hot columns together; each tree predicts only its OOB samples),
    mean ± s.d. of 5 shuffles (Breiman 2001).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from i18n import N_
from logic.common.features import Feature, FeatureStack, study_area
from logic.common.training import auto_spacing, background_candidates, sample_positions

MAX_FEATURES = ("sqrt", "log2", "all", "third")
MAX_FEATURE_LABELS = {"sqrt": N_("√p (classification default)"), "log2": N_("log2(p)"),
                      "third": N_("p / 3 (regression default)"), "all": N_("all p (bagging)")}
CLASS_WEIGHTS = ("none", "balanced")
CLASS_WEIGHT_LABELS = {"none": N_("none (sample ratio)"), "balanced": N_("balanced (deposits weighted up)")}
LOO_MAX = 150             # leave-one-out = one forest per sample: above this, 10-fold
ProgressFn = Optional[Callable[[float, str], None]]


@dataclass
class RFResult:
    score: np.ma.MaskedArray                 # mean tree vote
    score_std: np.ma.MaskedArray             # std of the tree votes
    importance: pd.DataFrame                 # per evidence layer
    importance_columns: pd.DataFrame         # per feature column (one-hot classes)
    sample_validation: pd.DataFrame          # OOB / cross-validation report
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


class RandomForestModel:
    """Random-forest prospectivity model (see module doc).

    n_trees, max_features ("sqrt" | "log2" | "third" | "all"), min_samples_leaf, max_depth (0 = unlimited),
    class_weight ("none" | "balanced"); training samples: ``neg_ratio`` non-deposit cells per deposit at least
    ``min_distance`` map units from every known deposit and ``min_spacing`` apart (< 0 = auto, 0 = none), ``repeats``
    random draws; ``cv_folds``: 0 = none (OOB only), -1 = leave-one-out (≤ 150 samples, else 10-fold), k ≥ 2 = k-fold.
    """

    def __init__(self, n_trees: int = 500, max_features: str = "sqrt", min_samples_leaf: int = 1, max_depth: int = 0,
                 class_weight: str = "none", neg_ratio: float = 1.0, min_distance: float = 0.0,
                 min_spacing: float = -1.0, res: Tuple[float, float] = (1.0, 1.0), repeats: int = 1,
                 cv_folds: int = 0, missing: str = "exclude", seed: int = 0, n_jobs: int = -1):
        if max_features not in MAX_FEATURES:
            raise ValueError(f"max_features must be one of {MAX_FEATURES}")
        if class_weight not in CLASS_WEIGHTS:
            raise ValueError(f"class_weight must be one of {CLASS_WEIGHTS}")
        self.n_trees, self.max_features = max(1, int(n_trees)), max_features
        self.min_samples_leaf, self.max_depth = max(1, int(min_samples_leaf)), max(0, int(max_depth))
        self.class_weight = class_weight
        self.neg_ratio, self.min_distance, self.min_spacing = float(neg_ratio), float(min_distance), float(min_spacing)
        self.res, self.repeats, self.cv_folds = tuple(res), max(1, int(repeats)), int(cv_folds)
        self.missing, self.seed, self.n_jobs = missing, int(seed), int(n_jobs)

    # -------------------------------------------------------------- forest
    def _forest(self, seed: int, n_features: int):
        from sklearn.ensemble import RandomForestClassifier
        mf = {"sqrt": "sqrt", "log2": "log2", "all": None, "third": max(1, n_features // 3)}[self.max_features]
        return RandomForestClassifier(
            n_estimators=self.n_trees, max_features=mf, min_samples_leaf=self.min_samples_leaf,
            max_depth=self.max_depth or None, class_weight="balanced" if self.class_weight == "balanced" else None,
            bootstrap=True, oob_score=True, n_jobs=self.n_jobs, random_state=seed)

    @staticmethod
    def _tree_votes(rf, X: np.ndarray, n_jobs: int):
        """Σ p_t and Σ p_t² over the trees (p_t = deposit probability of tree t) — mean and spread in one pass."""
        from joblib import Parallel, delayed
        k = list(rf.classes_).index(1)
        trees = rf.estimators_
        parts = np.array_split(np.arange(len(trees)), max(1, min(len(trees), n_jobs if n_jobs > 0 else 4)))

        def run(ids):
            s = np.zeros(X.shape[0])
            q = np.zeros(X.shape[0])
            for i in ids:
                p = trees[i].predict_proba(X, check_input=False)[:, k]
                s += p
                q += p * p
            return s, q
        out = Parallel(n_jobs=len(parts), prefer="threads")(delayed(run)(ids) for ids in parts if len(ids))
        return sum(o[0] for o in out), sum(o[1] for o in out)

    # -------------------------------------------------------------- full model
    def fit_predict(self, features: Sequence[Feature], dep_rows, dep_cols, study_mask: Optional[np.ndarray] = None,
                    known_deposits: Optional[Tuple[np.ndarray, np.ndarray]] = None,
                    progress: ProgressFn = None) -> RFResult:
        say = progress or (lambda f, m: None)
        say(0.02, N_("encoding evidence"))
        base = study_area(features, study_mask, self.missing)
        stack = FeatureStack(features, base, self.missing)
        msgs = list(stack.notes)
        dep_pos = np.unique(stack.positions(dep_rows, dep_cols))
        n_d = int(dep_pos.size)
        if n_d < 2:
            raise ValueError(f"Only {n_d} training deposit cell(s) inside the study area — need at least 2.")
        m = int(round(self.neg_ratio * n_d))
        if m < 1:
            raise ValueError("The number of non-deposit samples must be at least 1.")
        kr, kc = known_deposits if known_deposits is not None else (dep_rows, dep_cols)
        cand = background_candidates(base, kr, kc, self.min_distance, self.res)
        cand_pos = np.searchsorted(stack.cells, np.flatnonzero(cand))
        spacing = auto_spacing(cand_pos.size, m, self.res) if self.min_spacing < 0 else self.min_spacing
        cand_cells = stack.cells[cand_pos]
        width = base.shape[1]
        if n_d < 20:
            msgs.append(f"only {n_d} training deposits: the forest and its importances are unstable — prefer "
                        "several repeats and look at the OOB AUC")

        n = stack.n_cells
        s_sum, s_sq = np.zeros(n), np.zeros(n)
        n_votes = 0
        mdi = np.zeros(stack.n_columns)
        rep_rows, first = [], None
        for r in range(self.repeats):
            say(0.05 + 0.75 * r / self.repeats, N_("training the forest"))
            rng = np.random.default_rng(self.seed + 1000 * r)
            neg = sample_positions(cand_pos, cand_cells // width, cand_cells % width, m, rng, spacing, self.res)
            idx = np.r_[dep_pos, neg]
            X = stack.matrix(idx)
            y = np.r_[np.ones(n_d, int), np.zeros(m, int)]
            rf = self._forest(self.seed + r, stack.n_columns).fit(X, y)
            k = list(rf.classes_).index(1)
            oob = rf.oob_decision_function_[:, k]
            ok = np.isfinite(oob)
            oob_auc = _auc(oob[ok & (y == 1)], oob[ok & (y == 0)])
            mdi += rf.feature_importances_
            rep_rows.append(dict(repeat=r + 1, seed=self.seed + r, deposits=n_d, non_deposits=m,
                                 oob_auc=oob_auc, oob_accuracy=float(rf.oob_score_)))
            if first is None:
                first = (rf, X, y, oob, idx)
            say(0.05 + 0.75 * (r + 0.5) / self.repeats, N_("predicting the study area"))
            for s in range(0, n, 250_000):
                chunk = np.arange(s, min(n, s + 250_000))
                a, b = self._tree_votes(rf, stack.matrix(chunk), self.n_jobs)
                s_sum[chunk] += a
                s_sq[chunk] += b
            n_votes += len(rf.estimators_)
        if spacing > 0:
            msgs.append(f"non-deposit samples are at least {spacing:,.0f} map units apart"
                        + (" (auto: ½·√(area / n))" if self.min_spacing < 0 else ""))
        if self.repeats > 1:
            msgs.append(f"{self.repeats} random non-deposit samples: {self.repeats * self.n_trees} trees pooled; "
                        "importances averaged; OOB and cross-validation refer to sample 1")
        score = s_sum / n_votes
        std = np.sqrt(np.maximum(s_sq / n_votes - score ** 2, 0.0))

        say(0.85, N_("variable importance"))
        rf, X, y, oob, idx = first
        imp_cols, imp_layers = self._importance(rf, X, y, stack, mdi / self.repeats)
        say(0.92, N_("cross-validating"))
        val = self._validation(rf, X, y, oob, stack)

        def grid(v):
            out = np.zeros(base.shape)
            out.ravel()[stack.cells] = v
            return np.ma.array(out, mask=~base)

        rep = pd.DataFrame(rep_rows)
        stats = dict(oob_auc=float(rep.oob_auc.iloc[0]), oob_accuracy=float(rep.oob_accuracy.iloc[0]),
                     n_trees=self.n_trees * self.repeats, spacing=spacing)
        cv_auc = val.set_index("item")["value"].get("Cross-validated AUC")
        if cv_auc is not None:
            stats["cv_auc"] = float(cv_auc)
        summary = pd.DataFrame([
            ("Training deposits (cells)", str(n_d)), ("Non-deposit samples per draw", str(m)),
            ("Random draws (repeats)", str(self.repeats)), ("Trees per forest / pooled", f"{self.n_trees} / "
                                                                                         f"{n_votes}"),
            ("Features per split", f"{self.max_features} of {stack.n_columns} columns"),
            ("Min. samples per leaf / max. depth", f"{self.min_samples_leaf} / {self.max_depth or 'unlimited'}"),
            ("Class weight", self.class_weight),
            ("Min. distance of non-deposits from deposits", f"{self.min_distance:g}"),
            ("Min. spacing between non-deposits", f"{spacing:,.0f}"),
            ("Study area (cells)", f"{n:,}"), ("Evidence columns", ", ".join(stack.names)[:300]),
            ("OOB AUC (sample 1)", f"{stats['oob_auc']:.4f}"), ("OOB accuracy (sample 1)", f"{stats['oob_accuracy']:.4f}"),
            ("Mean score in the study area", f"{score.mean():.4f}"), ("Mean tree-vote std", f"{std.mean():.4f}"),
        ], columns=["item", "value"])
        params = dict(n_trees=self.n_trees, max_features=self.max_features, min_samples_leaf=self.min_samples_leaf,
                      max_depth=self.max_depth, class_weight=self.class_weight, neg_ratio=self.neg_ratio,
                      min_distance=self.min_distance, min_spacing=spacing, repeats=self.repeats,
                      cv_folds=self.cv_folds, missing=self.missing, seed=self.seed)
        say(1.0, N_("done"))
        return RFResult(grid(score), grid(std), imp_layers, imp_cols, val, summary, rep, base, n_d, m, stats,
                        params, msgs)

    # -------------------------------------------------------------- importance / validation
    @staticmethod
    def _oob_scorer(rf, X):
        """Function Xmat -> OOB votes: every sample is predicted only by the trees that did not see it."""
        k = list(rf.classes_).index(1)
        n = X.shape[0]
        try:
            from sklearn.ensemble._forest import _generate_unsampled_indices, _get_n_samples_bootstrap
            nb = _get_n_samples_bootstrap(n, rf.max_samples)
            oob_ids = [_generate_unsampled_indices(t.random_state, n, nb) for t in rf.estimators_]
        except Exception:  # noqa: BLE001 — private API changed: fall back to all trees (in-sample)
            oob_ids = [np.arange(n)] * len(rf.estimators_)

        def votes(Xm):
            s, c = np.zeros(n), np.zeros(n)
            for t, ids in zip(rf.estimators_, oob_ids):
                if ids.size:
                    s[ids] += t.predict_proba(np.ascontiguousarray(Xm[ids]), check_input=False)[:, k]
                    c[ids] += 1
            return np.divide(s, c, out=np.full(n, np.nan), where=c > 0)
        return votes

    def _importance(self, rf, X, y, stack: FeatureStack, mdi: np.ndarray):
        votes = self._oob_scorer(rf, X)

        def oob_auc(Xm):
            v = votes(Xm)
            ok = np.isfinite(v)
            return _auc(v[ok & (y == 1)], v[ok & (y == 0)])
        base_auc = oob_auc(X)
        rng = np.random.default_rng(self.seed + 17)
        perm = {}
        for layer, cols in stack.groups.items():              # Breiman (2001): permute among the OOB samples
            drops = []
            for _ in range(5):
                Xp = X.copy()
                order = rng.permutation(X.shape[0])
                Xp[:, cols] = X[order][:, cols]                  # shuffle the whole layer (all one-hot columns)
                drops.append(base_auc - oob_auc(Xp))
            perm[layer] = (float(np.mean(drops)), float(np.std(drops)))
        cols = pd.DataFrame(dict(column=stack.names, layer=stack.layer_of, impurity_pct=100 * mdi / mdi.sum()))
        layers = (cols.groupby("layer", sort=False)["impurity_pct"].sum().reset_index()
                  .assign(permutation_auc_drop=lambda d: [perm[l][0] for l in d.layer],
                          permutation_sd=lambda d: [perm[l][1] for l in d.layer])
                  .sort_values("permutation_auc_drop", ascending=False).reset_index(drop=True))
        layers.insert(0, "rank", np.arange(1, len(layers) + 1))
        return cols.sort_values("impurity_pct", ascending=False).reset_index(drop=True), layers

    @staticmethod
    def _split_scores(p, y):
        return p[y == 1], p[y == 0]

    def _validation(self, rf, X, y, oob, stack) -> pd.DataFrame:
        ok = np.isfinite(oob)
        pred = oob >= 0.5
        pos, neg = y == 1, y == 0
        tn, fp = int((neg & ~pred & ok).sum()), int((neg & pred & ok).sum())
        fn, tp = int((pos & ~pred & ok).sum()), int((pos & pred & ok).sum())
        items = [("Samples (deposits / non-deposits)", f"{int(pos.sum())} / {int(neg.sum())}"),
                 ("OOB confusion: true non-deposit → predicted non-deposit / deposit", f"{tn} / {fp}"),
                 ("OOB confusion: true deposit → predicted non-deposit / deposit", f"{fn} / {tp}"),
                 ("OOB accuracy", f"{(tn + tp) / max(1, ok.sum()):.4f}"),
                 ("OOB AUC", f"{_auc(oob[ok & pos], oob[ok & neg]):.4f}")]
        k = self.cv_folds
        n = y.size
        if k < 0 and n > LOO_MAX:
            k = 10
        if k != 0:
            from sklearn.base import clone
            if k < 0 or k >= n:
                fold = np.arange(n)
                name = "leave-one-out"
            else:
                fold = np.empty(n, int)
                rng = np.random.default_rng(self.seed + 7)
                for cls in (1, 0):
                    ids = rng.permutation(np.flatnonzero(y == cls))
                    fold[ids] = np.arange(ids.size) % k
                name = f"{k}-fold (stratified)" + (f"; leave-one-out needs ≤ {LOO_MAX} samples"
                                                     if self.cv_folds < 0 else "")
            held = np.empty(n)
            accs = []
            for f in np.unique(fold):
                te = fold == f
                m = clone(rf).set_params(oob_score=False).fit(X[~te], y[~te])
                held[te] = m.predict_proba(X[te])[:, list(m.classes_).index(1)] if 1 in m.classes_ else 0.0
                accs.append(float(((held[te] >= 0.5) == (y[te] == 1)).mean()))
            items += [("Cross-validation", name),
                      ("Cross-validated accuracy (mean of folds)", f"{np.mean(accs):.4f}"),
                      ("Cross-validated AUC", f"{_auc(held[pos], held[neg]):.4f}")]
        return pd.DataFrame(items, columns=["item", "value"])
