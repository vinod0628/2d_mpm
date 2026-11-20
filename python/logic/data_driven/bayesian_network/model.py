"""Bayesian network classifiers for mineral prospectivity mapping — own implementation (NumPy + networkx), no GUI.

Porwal, Carranza & Hale (2006): the deposit variable D and the (discretised) evidence layers X_1 … X_n form a Bayesian
network; the posterior P(D | x) of every cell follows from Bayes' rule over the network's factorisation.

* **Naive Bayes (NB)** — every X_i depends only on D (conditional independence, as in weights of evidence):
      P(D=d | x) ∝ P(d) Π_i P(x_i | d)
  Trained on all study-area cells, NB is exactly the weights-of-evidence model (log-odds = prior log-odds + Σ W).
* **Tree-augmented naive Bayes (TAN)** — Friedman, Geiger & Goldszmidt (1997): additionally every evidence layer may
  depend on one other evidence layer. The tree maximises Σ I(X_i; X_j | D) (conditional mutual information) —
  Chow & Liu (1968) maximum-weight spanning tree, rooted at the first layer, directed away from the root:
      P(D=d | x) ∝ P(d) P(x_r | d) Π_{i ≠ r} P(x_i | x_π(i), d)
  The dependency between correlated layers (e.g. As and Sb from one fluid, or a fault-controlled lithology and the
  distance to faults) is modelled instead of being double-counted.

Conditional probability tables (CPTs) are estimated with a Dirichlet prior (equivalent sample size α spread evenly,
BDeu-type; Heckerman et al. 1995): θ = (N_ijk + α/r_i) / (N_ij + α). Uncertainty: CPTs are drawn ``draws`` times from
their Dirichlet posteriors and the s.d. of the resulting posterior probabilities is reported (Bayesian uncertainty).
Training cells: all study-area cells (deposit cells = 1, the others = 0; probabilities per cell are calibrated like
weights of evidence) or deposits + random non-deposit cells (Porwal et al. 2006; probabilities depend on the ratio).
Continuous evidence is discretised into equal-area (quantile) classes; categorical evidence keeps its classes;
missing evidence is a class of its own or excluded.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from i18n import N_
from logic.common.features import Feature, study_area
from logic.common.training import auto_spacing, background_candidates, sample_positions

STRUCTURES = ("naive", "tan")
STRUCTURE_LABELS = {"naive": N_("naive Bayes (conditional independence)"),
                    "tan": N_("tree-augmented naive Bayes (TAN)")}
DESIGNS = ("all", "sampled")
DESIGN_LABELS = {"all": N_("all study-area cells"), "sampled": N_("deposits + random non-deposit cells")}
ProgressFn = Optional[Callable[[float, str], None]]


def _auc(pos: np.ndarray, neg: np.ndarray) -> float:
    from scipy.stats import rankdata
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    r = rankdata(np.r_[pos, neg])
    return float((r[: pos.size].sum() - pos.size * (pos.size + 1) / 2) / (pos.size * neg.size))


@dataclass
class Discrete:
    """One discretised evidence layer: class code 0 … k−1 per study cell and the class labels."""
    name: str
    codes: np.ndarray
    labels: List[str]

    @property
    def k(self) -> int:
        return len(self.labels)


def discretise(feature: Feature, cells: np.ndarray, n_classes: int, missing: str) -> Discrete:
    """Categorical → its classes; continuous → ``n_classes`` equal-area (quantile) classes of the study area.
    Missing values → an extra class "missing" (missing="class")."""
    arr = np.ma.asarray(feature.array)
    data = np.ma.getdata(arr).ravel()[cells].astype(float)
    known = ~np.ma.getmaskarray(arr).ravel()[cells] & np.isfinite(data)
    codes = np.full(cells.size, -1, int)
    labels = feature.labels or {}
    if feature.kind == "categorical":
        cls = np.unique(data[known])
        codes[known] = np.searchsorted(cls, data[known])
        names = [str(labels.get(float(c), f"{c:g}")) for c in cls]
    else:
        qs = np.unique(np.quantile(data[known], np.linspace(0, 1, n_classes + 1)[1:-1]))
        codes[known] = np.searchsorted(qs, data[known], side="right")
        edges = np.r_[np.min(data[known]), qs, np.max(data[known])]
        names = [f"{edges[i]:.4g} – {edges[i + 1]:.4g}" for i in range(len(qs) + 1)]
    if (~known).any():
        if missing == "class":
            codes[~known] = len(names)
            names.append("missing")
    return Discrete(feature.name, codes, names)


def mutual_information(a: np.ndarray, b: np.ndarray, d: np.ndarray, ka: int, kb: int) -> float:
    """Conditional mutual information I(A; B | D) (nats) from counts, D binary."""
    total = 0.0
    n = a.size
    for dv in (0, 1):
        m = d == dv
        nd = m.sum()
        if nd == 0:
            continue
        joint = np.bincount(a[m] * kb + b[m], minlength=ka * kb).reshape(ka, kb).astype(float)
        pa, pb = joint.sum(1, keepdims=True), joint.sum(0, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            t = joint * np.log(joint * nd / (pa * pb))
        total += np.nansum(t) / n
    return float(total)


@dataclass
class BNModelFit:
    structure: str
    parents: Dict[int, Optional[int]]           # evidence index -> parent evidence index (TAN) or None
    prior: np.ndarray                           # P(D)
    cpts: Dict[int, np.ndarray]                 # index -> array [d, (parent class,) class]
    counts: Dict[int, np.ndarray]               # same shape: raw counts (for Dirichlet draws)
    prior_counts: np.ndarray
    alpha: float
    edges: pd.DataFrame


def fit_bn(ev: Sequence[Discrete], d: np.ndarray, structure: str = "tan", alpha: float = 1.0,
           rows: Optional[np.ndarray] = None) -> BNModelFit:
    """Structure (TAN tree by conditional mutual information) and Dirichlet-smoothed CPTs from the rows ``rows``."""
    import networkx as nx
    idx = np.arange(d.size) if rows is None else np.asarray(rows)
    dd = d[idx].astype(int)
    codes = [e.codes[idx] for e in ev]
    ok = np.logical_and.reduce([c >= 0 for c in codes]) if codes else np.ones(idx.size, bool)
    dd, codes = dd[ok], [c[ok] for c in codes]
    n = len(ev)
    parents: Dict[int, Optional[int]] = {i: None for i in range(n)}
    edge_rows = []
    if structure == "tan" and n > 1:
        g = nx.Graph()
        for i in range(n):
            g.add_node(i)
            for j in range(i + 1, n):
                w = mutual_information(codes[i], codes[j], dd, ev[i].k, ev[j].k)
                g.add_edge(i, j, weight=w)
        tree = nx.maximum_spanning_tree(g)
        for parent, child in nx.bfs_edges(tree, 0):
            parents[child] = parent
            edge_rows.append(dict(parent=ev[parent].name, child=ev[child].name,
                                  cond_mutual_information=tree[parent][child]["weight"]))
    prior_counts = np.bincount(dd, minlength=2).astype(float)
    prior = (prior_counts + alpha / 2) / (prior_counts.sum() + alpha)
    cpts, counts = {}, {}
    for i in range(n):
        k = ev[i].k
        if parents[i] is None:
            c = np.zeros((2, k))
            np.add.at(c, (dd, codes[i]), 1.0)
            cpt = (c + alpha / k) / (c.sum(-1, keepdims=True) + alpha)
        else:
            kp = ev[parents[i]].k
            c = np.zeros((2, kp, k))
            np.add.at(c, (dd, codes[parents[i]], codes[i]), 1.0)
            cpt = (c + alpha / (k * kp)) / (c.sum(-1, keepdims=True) + alpha / kp)
        cpts[i], counts[i] = cpt, c
    return BNModelFit(structure, parents, prior, cpts, counts, prior_counts, alpha,
                      pd.DataFrame(edge_rows, columns=["parent", "child", "cond_mutual_information"]))


def log_odds(fit: BNModelFit, codes: Sequence[np.ndarray], cpts=None, prior=None) -> np.ndarray:
    """ln P(D=1 | x) − ln P(D=0 | x) for code rows (missing codes −1 contribute nothing)."""
    cpts = fit.cpts if cpts is None else cpts
    prior = fit.prior if prior is None else prior
    lo = np.full(codes[0].size, np.log(prior[1] / prior[0]))
    for i, cpt in cpts.items():
        c = codes[i]
        p = fit.parents[i]
        ok = c >= 0 if p is None else (c >= 0) & (codes[p] >= 0)
        if p is None:
            lo[ok] += np.log(cpt[1, c[ok]]) - np.log(cpt[0, c[ok]])
        else:
            pc = codes[p][ok]
            lo[ok] += np.log(cpt[1, pc, c[ok]]) - np.log(cpt[0, pc, c[ok]])
    return lo


def draw_cpts(fit: BNModelFit, rng: np.random.Generator):
    """One draw of all CPTs (and the prior) from their Dirichlet posteriors."""
    cpts = {}
    for i, c in fit.counts.items():
        k = c.shape[-1]
        conc = c + (fit.alpha / k if fit.parents[i] is None else fit.alpha / (k * c.shape[1]))
        g = rng.gamma(conc)
        cpts[i] = g / g.sum(-1, keepdims=True)
    gp = rng.gamma(fit.prior_counts + fit.alpha / 2)
    return cpts, gp / gp.sum()


@dataclass
class BNResult:
    posterior: np.ma.MaskedArray          # P(D | x) per raster cell (all-cells design) or score (sampled design)
    posterior_sd: np.ma.MaskedArray       # Dirichlet posterior s.d.
    unit_probability: Optional[np.ma.MaskedArray]
    edges: pd.DataFrame
    cpt_table: pd.DataFrame
    layers: pd.DataFrame                  # per layer: classes, I(X;D), parent, AUC drop when left out
    comparison: pd.DataFrame              # NB vs TAN fit
    summary: pd.DataFrame
    study_mask: np.ndarray
    n_deposits: int
    stats: Dict[str, float]
    params: Dict[str, object]
    messages: List[str] = field(default_factory=list)


class BayesianNetworkModel:
    """Bayesian network prospectivity model (see module doc)."""

    def __init__(self, structure: str = "tan", n_classes: int = 5, alpha: float = 1.0, design: str = "all",
                 missing: str = "class", draws: int = 50, unit_cell_km2: float = 1.0, cell_area_km2: float = 1.0,
                 neg_ratio: float = 5.0, min_distance: float = 0.0, min_spacing: float = -1.0,
                 res: Tuple[float, float] = (1.0, 1.0), seed: int = 0):
        if structure not in STRUCTURES:
            raise ValueError(f"structure must be one of {STRUCTURES}")
        if design not in DESIGNS:
            raise ValueError(f"design must be one of {DESIGNS}")
        if missing not in ("class", "exclude"):
            raise ValueError("missing must be 'class' or 'exclude'")
        self.structure, self.n_classes, self.alpha = structure, max(2, int(n_classes)), float(alpha)
        self.design, self.missing, self.draws = design, missing, max(0, int(draws))
        self.unit_cell_km2, self.cell_area_km2 = float(unit_cell_km2), float(cell_area_km2)
        self.neg_ratio, self.min_distance, self.min_spacing = float(neg_ratio), float(min_distance), float(min_spacing)
        self.res, self.seed = tuple(res), int(seed)

    def fit_predict(self, features: Sequence[Feature], dep_rows, dep_cols, study_mask: Optional[np.ndarray] = None,
                    known_deposits: Optional[Tuple[np.ndarray, np.ndarray]] = None,
                    progress: ProgressFn = None) -> BNResult:
        say = progress or (lambda f, m: None)
        say(0.03, N_("discretising evidence"))
        base = study_area(features, study_mask, "exclude" if self.missing == "exclude" else "keep")
        cells = np.flatnonzero(base)
        ev = [discretise(f, cells, self.n_classes, self.missing) for f in features]
        msgs = []
        width = base.shape[1]
        d = np.zeros(cells.size, int)
        flat = np.asarray(dep_rows, np.int64) * width + np.asarray(dep_cols, np.int64)
        pos = np.searchsorted(cells, flat)
        pos = np.clip(pos, 0, max(cells.size - 1, 0))
        dep_pos = np.unique(pos[cells[pos] == flat])
        d[dep_pos] = 1
        n_d = int(dep_pos.size)
        if n_d < 2:
            raise ValueError(f"Only {n_d} training deposit cell(s) inside the study area — need at least 2.")
        rows = None
        if self.design == "sampled":
            m = int(round(self.neg_ratio * n_d))
            kr, kc = known_deposits if known_deposits is not None else (dep_rows, dep_cols)
            cand = background_candidates(base, kr, kc, self.min_distance, self.res)
            cand_pos = np.searchsorted(cells, np.flatnonzero(cand))
            spacing = auto_spacing(cand_pos.size, m, self.res) if self.min_spacing < 0 else self.min_spacing
            cc = cells[cand_pos]
            neg = sample_positions(cand_pos, cc // width, cc % width, m, np.random.default_rng(self.seed), spacing,
                                   self.res)
            rows = np.r_[dep_pos, neg]
            msgs.append(f"{m} random non-deposit cells: the posterior depends on the deposit / non-deposit ratio")
        if n_d < 10:
            msgs.append(f"only {n_d} training deposits: the conditional probability tables are poorly determined — "
                        "use few classes and a TAN only if the layers are clearly dependent")

        say(0.2, N_("learning the network"))
        fit = fit_bn(ev, d, self.structure, self.alpha, rows)
        codes = [e.codes for e in ev]
        lo = log_odds(fit, codes)
        post = 1.0 / (1.0 + np.exp(-lo))

        say(0.5, N_("posterior uncertainty (Dirichlet draws)"))
        rng = np.random.default_rng(self.seed + 5)
        if self.draws:
            # unique evidence combinations: every draw evaluated once per combination
            combo = np.stack(codes, 1)
            uc, inv = np.unique(combo, axis=0, return_inverse=True)
            inv = inv.ravel()
            ucodes = [uc[:, i] for i in range(len(ev))]
            s1, s2 = np.zeros(len(uc)), np.zeros(len(uc))
            for _ in range(self.draws):
                cp, pr = draw_cpts(fit, rng)
                pdraw = 1.0 / (1.0 + np.exp(-log_odds(fit, ucodes, cp, pr)))
                s1 += pdraw
                s2 += pdraw * pdraw
            sd = np.sqrt(np.maximum(s2 / self.draws - (s1 / self.draws) ** 2, 0.0))[inv]
        else:
            sd = np.zeros(cells.size)

        say(0.7, N_("evaluating the structure"))
        dd_rows = np.arange(cells.size) if rows is None else rows
        y = d[dd_rows]

        def loglik(f):
            l = log_odds(f, [c[dd_rows] for c in codes])
            pp = np.clip(1.0 / (1.0 + np.exp(-l)), 1e-12, 1 - 1e-12)
            return float(np.sum(y * np.log(pp) + (1 - y) * np.log(1 - pp)))
        comp = []
        for s in STRUCTURES:
            f = fit if s == self.structure else fit_bn(ev, d, s, self.alpha, rows)
            ll = loglik(f)
            k = sum(c.size - c.shape[0] * (c.shape[1] if c.ndim == 3 else 1) for c in f.counts.values())
            l2 = log_odds(f, [c[dd_rows] for c in codes])
            comp.append(dict(structure=s, log_likelihood=ll, parameters=int(k), bic=-2 * ll + k * np.log(y.size),
                             training_auc=_auc(l2[y == 1], l2[y == 0]), edges=len(f.edges)))
        comp = pd.DataFrame(comp)

        layers = []
        full_auc = _auc(lo[dd_rows][y == 1], lo[dd_rows][y == 0])
        for i, e in enumerate(ev):
            joint = np.bincount(np.clip(e.codes[dd_rows], 0, None) * 2 + y, minlength=e.k * 2).reshape(e.k, 2) / y.size
            with np.errstate(divide="ignore", invalid="ignore"):
                mi = float(np.nansum(joint * np.log(joint / (joint.sum(1, keepdims=True) * joint.sum(0, keepdims=True)))))
            rest = [ev[j] for j in range(len(ev)) if j != i]
            if rest:
                f2 = fit_bn(rest, d, self.structure, self.alpha, rows)
                l2 = log_odds(f2, [r.codes[dd_rows] for r in rest])
                drop = full_auc - _auc(l2[y == 1], l2[y == 0])
            else:
                drop = float("nan")
            p = fit.parents[i]
            layers.append(dict(layer=e.name, classes=e.k, mutual_information_with_deposits=mi,
                               parent="D" if p is None else f"D, {ev[p].name}", auc_drop_without_layer=drop))
        layers = pd.DataFrame(layers).sort_values("auc_drop_without_layer", ascending=False).reset_index(drop=True)

        cpt_rows = []
        for i, e in enumerate(ev):
            cpt = fit.cpts[i]
            p = fit.parents[i]
            if p is None:
                for kk, lab in enumerate(e.labels):
                    cpt_rows.append(dict(layer=e.name, parent_class="", cls=lab, p_given_deposit=cpt[1, kk],
                                         p_given_barren=cpt[0, kk], log_ratio=np.log(cpt[1, kk] / cpt[0, kk]),
                                         deposits=int(fit.counts[i][1, kk]), cells=int(fit.counts[i][:, kk].sum())))
            else:
                for pk, plab in enumerate(ev[p].labels):
                    for kk, lab in enumerate(e.labels):
                        cpt_rows.append(dict(layer=e.name, parent_class=f"{ev[p].name} = {plab}", cls=lab,
                                             p_given_deposit=cpt[1, pk, kk], p_given_barren=cpt[0, pk, kk],
                                             log_ratio=np.log(cpt[1, pk, kk] / cpt[0, pk, kk]),
                                             deposits=int(fit.counts[i][1, pk, kk]),
                                             cells=int(fit.counts[i][:, pk, kk].sum())))
        cpt_table = pd.DataFrame(cpt_rows)

        def grid(v):
            out = np.zeros(base.shape)
            out.ravel()[cells] = v
            return np.ma.array(out, mask=~base)

        unit = None
        if self.design == "all":
            ratio = self.unit_cell_km2 / self.cell_area_km2
            unit = grid(1.0 - (1.0 - post) ** ratio)
            predicted = float(post.sum())
            msgs.append(f"predicted deposits Σ P = {predicted:.1f} vs observed {n_d} (ratio N/T = {n_d / predicted:.2f})")
        st = dict(training_auc=full_auc, deposits=n_d, structure=self.structure, edges=len(fit.edges),
                  predicted=float(post.sum()) if self.design == "all" else None,
                  log_likelihood=float(comp.set_index("structure").loc[self.structure, "log_likelihood"]))
        summary = pd.DataFrame([
            ("Structure", self.structure), ("Training cells", self.design),
            ("Training deposits (cells)", str(n_d)), ("Cells used for the tables", f"{y.size:,}"),
            ("Evidence layers / classes", ", ".join(f"{e.name} ({e.k})" for e in ev)[:300]),
            ("Classes of continuous layers", f"{self.n_classes} equal-area (quantile) classes"),
            ("Missing evidence", "own class" if self.missing == "class" else "excluded"),
            ("Dirichlet equivalent sample size α", f"{self.alpha:g}"),
            ("TAN edges (parent → child)", "; ".join(f"{r.parent} → {r.child}" for r in fit.edges.itertuples()) or "—"),
            ("Log-likelihood of the training cells", f"{st['log_likelihood']:.2f}"),
            ("Training AUC", f"{full_auc:.4f}"),
            ("Posterior s.d. from", f"{self.draws} Dirichlet draws of the tables" if self.draws else "—"),
        ], columns=["item", "value"])
        params = dict(structure=self.structure, n_classes=self.n_classes, alpha=self.alpha, design=self.design,
                      missing=self.missing, draws=self.draws, unit_cell_km2=self.unit_cell_km2,
                      neg_ratio=self.neg_ratio, seed=self.seed)
        say(1.0, N_("done"))
        return BNResult(grid(post), grid(sd), unit, fit.edges, cpt_table, layers, comp, summary, base, n_d, st, params,
                        msgs)
