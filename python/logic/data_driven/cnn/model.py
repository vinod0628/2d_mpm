"""Convolutional neural network (CNN) for mineral prospectivity mapping — TensorFlow / Keras, no GUI.

Patch-based CNN (LeCun et al. 1998; in MPM: Sun et al. 2020, Li et al. 2021): every cell is represented by the k × k
window of all evidence layers around it (channels = evidence columns, z-scored; categorical layers one-hot), so the
network sees the spatial context — the shape and neighbourhood of anomalies, contacts and structures — not only the
values at the cell. Architecture (Keras):

    Conv2D(f1, 3×3, ReLU, same) → Conv2D(f2, 3×3, ReLU, same) → MaxPool(2×2) → Flatten
    → Dense(units, ReLU) → Dropout(rate) → Dense(1, sigmoid)

trained with Adam on the binary cross-entropy. Defaults (8 and 16 filters, 16 dense neurons, dropout 0.5, learning
rate 3·10⁻⁴) are deliberately small: with tens of deposits a larger network (16 / 32 / 32, 10⁻³) over-fitted within a
few epochs (Carlin, 30 % spatial hold-out: test AUC 0.888 vs 0.863). Training samples are deposit cells and random non-deposit cells (as
for the other machine-learning methods), **augmented** by the 8 rotations / reflections of every patch (the dihedral
group D4) — geological patterns have no preferred orientation at patch scale and the augmentation multiplies the few
deposit examples (Sun et al. 2020).

Number of epochs: k stratified folds trained with the held-out AUC monitored (EarlyStopping, best weights restored);
E* = median best epoch; the final networks are trained for E* epochs on all (augmented) samples. Ensemble:
``networks`` networks with different initial weights per random non-deposit sample, R samples → score = mean output
(a favourability in [0, 1] that depends on the deposit / non-deposit ratio — not a probability per unit cell),
score_std = s.d. of the outputs. Device: TensorFlow uses a visible GPU automatically (TensorFlow ≥ 2.11 has no GPU
support on native Windows — there the CPU is used).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from i18n import N_
from logic.common.features import Feature, FeatureStack, study_area
from logic.common.training import auto_spacing, background_candidates, sample_positions

CHUNK = 50_000
ProgressFn = Optional[Callable[[float, str], None]]


def _tf():
    """Import TensorFlow lazily (≈ 10–30 s the first time); clear message if it is not installed."""
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    try:
        import tensorflow as tf
    except ImportError as exc:                                # e.g. a release build without TensorFlow
        raise RuntimeError("The CNN needs TensorFlow, which is not available in this installation.") from exc
    tf.get_logger().setLevel("ERROR")
    return tf


def parse_filters(text) -> Tuple[int, ...]:
    if isinstance(text, (tuple, list)):
        vals = tuple(int(v) for v in text)
    else:
        vals = tuple(int(v) for v in str(text).replace(";", ",").split(",") if v.strip())
    if not vals or any(v < 1 for v in vals):
        raise ValueError("convolution filters: give one or more positive counts, e.g. '16, 32'")
    return vals


def dihedral(patches: np.ndarray) -> np.ndarray:
    """The 8 rotations / reflections of (n, k, k, c) patches → (8n, k, k, c)."""
    out = []
    for flip in (False, True):
        p = patches[:, :, ::-1, :] if flip else patches
        for r in range(4):
            out.append(np.rot90(p, r, axes=(1, 2)))
    return np.concatenate(out, 0)


def _auc(pos: np.ndarray, neg: np.ndarray) -> float:
    from scipy.stats import rankdata
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    r = rankdata(np.r_[pos, neg])
    return float((r[: pos.size].sum() - pos.size * (pos.size + 1) / 2) / (pos.size * neg.size))


class PatchGrid:
    """z-scored evidence of the whole grid (H × W × channels), zero-padded by half a window; cells outside the
    study area are 0 (= the study-area mean). ``patches(pos)`` returns the k × k windows of study cells."""

    def __init__(self, stack: FeatureStack, shape: Tuple[int, int], k: int):
        self.k, self.h = k, k // 2
        H, W = shape
        p = stack.n_columns
        grid = np.zeros((H + 2 * self.h, W + 2 * self.h, p), np.float32)
        rows, cols = stack.cells // W, stack.cells % W
        for j in range(p):
            col = stack.column(j).astype(np.float64)
            sd = col.std() or 1.0
            grid[rows + self.h, cols + self.h, j] = ((col - col.mean()) / sd).astype(np.float32)
        self.grid, self.rows, self.cols = grid, rows, cols

    def patches(self, pos: np.ndarray) -> np.ndarray:
        pos = np.asarray(pos, np.int64)
        r, c = self.rows[pos], self.cols[pos]
        off = np.arange(self.k)
        return self.grid[(r[:, None] + off)[:, :, None], (c[:, None] + off)[:, None, :], :]   # (n, k, k, p)


@dataclass
class CNNResult:
    score: np.ma.MaskedArray
    score_std: np.ma.MaskedArray
    importance: pd.DataFrame
    curve: pd.DataFrame
    sample_validation: pd.DataFrame
    summary: pd.DataFrame
    repeats: pd.DataFrame
    study_mask: np.ndarray
    n_deposits: int
    n_negatives: int
    stats: Dict[str, float]
    params: Dict[str, object]
    messages: List[str] = field(default_factory=list)


class CNNModel:
    """Patch-CNN prospectivity model (see module doc)."""

    def __init__(self, patch: int = 9, filters=(8, 16), dense: int = 16, dropout: float = 0.5,
                 learning_rate: float = 3e-4, batch_size: int = 32, max_epochs: int = 200, patience: int = 20,
                 augment: bool = True, networks: int = 2, neg_ratio: float = 1.0, min_distance: float = 0.0,
                 min_spacing: float = -1.0, res: Tuple[float, float] = (1.0, 1.0), repeats: int = 1, cv_folds: int = 3,
                 missing: str = "exclude", seed: int = 0):
        if patch < 3 or patch % 2 == 0:
            raise ValueError("patch size must be an odd number ≥ 3")
        if cv_folds < 2:
            raise ValueError("cv_folds must be ≥ 2")
        self.patch, self.filters = int(patch), parse_filters(filters)
        self.dense, self.dropout = max(1, int(dense)), float(dropout)
        self.learning_rate, self.batch_size = float(learning_rate), max(1, int(batch_size))
        self.max_epochs, self.patience, self.augment = max(1, int(max_epochs)), max(1, int(patience)), bool(augment)
        self.networks = max(1, int(networks))
        self.neg_ratio, self.min_distance, self.min_spacing = float(neg_ratio), float(min_distance), float(min_spacing)
        self.res, self.repeats, self.cv_folds = tuple(res), max(1, int(repeats)), int(cv_folds)
        self.missing, self.seed = missing, int(seed)

    # -------------------------------------------------------------- network
    def build(self, channels: int, seed: int):
        tf = _tf()
        keras = tf.keras
        keras.utils.set_random_seed(seed)
        L = keras.layers
        layers = [L.Input((self.patch, self.patch, channels))]
        for f in self.filters:
            layers.append(L.Conv2D(f, 3, activation="relu", padding="same"))
        if self.patch >= 4:
            layers.append(L.MaxPooling2D(2))
        layers += [L.Flatten(), L.Dense(self.dense, activation="relu"), L.Dropout(self.dropout),
                   L.Dense(1, activation="sigmoid")]
        m = keras.Sequential(layers)
        m.compile(optimizer=keras.optimizers.Adam(self.learning_rate), loss="binary_crossentropy",
                  metrics=[keras.metrics.AUC(name="auc")])
        return m

    def _train_set(self, P, y):
        if self.augment:
            return dihedral(P), np.tile(y, 8).astype(np.float32)
        return P, y.astype(np.float32)

    def fit_net(self, P, y, epochs, seed, val=None):
        tf = _tf()
        m = self.build(P.shape[-1], seed)
        Xa, ya = self._train_set(P, y)
        cb = []
        if val is not None:
            cb = [tf.keras.callbacks.EarlyStopping(monitor="val_auc", mode="max", patience=self.patience,
                                                   restore_best_weights=True)]
        h = m.fit(Xa, ya, epochs=epochs, batch_size=self.batch_size, shuffle=True, verbose=0, callbacks=cb,
                  validation_data=None if val is None else (val[0], val[1].astype(np.float32)))
        return m, h.history

    @staticmethod
    def predict(m, P):
        return m.predict(P, batch_size=4096, verbose=0).ravel()

    def folds(self, y, seed):
        k = min(self.cv_folds, int(min((y == 1).sum(), (y == 0).sum())))
        fold = np.empty(y.size, int)
        rng = np.random.default_rng(seed)
        for cls in (1, 0):
            ids = rng.permutation(np.flatnonzero(y == cls))
            fold[ids] = np.arange(ids.size) % k
        return fold

    def cross_validate(self, P, y, seed):
        fold = self.folds(y, seed)
        held = np.empty(y.size)
        curves, bests, models = [], [], []
        for f in np.unique(fold):
            te = fold == f
            m, hist = self.fit_net(P[~te], y[~te], self.max_epochs, seed + 31 * int(f), val=(P[te], y[te]))
            held[te] = self.predict(m, P[te])
            curves.append(hist.get("val_auc", []))
            tr = hist.get("auc", [])
            bests.append(int(np.argmax(hist["val_auc"])) + 1 if hist.get("val_auc") else self.max_epochs)
            models.append((te, m, tr))
        n = max(len(c) for c in curves)
        pad = lambda c: np.r_[c, np.full(n - len(c), np.nan)]                       # noqa: E731
        tr_curves = [pad(t) for _, _, t in models]
        curve = pd.DataFrame(dict(epoch=np.arange(1, n + 1), train_auc=np.nanmean(tr_curves, 0),
                                  cv_auc=np.nanmean([pad(c) for c in curves], 0)))
        e_best = int(np.median(bests))
        return e_best, held, curve, [(te, m) for te, m, _ in models], fold

    # -------------------------------------------------------------- full model
    def fit_predict(self, features: Sequence[Feature], dep_rows, dep_cols, study_mask: Optional[np.ndarray] = None,
                    known_deposits: Optional[Tuple[np.ndarray, np.ndarray]] = None,
                    progress: ProgressFn = None) -> CNNResult:
        say = progress or (lambda f, m: None)
        say(0.01, N_("loading TensorFlow"))
        tf = _tf()
        try:
            tf.config.experimental.enable_op_determinism()
        except Exception:  # noqa: BLE001
            pass
        device = "GPU" if tf.config.list_physical_devices("GPU") else "CPU"
        say(0.03, N_("encoding evidence"))
        base = study_area(features, study_mask, self.missing)
        stack = FeatureStack(features, base, self.missing)
        msgs = list(stack.notes)
        pg = PatchGrid(stack, base.shape, self.patch)
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

        n = stack.n_cells
        s1, s2, cnt = np.zeros(n), np.zeros(n), 0
        rep_rows, first = [], None
        for r in range(self.repeats):
            frac = r / self.repeats
            step = 0.85 / self.repeats
            rng = np.random.default_rng(self.seed + 1000 * r)
            neg = sample_positions(cand_pos, cand_cells // width, cand_cells % width, m, rng, spacing, self.res)
            idx = np.r_[dep_pos, neg]
            P = pg.patches(idx)
            y = np.r_[np.ones(n_d, int), np.zeros(neg.size, int)]
            say(0.05 + 0.85 * frac, N_("cross-validating the number of epochs"))
            e_best, held, curve, fold_models, fold = self.cross_validate(P, y, self.seed + 7 + r)
            say(0.05 + 0.85 * frac + 0.4 * step, N_("training the networks"))
            nets = [self.fit_net(P, y, e_best, self.seed + 1000 * r + 101 * j)[0] for j in range(self.networks)]
            say(0.05 + 0.85 * frac + 0.6 * step, N_("predicting the study area"))
            for s in range(0, n, CHUNK):
                ch = np.arange(s, min(n, s + CHUNK))
                Pc = pg.patches(ch)
                for net in nets:
                    pr = self.predict(net, Pc)
                    s1[ch] += pr
                    s2[ch] += pr * pr
                say(0.05 + 0.85 * frac + step * (0.6 + 0.4 * min(1.0, (s + CHUNK) / n)), N_("predicting the study area"))
            cnt += len(nets)
            p_tr = np.mean([self.predict(net, P) for net in nets], 0)
            rep_rows.append(dict(repeat=r + 1, seed=self.seed + 1000 * r, deposits=n_d, non_deposits=int(neg.size),
                                 epochs=e_best, networks=len(nets), training_auc=_auc(p_tr[y == 1], p_tr[y == 0]),
                                 cv_auc=_auc(held[y == 1], held[y == 0])))
            if first is None:
                first = dict(P=P, y=y, held=held, p_tr=p_tr, curve=curve, fold_models=fold_models, fold=fold,
                             e_best=e_best, n_params=int(nets[0].count_params()))
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
                  epochs=int(first["e_best"]), networks=int(cnt), parameters=first["n_params"], device=device,
                  patch_m=float(self.patch * self.res[0]), spacing=spacing)
        arch = (f"{self.patch}×{self.patch}×{stack.n_columns} → " + " → ".join(f"conv {f}" for f in self.filters)
                + (" → max-pool" if self.patch >= 4 else "") + f" → dense {self.dense} → dropout {self.dropout:g} → 1")
        summary = pd.DataFrame([
            ("Training deposits (cells)", str(n_d)), ("Non-deposit samples per draw", str(m)),
            ("Random draws × networks", f"{R} × {self.networks} = {cnt}"),
            ("Patch (window)", f"{self.patch} × {self.patch} cells = {st['patch_m']:,.0f} × {st['patch_m']:,.0f} map units"),
            ("Architecture", arch), ("Trainable parameters", f"{st['parameters']:,}"),
            ("Augmentation", "8 rotations / reflections" if self.augment else "none"),
            ("Optimiser", f"Adam, learning rate {self.learning_rate:g}, batch {self.batch_size}"),
            ("Epochs E* (median best of the folds)", f"{st['epochs']} (max. {self.max_epochs}, patience {self.patience})"),
            ("Device", device),
            ("Evidence channels", ", ".join(stack.names)[:300]), ("Study area (cells)", f"{n:,}"),
            ("Min. distance of non-deposits from deposits", f"{self.min_distance:g}"),
            ("Min. spacing between non-deposits", f"{spacing:,.0f}"),
            ("Training AUC (samples)", f"{st['training_auc']:.4f}"),
            ("Cross-validated AUC (samples)", f"{st['cv_auc']:.4f}"),
            ("Mean score in the study area", f"{score.mean():.4f}"),
        ], columns=["item", "value"])
        params = dict(patch=self.patch, filters=list(self.filters), dense=self.dense, dropout=self.dropout,
                      learning_rate=self.learning_rate, batch_size=self.batch_size, max_epochs=self.max_epochs,
                      patience=self.patience, augment=self.augment, networks=self.networks, neg_ratio=self.neg_ratio,
                      min_distance=self.min_distance, min_spacing=spacing, repeats=R, cv_folds=self.cv_folds,
                      missing=self.missing, seed=self.seed)
        say(1.0, N_("done"))
        return CNNResult(grid(score), grid(std), imp, first["curve"], val, summary, rep, base, n_d, m, st, params, msgs)

    # -------------------------------------------------------------- importance / validation
    def _importance(self, first, stack: FeatureStack) -> pd.DataFrame:
        """Drop of the CV AUC when the patches of one evidence layer are exchanged between held-out samples."""
        P, y, held = first["P"], first["y"], first["held"]
        base_auc = _auc(held[y == 1], held[y == 0])
        rng = np.random.default_rng(self.seed + 17)
        rows = []
        for layer, cols in stack.groups.items():
            drops = []
            for _ in range(3):
                h = np.empty(y.size)
                for te, net in first["fold_models"]:
                    Pt = P[te].copy()
                    Pt[..., cols] = Pt[rng.permutation(Pt.shape[0])][..., cols]
                    h[te] = self.predict(net, Pt)
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
                 ("Epochs E* (median best of the folds)", str(first["e_best"])),
                 ("CV confusion: true non-deposit → predicted non-deposit / deposit", f"{ctn} / {cfp}"),
                 ("CV confusion: true deposit → predicted non-deposit / deposit", f"{cfn} / {ctp}"),
                 ("Cross-validated accuracy (mean of folds)", f"{np.mean(accs):.4f}"),
                 ("Cross-validated AUC", f"{_auc(held[pos], held[neg]):.4f}")]
        return pd.DataFrame(items, columns=["item", "value"])
