"""Logistic-regression parameter dialog and results dialog."""
from __future__ import annotations

import numpy as np
import pandas as pd
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QAbstractItemView, QCheckBox, QComboBox, QDialog, QDoubleSpinBox, QFileDialog,
                             QFormLayout, QGridLayout, QGroupBox, QHBoxLayout, QHeaderView, QLabel, QLineEdit,
                             QMessageBox, QPushButton, QSpinBox, QTableWidget, QTableWidgetItem, QTabWidget,
                             QVBoxLayout, QWidget)

from core.layers import LayerStore, RasterLayer
from gui import theme
from gui.widgets import (BaseDialog, LayerCombo, fit_to_screen, hint, make_table, metric_card, split_method_combo,
                         thinning_spin)
from i18n import N_, tr

from .model import (DESIGN_LABELS, ESTIMATOR_LABELS, MAX_CLASSES, MISSING_LABELS, SELECTION_LABELS,
                    TRANSFORM_LABELS, VAR_KIND_LABELS)

_LAST = {}          # remembers settings during the session
CV_CHOICES = [(-1, N_("leave-one-out")), (3, N_("3-fold")), (10, N_("10-fold")), (0, N_("none"))]


def _sample(lyr: RasterLayer, n: int = 400) -> np.ndarray:
    d = lyr.data
    return np.ma.compressed(d[:: max(1, d.shape[0] // n), :: max(1, d.shape[1] // n)])


class LogisticRegressionDialog(BaseDialog):
    COLS = [N_("Use"), N_("Evidence layer"), N_("Values"), N_("Variable type"), N_("Transform")]

    def __init__(self, store: LayerStore, parent=None):
        super().__init__(tr("Logistic Regression"),
                         tr("Data-driven model logit P = β0 + Σ βj·xj (Agterberg et al. 1993). Coefficients "
                            "are estimated jointly, so correlated layers are not double-counted. Continuous = values "
                            "(also ordered classes, e.g. distance classes); categorical = one 0/1 dummy per class, "
                            "the largest class is the reference."),
                         parent, ok_text=tr("Run model"))
        self.store = store

        # ---------------------------------------------------------------- training data
        g1 = QGroupBox(tr("Training data"))
        f1 = QFormLayout(g1)
        self.deposits = LayerCombo(store, "point")
        self.mask = LayerCombo(store, "raster", allow_none=True, none_text=tr("— cells with evidence —"))
        sm = next((l for l in store.rasters() if "mask" in l.name.lower()), None)
        if sm is not None:
            self.mask.set_layer(sm)
        else:
            self.mask.setCurrentIndex(0)
        dep = next((l for l in store.vectors("point") if "deposit" in l.name.lower()), None)
        if dep is not None:
            self.deposits.set_layer(dep)
        self.test_pct = QSpinBox()
        self.test_pct.setRange(0, 50)
        self.test_pct.setSuffix(" %")
        self.test_pct.setValue(_LAST.get("test_pct", 20))
        self.seed = QSpinBox()
        self.seed.setRange(0, 10 ** 6)
        self.seed.setValue(_LAST.get("seed", 42))
        self.split = split_method_combo(_LAST.get("split", "spatial"))
        self.thin = thinning_spin(_LAST.get("thin", 0.0))
        row = QHBoxLayout()
        row.addWidget(self.test_pct)
        row.addWidget(self.split)
        row.addWidget(QLabel(tr("random seed")))
        row.addWidget(self.seed)
        row.addStretch(1)
        self.negatives = LayerCombo(store, "point", allow_none=True, none_text=tr("— all background cells —"))
        self.negatives.setCurrentIndex(0)
        self.negatives.setToolTip(tr("Optional known non-deposit points used as ROC negatives instead of all "
                                     "background cells."))
        f1.addRow(tr("Deposits (points)"), self.deposits)
        f1.addRow(tr("Study area mask"), self.mask)
        f1.addRow(tr("Hold out for testing"), row)
        f1.addRow(tr("Thin deposits (min. distance)"), self.thin)
        f1.addRow(tr("Absence points (ROC)"), self.negatives)
        self.body.addWidget(g1)

        # ---------------------------------------------------------------- evidence
        g2 = QGroupBox(tr("Evidence layers"))
        v2 = QVBoxLayout(g2)
        self.table = QTableWidget(0, len(self.COLS))
        self.table.setHorizontalHeaderLabels([tr(c) for c in self.COLS])
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionMode(QAbstractItemView.NoSelection)
        self.table.setMinimumHeight(170)
        self.table.verticalHeader().setDefaultSectionSize(30)
        hh = self.table.horizontalHeader()
        hh.setSectionResizeMode(QHeaderView.Fixed)
        hh.setSectionResizeMode(1, QHeaderView.Stretch)
        for col, w in ((0, 44), (2, 110), (3, 250), (4, 110)):
            self.table.setColumnWidth(col, w)
        self._rows = []
        prev = _LAST.get("evidence", {})
        for lyr in store.rasters():
            if "mask" in lyr.name.lower():
                continue
            self._add_row(lyr, prev.get(lyr.name))
        v2.addWidget(self.table, 1)
        note = hint(tr("Tip: keep about 10 deposits per coefficient (a categorical layer uses classes − 1)."))
        v2.addWidget(note)
        self.body.addWidget(g2, 1)

        # ---------------------------------------------------------------- training samples
        g3 = QGroupBox(tr("Training samples"))
        f3 = QGridLayout(g3)
        self.design = QComboBox()
        for k, v in DESIGN_LABELS.items():
            self.design.addItem(tr(v), k)
        self.design.setCurrentIndex(list(DESIGN_LABELS).index(_LAST.get("design", "all")))
        self.neg_ratio = QDoubleSpinBox()
        self.neg_ratio.setRange(0.1, 100)
        self.neg_ratio.setDecimals(1)
        self.neg_ratio.setValue(_LAST.get("neg_ratio", 1.0))
        self.min_dist = QDoubleSpinBox()
        self.min_dist.setRange(0, 1e7)
        self.min_dist.setDecimals(0)
        self.min_dist.setSingleStep(100)
        cell = next((l.grid.res[0] for l in store.rasters()), 100.0)
        self.min_dist.setValue(_LAST.get("min_dist", 4 * cell))
        self.min_dist.setToolTip(tr("Non-deposit samples keep at least this distance (map units) from every known "
                                    "deposit."))
        self.spacing = QDoubleSpinBox()
        self.spacing.setRange(-1, 1e7)
        self.spacing.setDecimals(0)
        self.spacing.setSingleStep(100)
        self.spacing.setSpecialValueText(tr("auto"))
        self.spacing.setValue(_LAST.get("spacing", -1))
        self.spacing.setToolTip(tr("Minimum distance between non-deposit samples; auto = ½·√(available area / "
                                   "number of samples) spreads them evenly; 0 = no constraint."))
        self.repeats = QSpinBox()
        self.repeats.setRange(1, 200)
        self.repeats.setValue(_LAST.get("repeats", 10))
        self.repeats.setToolTip(tr("Fit the model on several random non-deposit samples and pool the results "
                                   "(Zuo & Wang 2020); the spread between samples is added to the uncertainty."))
        self.prior = QCheckBox(tr("prior correction → probability per unit cell (King & Zeng 2001)"))
        self.prior.setChecked(_LAST.get("prior", True))
        self.cv = QComboBox()
        for k, v in CV_CHOICES:
            self.cv.addItem(tr(v), k)
        self.cv.setCurrentIndex([k for k, _ in CV_CHOICES].index(_LAST.get("cv", -1)))
        f3.addWidget(QLabel(tr("Training cells")), 0, 0)
        f3.addWidget(self.design, 0, 1, 1, 3)
        f3.addWidget(QLabel(tr("Non-deposit samples per deposit")), 1, 0)
        f3.addWidget(self.neg_ratio, 1, 1)
        f3.addWidget(QLabel(tr("Min. distance from deposits")), 1, 2)
        f3.addWidget(self.min_dist, 1, 3)
        f3.addWidget(QLabel(tr("Min. spacing between samples")), 2, 0)
        f3.addWidget(self.spacing, 2, 1)
        f3.addWidget(QLabel(tr("Random samples (repeats)")), 3, 0)
        f3.addWidget(self.repeats, 3, 1)
        f3.addWidget(QLabel(tr("Cross-validation")), 3, 2)
        f3.addWidget(self.cv, 3, 3)
        f3.addWidget(self.prior, 4, 0, 1, 4)
        self.body.addWidget(g3)
        self._sampled_widgets = [self.neg_ratio, self.min_dist, self.spacing, self.repeats, self.prior, self.cv]
        self.design.currentIndexChanged.connect(self._sync)

        # ---------------------------------------------------------------- model options
        g4 = QGroupBox(tr("Model options"))
        f4 = QGridLayout(g4)
        self.estimator = QComboBox()
        for k, v in ESTIMATOR_LABELS.items():
            self.estimator.addItem(tr(v), k)
        self.estimator.setCurrentIndex(list(ESTIMATOR_LABELS).index(_LAST.get("estimator", "firth")))
        self.ridge = QDoubleSpinBox()
        self.ridge.setRange(0.001, 1e6)
        self.ridge.setDecimals(3)
        self.ridge.setValue(_LAST.get("ridge", 1.0))
        self.ridge.setPrefix("λ = ")
        self.selection = QComboBox()
        for k, v in SELECTION_LABELS.items():
            self.selection.addItem(tr(v), k)
        self.selection.setCurrentIndex(list(SELECTION_LABELS).index(_LAST.get("selection", "none")))
        self.alpha = QDoubleSpinBox()
        self.alpha.setRange(0.001, 0.5)
        self.alpha.setDecimals(3)
        self.alpha.setSingleStep(0.01)
        self.alpha.setValue(_LAST.get("alpha", 0.10))
        self.alpha.setPrefix(tr("remove if p > "))
        self.missing = QComboBox()
        for k, v in MISSING_LABELS.items():
            self.missing.addItem(tr(v), k)
        self.missing.setCurrentIndex(list(MISSING_LABELS).index(_LAST.get("missing", "exclude")))
        self.unit = QDoubleSpinBox()
        self.unit.setRange(1e-6, 1e6)
        self.unit.setDecimals(4)
        self.unit.setValue(_LAST.get("unit", 1.0))
        self.unit.setSuffix(" km²")
        self.cell_info = QLabel("")
        self.cell_info.setObjectName("hint")
        self.align = QCheckBox(tr("Resample evidence to the first layer's grid if grids differ"))
        self.align.setChecked(True)
        self.o_std = QCheckBox(tr("std. deviation"))
        self.o_std.setChecked(True)
        self.o_t = QCheckBox(tr("t-value (P / s(P))"))
        self.o_t.setChecked(True)
        self.o_logit = QCheckBox(tr("linear predictor (logit)"))
        self.prefix = QLineEdit(_LAST.get("prefix", "lr"))
        f4.addWidget(QLabel(tr("Estimator")), 0, 0)
        f4.addWidget(self.estimator, 0, 1)
        f4.addWidget(self.ridge, 0, 2)
        f4.addWidget(QLabel(tr("Variable selection")), 1, 0)
        f4.addWidget(self.selection, 1, 1)
        f4.addWidget(self.alpha, 1, 2)
        f4.addWidget(QLabel(tr("Missing evidence")), 2, 0)
        f4.addWidget(self.missing, 2, 1, 1, 2)
        f4.addWidget(QLabel(tr("Unit cell area")), 3, 0)
        f4.addWidget(self.unit, 3, 1)
        f4.addWidget(self.cell_info, 3, 2)
        f4.addWidget(self.align, 4, 0, 1, 3)
        f4.addWidget(QLabel(tr("Outputs")), 5, 0)
        outs = QHBoxLayout()
        outs.addWidget(QLabel(tr("probability +")))
        for w in (self.o_std, self.o_t, self.o_logit):
            outs.addWidget(w)
        outs.addStretch(1)
        f4.addLayout(outs, 5, 1, 1, 2)
        f4.addWidget(QLabel(tr("Output name prefix")), 6, 0)
        f4.addWidget(self.prefix, 6, 1)
        self.body.addWidget(g4)
        self.estimator.currentIndexChanged.connect(self._sync)
        self.selection.currentIndexChanged.connect(self._sync)
        self._sync()
        self._update_cell_info()
        self.resize(1000, 900)

    # ------------------------------------------------------------------ evidence rows
    def _add_row(self, lyr: RasterLayer, prev=None):
        r = self.table.rowCount()
        self.table.insertRow(r)
        vals = _sample(lyr)
        ncls = int(np.unique(vals).size)
        use = QCheckBox()
        use.setChecked(bool(prev and prev.get("use")))
        cell = QWidget()
        cl = QHBoxLayout(cell)
        cl.setContentsMargins(8, 0, 0, 0)
        cl.addWidget(use)
        self.table.setCellWidget(r, 0, cell)
        name = QTableWidgetItem(lyr.name)
        name.setIcon(theme.icon("mdi6.shape-outline" if lyr.categorical else "mdi6.grid",
                                "#b18cf2" if lyr.categorical else "#4fc1c9"))
        self.table.setItem(r, 1, name)
        info = QTableWidgetItem(tr("{n} classes", n=ncls) if lyr.categorical or ncls <= 32
                                else tr("continuous"))
        info.setTextAlignment(Qt.AlignCenter)
        info.setToolTip(tr("range {lo:g} – {hi:g}", lo=float(vals.min()) if vals.size else 0,
                           hi=float(vals.max()) if vals.size else 0))
        self.table.setItem(r, 2, info)
        kind = QComboBox()
        for k, v in VAR_KIND_LABELS.items():
            kind.addItem(tr(v), k)
        default = "categorical" if (lyr.categorical and ncls <= MAX_CLASSES) else "continuous"
        kind.setCurrentIndex(list(VAR_KIND_LABELS).index(prev.get("kind", default) if prev else default))
        trans = QComboBox()
        for k, v in TRANSFORM_LABELS.items():
            trans.addItem(tr(v), k)
        trans.setCurrentIndex(list(TRANSFORM_LABELS).index(prev.get("transform", "none") if prev else "none"))

        def sync(_i=None):
            trans.setEnabled(kind.currentData() == "continuous")
        kind.currentIndexChanged.connect(sync)
        sync()
        self.table.setCellWidget(r, 3, kind)
        self.table.setCellWidget(r, 4, trans)
        use.toggled.connect(lambda _c: self._update_cell_info())
        self._rows.append(dict(layer=lyr, use=use, kind=kind, transform=trans, n_classes=ncls,
                               min=float(vals.min()) if vals.size else np.nan))

    def _selected(self):
        return [r for r in self._rows if r["use"].isChecked()]

    def _sync(self, _i=None):
        sampled = self.design.currentData() == "sampled"
        for w in self._sampled_widgets:
            w.setEnabled(sampled)
        self.ridge.setEnabled(self.estimator.currentData() == "ridge")
        self.alpha.setEnabled(self.selection.currentData() == "backward_lr")

    def _update_cell_info(self):
        sel = self._selected()
        lyr = sel[0]["layer"] if sel else (self._rows[0]["layer"] if self._rows else None)
        if lyr is None:
            return
        a = lyr.grid.cell_area / 1e6
        txt = tr("raster cell = {a:.6g} km² ({rx:g} × {ry:g})", a=a, rx=lyr.grid.res[0], ry=lyr.grid.res[1])
        if lyr.grid.is_geographic:
            txt = tr("⚠ grid is in degrees — reproject to a metric CRS for meaningful areas")
        self.cell_info.setText(txt)

    def validate(self):
        if self.deposits.layer() is None:
            return tr("Add a point layer with known deposits (training points).")
        sel = self._selected()
        if not sel:
            return tr("Tick at least one evidence layer.")
        for r in sel:
            name = r["layer"].name
            if r["kind"].currentData() == "categorical" and r["n_classes"] > MAX_CLASSES:
                return tr("'{layer}' has more than {n} distinct values — use it as continuous or reclassify it.",
                          layer=name, n=MAX_CLASSES)
            if r["kind"].currentData() == "continuous" and r["transform"].currentData() == "log10" \
                    and not r["min"] > 0:
                return tr("'{layer}': log10 needs values > 0 (minimum {v:g}).", layer=name, v=r["min"])
        g0 = sel[0]["layer"].grid
        if not self.align.isChecked():
            for r in sel[1:]:
                if not r["layer"].grid.same_as(g0):
                    return tr("'{layer}' is not on the grid of '{ref}'. Enable resampling "
                              "or use Data ▸ Align raster to grid.", layer=r['layer'].name, ref=sel[0]['layer'].name)
        if g0.is_geographic:
            return tr("The evidence grid is in geographic degrees. Reproject it to a metric CRS first.")
        return None

    def params(self) -> dict:
        sel = self._selected()
        _LAST.update(test_pct=self.test_pct.value(), seed=self.seed.value(), unit=self.unit.value(),
                     prefix=self.prefix.text(), design=self.design.currentData(), neg_ratio=self.neg_ratio.value(),
                     min_dist=self.min_dist.value(), repeats=self.repeats.value(), prior=self.prior.isChecked(),
                     split=self.split.currentData(), thin=self.thin.value(), spacing=self.spacing.value(),
                     cv=self.cv.currentData(), estimator=self.estimator.currentData(), ridge=self.ridge.value(),
                     selection=self.selection.currentData(), alpha=self.alpha.value(),
                     missing=self.missing.currentData(),
                     evidence={r["layer"].name: dict(use=r["use"].isChecked(), kind=r["kind"].currentData(),
                                                     transform=r["transform"].currentData()) for r in self._rows})
        return dict(
            evidence=[dict(name=r["layer"].name, data=r["layer"].data, grid=r["layer"].grid,
                           labels=r["layer"].labels(), kind=r["kind"].currentData(),
                           transform=r["transform"].currentData() if r["kind"].currentData() == "continuous"
                           else "none") for r in sel],
            deposits=self.deposits.layer().gdf, deposits_name=self.deposits.layer().name,
            negatives=self.negatives.layer().gdf if self.negatives.layer() is not None else None,
            mask=(self.mask.layer().data, self.mask.layer().grid) if self.mask.layer() is not None else None,
            test_fraction=self.test_pct.value() / 100.0, seed=self.seed.value(), unit_cell_km2=self.unit.value(),
            split_method=self.split.currentData(), thin_distance=self.thin.value(), min_spacing=self.spacing.value(),
            design=self.design.currentData(), neg_ratio=self.neg_ratio.value(), min_distance=self.min_dist.value(),
            repeats=self.repeats.value(), prior_correction=self.prior.isChecked(), cv_folds=self.cv.currentData(),
            estimator=self.estimator.currentData(), ridge=self.ridge.value(), selection=self.selection.currentData(),
            alpha=self.alpha.value(), missing=self.missing.currentData(), auto_align=self.align.isChecked(),
            out_std=self.o_std.isChecked(), out_t=self.o_t.isChecked(), out_logit=self.o_logit.isChecked(),
            prefix=self.prefix.text().strip() or "lr")


# ============================================================================== results
class LogisticRegressionResultsDialog(QDialog):
    def __init__(self, result, window=None, parent=None):
        super().__init__(parent)
        self.result = result
        self.main_window = window          # not "self.window": that would hide QWidget.window()
        res = result.extra["lr"]
        self.setWindowTitle(tr("Logistic Regression — results"))
        lay = QVBoxLayout(self)
        lay.setContentsMargins(14, 12, 14, 12)
        t = QLabel(tr("Logistic Regression — results"))
        t.setObjectName("title")
        lay.addWidget(t)
        ev = result.evaluation
        st = res.stats
        cards = QHBoxLayout()
        cards.addWidget(metric_card(f"{res.n_deposits}", tr("training deposits (cells)")))
        cards.addWidget(metric_card(f"{st.get('mcfadden', float('nan')):.3f}", tr("McFadden pseudo-R²")))
        cards.addWidget(metric_card(f"{st['predicted']:.1f} / {res.n_deposits}",
                                    tr("predicted / observed deposits")))
        if ev is not None:
            cards.addWidget(metric_card(f"{ev.curves['training'].auc:.3f}", tr("ROC AUC — training")))
            if "test" in ev.curves:
                cards.addWidget(metric_card(f"{ev.curves['test'].auc:.3f}", tr("ROC AUC — test")))
        if len(res.sample_validation):
            sv = res.sample_validation.set_index("item")["value"]
            cv = sv.get("Cross-validated AUC (pooled held-out scores)")
            if cv is not None:
                cards.addWidget(metric_card(str(cv), tr("cross-validated AUC (samples)")))
        cards.addStretch(1)
        lay.addLayout(cards)
        if res.messages:
            w = hint("⚠ " + "\n⚠ ".join(res.messages))
            w.setStyleSheet(f"color:{theme.WARN}")
            lay.addWidget(w)

        self.tabs = QTabWidget()
        self.tabs.addTab(self._tab(res.fit_table, tr("<b>Model fit</b> — likelihood-ratio χ² against the "
                                                     "intercept-only model, information criteria and calibration.")),
                         tr("Summary"))
        sig = theme.ACCENT_DARK
        self.tabs.addTab(self._tab(res.coefficients, tr(
            "<b>Coefficients</b> per unit of the evidence value (continuous) or relative to the reference class "
            "(categorical); B_std = per standard deviation. Highlighted: p < 0.05."),
            highlight=lambda r: sig if r.get("p_value", 1) < 0.05 else None), tr("Coefficients"))
        self.tabs.addTab(self._tab(res.variables, tr(
            "<b>Evidence layers</b> — Wald and likelihood-ratio tests of each layer (all its coefficients = 0) and "
            "collinearity: GVIF^(1/(2·df)) > ~2.2 (VIF > 5) indicates strongly correlated layers."),
            highlight=lambda r: "#5a3a3a" if r.get("gvif_adj", 0) > 2.24 else None), tr("Evidence layers"))
        gof = QWidget()
        gl = QVBoxLayout(gof)
        gl.addWidget(QLabel(tr("<b>Hosmer–Lemeshow groups</b> (cells sorted by probability): observed vs expected "
                               "deposits; a small p-value means poor calibration.")))
        gl.addWidget(make_table(res.hosmer_lemeshow), 1)
        self.tabs.addTab(gof, tr("Goodness of fit"))
        if len(res.sample_validation):
            self.tabs.addTab(self._tab(res.sample_validation, tr(
                "<b>Deposit / non-deposit samples</b> — confusion matrix (p ≥ 0.5 on the sample scale), training AUC "
                "and cross-validation.")), tr("Sample validation"))
        if len(res.selection):
            self.tabs.addTab(self._tab(res.selection, tr("<b>Backward elimination</b> — layers removed step by "
                                                         "step.")), tr("Selection"))
        if len(res.repeats):
            self.tabs.addTab(self._tab(res.repeats, tr("<b>Random non-deposit samples</b> — coefficients and "
                                                       "training AUC of every sample (pooled with Rubin's rules).")),
                             tr("Repeats"))
        if len(res.correlation):
            self.tabs.addTab(self._tab(res.correlation, tr("<b>Correlation</b> of the design columns (z-scores "
                                                           "and class dummies) in the training rows.")),
                             tr("Correlation"))
        lay.addWidget(self.tabs, 1)

        btns = QHBoxLayout()
        b_roc = QPushButton(theme.icon("mdi6.chart-bell-curve-cumulative", "white"), "ROC / AUC…")
        b_roc.setProperty("primary", True)
        b_roc.clicked.connect(self.show_evaluation)
        b_roc.setEnabled(ev is not None)
        b_x = QPushButton(theme.icon("mdi6.microsoft-excel"), tr("Export tables (Excel)…"))
        b_x.clicked.connect(self.export_excel)
        close = QPushButton(tr("Close"))
        close.clicked.connect(self.accept)
        btns.addWidget(b_roc)
        btns.addWidget(b_x)
        btns.addStretch(1)
        btns.addWidget(close)
        lay.addLayout(btns)
        fit_to_screen(self, 1100, 740)

    @staticmethod
    def _tab(df: pd.DataFrame, text: str, highlight=None) -> QWidget:
        w = QWidget()
        l = QVBoxLayout(w)
        lab = QLabel(text)
        lab.setWordWrap(True)
        l.addWidget(lab)
        l.addWidget(make_table(df, highlight=highlight), 1)
        return w

    def show_evaluation(self):
        from gui.dialogs.evaluation_dialogs import EvaluationDialog
        if self.result.evaluation is not None:
            dlg = EvaluationDialog(self.result.evaluation, tr("Logistic Regression"), self)
            dlg.show()

    def export_excel(self):
        p, _ = QFileDialog.getSaveFileName(self, tr("Export logistic regression tables"), "lr_results.xlsx",
                                           "Excel (*.xlsx)")
        if not p:
            return
        res = self.result.extra["lr"]
        with pd.ExcelWriter(p) as xw:
            for name, df in (("model_fit", res.fit_table), ("coefficients", res.coefficients),
                             ("variables", res.variables), ("hosmer_lemeshow", res.hosmer_lemeshow),
                             ("sample_validation", res.sample_validation), ("selection", res.selection),
                             ("repeats", res.repeats), ("correlation", res.correlation)):
                if len(df):
                    df.to_excel(xw, sheet_name=name, index=False)
            if self.result.evaluation is not None:
                pd.DataFrame(self.result.evaluation.summary_rows()).to_excel(xw, sheet_name="validation", index=False)
        QMessageBox.information(self, tr("Export"), tr("Saved {path}", path=p))
