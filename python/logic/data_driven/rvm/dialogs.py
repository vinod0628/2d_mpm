"""Relevance-vector-machine parameter dialog and results dialog."""
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
from gui.widgets import (BaseDialog, LayerCombo, MplWidget, fit_to_screen, hint, make_table, metric_card,
                         split_method_combo, thinning_spin)
from i18n import N_, tr
from logic.common.features import FEATURE_KIND_LABELS, MAX_CLASSES, MISSING_RULE_LABELS

from .model import KERNEL_LABELS

_LAST = {}          # remembers settings during the session
CV_CHOICES = [(5, N_("5-fold")), (10, N_("10-fold")), (3, N_("3-fold"))]


def _sample(lyr: RasterLayer, n: int = 400) -> np.ndarray:
    d = lyr.data
    return np.ma.compressed(d[:: max(1, d.shape[0] // n), :: max(1, d.shape[1] // n)])


class RVMDialog(BaseDialog):
    COLS = [N_("Use"), N_("Evidence layer"), N_("Values"), N_("Variable type")]

    def __init__(self, store: LayerStore, parent=None):
        super().__init__(tr("Relevance Vector Machine"),
                         tr("Sparse Bayesian model (Tipping 2001): a logistic model on kernel functions centred on the "
                            "training samples; the Bayesian evidence keeps only a few of them (relevance vectors). "
                            "Gives a probability with its predictive uncertainty. Evidence is scaled (z-scores); "
                            "categorical = one 0/1 column per class. The score ranks prospectivity — it is not a "
                            "probability per unit cell."),
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
        for col, w in ((0, 44), (2, 110), (3, 250)):
            self.table.setColumnWidth(col, w)
        self._rows = []
        prev = _LAST.get("evidence", {})
        for lyr in store.rasters():
            if "mask" in lyr.name.lower():
                continue
            self._add_row(lyr, prev.get(lyr.name))
        v2.addWidget(self.table, 1)
        v2.addWidget(hint(tr("Tip: use log-transformed geochemistry — after scaling, every column has the same "
                             "influence on the kernel distance.")))
        self.body.addWidget(g2, 1)

        # ---------------------------------------------------------------- training samples
        g3 = QGroupBox(tr("Training samples"))
        f3 = QGridLayout(g3)
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
        self.repeats.setRange(1, 100)
        self.repeats.setValue(_LAST.get("repeats", 5))
        self.repeats.setToolTip(tr("Train one RVM per random non-deposit sample and average the scores "
                                   "(Zuo & Wang 2020); the spread is the score's std. deviation."))
        self.cv = QComboBox()
        for k, v in CV_CHOICES:
            self.cv.addItem(tr(v), k)
        self.cv.setCurrentIndex([k for k, _ in CV_CHOICES].index(_LAST.get("cv", 5)))
        self.cv.setToolTip(tr("Folds for the kernel-width tuning, the cross-validated AUC and the variable "
                              "importance."))
        f3.addWidget(QLabel(tr("Non-deposit samples per deposit")), 0, 0)
        f3.addWidget(self.neg_ratio, 0, 1)
        f3.addWidget(QLabel(tr("Min. distance from deposits")), 0, 2)
        f3.addWidget(self.min_dist, 0, 3)
        f3.addWidget(QLabel(tr("Min. spacing between samples")), 1, 0)
        f3.addWidget(self.spacing, 1, 1)
        f3.addWidget(QLabel(tr("Random samples (repeats)")), 1, 2)
        f3.addWidget(self.repeats, 1, 3)
        f3.addWidget(QLabel(tr("Cross-validation")), 2, 0)
        f3.addWidget(self.cv, 2, 1)
        self.body.addWidget(g3)

        # ---------------------------------------------------------------- model options
        g4 = QGroupBox(tr("Model options"))
        f4 = QGridLayout(g4)
        self.kernel = QComboBox()
        for k, v in KERNEL_LABELS.items():
            self.kernel.addItem(tr(v), k)
        self.kernel.setCurrentIndex(list(KERNEL_LABELS).index(_LAST.get("kernel", "rbf")))
        self.kernel.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.kernel.setMinimumContentsLength(18)
        self.tune = QCheckBox(tr("Tune the kernel width γ (cross-validated AUC)"))
        self.tune.setChecked(_LAST.get("tune", True))
        self.gamma = QDoubleSpinBox()
        self.gamma.setRange(0, 1000)
        self.gamma.setDecimals(4)
        self.gamma.setSingleStep(0.01)
        self.gamma.setSpecialValueText(tr("auto (1 / p)"))
        self.gamma.setValue(_LAST.get("gamma", 0.0))
        self.gamma.setToolTip(tr("Kernel width: small γ = smooth, regional response; large γ = local response around "
                                 "individual samples. auto = 1 / number of columns."))
        self.missing = QComboBox()
        for k, v in MISSING_RULE_LABELS.items():
            self.missing.addItem(tr(v), k)
        self.missing.setCurrentIndex(list(MISSING_RULE_LABELS).index(_LAST.get("missing", "exclude")))
        self.missing.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.missing.setMinimumContentsLength(18)
        self.align = QCheckBox(tr("Resample evidence to the first layer's grid if grids differ"))
        self.align.setChecked(True)
        self.o_psd = QCheckBox(tr("predictive std. deviation"))
        self.o_psd.setChecked(True)
        self.o_std = QCheckBox(tr("std. deviation over the samples"))
        self.o_std.setChecked(True)
        self.prefix = QLineEdit(_LAST.get("prefix", "rvm"))
        f4.addWidget(QLabel(tr("Basis functions")), 0, 0)
        f4.addWidget(self.kernel, 0, 1)
        f4.addWidget(self.tune, 0, 2, 1, 2)
        f4.addWidget(QLabel(tr("γ (kernel width)")), 1, 0)
        f4.addWidget(self.gamma, 1, 1)
        f4.addWidget(QLabel(tr("Missing evidence")), 1, 2)
        f4.addWidget(self.missing, 1, 3)
        f4.addWidget(self.align, 2, 0, 1, 4)
        f4.addWidget(QLabel(tr("Outputs")), 3, 0)
        outs = QHBoxLayout()
        outs.addWidget(QLabel(tr("score + relevance vectors +")))
        outs.addWidget(self.o_psd)
        outs.addWidget(self.o_std)
        outs.addStretch(1)
        f4.addLayout(outs, 3, 1, 1, 3)
        f4.addWidget(QLabel(tr("Output name prefix")), 4, 0)
        f4.addWidget(self.prefix, 4, 1)
        self.body.addWidget(g4)
        self.kernel.currentIndexChanged.connect(self._sync)
        self.tune.toggled.connect(self._sync)
        self._sync()
        self.resize(1000, 900)

    def _sync(self):
        rbf = self.kernel.currentData() == "rbf"
        self.tune.setEnabled(rbf)
        self.gamma.setEnabled(rbf and not self.tune.isChecked())

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
        info = QTableWidgetItem(tr("{n} classes", n=ncls) if lyr.categorical or ncls <= 32 else tr("continuous"))
        info.setTextAlignment(Qt.AlignCenter)
        info.setToolTip(tr("range {lo:g} – {hi:g}", lo=float(vals.min()) if vals.size else 0,
                           hi=float(vals.max()) if vals.size else 0))
        self.table.setItem(r, 2, info)
        kind = QComboBox()
        for k, v in FEATURE_KIND_LABELS.items():
            kind.addItem(tr(v), k)
        default = "categorical" if (lyr.categorical and ncls <= MAX_CLASSES) else "continuous"
        kind.setCurrentIndex(list(FEATURE_KIND_LABELS).index(prev.get("kind", default) if prev else default))
        self.table.setCellWidget(r, 3, kind)
        self._rows.append(dict(layer=lyr, use=use, kind=kind, n_classes=ncls))

    def _selected(self):
        return [r for r in self._rows if r["use"].isChecked()]

    def validate(self):
        if self.deposits.layer() is None:
            return tr("Add a point layer with known deposits (training points).")
        sel = self._selected()
        if not sel:
            return tr("Tick at least one evidence layer.")
        for r in sel:
            if r["kind"].currentData() == "categorical" and r["n_classes"] > MAX_CLASSES:
                return tr("'{layer}' has more than {n} distinct values — use it as continuous or reclassify it.",
                          layer=r["layer"].name, n=MAX_CLASSES)
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
        _LAST.update(test_pct=self.test_pct.value(), seed=self.seed.value(), split=self.split.currentData(),
                     thin=self.thin.value(), neg_ratio=self.neg_ratio.value(), min_dist=self.min_dist.value(),
                     spacing=self.spacing.value(), repeats=self.repeats.value(), cv=self.cv.currentData(),
                     kernel=self.kernel.currentData(), tune=self.tune.isChecked(), gamma=self.gamma.value(),
                     missing=self.missing.currentData(), prefix=self.prefix.text(),
                     evidence={r["layer"].name: dict(use=r["use"].isChecked(), kind=r["kind"].currentData())
                               for r in self._rows})
        return dict(
            evidence=[dict(name=r["layer"].name, data=r["layer"].data, grid=r["layer"].grid,
                           labels=r["layer"].labels(), kind=r["kind"].currentData()) for r in sel],
            deposits=self.deposits.layer().gdf, deposits_name=self.deposits.layer().name,
            negatives=self.negatives.layer().gdf if self.negatives.layer() is not None else None,
            mask=(self.mask.layer().data, self.mask.layer().grid) if self.mask.layer() is not None else None,
            test_fraction=self.test_pct.value() / 100.0, seed=self.seed.value(),
            split_method=self.split.currentData(), thin_distance=self.thin.value(),
            neg_ratio=self.neg_ratio.value(), min_distance=self.min_dist.value(), min_spacing=self.spacing.value(),
            repeats=self.repeats.value(), cv_folds=self.cv.currentData(), kernel=self.kernel.currentData(),
            tune=self.tune.isChecked(), gamma=self.gamma.value(), missing=self.missing.currentData(),
            auto_align=self.align.isChecked(), out_pred_sd=self.o_psd.isChecked(), out_std=self.o_std.isChecked(),
            prefix=self.prefix.text().strip() or "rvm")


# ============================================================================== results
class RVMResultsDialog(QDialog):
    def __init__(self, result, window=None, parent=None):
        super().__init__(parent)
        self.result = result
        self.main_window = window          # not "self.window": that would hide QWidget.window()
        res = result.extra["rvm"]
        self.setWindowTitle(tr("Relevance Vector Machine — results"))
        lay = QVBoxLayout(self)
        lay.setContentsMargins(14, 12, 14, 12)
        t = QLabel(tr("Relevance Vector Machine — results"))
        t.setObjectName("title")
        lay.addWidget(t)
        ev = result.evaluation
        cards = QHBoxLayout()
        cards.addWidget(metric_card(f"{res.n_deposits} / {res.n_negatives}", tr("training deposits / non-deposits")))
        cards.addWidget(metric_card(f"{res.stats['relevance_vectors']} / {res.stats['samples']}",
                                    tr("relevance vectors / samples") if res.params["kernel"] == "rbf"
                                    else tr("relevant columns / samples")))
        cards.addWidget(metric_card(f"{res.stats['cv_auc']:.3f}", tr("cross-validated AUC (samples)")))
        if ev is not None:
            cards.addWidget(metric_card(f"{ev.curves['training'].auc:.3f}", tr("ROC AUC — training")))
            if "test" in ev.curves:
                cards.addWidget(metric_card(f"{ev.curves['test'].auc:.3f}", tr("ROC AUC — test")))
        cards.addStretch(1)
        lay.addLayout(cards)
        if result.messages:
            w = hint("⚠ " + "\n⚠ ".join(result.messages))
            w.setStyleSheet(f"color:{theme.WARN}")
            lay.addWidget(w)

        self.tabs = QTabWidget()
        self.tabs.addTab(self._tab(res.summary, tr("<b>Model</b> — basis, kernel width, relevance vectors, "
                                                   "evidence and cross-validated performance.")), tr("Summary"))
        self.tabs.addTab(self._tab(res.relevance, tr(
            "<b>Relevance vectors</b> — the training samples (RBF basis) or evidence columns (linear basis) kept by "
            "the Bayesian evidence, with their posterior weights; RBF: evidence values of each sample (also a point "
            "layer on the map).")), tr("Relevance vectors"))
        if len(res.tuning):
            self.tabs.addTab(self._tuning_tab(res), tr("Kernel width"))
        self.tabs.addTab(self._importance_tab(res), tr("Variable importance"))
        self.tabs.addTab(self._tab(res.sample_validation, tr(
            "<b>Training samples</b> — confusion matrices at probability 0.5 for the training samples and for the "
            "held-out cross-validation predictions, accuracy and AUC.")), tr("Sample validation"))
        if len(res.repeats) > 1:
            self.tabs.addTab(self._tab(res.repeats, tr("<b>Random non-deposit samples</b> — one RVM per sample; the "
                                                       "scores are averaged.")), tr("Repeats"))
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
        fit_to_screen(self, 1080, 740)

    @staticmethod
    def _tab(df: pd.DataFrame, text: str) -> QWidget:
        w = QWidget()
        l = QVBoxLayout(w)
        lab = QLabel(text)
        lab.setWordWrap(True)
        l.addWidget(lab)
        l.addWidget(make_table(df), 1)
        return w

    def _tuning_tab(self, res) -> QWidget:
        w = QWidget()
        l = QVBoxLayout(w)
        lab = QLabel(tr("<b>Kernel width</b> — cross-validated AUC of every γ on the first training sample; the best "
                        "γ is used (ties → smaller γ)."))
        lab.setWordWrap(True)
        l.addWidget(lab)
        row = QHBoxLayout()
        row.addWidget(make_table(res.tuning.drop(columns=["log2_gamma"])), 1)
        canvas = MplWidget(5, 3)
        canvas.setMinimumHeight(200)
        ax = canvas.fig.add_subplot(111)
        ax.plot(res.tuning.log2_gamma, res.tuning.cv_auc, "o-", color=theme.SERIES[0])
        ax.axvline(np.log2(res.stats["gamma"]), color="r", lw=1, ls="--")
        ax.set_xlabel("log2 γ", fontsize=8)
        ax.set_ylabel("CV AUC", fontsize=8)
        ax.tick_params(labelsize=8)
        row.addWidget(canvas, 1)
        l.addLayout(row, 1)
        self.tuning_canvas = canvas
        return w

    def _importance_tab(self, res) -> QWidget:
        w = QWidget()
        l = QVBoxLayout(w)
        lab = QLabel(tr("<b>Variable importance</b> — drop of the cross-validated AUC when the values of a layer are "
                        "shuffled in the held-out folds (mean ± s.d. of 5 shuffles); linear basis: also the columns "
                        "kept by relevance determination and their weights."))
        lab.setWordWrap(True)
        l.addWidget(lab)
        row = QHBoxLayout()
        row.addWidget(make_table(res.importance), 1)
        canvas = MplWidget(5, 3)
        canvas.setMinimumHeight(200)
        ax = canvas.fig.add_subplot(111)
        imp = res.importance.iloc[::-1]
        y = np.arange(len(imp))
        ax.barh(y, imp.permutation_auc_drop, xerr=imp.permutation_sd, color=theme.SERIES[0], ecolor=theme.MUTED)
        ax.set_yticks(y)
        ax.set_yticklabels(imp.layer, fontsize=8)
        ax.tick_params(axis="x", labelsize=8)
        ax.set_xlabel(tr("CV AUC drop when shuffled"), fontsize=8)
        ax.axvline(0, color=theme.MUTED, lw=0.8)
        row.addWidget(canvas, 1)
        l.addLayout(row, 1)
        self.importance_canvas = canvas
        return w

    def show_evaluation(self):
        from gui.dialogs.evaluation_dialogs import EvaluationDialog
        if self.result.evaluation is not None:
            dlg = EvaluationDialog(self.result.evaluation, tr("Relevance Vector Machine"), self)
            dlg.show()

    def export_excel(self):
        p, _ = QFileDialog.getSaveFileName(self, tr("Export RVM tables"), "rvm_results.xlsx", "Excel (*.xlsx)")
        if not p:
            return
        res = self.result.extra["rvm"]
        with pd.ExcelWriter(p) as xw:
            for name, df in (("summary", res.summary), ("relevance_vectors", res.relevance), ("tuning", res.tuning),
                             ("importance", res.importance), ("sample_validation", res.sample_validation),
                             ("repeats", res.repeats)):
                if len(df):
                    df.to_excel(xw, sheet_name=name, index=False)
            if self.result.evaluation is not None:
                pd.DataFrame(self.result.evaluation.summary_rows()).to_excel(xw, sheet_name="validation", index=False)
        QMessageBox.information(self, tr("Export"), tr("Saved {path}", path=p))
