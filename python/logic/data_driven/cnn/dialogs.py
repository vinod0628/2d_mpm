"""Convolutional-neural-network (CNN) parameter dialog and results dialog."""
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

from .model import parse_filters

_LAST = {}          # remembers settings during the session
CV_CHOICES = [(3, N_("3-fold")), (5, N_("5-fold")), (10, N_("10-fold"))]


def _sample(lyr: RasterLayer, n: int = 400) -> np.ndarray:
    d = lyr.data
    return np.ma.compressed(d[:: max(1, d.shape[0] // n), :: max(1, d.shape[1] // n)])


class CNNDialog(BaseDialog):
    COLS = [N_("Use"), N_("Evidence layer"), N_("Values"), N_("Variable type")]

    def __init__(self, store: LayerStore, parent=None):
        super().__init__(tr("Convolutional Neural Network (CNN)"),
                         tr("Patch-based CNN (TensorFlow / Keras): each cell is represented by the window of all evidence "
                            "layers around it, so the network learns the spatial context of deposits. Training patches "
                            "are augmented by rotations / reflections; several networks are averaged. The score ranks "
                            "prospectivity — it is not a probability per unit cell. TensorFlow uses an NVIDIA GPU when "
                            "it can see one; on native Windows it runs on the CPU."),
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
        self.repeats.setToolTip(tr("Train CNNs for each random non-deposit sample and average the scores "
                                   "(Zuo & Wang 2020); the spread is the score's std. deviation."))
        self.cv = QComboBox()
        for k, v in CV_CHOICES:
            self.cv.addItem(tr(v), k)
        self.cv.setCurrentIndex([k for k, _ in CV_CHOICES].index(_LAST.get("cv", 3)))
        self.cv.setToolTip(tr("Folds for the number of epochs (early stopping), the cross-validated AUC and the "
                              "variable importance."))
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
        self.patch = QSpinBox()
        self.patch.setRange(3, 33)
        self.patch.setSingleStep(2)
        self.patch.setValue(_LAST.get("patch", 9))
        self.patch.setToolTip(tr("Window size in cells (odd): the spatial context the network sees around a cell."))
        self.patch_info = QLabel("")
        self.patch_info.setObjectName("hint")
        self.filters = QLineEdit(_LAST.get("filters", "8, 16"))
        self.filters.setToolTip(tr("Filters of each 3 × 3 convolution layer, separated by commas."))
        self.dense = QSpinBox()
        self.dense.setRange(1, 1024)
        self.dense.setValue(_LAST.get("dense", 16))
        self.dropout = QDoubleSpinBox()
        self.dropout.setRange(0, 0.9)
        self.dropout.setDecimals(2)
        self.dropout.setSingleStep(0.05)
        self.dropout.setValue(_LAST.get("dropout", 0.5))
        self.lr = QDoubleSpinBox()
        self.lr.setRange(1e-5, 1.0)
        self.lr.setDecimals(5)
        self.lr.setSingleStep(0.0005)
        self.lr.setValue(_LAST.get("lr", 0.0003))
        self.batch = QSpinBox()
        self.batch.setRange(1, 4096)
        self.batch.setValue(_LAST.get("batch", 32))
        self.epochs = QSpinBox()
        self.epochs.setRange(5, 5000)
        self.epochs.setSingleStep(50)
        self.epochs.setValue(_LAST.get("epochs", 200))
        self.patience = QSpinBox()
        self.patience.setRange(1, 1000)
        self.patience.setValue(_LAST.get("patience", 20))
        self.augment = QCheckBox(tr("Augment training patches (8 rotations / reflections)"))
        self.augment.setChecked(_LAST.get("augment", True))
        self.networks = QSpinBox()
        self.networks.setRange(1, 50)
        self.networks.setValue(_LAST.get("networks", 2))
        self.networks.setToolTip(tr("Networks with different initial weights per random sample; their outputs are "
                                    "averaged and their spread is the uncertainty."))
        self.missing = QComboBox()
        for k, v in MISSING_RULE_LABELS.items():
            self.missing.addItem(tr(v), k)
        self.missing.setCurrentIndex(list(MISSING_RULE_LABELS).index(_LAST.get("missing", "exclude")))
        self.missing.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.missing.setMinimumContentsLength(16)
        self.align = QCheckBox(tr("Resample evidence to the first layer's grid if grids differ"))
        self.align.setChecked(True)
        self.o_std = QCheckBox(tr("std. deviation of the network outputs"))
        self.o_std.setChecked(True)
        self.prefix = QLineEdit(_LAST.get("prefix", "cnn"))
        grid = [(tr("Patch size (cells)"), self.patch, None, self.patch_info),
                (tr("Convolution filters"), self.filters, tr("Dense neurons"), self.dense),
                (tr("Dropout"), self.dropout, tr("Learning rate (Adam)"), self.lr),
                (tr("Batch size"), self.batch, tr("Max. epochs"), self.epochs),
                (tr("Early stopping (epochs)"), self.patience, tr("Networks per sample"), self.networks),
                (tr("Missing evidence"), self.missing, None, None)]
        for i, (a, wa, b, wb) in enumerate(grid):
            f4.addWidget(QLabel(a), i, 0)
            f4.addWidget(wa, i, 1)
            if b is not None:
                f4.addWidget(QLabel(b), i, 2)
                f4.addWidget(wb, i, 3)
            elif wb is not None:
                f4.addWidget(wb, i, 2, 1, 2)
        i = len(grid)
        f4.addWidget(self.augment, i, 0, 1, 4)
        f4.addWidget(self.align, i + 1, 0, 1, 4)
        f4.addWidget(QLabel(tr("Outputs")), i + 2, 0)
        outs = QHBoxLayout()
        outs.addWidget(QLabel(tr("score +")))
        outs.addWidget(self.o_std)
        outs.addStretch(1)
        f4.addLayout(outs, i + 2, 1, 1, 3)
        f4.addWidget(QLabel(tr("Output name prefix")), i + 3, 0)
        f4.addWidget(self.prefix, i + 3, 1)
        self.body.addWidget(g4)
        self.patch.valueChanged.connect(self._patch_info)
        self._patch_info()
        self.resize(1000, 900)

    def _patch_info(self):
        cell = next((l.grid.res[0] for l in self.store.rasters()), None)
        k = self.patch.value()
        self.patch_info.setText(tr("window {w:,.0f} × {w:,.0f} map units", w=k * cell) if cell else "")

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
                     patch=self.patch.value(), filters=self.filters.text(), dense=self.dense.value(),
                     dropout=self.dropout.value(), lr=self.lr.value(), batch=self.batch.value(),
                     epochs=self.epochs.value(), patience=self.patience.value(), augment=self.augment.isChecked(),
                     networks=self.networks.value(),
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
            repeats=self.repeats.value(), cv_folds=self.cv.currentData(), patch=self.patch.value(),
            filters=parse_filters(self.filters.text()), dense=self.dense.value(), dropout=self.dropout.value(),
            learning_rate=self.lr.value(), batch_size=self.batch.value(), max_epochs=self.epochs.value(),
            patience=self.patience.value(), augment=self.augment.isChecked(), networks=self.networks.value(),
            missing=self.missing.currentData(), auto_align=self.align.isChecked(), out_std=self.o_std.isChecked(),
            prefix=self.prefix.text().strip() or "cnn")


# ============================================================================== results
class CNNResultsDialog(QDialog):
    def __init__(self, result, window=None, parent=None):
        super().__init__(parent)
        self.result = result
        self.main_window = window          # not "self.window": that would hide QWidget.window()
        res = result.extra["cnn"]
        self.setWindowTitle(tr("Convolutional Neural Network (CNN) — results"))
        lay = QVBoxLayout(self)
        lay.setContentsMargins(14, 12, 14, 12)
        t = QLabel(tr("Convolutional Neural Network (CNN) — results"))
        t.setObjectName("title")
        lay.addWidget(t)
        ev = result.evaluation
        cards = QHBoxLayout()
        cards.addWidget(metric_card(f"{res.n_deposits} / {res.n_negatives}", tr("training deposits / non-deposits")))
        cards.addWidget(metric_card(f"{res.stats['cv_auc']:.3f}", tr("cross-validated AUC (samples)")))
        cards.addWidget(metric_card(f"{res.stats['epochs']}", tr("epochs (early stopping)")))
        cards.addWidget(metric_card(f"{res.stats['networks']}", tr("networks averaged")))
        cards.addWidget(metric_card(res.stats["device"], tr("device")))
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
        self.tabs.addTab(self._tab(res.summary, tr("<b>Model</b> — patch, architecture, augmentation, epochs and "
                                                   "cross-validated performance.")), tr("Summary"))
        self.tabs.addTab(self._curve_tab(res), tr("Learning curve"))
        self.tabs.addTab(self._importance_tab(res), tr("Variable importance"))
        self.tabs.addTab(self._tab(res.sample_validation, tr(
            "<b>Training samples</b> — confusion matrices at output 0.5 for the training samples and for the held-out "
            "cross-validation predictions, accuracy and AUC.")), tr("Sample validation"))
        if len(res.repeats) > 1:
            self.tabs.addTab(self._tab(res.repeats, tr("<b>Random non-deposit samples</b> — networks per sample; all "
                                                       "outputs are averaged.")), tr("Repeats"))
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

    def _curve_tab(self, res) -> QWidget:
        w = QWidget()
        l = QVBoxLayout(w)
        lab = QLabel(tr("<b>Learning curve</b> — mean AUC of the training and held-out folds after every epoch (first "
                        "sample); the final networks are trained for the median best epoch of the folds."))
        lab.setWordWrap(True)
        l.addWidget(lab)
        canvas = MplWidget(6, 3.2)
        canvas.setMinimumHeight(220)
        ax = canvas.fig.add_subplot(111)
        c = res.curve
        ax.plot(c["epoch"], c.train_auc, color=theme.SERIES[1], label=tr("training folds"))
        ax.plot(c["epoch"], c.cv_auc, color=theme.SERIES[0], label=tr("held-out folds"))
        ax.axvline(res.stats["epochs"], color="r", lw=1, ls="--")
        ax.set_xlabel(tr("epoch"), fontsize=8)
        ax.set_ylabel("AUC", fontsize=8)
        ax.tick_params(labelsize=8)
        ax.legend(fontsize=8)
        l.addWidget(canvas, 1)
        self.curve_canvas = canvas
        return w

    def _importance_tab(self, res) -> QWidget:
        w = QWidget()
        l = QVBoxLayout(w)
        lab = QLabel(tr("<b>Variable importance</b> — drop of the cross-validated AUC when the windows of a layer are "
                        "exchanged between held-out samples (mean ± s.d. of 3 shuffles)."))
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
            dlg = EvaluationDialog(self.result.evaluation, tr("Convolutional Neural Network (CNN)"), self)
            dlg.show()

    def export_excel(self):
        p, _ = QFileDialog.getSaveFileName(self, tr("Export CNN tables"), "cnn_results.xlsx", "Excel (*.xlsx)")
        if not p:
            return
        res = self.result.extra["cnn"]
        with pd.ExcelWriter(p) as xw:
            for name, df in (("summary", res.summary), ("learning_curve", res.curve), ("importance", res.importance),
                             ("sample_validation", res.sample_validation), ("repeats", res.repeats)):
                if len(df):
                    df.to_excel(xw, sheet_name=name, index=False)
            if self.result.evaluation is not None:
                pd.DataFrame(self.result.evaluation.summary_rows()).to_excel(xw, sheet_name="validation", index=False)
        QMessageBox.information(self, tr("Export"), tr("Saved {path}", path=p))
