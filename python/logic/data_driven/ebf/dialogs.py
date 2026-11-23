"""Evidential-belief-function parameter dialog and results dialog."""
from __future__ import annotations

import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QAbstractItemView, QCheckBox, QComboBox, QGridLayout, QGroupBox, QHBoxLayout,
                             QHeaderView, QLabel, QLineEdit, QSpinBox, QTableWidget, QTableWidgetItem, QVBoxLayout,
                             QWidget)

from core.layers import LayerStore, RasterLayer
from gui import theme
from gui.widgets import (BaseDialog, LayerCombo, MplWidget, hint, raster_value_kind, split_method_combo,
                         thinning_spin)
from i18n import N_, tr
from logic.hybrid.common.dialogs import HybridResultsDialog

from .model import CLASSING_LABELS, MAX_QUANTILES, MAX_VALUES, SCORE_LABELS

_LAST = {}          # remembers settings during the session


def default_classing(layer: RasterLayer, n_values: int, kind: str) -> str:
    """Class values for class rasters and integer layers with few values, quantile classes otherwise."""
    t = layer.class_table
    if layer.categorical or (t is not None and {"lower", "upper"} <= set(t.columns)):
        return "values"
    return "values" if kind == "integer" and n_values <= 32 else "quantile"


class EBFDialog(BaseDialog):
    COLS = [N_("Use"), N_("Evidence layer"), N_("Values"), N_("Classes"), N_("Quantiles")]

    def __init__(self, store: LayerStore, parent=None):
        super().__init__(tr("Evidential Belief Functions"),
                         tr("Data-driven Dempster–Shafer model (An et al., 1994; Carranza & Hale, 2003): every class "
                            "of every layer gets a belief Bel (support for deposits), disbelief Dis, uncertainty "
                            "Unc and plausibility Pls from the training deposits; the layers are combined with "
                            "Dempster's rule. Class rasters use their classes, continuous layers quantile classes."),
                         parent, ok_text=tr("Run model"))
        self.store = store

        # ---------------------------------------------------------------- training data
        g1 = QGroupBox(tr("Training data"))
        f1 = QGridLayout(g1)
        self.deposits = LayerCombo(store, "point")
        dep = next((l for l in store.vectors("point") if "deposit" in l.name.lower()), None)
        if dep is not None:
            self.deposits.set_layer(dep)
        self.mask = LayerCombo(store, "raster", allow_none=True, none_text=tr("— cells with evidence —"))
        sm = next((l for l in store.rasters() if "mask" in l.name.lower()), None)
        if sm is not None:
            self.mask.set_layer(sm)
        else:
            self.mask.setCurrentIndex(0)
        self.test_pct = QSpinBox()
        self.test_pct.setRange(0, 50)
        self.test_pct.setSuffix(" %")
        self.test_pct.setValue(_LAST.get("test_pct", 20))
        self.seed = QSpinBox()
        self.seed.setRange(0, 10 ** 6)
        self.seed.setValue(_LAST.get("seed", 42))
        self.split = split_method_combo(_LAST.get("split", "spatial"))
        self.thin = thinning_spin(_LAST.get("thin", 0.0))
        self.negatives = LayerCombo(store, "point", allow_none=True, none_text=tr("— all background cells —"))
        self.negatives.setCurrentIndex(0)
        self.negatives.setToolTip(tr("Optional known non-deposit points used as ROC negatives instead of all "
                                     "background cells."))
        row = QHBoxLayout()
        row.addWidget(self.test_pct)
        row.addWidget(self.split)
        row.addWidget(QLabel(tr("random seed")))
        row.addWidget(self.seed)
        row.addStretch(1)
        f1.addWidget(QLabel(tr("Deposits (points)")), 0, 0)
        f1.addWidget(self.deposits, 0, 1)
        f1.addWidget(QLabel(tr("Study area mask")), 0, 2)
        f1.addWidget(self.mask, 0, 3)
        f1.addWidget(QLabel(tr("Hold out for testing")), 1, 0)
        f1.addLayout(row, 1, 1, 1, 3)
        f1.addWidget(QLabel(tr("Thin deposits (min. distance)")), 2, 0)
        f1.addWidget(self.thin, 2, 1)
        f1.addWidget(QLabel(tr("Absence points (ROC)")), 2, 2)
        f1.addWidget(self.negatives, 2, 3)
        f1.setColumnStretch(1, 1)
        f1.setColumnStretch(3, 1)
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
        for col, w in ((0, 44), (2, 64), (3, 170), (4, 84)):
            self.table.setColumnWidth(col, w)
        self._rows = []
        prev = _LAST.get("evidence", {})
        for lyr in store.rasters():
            if "mask" in lyr.name.lower():
                continue
            self._add_row(lyr, prev.get(lyr.name))
        v2.addWidget(self.table, 1)
        v2.addWidget(hint(tr("Class values: one belief function per class (class rasters, e.g. geology or "
                             "distance classes). Quantile classes: equal-area classes of the values inside the study "
                             "area (continuous layers). Few deposits → few classes.")))
        self.body.addWidget(g2, 1)

        # ---------------------------------------------------------------- model options
        g3 = QGroupBox(tr("Model options"))
        f3 = QGridLayout(g3)
        self.missing = QComboBox()
        self.missing.addItem(tr("vacuous belief (Unc = 1, keep area)"), "vacuous")
        self.missing.addItem(tr("exclude cells from study area"), "exclude")
        self.missing.setToolTip(tr("A cell without evidence of a layer gets Bel 0, Dis 0 and Unc 1 for that layer — "
                                   "the identity of Dempster's rule: only the other layers decide."))
        self.zero = QComboBox()
        self.zero.addItem(tr("Bel = 0 (original EBF)"), "zero")
        self.zero.addItem(tr("0.5-deposit correction (small Bel)"), "haldane")
        self.score = QComboBox()
        for k, v in SCORE_LABELS.items():
            self.score.addItem(tr(v), k)
        self.score.setCurrentIndex(list(SCORE_LABELS).index(_LAST.get("score", "bel")))
        self.score.setToolTip(tr("Bel is the usual EBF prospectivity map. Pls = 1 − Dis also counts the uncertain "
                                 "support (e.g. cells without evidence); the mid-point lies between them."))
        self.align = QCheckBox(tr("Resample evidence to the first layer's grid if grids differ"))
        self.align.setChecked(True)
        self.o_maps = QCheckBox(tr("Dis, Unc and Pls maps"))
        self.o_maps.setChecked(_LAST.get("out_maps", True))
        self.o_layers = QCheckBox(tr("belief map of every layer"))
        self.o_layers.setChecked(_LAST.get("out_layers", False))
        self.prefix = QLineEdit(_LAST.get("prefix", "ebf"))
        f3.addWidget(QLabel(tr("Missing evidence")), 0, 0)
        f3.addWidget(self.missing, 0, 1)
        f3.addWidget(QLabel(tr("Classes without deposits")), 0, 2)
        f3.addWidget(self.zero, 0, 3)
        f3.addWidget(QLabel(tr("Prospectivity score")), 1, 0)
        f3.addWidget(self.score, 1, 1)
        f3.addWidget(QLabel(tr("Output name prefix")), 1, 2)
        f3.addWidget(self.prefix, 1, 3)
        f3.addWidget(QLabel(tr("Outputs")), 2, 0)
        outs = QHBoxLayout()
        outs.addWidget(QLabel(tr("score +")))
        outs.addWidget(self.o_maps)
        outs.addWidget(self.o_layers)
        outs.addStretch(1)
        f3.addLayout(outs, 2, 1, 1, 3)
        f3.addWidget(self.align, 3, 0, 1, 4)
        f3.setColumnStretch(1, 1)
        f3.setColumnStretch(3, 1)
        self.body.addWidget(g3)
        self.resize(940, 760)

    def _add_row(self, lyr: RasterLayer, prev=None):
        r = self.table.rowCount()
        self.table.insertRow(r)
        nval, kind = raster_value_kind(lyr)
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
        info = QTableWidgetItem(str(nval) if kind != "real" else tr("real"))
        info.setTextAlignment(Qt.AlignCenter)
        self.table.setItem(r, 2, info)
        cls = QComboBox()
        for k, v in CLASSING_LABELS.items():
            cls.addItem(tr(v), k)
        cls.setCurrentIndex(list(CLASSING_LABELS).index(prev.get("classing") if prev else
                                                         default_classing(lyr, nval, kind)))
        n = QSpinBox()
        n.setRange(2, MAX_QUANTILES)
        n.setValue(prev.get("n_classes", 10) if prev else 10)
        cls.currentIndexChanged.connect(lambda _i: n.setEnabled(cls.currentData() == "quantile"))
        n.setEnabled(cls.currentData() == "quantile")
        self.table.setCellWidget(r, 3, cls)
        self.table.setCellWidget(r, 4, n)
        self._rows.append(dict(layer=lyr, use=use, classing=cls, n=n, n_values=nval, kind=kind))

    def _selected(self):
        return [r for r in self._rows if r["use"].isChecked()]

    def validate(self):
        if self.deposits.layer() is None:
            return tr("Add a point layer with known deposits (training points).")
        sel = self._selected()
        if not sel:
            return tr("Tick at least one evidence layer.")
        for r in sel:
            if r["classing"].currentData() == "values" and (r["kind"] == "real" or r["n_values"] > MAX_VALUES):
                return tr("'{layer}' has real values or more than {n} distinct values — use quantile classes.",
                          layer=r["layer"].name, n=MAX_VALUES)
        g0 = sel[0]["layer"].grid
        if not self.align.isChecked():
            for r in sel[1:]:
                if not r["layer"].grid.same_as(g0):
                    return tr("'{layer}' is not on the grid of '{ref}'. Enable resampling "
                              "or use Data ▸ Align raster to grid.", layer=r['layer'].name, ref=sel[0]['layer'].name)
        return None

    def params(self) -> dict:
        sel = self._selected()
        _LAST.update(test_pct=self.test_pct.value(), seed=self.seed.value(), split=self.split.currentData(),
                     thin=self.thin.value(), prefix=self.prefix.text(), score=self.score.currentData(),
                     out_maps=self.o_maps.isChecked(), out_layers=self.o_layers.isChecked(),
                     evidence={r["layer"].name: dict(use=r["use"].isChecked(), classing=r["classing"].currentData(),
                                                     n_classes=r["n"].value()) for r in self._rows})
        return dict(
            evidence=[dict(name=r["layer"].name, data=r["layer"].data, grid=r["layer"].grid,
                           labels=r["layer"].labels(), classing=r["classing"].currentData(), n_classes=r["n"].value())
                      for r in sel],
            deposits=self.deposits.layer().gdf, deposits_name=self.deposits.layer().name,
            negatives=self.negatives.layer().gdf if self.negatives.layer() is not None else None,
            mask=(self.mask.layer().data, self.mask.layer().grid) if self.mask.layer() is not None else None,
            test_fraction=self.test_pct.value() / 100.0, seed=self.seed.value(), split_method=self.split.currentData(),
            thin_distance=self.thin.value(), missing=self.missing.currentData(), zero_deposit=self.zero.currentData(),
            score=self.score.currentData(), auto_align=self.align.isChecked(), out_maps=self.o_maps.isChecked(),
            out_layers=self.o_layers.isChecked(), prefix=self.prefix.text().strip() or "ebf")


# ============================================================================== results
class EBFResultsDialog(HybridResultsDialog):
    """Generic results window; class tables get a Bel / Unc / Dis stacked bar chart, the layer table the AUC of
    every layer's belief alone."""

    def __init__(self, result, window=None, parent=None):
        super().__init__(result, tr("Evidential Belief Functions"), window, parent)

    def _chart(self, key, df):
        if key.startswith("classes_") and {"Bel", "Dis", "Unc"} <= set(df.columns):
            canvas = MplWidget(4.5, 3)
            canvas.setMinimumHeight(200)
            ax = canvas.fig.add_subplot(111)
            y = np.arange(len(df))
            left = np.zeros(len(df))
            for col, color, lab in (("Bel", theme.SERIES[0], "Bel"), ("Unc", theme.MUTED, "Unc"),
                                    ("Dis", theme.SERIES[1], "Dis")):
                ax.barh(y, df[col], left=left, color=color, label=lab, height=0.75)
                left = left + df[col].values
            ax.set_yticks(y)
            ax.set_yticklabels([t if len(t) <= 14 else t[:13] + "…" for t in df["label"].astype(str)], fontsize=7)
            ax.invert_yaxis()
            ax.set_xlim(0, 1)
            ax.legend(fontsize=7, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0), frameon=False)
            ax.tick_params(axis="x", labelsize=7)
            canvas.fig.set_tight_layout(False)              # fixed margins: long class labels + legend on top
            canvas.fig.subplots_adjust(left=0.3, right=0.96, top=0.86, bottom=0.1)
            self.canvases.append(canvas)
            return canvas
        elif key == "layers" and "single_layer_auc" in df.columns:
            canvas = MplWidget(4.5, 3)
            canvas.setMinimumHeight(200)
            ax = canvas.fig.add_subplot(111)
            y = np.arange(len(df))
            ax.barh(y, df.single_layer_auc, color=theme.SERIES[0])
            ax.axvline(0.5, color=theme.MUTED, lw=0.8, ls="--")
            ax.set_yticks(y)
            ax.set_yticklabels(df.layer.astype(str), fontsize=7)
            ax.invert_yaxis()
            ax.set_xlim(0, 1)
            ax.set_xlabel(tr("AUC of the layer's belief (training)"), fontsize=7)
            ax.tick_params(axis="x", labelsize=7)
        else:
            return None
        canvas.fig.tight_layout()
        self.canvases.append(canvas)
        return canvas
