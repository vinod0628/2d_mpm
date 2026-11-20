"""Bayesian-network parameter dialog and results dialog."""
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

from .model import DESIGN_LABELS, STRUCTURE_LABELS

_LAST = {}          # remembers settings during the session


def _sample(lyr: RasterLayer, n: int = 400) -> np.ndarray:
    d = lyr.data
    return np.ma.compressed(d[:: max(1, d.shape[0] // n), :: max(1, d.shape[1] // n)])


class BayesianNetworkDialog(BaseDialog):
    COLS = [N_("Use"), N_("Evidence layer"), N_("Values"), N_("Variable type")]

    def __init__(self, store: LayerStore, parent=None):
        super().__init__(tr("Bayesian Network"),
                         tr("Bayesian network classifier (Porwal et al. 2006): deposit and evidence layers form a network; "
                            "naive Bayes treats the layers as independent (as weights of evidence), TAN also learns "
                            "which layers depend on each other. Continuous layers are divided into equal-area "
                            "classes; categorical layers keep their classes."),
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
        v2.addWidget(hint(tr("Continuous = divided into equal-area classes; categorical = its own classes. TAN "
                             "learns dependencies between layers (e.g. two pathfinder elements of one fluid).")))
        self.body.addWidget(g2, 1)

        # ---------------------------------------------------------------- model options
        g3 = QGroupBox(tr("Network and training cells"))
        f3 = QGridLayout(g3)
        self.structure = QComboBox()
        for k, v in STRUCTURE_LABELS.items():
            self.structure.addItem(tr(v), k)
        self.structure.setCurrentIndex(list(STRUCTURE_LABELS).index(_LAST.get("structure", "tan")))
        self.design = QComboBox()
        for k, v in DESIGN_LABELS.items():
            self.design.addItem(tr(v), k)
        self.design.setCurrentIndex(list(DESIGN_LABELS).index(_LAST.get("design", "all")))
        self.design.setToolTip(tr("All study-area cells: probabilities per cell (as weights of evidence). Deposits + "
                                  "random non-deposit cells: as Porwal et al. (2006); the posterior then depends on "
                                  "the ratio."))
        self.neg_ratio = QDoubleSpinBox()
        self.neg_ratio.setRange(0.1, 100)
        self.neg_ratio.setDecimals(1)
        self.neg_ratio.setValue(_LAST.get("neg_ratio", 5.0))
        self.min_dist = QDoubleSpinBox()
        self.min_dist.setRange(0, 1e7)
        self.min_dist.setDecimals(0)
        self.min_dist.setSingleStep(100)
        cell = next((l.grid.res[0] for l in store.rasters()), 100.0)
        self.min_dist.setValue(_LAST.get("min_dist", 4 * cell))
        self.classes = QSpinBox()
        self.classes.setRange(2, 20)
        self.classes.setValue(_LAST.get("classes", 5))
        self.classes.setToolTip(tr("Continuous layers are divided into this many equal-area (quantile) classes; "
                                   "categorical layers keep their classes."))
        self.alpha = QDoubleSpinBox()
        self.alpha.setRange(0.01, 1000)
        self.alpha.setDecimals(2)
        self.alpha.setValue(_LAST.get("alpha", 1.0))
        self.alpha.setToolTip(tr("Dirichlet prior (equivalent sample size) of the probability tables: protects "
                                 "classes with few cells or no deposits from zero probabilities."))
        self.draws = QSpinBox()
        self.draws.setRange(0, 1000)
        self.draws.setValue(_LAST.get("draws", 50))
        self.draws.setSpecialValueText(tr("off"))
        self.draws.setToolTip(tr("Number of random draws of the probability tables for the posterior std. "
                                 "deviation."))
        self.unit = QDoubleSpinBox()
        self.unit.setRange(1e-6, 1e6)
        self.unit.setDecimals(4)
        self.unit.setValue(_LAST.get("unit", 1.0))
        self.unit.setSuffix(" km²")
        self.missing = QComboBox()
        self.missing.addItem(tr("own class 'missing'"), "class")
        self.missing.addItem(tr("exclude cells from study area"), "exclude")
        self.missing.setCurrentIndex(0 if _LAST.get("missing", "class") == "class" else 1)
        self.align = QCheckBox(tr("Resample evidence to the first layer's grid if grids differ"))
        self.align.setChecked(True)
        self.o_unit = QCheckBox(tr("probability per unit cell"))
        self.o_unit.setChecked(True)
        self.o_sd = QCheckBox(tr("posterior std. deviation"))
        self.o_sd.setChecked(True)
        self.prefix = QLineEdit(_LAST.get("prefix", "bn"))
        for cb in (self.structure, self.design, self.missing):
            cb.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
            cb.setMinimumContentsLength(16)
        grid = [(tr("Structure"), self.structure, tr("Training cells"), self.design),
                (tr("Non-deposit samples per deposit"), self.neg_ratio, tr("Min. distance from deposits"), self.min_dist),
                (tr("Classes of continuous layers"), self.classes, tr("Dirichlet prior α"), self.alpha),
                (tr("Uncertainty draws"), self.draws, tr("Unit cell area"), self.unit),
                (tr("Missing evidence"), self.missing, None, None)]
        for i, (a, wa, b, wb) in enumerate(grid):
            f3.addWidget(QLabel(a), i, 0)
            f3.addWidget(wa, i, 1)
            if b is not None:
                f3.addWidget(QLabel(b), i, 2)
                f3.addWidget(wb, i, 3)
        i = len(grid)
        f3.addWidget(self.align, i, 0, 1, 4)
        f3.addWidget(QLabel(tr("Outputs")), i + 1, 0)
        outs = QHBoxLayout()
        outs.addWidget(QLabel(tr("posterior probability +")))
        outs.addWidget(self.o_unit)
        outs.addWidget(self.o_sd)
        outs.addStretch(1)
        f3.addLayout(outs, i + 1, 1, 1, 3)
        f3.addWidget(QLabel(tr("Output name prefix")), i + 2, 0)
        f3.addWidget(self.prefix, i + 2, 1)
        self.body.addWidget(g3)
        self.design.currentIndexChanged.connect(self._sync)
        self._sync()
        self.resize(1000, 900)

    def _sync(self):
        sampled = self.design.currentData() == "sampled"
        self.neg_ratio.setEnabled(sampled)
        self.min_dist.setEnabled(sampled)
        self.unit.setEnabled(not sampled)
        self.o_unit.setEnabled(not sampled)

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
                     thin=self.thin.value(), structure=self.structure.currentData(), design=self.design.currentData(),
                     neg_ratio=self.neg_ratio.value(), min_dist=self.min_dist.value(), classes=self.classes.value(),
                     alpha=self.alpha.value(), draws=self.draws.value(), unit=self.unit.value(),
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
            structure=self.structure.currentData(), design=self.design.currentData(),
            neg_ratio=self.neg_ratio.value(), min_distance=self.min_dist.value(), min_spacing=-1.0,
            n_classes=self.classes.value(), alpha=self.alpha.value(), draws=self.draws.value(),
            unit_cell_km2=self.unit.value(), missing=self.missing.currentData(), auto_align=self.align.isChecked(),
            out_unit=self.o_unit.isChecked(), out_sd=self.o_sd.isChecked(), prefix=self.prefix.text().strip() or "bn")


# ============================================================================== results
class BayesianNetworkResultsDialog(QDialog):
    def __init__(self, result, window=None, parent=None):
        super().__init__(parent)
        self.result = result
        self.main_window = window          # not "self.window": that would hide QWidget.window()
        res = result.extra["bn"]
        self.setWindowTitle(tr("Bayesian Network — results"))
        lay = QVBoxLayout(self)
        lay.setContentsMargins(14, 12, 14, 12)
        t = QLabel(tr("Bayesian Network — results"))
        t.setObjectName("title")
        lay.addWidget(t)
        ev = result.evaluation
        cards = QHBoxLayout()
        cards.addWidget(metric_card(f"{res.n_deposits}", tr("training deposits (cells)")))
        cards.addWidget(metric_card(tr("TAN") if res.params["structure"] == "tan" else tr("naive Bayes"),
                                    tr("structure")))
        cards.addWidget(metric_card(f"{res.stats['edges']}", tr("dependencies between layers")))
        if res.stats.get("predicted"):
            cards.addWidget(metric_card(f"{res.n_deposits / res.stats['predicted']:.2f}",
                                        tr("observed / predicted deposits")))
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
        self.tabs.addTab(self._tab(res.summary, tr("<b>Model</b> — structure, classes, prior and fit.")),
                         tr("Summary"))
        self.tabs.addTab(self._network_tab(res), tr("Network"))
        self.tabs.addTab(self._tab(res.cpt_table, tr(
            "<b>Conditional probability tables</b> — P(class | deposit) and P(class | no deposit) of every layer "
            "(TAN: also given the class of the parent layer); log ratio > 0 = class favourable.")),
            tr("Probability tables"))
        self.tabs.addTab(self._tab(res.layers, tr(
            "<b>Evidence layers</b> — classes, mutual information with the deposits, parents in the network and drop "
            "of the training AUC when the layer is left out.")), tr("Evidence layers"))
        self.tabs.addTab(self._tab(res.comparison, tr(
            "<b>Naive Bayes vs TAN</b> — log-likelihood, number of parameters, BIC and training AUC of both "
            "structures on the same cells.")), tr("Structure comparison"))
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

    def _network_tab(self, res) -> QWidget:
        import networkx as nx
        w = QWidget()
        l = QVBoxLayout(w)
        lab = QLabel(tr("<b>Network</b> — the deposit node D is a parent of every evidence layer; TAN arrows between "
                        "layers show the learned dependencies (width ∝ conditional mutual information)."))
        lab.setWordWrap(True)
        l.addWidget(lab)
        row = QHBoxLayout()
        row.addWidget(make_table(res.edges), 1)
        canvas = MplWidget(5, 3.6)
        canvas.setMinimumHeight(240)
        ax = canvas.fig.add_subplot(111)
        g = nx.DiGraph()
        layers = list(res.layers.layer)
        g.add_node("D")
        for name in layers:
            g.add_edge("D", name)
        for e in res.edges.itertuples():
            g.add_edge(e.parent, e.child)
        pos = {"D": (0.0, 1.0)}
        for i, name in enumerate(layers):
            pos[name] = ((i - (len(layers) - 1) / 2) / max(len(layers) - 1, 1) * 2.0, 0.0)
        nx.draw_networkx_nodes(g, pos, ax=ax, nodelist=["D"], node_color="#e53935", node_size=700)
        nx.draw_networkx_nodes(g, pos, ax=ax, nodelist=layers, node_color=theme.SERIES[0], node_size=500)
        nx.draw_networkx_edges(g, pos, ax=ax, edgelist=[("D", n) for n in layers], edge_color=theme.MUTED, arrows=True)
        if len(res.edges):
            mi = res.edges.cond_mutual_information.values
            width = 1 + 5 * mi / max(mi.max(), 1e-12)
            nx.draw_networkx_edges(g, pos, ax=ax, edgelist=[(e.parent, e.child) for e in res.edges.itertuples()],
                                   edge_color="#fb8c00", width=width, arrows=True, connectionstyle="arc3,rad=0.35")
        nx.draw_networkx_labels(g, pos, ax=ax, font_size=7, font_color=theme.TEXT if hasattr(theme, "TEXT") else "k")
        ax.set_xlim(-1.35, 1.35)
        ax.set_ylim(-0.75, 1.2)                          # room for the curved TAN arrows below the layer nodes
        ax.set_axis_off()
        row.addWidget(canvas, 2)
        l.addLayout(row, 1)
        self.network_canvas = canvas
        return w

    def show_evaluation(self):
        from gui.dialogs.evaluation_dialogs import EvaluationDialog
        if self.result.evaluation is not None:
            dlg = EvaluationDialog(self.result.evaluation, tr("Bayesian Network"), self)
            dlg.show()

    def export_excel(self):
        p, _ = QFileDialog.getSaveFileName(self, tr("Export Bayesian network tables"), "bn_results.xlsx",
                                           "Excel (*.xlsx)")
        if not p:
            return
        res = self.result.extra["bn"]
        with pd.ExcelWriter(p) as xw:
            for name, df in (("summary", res.summary), ("network_edges", res.edges), ("probability_tables", res.cpt_table),
                             ("evidence_layers", res.layers), ("structure_comparison", res.comparison)):
                if len(df):
                    df.to_excel(xw, sheet_name=name, index=False)
            if self.result.evaluation is not None:
                pd.DataFrame(self.result.evaluation.summary_rows()).to_excel(xw, sheet_name="validation", index=False)
        QMessageBox.information(self, tr("Export"), tr("Saved {path}", path=p))
