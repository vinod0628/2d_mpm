"""Logistic fuzzification + geometric average: parameter dialog and results dialog."""
from __future__ import annotations

import numpy as np
import pandas as pd
from PyQt5.QtWidgets import (QAbstractItemView, QCheckBox, QComboBox, QDialog, QDoubleSpinBox, QFileDialog,
                             QFormLayout, QGridLayout, QGroupBox, QHBoxLayout, QHeaderView, QLabel, QLineEdit,
                             QMessageBox, QPushButton, QTableWidget, QTableWidgetItem, QTabWidget, QVBoxLayout,
                             QWidget)

from core.layers import LayerStore, RasterLayer
from gui import theme
from gui.widgets import BaseDialog, LayerCombo, MplWidget, fit_to_screen, hint, make_table, metric_card
from i18n import N_, tr

from .model import DIRECTION_LABELS, WEIGHTING_LABELS, fuzzify, parse_range, parse_scores

_LAST = {}


def _sample(lyr: RasterLayer, n: int = 400) -> np.ndarray:
    d = lyr.data
    return np.ma.compressed(d[:: max(1, d.shape[0] // n), :: max(1, d.shape[1] // n)]).astype(float)


def _spec(direction, vals):
    if not vals.size:
        return ""
    if direction == "classes":
        return "; ".join(f"{c:g}=0.5" for c in np.unique(vals))
    return f"{np.min(vals):.6g}, {np.max(vals):.6g}"


class LogisticGeometricDialog(BaseDialog):
    COLS = [N_("Use"), N_("Evidence layer"), N_("Evidence type"), N_("Range (min, max) / class scores"),
            N_("Weight")]

    def __init__(self, store: LayerStore, parent=None):
        super().__init__(tr("Logistic Fuzzification + Geometric Average"),
                         tr("Yousefi & Carranza (2015): every continuous layer is fuzzified by a logistic function "
                            "whose inflection and slope come from its value range (min → 0.01, max → 0.99; no expert "
                            "thresholds); categorical layers get a score per class. The fuzzified layers are combined "
                            "by their geometric average. Deposits are used only for validation (and, optionally, "
                            "P–A weights)."),
                         parent, ok_text=tr("Run model"))
        self.store = store
        g1 = QGroupBox(tr("Study area and validation"))
        f1 = QFormLayout(g1)
        self.mask = LayerCombo(store, "raster", allow_none=True, none_text=tr("— cells with evidence —"))
        sm = next((l for l in store.rasters() if "mask" in l.name.lower()), None)
        if sm is not None:
            self.mask.set_layer(sm)
        else:
            self.mask.setCurrentIndex(0)
        self.deposits = LayerCombo(store, "point", allow_none=True, none_text=tr("— no validation —"))
        dep = next((l for l in store.vectors("point") if "deposit" in l.name.lower()), None)
        if dep is not None:
            self.deposits.set_layer(dep)
        else:
            self.deposits.setCurrentIndex(0)
        self.negatives = LayerCombo(store, "point", allow_none=True, none_text=tr("— all background cells —"))
        self.negatives.setCurrentIndex(0)
        f1.addRow(tr("Study area mask"), self.mask)
        f1.addRow(tr("Deposits (validation)"), self.deposits)
        f1.addRow(tr("Absence points (ROC)"), self.negatives)
        self.body.addWidget(g1)

        g2 = QGroupBox(tr("Evidence layers"))
        v2 = QVBoxLayout(g2)
        self.table = QTableWidget(0, len(self.COLS))
        self.table.setHorizontalHeaderLabels([tr(c) for c in self.COLS])
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionMode(QAbstractItemView.NoSelection)
        self.table.setMinimumHeight(180)
        self.table.verticalHeader().setDefaultSectionSize(30)
        hh = self.table.horizontalHeader()
        hh.setSectionResizeMode(QHeaderView.Fixed)
        hh.setSectionResizeMode(1, QHeaderView.Stretch)
        hh.setSectionResizeMode(3, QHeaderView.Stretch)
        for col, w in ((0, 44), (2, 210), (4, 70)):
            self.table.setColumnWidth(col, w)
        self._rows = []
        prev = _LAST.get("evidence", {})
        for lyr in store.rasters():
            if "mask" in lyr.name.lower():
                continue
            self._add_row(lyr, prev.get(lyr.name))
        v2.addWidget(self.table, 1)
        v2.addWidget(hint(tr("Positive: high values favourable (e.g. pathfinder elements); negative: low values "
                             "favourable (e.g. distance to faults). The range defaults to the data minimum and maximum."
                             )))
        self.body.addWidget(g2, 1)

        g3 = QGroupBox(tr("Combination"))
        f3 = QGridLayout(g3)
        self.weighting = QComboBox()
        for k, v in WEIGHTING_LABELS.items():
            self.weighting.addItem(tr(v), k)
        self.weighting.setCurrentIndex(list(WEIGHTING_LABELS).index(_LAST.get("weighting", "equal")))
        self.o_fz = QCheckBox(tr("fuzzified maps"))
        self.o_fz.setChecked(_LAST.get("out_fz", False))
        self.prefix = QLineEdit(_LAST.get("prefix", "lgfa"))
        f3.addWidget(QLabel(tr("Weights")), 0, 0)
        f3.addWidget(self.weighting, 0, 1, 1, 3)
        f3.addWidget(QLabel(tr("Outputs")), 1, 0)
        outs = QHBoxLayout()
        outs.addWidget(QLabel(tr("geometric average +")))
        outs.addWidget(self.o_fz)
        outs.addStretch(1)
        f3.addLayout(outs, 1, 1, 1, 3)
        f3.addWidget(QLabel(tr("Output name prefix")), 2, 0)
        f3.addWidget(self.prefix, 2, 1)
        self.body.addWidget(g3)
        self.weighting.currentIndexChanged.connect(self._sync)
        self._sync()
        self.resize(1000, 860)

    def _sync(self):
        custom = self.weighting.currentData() == "custom"
        for r in self._rows:
            r["weight"].setEnabled(custom)

    def _add_row(self, lyr: RasterLayer, prev=None):
        r = self.table.rowCount()
        self.table.insertRow(r)
        vals = _sample(lyr)
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
        dirn = QComboBox()
        for k, v in DIRECTION_LABELS.items():
            dirn.addItem(tr(v), k)
        default = "classes" if lyr.categorical else "positive"
        dirn.setCurrentIndex(list(DIRECTION_LABELS).index(prev.get("direction", default) if prev else default))
        self.table.setCellWidget(r, 2, dirn)
        spec = QLineEdit(prev.get("spec") if prev else _spec(dirn.currentData(), vals))
        self.table.setCellWidget(r, 3, spec)
        wt = QDoubleSpinBox()
        wt.setRange(0, 100)
        wt.setDecimals(2)
        wt.setValue(prev.get("weight", 1.0) if prev else 1.0)
        self.table.setCellWidget(r, 4, wt)

        def changed(_=None, d=dirn, s=spec, v=vals):
            old_cls = "=" in s.text()
            if (d.currentData() == "classes") != old_cls:
                s.setText(_spec(d.currentData(), v))
        dirn.currentIndexChanged.connect(changed)
        self._rows.append(dict(layer=lyr, use=use, direction=dirn, spec=spec, weight=wt))

    def _selected(self):
        return [r for r in self._rows if r["use"].isChecked()]

    def validate(self):
        sel = self._selected()
        if not sel:
            return tr("Tick at least one evidence layer.")
        for r in sel:
            try:
                (parse_scores if r["direction"].currentData() == "classes" else parse_range)(r["spec"].text())
            except ValueError as exc:
                return f"{r['layer'].name}: {exc}"
        if self.weighting.currentData() == "pa" and self.deposits.layer() is None:
            return tr("P–A weights need a deposit layer.")
        return None

    def params(self) -> dict:
        sel = self._selected()
        _LAST.update(weighting=self.weighting.currentData(), out_fz=self.o_fz.isChecked(), prefix=self.prefix.text(),
                     evidence={r["layer"].name: dict(use=r["use"].isChecked(), direction=r["direction"].currentData(),
                                                     spec=r["spec"].text(), weight=r["weight"].value())
                               for r in self._rows})
        return dict(
            evidence=[dict(name=r["layer"].name, data=r["layer"].data, grid=r["layer"].grid,
                           direction=r["direction"].currentData(), spec=r["spec"].text(), weight=r["weight"].value())
                      for r in sel],
            weighting=self.weighting.currentData(),
            deposits=self.deposits.layer().gdf if self.deposits.layer() is not None else None,
            deposits_name=self.deposits.layer().name if self.deposits.layer() is not None else None,
            negatives=self.negatives.layer().gdf if self.negatives.layer() is not None else None,
            mask=(self.mask.layer().data, self.mask.layer().grid) if self.mask.layer() is not None else None,
            out_fuzzy=self.o_fz.isChecked(), prefix=self.prefix.text().strip() or "lgfa")


class LogisticGeometricResultsDialog(QDialog):
    def __init__(self, result, window=None, parent=None):
        super().__init__(parent)
        self.result = result
        self.main_window = window
        res = result.extra["lg"]
        self.setWindowTitle(tr("Logistic Fuzzification + Geometric Average — results"))
        lay = QVBoxLayout(self)
        lay.setContentsMargins(14, 12, 14, 12)
        t = QLabel(tr("Logistic Fuzzification + Geometric Average — results"))
        t.setObjectName("title")
        lay.addWidget(t)
        ev = result.evaluation
        cards = QHBoxLayout()
        cards.addWidget(metric_card(f"{len(res.fuzzified)}", tr("evidence layers")))
        cards.addWidget(metric_card(f"{res.favourability.compressed().mean():.3f}", tr("mean favourability")))
        if ev is not None:
            c = ev.curves["training"]
            cards.addWidget(metric_card(f"{c.auc:.3f}", tr("ROC AUC — all deposits")))
            pa = c.pa_intersection
            if pa:
                cards.addWidget(metric_card(f"{pa.get('prediction', float('nan')):.0f} % / "
                                            f"{pa.get('area', float('nan')):.0f} %",
                                            tr("P–A: deposits / area at the intersection")))
        cards.addStretch(1)
        lay.addLayout(cards)
        if result.messages:
            lay.addWidget(hint("⚠ " + "\n⚠ ".join(result.messages)))
        self.tabs = QTabWidget()
        w = QWidget()
        l = QVBoxLayout(w)
        lab = QLabel(tr("<b>Evidence</b> — logistic parameters (inflection i, slope s) or class scores, mean fuzzy "
                        "score, weight; with P–A weights also the P–A intersection and normalised density."))
        lab.setWordWrap(True)
        l.addWidget(lab)
        l.addWidget(make_table(res.table), 1)
        self.tabs.addTab(w, tr("Evidence"))
        self.tabs.addTab(self._curves_tab(result.extra["layers"]), tr("Fuzzification"))
        lay.addWidget(self.tabs, 1)
        btns = QHBoxLayout()
        b_roc = QPushButton(theme.icon("mdi6.chart-bell-curve-cumulative", "white"), "ROC / P–A…")
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
        fit_to_screen(self, 1040, 700)

    def _curves_tab(self, layers):
        w = QWidget()
        l = QVBoxLayout(w)
        canvas = MplWidget(7, 3.5)
        n = len(layers)
        cols = min(n, 3)
        rows = int(np.ceil(n / cols))
        for i, ly in enumerate(layers):
            ax = canvas.fig.add_subplot(rows, cols, i + 1)
            v = np.ma.compressed(np.ma.asarray(ly.array)).astype(float)
            if v.size > 200000:
                v = v[:: v.size // 200000]
            if ly.direction == "classes":
                cls, cnt = np.unique(v, return_counts=True)
                ax.bar(range(len(cls)), cnt / cnt.max(), color=theme.MUTED, alpha=0.4)
                ax.plot(range(len(cls)), fuzzify(ly, cls)[0], "o-", color=theme.SERIES[0])
                ax.set_xticks(range(len(cls)))
                ax.set_xticklabels([f"{c:g}" for c in cls], fontsize=6)
            else:
                lo, hi = (np.min(v), np.max(v)) if v.size else (0, 1)
                h, e = np.histogram(v, bins=30, range=(lo, hi))
                ax.bar((e[:-1] + e[1:]) / 2, h / max(h.max(), 1), width=np.diff(e), color=theme.MUTED, alpha=0.4)
                xs = np.linspace(lo, hi, 200)
                ax.plot(xs, fuzzify(ly, xs)[0], color=theme.SERIES[0])
            ax.set_ylim(0, 1.05)
            ax.set_title(f"{ly.name} ({ly.direction})", fontsize=7)
            ax.tick_params(labelsize=6)
        canvas.fig.tight_layout()
        l.addWidget(canvas, 1)
        self.curves_canvas = canvas
        return w

    def show_evaluation(self):
        from gui.dialogs.evaluation_dialogs import EvaluationDialog
        if self.result.evaluation is not None:
            EvaluationDialog(self.result.evaluation, tr("Logistic Fuzzification + Geometric Average"), self).show()

    def export_excel(self):
        p, _ = QFileDialog.getSaveFileName(self, tr("Export tables"), "lgfa_results.xlsx", "Excel (*.xlsx)")
        if not p:
            return
        with pd.ExcelWriter(p) as xw:
            self.result.extra["lg"].table.to_excel(xw, sheet_name="evidence", index=False)
            if self.result.evaluation is not None:
                pd.DataFrame(self.result.evaluation.summary_rows()).to_excel(xw, sheet_name="validation", index=False)
        QMessageBox.information(self, tr("Export"), tr("Saved {path}", path=p))
