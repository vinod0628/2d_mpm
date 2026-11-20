"""Fuzzy logic overlay: parameter dialog (memberships, inference network) and results dialog."""
from __future__ import annotations

import numpy as np
import pandas as pd
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QAbstractItemView, QCheckBox, QComboBox, QDialog, QDoubleSpinBox, QFileDialog,
                             QFormLayout, QGridLayout, QGroupBox, QHBoxLayout, QHeaderView, QLabel, QLineEdit,
                             QMessageBox, QPushButton, QTableWidget, QTableWidgetItem, QTabWidget, QVBoxLayout,
                             QWidget)

from core.layers import LayerStore, RasterLayer
from gui import theme
from gui.widgets import BaseDialog, LayerCombo, MplWidget, fit_to_screen, hint, make_table, metric_card
from i18n import N_, tr

from .model import MEMBERSHIP_LABELS, OPERATOR_LABELS, PARAM_HINTS, default_params, membership, parse_params

_LAST = {}


def _sample(lyr: RasterLayer, n: int = 400) -> np.ndarray:
    d = lyr.data
    return np.ma.compressed(d[:: max(1, d.shape[0] // n), :: max(1, d.shape[1] // n)]).astype(float)


def _op_combo(current="and"):
    cb = QComboBox()
    for k, v in OPERATOR_LABELS.items():
        cb.addItem(tr(v), k)
    cb.setCurrentIndex(list(OPERATOR_LABELS).index(current))
    return cb


class FuzzyLogicDialog(BaseDialog):
    COLS = [N_("Use"), N_("Evidence layer"), N_("Membership function"), N_("Parameters"), N_("Group")]

    def __init__(self, store: LayerStore, parent=None):
        super().__init__(tr("Fuzzy Logic Overlay"),
                         tr("Knowledge-driven model (Bonham-Carter 1994): give every evidence layer a fuzzy "
                            "membership (0 = unfavourable … 1 = favourable for the deposit model), group the layers "
                            "into mineral-system components (e.g. source, pathway, trap) and choose how groups and "
                            "layers are combined. Deposits are used only for validation."),
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
        f1.addRow(tr("Deposits (validation only)"), self.deposits)
        f1.addRow(tr("Absence points (ROC)"), self.negatives)
        self.body.addWidget(g1)

        g2 = QGroupBox(tr("Evidence layers and memberships"))
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
        for col, w in ((0, 44), (2, 190), (4, 90)):
            self.table.setColumnWidth(col, w)
        self._rows = []
        prev = _LAST.get("evidence", {})
        for lyr in store.rasters():
            if "mask" in lyr.name.lower():
                continue
            self._add_row(lyr, prev.get(lyr.name))
        v2.addWidget(self.table, 1)
        v2.addWidget(hint(tr("Parameters (evidence units): linear / S / Z = a, b; logistic = inflection, slope; "
                             "Gaussian = centre, sigma; large / small = midpoint, spread; per class = class=membership; "
                             "… The defaults are taken from the data (2nd–98th percentiles) — edit them to express the "
                             "deposit model.")))
        self.body.addWidget(g2, 1)

        g3 = QGroupBox(tr("Inference network"))
        f3 = QGridLayout(g3)
        self.group_op = _op_combo(_LAST.get("group_op", "and"))
        self.group_gamma = QDoubleSpinBox()
        self.group_gamma.setRange(0, 1)
        self.group_gamma.setSingleStep(0.05)
        self.group_gamma.setValue(_LAST.get("group_gamma", 0.9))
        self.final_op = _op_combo(_LAST.get("final_op", "gamma"))
        self.final_gamma = QDoubleSpinBox()
        self.final_gamma.setRange(0, 1)
        self.final_gamma.setSingleStep(0.05)
        self.final_gamma.setValue(_LAST.get("final_gamma", 0.9))
        self.missing = QComboBox()
        self.missing.addItem(tr("exclude cells from study area"), "exclude")
        self.missing.addItem(tr("membership 0.5 (no information)"), "value")
        self.missing.setCurrentIndex(0 if _LAST.get("missing", "exclude") == "exclude" else 1)
        self.o_groups = QCheckBox(tr("group maps"))
        self.o_groups.setChecked(True)
        self.o_mu = QCheckBox(tr("membership maps"))
        self.o_mu.setChecked(_LAST.get("out_mu", False))
        self.prefix = QLineEdit(_LAST.get("prefix", "fuzzy"))
        f3.addWidget(QLabel(tr("Within each group")), 0, 0)
        f3.addWidget(self.group_op, 0, 1)
        f3.addWidget(QLabel(tr("γ")), 0, 2)
        f3.addWidget(self.group_gamma, 0, 3)
        f3.addWidget(QLabel(tr("Between groups")), 1, 0)
        f3.addWidget(self.final_op, 1, 1)
        f3.addWidget(QLabel(tr("γ")), 1, 2)
        f3.addWidget(self.final_gamma, 1, 3)
        f3.addWidget(QLabel(tr("Missing evidence")), 2, 0)
        f3.addWidget(self.missing, 2, 1, 1, 3)
        f3.addWidget(QLabel(tr("Outputs")), 3, 0)
        outs = QHBoxLayout()
        outs.addWidget(QLabel(tr("favourability +")))
        outs.addWidget(self.o_groups)
        outs.addWidget(self.o_mu)
        outs.addStretch(1)
        f3.addLayout(outs, 3, 1, 1, 3)
        f3.addWidget(QLabel(tr("Output name prefix")), 4, 0)
        f3.addWidget(self.prefix, 4, 1)
        f3.addWidget(hint(tr("Typical: AND or gamma within a group whose components must all be present; OR within "
                             "a group of alternative proxies; gamma (0.8–0.95) between groups.")), 5, 0, 1, 4)
        self.body.addWidget(g3)
        self.resize(1000, 900)

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
        name.setToolTip(tr("range {lo:g} – {hi:g}", lo=float(vals.min()) if vals.size else 0,
                           hi=float(vals.max()) if vals.size else 0))
        self.table.setItem(r, 1, name)
        func = QComboBox()
        for k, v in MEMBERSHIP_LABELS.items():
            func.addItem(tr(v), k)
        default = "classes" if lyr.categorical else "linear"
        func.setCurrentIndex(list(MEMBERSHIP_LABELS).index(prev.get("function", default) if prev else default))
        self.table.setCellWidget(r, 2, func)
        par = QLineEdit(prev.get("params") if prev else default_params(func.currentData(), vals))
        par.setToolTip(PARAM_HINTS[func.currentData()])
        self.table.setCellWidget(r, 3, par)
        grp = QLineEdit(prev.get("group", "1") if prev else "1")
        grp.setToolTip(tr("Layers with the same group name are combined first (e.g. source, pathway, trap)."))
        self.table.setCellWidget(r, 4, grp)

        def changed(_=None, f=func, p=par, v=vals):
            p.setText(default_params(f.currentData(), v))
            p.setToolTip(PARAM_HINTS[f.currentData()])
        func.currentIndexChanged.connect(changed)
        self._rows.append(dict(layer=lyr, use=use, func=func, params=par, group=grp))

    def _selected(self):
        return [r for r in self._rows if r["use"].isChecked()]

    def validate(self):
        sel = self._selected()
        if not sel:
            return tr("Tick at least one evidence layer.")
        for r in sel:
            try:
                parse_params(r["func"].currentData(), r["params"].text())
            except ValueError as exc:
                return f"{r['layer'].name}: {exc}"
        return None

    def params(self) -> dict:
        sel = self._selected()
        _LAST.update(group_op=self.group_op.currentData(), group_gamma=self.group_gamma.value(),
                     final_op=self.final_op.currentData(), final_gamma=self.final_gamma.value(),
                     missing=self.missing.currentData(), out_mu=self.o_mu.isChecked(), prefix=self.prefix.text(),
                     evidence={r["layer"].name: dict(use=r["use"].isChecked(), function=r["func"].currentData(),
                                                     params=r["params"].text(), group=r["group"].text())
                               for r in self._rows})
        groups = sorted({r["group"].text().strip() or "1" for r in sel})
        op = (self.group_op.currentData(), self.group_gamma.value())
        return dict(
            evidence=[dict(name=r["layer"].name, data=r["layer"].data, grid=r["layer"].grid,
                           function=r["func"].currentData(), params=r["params"].text(),
                           group=r["group"].text().strip() or "1") for r in sel],
            group_ops={g: op for g in groups}, final_op=(self.final_op.currentData(), self.final_gamma.value()),
            deposits=self.deposits.layer().gdf if self.deposits.layer() is not None else None,
            deposits_name=self.deposits.layer().name if self.deposits.layer() is not None else None,
            negatives=self.negatives.layer().gdf if self.negatives.layer() is not None else None,
            mask=(self.mask.layer().data, self.mask.layer().grid) if self.mask.layer() is not None else None,
            missing=self.missing.currentData(), missing_value=0.5, auto_align=True,
            out_groups=self.o_groups.isChecked(), out_memberships=self.o_mu.isChecked(),
            prefix=self.prefix.text().strip() or "fuzzy")


class FuzzyLogicResultsDialog(QDialog):
    def __init__(self, result, window=None, parent=None):
        super().__init__(parent)
        self.result = result
        self.main_window = window
        res = result.extra["fuzzy"]
        self.setWindowTitle(tr("Fuzzy Logic Overlay — results"))
        lay = QVBoxLayout(self)
        lay.setContentsMargins(14, 12, 14, 12)
        t = QLabel(tr("Fuzzy Logic Overlay — results"))
        t.setObjectName("title")
        lay.addWidget(t)
        ev = result.evaluation
        cards = QHBoxLayout()
        fav = res.favourability.compressed()
        cards.addWidget(metric_card(f"{len(res.memberships)}", tr("evidence layers")))
        cards.addWidget(metric_card(f"{len(res.groups)}", tr("groups")))
        cards.addWidget(metric_card(f"{fav.mean():.3f}", tr("mean favourability")))
        if ev is not None:
            cards.addWidget(metric_card(f"{ev.curves['training'].auc:.3f}", tr("ROC AUC — all deposits")))
        cards.addStretch(1)
        lay.addLayout(cards)
        if result.messages:
            w = hint("⚠ " + "\n⚠ ".join(result.messages))
            w.setStyleSheet(f"color:{theme.WARN}")
            lay.addWidget(w)
        self.tabs = QTabWidget()
        self.tabs.addTab(self._tab(res.network, tr("<b>Inference network</b> — groups, their inputs and operators, "
                                                   "and the final combination.")), tr("Inference network"))
        self.tabs.addTab(self._curves_tab(result.extra["layers"]), tr("Membership functions"))
        self.tabs.addTab(self._tab(res.layer_table, tr("<b>Memberships</b> — statistics of every membership map in "
                                                       "the study area.")), tr("Memberships"))
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
    def _tab(df, text):
        w = QWidget()
        l = QVBoxLayout(w)
        lab = QLabel(text)
        lab.setWordWrap(True)
        l.addWidget(lab)
        l.addWidget(make_table(df), 1)
        return w

    def _curves_tab(self, layers):
        w = QWidget()
        l = QVBoxLayout(w)
        lab = QLabel(tr("<b>Membership functions</b> — membership against the evidence value (line) over the value "
                        "histogram of the study area (bars)."))
        lab.setWordWrap(True)
        l.addWidget(lab)
        canvas = MplWidget(7, 3.5)
        n = len(layers)
        cols = min(n, 3)
        rows = int(np.ceil(n / cols))
        for i, ly in enumerate(layers):
            ax = canvas.fig.add_subplot(rows, cols, i + 1)
            v = np.ma.compressed(np.ma.asarray(ly.array)).astype(float)
            if v.size > 200000:
                v = v[:: v.size // 200000]
            if ly.kind == "classes":
                cls, cnt = np.unique(v, return_counts=True)
                ax.bar(range(len(cls)), cnt / cnt.max(), color=theme.MUTED, alpha=0.4)
                ax.plot(range(len(cls)), membership(cls, ly.kind, ly.params), "o-", color=theme.SERIES[0])
                ax.set_xticks(range(len(cls)))
                ax.set_xticklabels([f"{c:g}" for c in cls], fontsize=6)
            else:
                lo, hi = np.percentile(v, [0.5, 99.5]) if v.size else (0, 1)
                h, e = np.histogram(v, bins=30, range=(lo, hi))
                ax.bar((e[:-1] + e[1:]) / 2, h / max(h.max(), 1), width=np.diff(e), color=theme.MUTED, alpha=0.4)
                xs = np.linspace(lo, hi, 200)
                ax.plot(xs, membership(xs, ly.kind, ly.params), color=theme.SERIES[0])
            ax.set_ylim(0, 1.05)
            ax.set_title(f"{ly.name} ({ly.kind})", fontsize=7)
            ax.tick_params(labelsize=6)
        canvas.fig.tight_layout()
        l.addWidget(canvas, 1)
        self.curves_canvas = canvas
        return w

    def show_evaluation(self):
        from gui.dialogs.evaluation_dialogs import EvaluationDialog
        if self.result.evaluation is not None:
            EvaluationDialog(self.result.evaluation, tr("Fuzzy Logic Overlay"), self).show()

    def export_excel(self):
        p, _ = QFileDialog.getSaveFileName(self, tr("Export fuzzy logic tables"), "fuzzy_results.xlsx",
                                           "Excel (*.xlsx)")
        if not p:
            return
        res = self.result.extra["fuzzy"]
        with pd.ExcelWriter(p) as xw:
            res.network.to_excel(xw, sheet_name="inference_network", index=False)
            res.layer_table.to_excel(xw, sheet_name="memberships", index=False)
            if self.result.evaluation is not None:
                pd.DataFrame(self.result.evaluation.summary_rows()).to_excel(xw, sheet_name="validation", index=False)
        QMessageBox.information(self, tr("Export"), tr("Saved {path}", path=p))
