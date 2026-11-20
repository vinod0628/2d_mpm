"""Boolean logic overlay: parameter dialog and results dialog."""
from __future__ import annotations

import numpy as np
import pandas as pd
from PyQt5.QtWidgets import (QAbstractItemView, QCheckBox, QComboBox, QDialog, QFileDialog, QFormLayout, QGridLayout,
                             QGroupBox, QHBoxLayout, QHeaderView, QLabel, QLineEdit, QMessageBox, QPushButton,
                             QTableWidget, QTableWidgetItem, QTabWidget, QVBoxLayout, QWidget)

from core.layers import LayerStore, RasterLayer
from gui import theme
from gui.widgets import BaseDialog, LayerCombo, fit_to_screen, hint, make_table, metric_card
from i18n import N_, tr

from .model import BOOL_OP_LABELS, CONDITION_LABELS, VALUE_HINTS, parse_values

_LAST = {}


def _sample(lyr: RasterLayer, n: int = 400) -> np.ndarray:
    d = lyr.data
    return np.ma.compressed(d[:: max(1, d.shape[0] // n), :: max(1, d.shape[1] // n)]).astype(float)


def _default(cond, vals):
    if not vals.size:
        return ""
    if cond == "in":
        return ", ".join(f"{c:g}" for c in np.unique(vals)[:1])
    p25, p75 = np.percentile(vals, [25, 75])
    if cond == "ge":
        return f"{p75:.6g}"
    if cond == "le":
        return f"{p25:.6g}"
    return f"{p25:.6g}, {p75:.6g}"


def _bool_combo(current="and"):
    cb = QComboBox()
    for k, v in BOOL_OP_LABELS.items():
        cb.addItem(tr(v), k)
    cb.setCurrentIndex(list(BOOL_OP_LABELS).index(current))
    return cb


class BooleanLogicDialog(BaseDialog):
    COLS = [N_("Use"), N_("Evidence layer"), N_("Condition"), N_("Value(s)"), N_("Group")]

    def __init__(self, store: LayerStore, parent=None):
        super().__init__(tr("Boolean Logic Overlay"),
                         tr("Knowledge-driven model (Bonham-Carter 1994): every evidence layer becomes true / false by "
                            "a condition from the deposit model; the conditions are combined with AND / OR within "
                            "groups (mineral-system components) and between groups. Deposits are used only for "
                            "validation."),
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

        g2 = QGroupBox(tr("Evidence layers and conditions"))
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
        for col, w in ((0, 44), (2, 160), (4, 90)):
            self.table.setColumnWidth(col, w)
        self._rows = []
        prev = _LAST.get("evidence", {})
        for lyr in store.rasters():
            if "mask" in lyr.name.lower():
                continue
            self._add_row(lyr, prev.get(lyr.name))
        v2.addWidget(self.table, 1)
        v2.addWidget(hint(tr("Values in evidence units: ≥ / ≤ = threshold; between / outside = a, b; classes = list "
                             "of class values. Defaults are quartiles of the data — set them from the deposit model.")))
        self.body.addWidget(g2, 1)

        g3 = QGroupBox(tr("Combination"))
        f3 = QGridLayout(g3)
        self.group_op = _bool_combo(_LAST.get("group_op", "and"))
        self.final_op = _bool_combo(_LAST.get("final_op", "and"))
        self.o_cond = QCheckBox(tr("condition maps"))
        self.o_cond.setChecked(_LAST.get("out_cond", False))
        self.prefix = QLineEdit(_LAST.get("prefix", "boolean"))
        f3.addWidget(QLabel(tr("Within each group")), 0, 0)
        f3.addWidget(self.group_op, 0, 1)
        f3.addWidget(QLabel(tr("Between groups")), 0, 2)
        f3.addWidget(self.final_op, 0, 3)
        f3.addWidget(QLabel(tr("Outputs")), 1, 0)
        outs = QHBoxLayout()
        outs.addWidget(QLabel(tr("favourable area + conditions met +")))
        outs.addWidget(self.o_cond)
        outs.addStretch(1)
        f3.addLayout(outs, 1, 1, 1, 3)
        f3.addWidget(QLabel(tr("Output name prefix")), 2, 0)
        f3.addWidget(self.prefix, 2, 1)
        self.body.addWidget(g3)
        self.resize(1000, 880)

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
        cond = QComboBox()
        for k, v in CONDITION_LABELS.items():
            cond.addItem(tr(v), k)
        default = "in" if lyr.categorical else "ge"
        cond.setCurrentIndex(list(CONDITION_LABELS).index(prev.get("cond", default) if prev else default))
        self.table.setCellWidget(r, 2, cond)
        val = QLineEdit(prev.get("values") if prev else _default(cond.currentData(), vals))
        val.setToolTip(VALUE_HINTS[cond.currentData()])
        self.table.setCellWidget(r, 3, val)
        grp = QLineEdit(prev.get("group", "1") if prev else "1")
        self.table.setCellWidget(r, 4, grp)

        def changed(_=None, c=cond, v=val, x=vals):
            v.setText(_default(c.currentData(), x))
            v.setToolTip(VALUE_HINTS[c.currentData()])
        cond.currentIndexChanged.connect(changed)
        self._rows.append(dict(layer=lyr, use=use, cond=cond, values=val, group=grp))

    def _selected(self):
        return [r for r in self._rows if r["use"].isChecked()]

    def validate(self):
        sel = self._selected()
        if not sel:
            return tr("Tick at least one evidence layer.")
        for r in sel:
            try:
                parse_values(r["cond"].currentData(), r["values"].text())
            except ValueError as exc:
                return f"{r['layer'].name}: {exc}"
        return None

    def params(self) -> dict:
        sel = self._selected()
        _LAST.update(group_op=self.group_op.currentData(), final_op=self.final_op.currentData(),
                     out_cond=self.o_cond.isChecked(), prefix=self.prefix.text(),
                     evidence={r["layer"].name: dict(use=r["use"].isChecked(), cond=r["cond"].currentData(),
                                                     values=r["values"].text(), group=r["group"].text())
                               for r in self._rows})
        groups = sorted({r["group"].text().strip() or "1" for r in sel})
        return dict(
            evidence=[dict(name=r["layer"].name, data=r["layer"].data, grid=r["layer"].grid,
                           cond=r["cond"].currentData(), values=r["values"].text(),
                           group=r["group"].text().strip() or "1") for r in sel],
            group_ops={g: self.group_op.currentData() for g in groups}, final_op=self.final_op.currentData(),
            deposits=self.deposits.layer().gdf if self.deposits.layer() is not None else None,
            deposits_name=self.deposits.layer().name if self.deposits.layer() is not None else None,
            negatives=self.negatives.layer().gdf if self.negatives.layer() is not None else None,
            mask=(self.mask.layer().data, self.mask.layer().grid) if self.mask.layer() is not None else None,
            out_conditions=self.o_cond.isChecked(), prefix=self.prefix.text().strip() or "boolean")


class BooleanLogicResultsDialog(QDialog):
    def __init__(self, result, window=None, parent=None):
        super().__init__(parent)
        self.result = result
        self.main_window = window
        res = result.extra["boolean"]
        self.setWindowTitle(tr("Boolean Logic Overlay — results"))
        lay = QVBoxLayout(self)
        lay.setContentsMargins(14, 12, 14, 12)
        t = QLabel(tr("Boolean Logic Overlay — results"))
        t.setObjectName("title")
        lay.addWidget(t)
        ev = result.evaluation
        cards = QHBoxLayout()
        fav = res.favourable.compressed()
        cards.addWidget(metric_card(f"{100 * fav.mean():.1f} %", tr("favourable area")))
        comb = res.table[res.table.layer == "FAVOURABLE (combined)"]
        if len(comb):
            cards.addWidget(metric_card(f"{comb.deposits_pct.iloc[0]:.0f} %", tr("deposits in the favourable area")))
        if ev is not None:
            cards.addWidget(metric_card(f"{ev.curves['training'].auc:.3f}", tr("ROC AUC of the condition count")))
        cards.addStretch(1)
        lay.addLayout(cards)
        if result.messages:
            w = hint("⚠ " + "\n⚠ ".join(result.messages))
            lay.addWidget(w)
        self.tabs = QTabWidget()
        for df, text, title in ((res.table, tr("<b>Conditions</b> — area of each condition, deposits inside and the "
                                               "ratio of deposit share to area share (> 1 = better than random)."),
                                 tr("Conditions")),
                                (res.network, tr("<b>Combination</b> — groups, operators and favourable area."),
                                 tr("Combination"))):
            w = QWidget()
            l = QVBoxLayout(w)
            lab = QLabel(text)
            lab.setWordWrap(True)
            l.addWidget(lab)
            l.addWidget(make_table(df), 1)
            self.tabs.addTab(w, title)
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
        fit_to_screen(self, 1000, 640)

    def show_evaluation(self):
        from gui.dialogs.evaluation_dialogs import EvaluationDialog
        if self.result.evaluation is not None:
            EvaluationDialog(self.result.evaluation, tr("Boolean Logic Overlay"), self).show()

    def export_excel(self):
        p, _ = QFileDialog.getSaveFileName(self, tr("Export Boolean logic tables"), "boolean_results.xlsx",
                                           "Excel (*.xlsx)")
        if not p:
            return
        res = self.result.extra["boolean"]
        with pd.ExcelWriter(p) as xw:
            res.table.to_excel(xw, sheet_name="conditions", index=False)
            res.network.to_excel(xw, sheet_name="combination", index=False)
            if self.result.evaluation is not None:
                pd.DataFrame(self.result.evaluation.summary_rows()).to_excel(xw, sheet_name="validation", index=False)
        QMessageBox.information(self, tr("Export"), tr("Saved {path}", path=p))
