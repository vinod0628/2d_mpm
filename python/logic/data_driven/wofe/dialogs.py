"""Weights-of-Evidence parameter dialog and results dialog."""
from __future__ import annotations

import numpy as np
import pandas as pd
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QAbstractItemView, QCheckBox, QComboBox, QDialog, QDoubleSpinBox, QFileDialog,
                             QFormLayout, QGridLayout, QGroupBox, QHBoxLayout, QHeaderView, QLabel, QLineEdit,
                             QPushButton, QSpinBox, QTableWidget, QTableWidgetItem, QTabWidget, QVBoxLayout,
                             QWidget, QMessageBox)

from core.layers import LayerStore, RasterLayer
from gui import theme
from gui.widgets import (BaseDialog, LayerCombo, fit_to_screen, hint, make_table, metric_card, raster_value_kind,
                         split_method_combo, thinning_spin)
from i18n import N_, tr

from .model import CRITERION_LABELS, MAX_CATEGORICAL, WEIGHT_TYPE_LABELS

_LAST = {}          # remembers settings during the session


def _default_type(layer: RasterLayer, n_values: int = 0) -> str:
    t = layer.class_table
    if t is not None and {"lower", "upper"} <= set(t.columns):
        return "ascending"            # classified continuous data
    if not layer.categorical and n_values > 32:
        return "ascending"            # integer layer with many values: every value is a candidate threshold
    return "categorical"


class WofEDialog(BaseDialog):
    COLS = [N_("Use"), N_("Evidence layer"), N_("Classes"), N_("Weight type"), N_("Generalisation"),
            N_("Min Stud(C)")]

    def __init__(self, store: LayerStore, parent=None):
        super().__init__(tr("Weights of Evidence"),
                         tr("Bayesian data-driven model (Bonham-Carter et al., 1989). Evidence layers must be integer "
                            "(class) rasters; real-valued data → Data ▸ Reclassify raster first. Cumulative ascending "
                            "= small values favourable (e.g. distance to faults), descending = large values "
                            "favourable (e.g. geochemical anomalies)."),
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
        for col, w in ((0, 44), (2, 70), (3, 250), (4, 190), (5, 96)):
            self.table.setColumnWidth(col, w)
        self._rows = []
        prev = _LAST.get("evidence", {})
        for lyr in store.rasters():
            if "mask" in lyr.name.lower():
                continue
            self._add_row(lyr, prev.get(lyr.name))
        v2.addWidget(self.table, 1)
        note = hint(tr("Cumulative types: every class value is a candidate threshold; the threshold with the "
                       "largest C whose Stud(C) ≥ the minimum is selected."))
        note.setToolTip(tr("'Maximum studentized contrast' ignores the minimum and picks the most significant "
                           "threshold. Categorical: one weight per class (at most {n} classes).", n=MAX_CATEGORICAL))
        v2.addWidget(note)
        self.body.addWidget(g2, 1)

        # ---------------------------------------------------------------- model options
        g3 = QGroupBox(tr("Model options"))
        f3 = QGridLayout(g3)
        self.unit = QDoubleSpinBox()
        self.unit.setRange(1e-6, 1e6)
        self.unit.setDecimals(4)
        self.unit.setValue(_LAST.get("unit", 1.0))
        self.unit.setSuffix(" km²")
        self.cell_info = QLabel("")
        self.cell_info.setObjectName("hint")
        self.missing = QComboBox()
        self.missing.addItem(tr("weight 0 + missing-data variance (keep area)"), "zero")
        self.missing.addItem(tr("exclude cells from study area"), "exclude")
        self.zero = QComboBox()
        self.zero.addItem(tr("weight 0"), "zero")
        self.zero.addItem(tr("0.5-deposit correction (negative weight)"), "haldane")
        self.align = QCheckBox(tr("Resample evidence to the first layer's grid if grids differ (nearest)"))
        self.align.setChecked(True)
        self.o_std = QCheckBox(tr("posterior std. deviation"))
        self.o_std.setChecked(True)
        self.o_conf = QCheckBox(tr("confidence (P / s(P))"))
        self.o_conf.setChecked(True)
        self.o_w = QCheckBox(tr("generalised weight rasters"))
        self.prefix = QLineEdit(_LAST.get("prefix", "wofe"))
        f3.addWidget(QLabel(tr("Unit cell area")), 0, 0)
        f3.addWidget(self.unit, 0, 1)
        f3.addWidget(self.cell_info, 0, 2, 1, 2)
        f3.addWidget(QLabel(tr("Missing evidence")), 1, 0)
        f3.addWidget(self.missing, 1, 1)
        f3.addWidget(QLabel(tr("Classes without deposits")), 1, 2)
        f3.addWidget(self.zero, 1, 3)
        f3.addWidget(self.align, 2, 0, 1, 4)
        f3.addWidget(QLabel(tr("Outputs")), 3, 0)
        outs = QHBoxLayout()
        lab = QLabel(tr("posterior probability +"))
        outs.addWidget(lab)
        for w in (self.o_std, self.o_conf, self.o_w):
            outs.addWidget(w)
        outs.addStretch(1)
        f3.addLayout(outs, 3, 1, 1, 3)
        f3.addWidget(QLabel(tr("Output name prefix")), 4, 0)
        f3.addWidget(self.prefix, 4, 1)
        self.body.addWidget(g3)
        self._update_cell_info()
        self.resize(1000, 860)

    def _add_row(self, lyr: RasterLayer, prev=None):
        r = self.table.rowCount()
        self.table.insertRow(r)
        ncls, kind = raster_value_kind(lyr)
        usable = kind != "real"                  # WofE works on integer class rasters (any number of classes)
        use = QCheckBox()
        use.setChecked(bool(prev and prev.get("use")) and usable)
        use.setEnabled(usable)
        cell = QWidget()
        cl = QHBoxLayout(cell)
        cl.setContentsMargins(8, 0, 0, 0)
        cl.addWidget(use)
        self.table.setCellWidget(r, 0, cell)
        name = QTableWidgetItem(lyr.name)
        name.setIcon(theme.icon("mdi6.shape-outline" if lyr.categorical else "mdi6.grid",
                                "#b18cf2" if lyr.categorical else "#4fc1c9"))
        why = tr("Real values: reclassify this layer into integer classes first (Data ▸ Reclassify raster), "
                 "then use it as cumulative ascending or descending.")
        info = QTableWidgetItem(str(ncls) if usable else "—")
        info.setTextAlignment(Qt.AlignCenter)
        if not usable:
            name.setForeground(Qt.gray)
            name.setToolTip(why)
            info.setForeground(Qt.gray)
            info.setToolTip(why)
        self.table.setItem(r, 1, name)
        self.table.setItem(r, 2, info)
        wt = QComboBox()
        for k, v in WEIGHT_TYPE_LABELS.items():
            wt.addItem(tr(v), k)
        wt.setCurrentIndex(list(WEIGHT_TYPE_LABELS).index(prev.get("type") if prev else _default_type(lyr, ncls)))
        crit = QComboBox()
        for k, v in CRITERION_LABELS.items():
            crit.addItem(tr(v), k)
        # many thresholds (integer layer with many values, e.g. distance in metres): maximum contrast tends to pick
        # tiny patterns with 1-2 deposits; the studentized contrast penalises them -> default for such layers
        fine = usable and not lyr.categorical and ncls > 32
        crit.setCurrentIndex(list(CRITERION_LABELS).index(
            prev.get("criterion", "max_contrast") if prev else ("max_studentized" if fine else "max_contrast")))
        if fine:
            crit.setToolTip(tr("Many values: every value is a threshold. 'Maximum studentized contrast' is "
                               "recommended — 'maximum contrast' can pick tiny patterns holding only 1–2 deposits."))
        ms = QDoubleSpinBox()
        ms.setRange(0, 10)
        ms.setSingleStep(0.5)
        ms.setValue(prev.get("min_stud", 1.96) if prev else 1.96)

        def sync(_i=None):
            cum = wt.currentData() != "categorical"
            crit.setEnabled(cum)
            ms.setEnabled(cum and crit.currentData() == "max_contrast")
        wt.currentIndexChanged.connect(sync)
        crit.currentIndexChanged.connect(sync)
        sync()
        for w in (wt, crit, ms):
            w.setEnabled(w.isEnabled() and usable)
        self.table.setCellWidget(r, 3, wt)
        self.table.setCellWidget(r, 4, crit)
        self.table.setCellWidget(r, 5, ms)
        use.toggled.connect(lambda _c: self._update_cell_info())
        self._rows.append(dict(layer=lyr, use=use, type=wt, criterion=crit, min_stud=ms, n_values=ncls))

    def _selected(self):
        return [r for r in self._rows if r["use"].isChecked()]

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
            if r["type"].currentData() == "categorical" and not r["layer"].categorical \
                    and r["n_values"] > MAX_CATEGORICAL:
                return tr("'{layer}' has more than {n} distinct values — use a cumulative type (every value is a "
                          "threshold) or reclassify it.", layer=r["layer"].name, n=MAX_CATEGORICAL)
            if r["layer"].data.dtype.kind == "f" and not r["layer"].categorical:
                return tr("'{layer}' has real values — reclassify it into integer classes first "
                          "(Data ▸ Reclassify raster).", layer=r["layer"].name)
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
                     split=self.split.currentData(), thin=self.thin.value(),
                     prefix=self.prefix.text(),
                     evidence={r["layer"].name: dict(use=r["use"].isChecked(), type=r["type"].currentData(),
                                                     criterion=r["criterion"].currentData(),
                                                     min_stud=r["min_stud"].value()) for r in self._rows})
        return dict(
            evidence=[dict(name=r["layer"].name, data=r["layer"].data, grid=r["layer"].grid,
                           labels=r["layer"].labels(), type=r["type"].currentData(),
                           criterion=r["criterion"].currentData(), min_stud=r["min_stud"].value()) for r in sel],
            deposits=self.deposits.layer().gdf, deposits_name=self.deposits.layer().name,
            negatives=self.negatives.layer().gdf if self.negatives.layer() is not None else None,
            mask=(self.mask.layer().data, self.mask.layer().grid) if self.mask.layer() is not None else None,
            test_fraction=self.test_pct.value() / 100.0, seed=self.seed.value(), unit_cell_km2=self.unit.value(),
            split_method=self.split.currentData(), thin_distance=self.thin.value(),
            missing=self.missing.currentData(), zero_deposit=self.zero.currentData(), auto_align=self.align.isChecked(),
            out_std=self.o_std.isChecked(), out_conf=self.o_conf.isChecked(), out_weights=self.o_w.isChecked(),
            prefix=self.prefix.text().strip() or "wofe")


# ============================================================================== results
WEIGHT_COLS = ["cls", "label", "cells", "area", "deposits", "cum_area", "cum_deposits", "W+", "s(W+)", "W-",
               "s(W-)", "C", "s(C)", "Stud(C)", "selected", "gen_class", "weight", "s_weight"]


class WofEResultsDialog(QDialog):
    def __init__(self, result, window=None, parent=None):
        super().__init__(parent)
        self.result = result
        self.main_window = window          # not "self.window": that would hide QWidget.window()
        res = result.extra["wofe"]
        self.setWindowTitle(tr("Weights of Evidence — results"))
        lay = QVBoxLayout(self)
        lay.setContentsMargins(14, 12, 14, 12)
        t = QLabel(tr("Weights of Evidence — results"))
        t.setObjectName("title")
        lay.addWidget(t)
        ev = result.evaluation
        cards = QHBoxLayout()
        cards.addWidget(metric_card(f"{res.n_deposits}", tr("training deposits (cells)")))
        cards.addWidget(metric_card(f"{res.prior:.5f}", tr("prior probability")))
        ci = res.ci["CI ratio N(D)/T"]
        cards.addWidget(metric_card(f"{ci:.2f}", tr("CI ratio N(D)/T  (≥ 0.85 ok)")))
        if ev is not None:
            cards.addWidget(metric_card(f"{ev.curves['training'].auc:.3f}", tr("ROC AUC — training")))
            if "test" in ev.curves:
                cards.addWidget(metric_card(f"{ev.curves['test'].auc:.3f}", tr("ROC AUC — test")))
        cards.addStretch(1)
        lay.addLayout(cards)
        if res.messages or any(l.messages for l in res.layers):
            msgs = list(res.messages) + [f"{l.name}: {m}" for l in res.layers for m in l.messages]
            w = hint("⚠ " + "\n⚠ ".join(msgs))
            w.setStyleSheet(f"color:{theme.WARN}")
            lay.addWidget(w)

        tabs = QTabWidget()
        tabs.addTab(self._summary_tab(res), tr("Summary"))
        for lw in res.layers:
            cols = [c for c in WEIGHT_COLS if c in lw.table.columns]
            sel_col = theme.ACCENT_DARK
            tab = QWidget()
            tl = QVBoxLayout(tab)
            head = tr("<b>{layer}</b> — {type}; {n} deposits on known evidence, area {area:,.1f} unit cells.",
                      layer=lw.name, type=tr(WEIGHT_TYPE_LABELS[lw.weight_type]), n=lw.n_deposits, area=lw.area)
            if lw.threshold is not None:
                head += " " + tr("Selected threshold: class <b>{cls:g}</b> (C = {c:.3f}, Stud(C) = {stud:.2f}).",
                                 cls=lw.threshold, c=lw.contrast, stud=lw.studentized)
            tl.addWidget(QLabel(head))
            tl.addWidget(make_table(lw.table[cols], highlight=lambda row: sel_col if row.get("selected", False) else None), 3)
            tl.addWidget(QLabel(tr("Generalised weights used in the posterior:")))
            tl.addWidget(make_table(lw.gen_table), 1)
            tabs.addTab(tab, lw.name)
        ci_tab = QWidget()
        cl = QVBoxLayout(ci_tab)
        cl.addWidget(QLabel(tr("<b>Overall / Agterberg–Cheng tests</b> — T = Σ area × posterior = predicted number "
                               "of deposits. N(D)/T < 0.85, a small one-tailed p or a probability of conditional "
                               "dependence > 95 % suggests conditional dependence.")))
        cl.addWidget(make_table(pd.DataFrame([(tr(k), v) for k, v in res.ci.items()],
                                             columns=[tr("statistic"), tr("value")])), 1)
        cl.addWidget(QLabel(tr("<b>Pairwise χ² tests</b> on deposit cells (generalised classes); p < 0.05 → the two "
                               "layers are not conditionally independent; consider combining or dropping one.")))
        cl.addWidget(make_table(res.pairwise, highlight=lambda r: "#5a3a3a" if r.get("p_value", 1) < 0.05 else None), 1)
        tabs.addTab(ci_tab, tr("Conditional independence"))
        lay.addWidget(tabs, 1)

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
        fit_to_screen(self, 1080, 720)

    def _summary_tab(self, res):
        w = QWidget()
        l = QVBoxLayout(w)
        p = res.params
        rows = [(tr("Evidence layers"), ", ".join(x.name for x in res.layers)),
                (tr("Deposits layer"), self.result.extra.get("deposits_name", "")),
                (tr("Training / test deposit cells"),
                 f"{res.n_deposits} / {len(self.result.extra.get('test_cells', ([],))[0])}"),
                (tr("Study area"), tr("{n:,.2f} unit cells of {unit:g} km²", n=res.n_area, unit=p['unit_cell_km2'])),
                (tr("Raster cell area"), f"{p['cell_area_km2']:.6g} km²"),
                (tr("Missing evidence"), p["missing"]), (tr("Zero-deposit classes"), p["zero_deposit"]),
                (tr("Prior probability"), f"{res.prior:.6f}")]
        for lw in res.layers:
            g = lw.gen_table
            rows.append((f"{lw.name}", "; ".join(f"{a}: W = {b:.3f} ± {c:.3f}" for a, b, c in
                                                zip(g["label"].astype(str), g["weight"], g["s_weight"]))[:400]))
        l.addWidget(make_table(pd.DataFrame(rows, columns=[tr("item"), tr("value")])))
        return w

    def show_evaluation(self):
        from gui.dialogs.evaluation_dialogs import EvaluationDialog
        if self.result.evaluation is not None:
            dlg = EvaluationDialog(self.result.evaluation, tr("Weights of Evidence"), self)
            dlg.show()

    def export_excel(self):
        p, _ = QFileDialog.getSaveFileName(self, tr("Export WofE tables"), "wofe_results.xlsx", "Excel (*.xlsx)")
        if not p:
            return
        res = self.result.extra["wofe"]
        with pd.ExcelWriter(p) as xw:
            res.summary_table().to_excel(xw, sheet_name="summary", index=False)
            for lw in res.layers:
                lw.table.to_excel(xw, sheet_name=lw.name[:28], index=False)
            pd.DataFrame(list(res.ci.items()), columns=["statistic", "value"]).to_excel(xw, sheet_name="ci_tests", index=False)
            res.pairwise.to_excel(xw, sheet_name="pairwise_chi2", index=False)
            if self.result.evaluation is not None:
                pd.DataFrame(self.result.evaluation.summary_rows()).to_excel(xw, sheet_name="validation", index=False)
        QMessageBox.information(self, tr("Export"), tr("Saved {path}", path=p))
