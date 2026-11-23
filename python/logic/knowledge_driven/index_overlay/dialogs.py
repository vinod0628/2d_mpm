"""Index overlay: parameter dialog (evidence scores, weights, AHP matrix) and results dialog."""
from __future__ import annotations

import numpy as np
import pandas as pd
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QAbstractItemView, QCheckBox, QComboBox, QDialog, QDoubleSpinBox, QFileDialog,
                             QGridLayout, QGroupBox, QHBoxLayout, QHeaderView, QLabel, QLineEdit, QMessageBox,
                             QPushButton, QTableWidget, QTableWidgetItem, QTabWidget, QVBoxLayout, QWidget)

from core.layers import LayerStore, RasterLayer
from gui import theme
from gui.widgets import BaseDialog, LayerCombo, MplWidget, fit_to_screen, hint, make_table, metric_card
from i18n import N_, tr

from .model import (RULE_LABELS, WEIGHTING_LABELS, ahp_weights, fuzzy_ahp_weights, parse_judgement, parse_range,
                    parse_scores, parse_threshold)

_LAST = {}
MISSING_LABELS = {"exclude": N_("exclude cells with missing evidence"),
                  "renormalise": N_("use the layers with data (weights renormalised)")}


def _sample(lyr: RasterLayer, n: int = 400) -> np.ndarray:
    d = lyr.data
    return np.ma.compressed(d[:: max(1, d.shape[0] // n), :: max(1, d.shape[1] // n)]).astype(float)


def default_spec(rule: str, vals: np.ndarray) -> str:
    if not vals.size:
        return ""
    if rule == "classes":
        return "; ".join(f"{c:g}=0.5" for c in np.unique(vals)[:40])
    lo, hi = np.percentile(vals, [2, 98])
    if rule in ("positive", "negative"):
        return f"{lo:.6g}, {hi:.6g}"
    return f"{np.percentile(vals, 75 if rule == 'ge' else 25):.6g}"


def _check_spec(rule, text):
    if rule == "classes":
        parse_scores(text)
    elif rule in ("positive", "negative"):
        parse_range(text)
    else:
        parse_threshold(text)


class IndexOverlayDialog(BaseDialog):
    COLS = [N_("Use"), N_("Evidence layer"), N_("Scoring"), N_("Scores / range / threshold"), N_("Weight")]

    TITLE = N_("Index Overlay (AHP / Fuzzy AHP weights)")
    SUBTITLE = N_("Every evidence layer is turned into scores between 0 and 1 — a score per class, a linear "
                  "rescaling between two values in the favourable direction, or a binary condition — and "
                  "the layers are combined by their weighted mean. Weights: equal, custom, or derived from "
                  "pairwise comparisons of the layers (AHP; fuzzy AHP for uncertain judgements). Deposits "
                  "are used only for validation.")
    WEIGHTINGS = WEIGHTING_LABELS
    PREFIX = "index"
    LAST = _LAST                     # remembered settings (one dict per tool)

    def __init__(self, store: LayerStore, parent=None):
        super().__init__(tr(self.TITLE), tr(self.SUBTITLE), parent, ok_text=tr("Run model"))
        last = self.LAST
        self.store = store
        g1 = QGroupBox(tr("Study area and validation"))
        f1 = QGridLayout(g1)
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
        f1.addWidget(QLabel(tr("Study area mask")), 0, 0)
        f1.addWidget(self.mask, 0, 1)
        f1.addWidget(QLabel(tr("Deposits (validation)")), 1, 0)
        f1.addWidget(self.deposits, 1, 1)
        f1.addWidget(QLabel(tr("Absence points (ROC)")), 2, 0)
        f1.addWidget(self.negatives, 2, 1)
        self.body.addWidget(g1)

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
        hh.setSectionResizeMode(3, QHeaderView.Stretch)
        for col, w in ((0, 44), (2, 220), (4, 70)):
            self.table.setColumnWidth(col, w)
        self._rows = []
        prev = last.get("evidence", {})
        for lyr in store.rasters():
            if "mask" in lyr.name.lower():
                continue
            self._add_row(lyr, prev.get(lyr.name))
        v2.addWidget(self.table, 1)
        v2.addWidget(hint(tr("Scores are 0 (unfavourable) … 1 (favourable). Linear: values outside the range are "
                             "clipped; the default range is the 2nd–98th percentile. Binary: 1 where the condition "
                             "holds (binary index overlay).")))
        self.body.addWidget(g2, 1)

        g3 = QGroupBox(tr("Layer weights"))
        v3 = QVBoxLayout(g3)
        row = QHBoxLayout()
        self.weighting = QComboBox()
        for k, v in self.WEIGHTINGS.items():
            self.weighting.addItem(tr(v), k)
        self.weighting.setCurrentIndex(max(0, self.weighting.findData(last.get("weighting", "equal"))))
        self.delta = QDoubleSpinBox()
        self.delta.setRange(0.0, 4.0)
        self.delta.setDecimals(2)
        self.delta.setSingleStep(0.5)
        self.delta.setValue(last.get("delta", 1.0))
        self.delta.setToolTip(tr("Fuzziness δ of every judgement a: triangular fuzzy number (a − δ, a, a + δ)."))
        row.addWidget(QLabel(tr("Weights")))
        row.addWidget(self.weighting, 1)
        row.addWidget(QLabel(tr("fuzziness δ")))
        row.addWidget(self.delta)
        v3.addLayout(row)
        self.matrix = QTableWidget(0, 0)
        self.matrix.setMinimumHeight(150)
        self.matrix.setToolTip(tr("Pairwise comparison (Saaty 1–9 scale): how much more important the row layer is "
                                  "than the column layer — 1 equal, 3 moderate, 5 strong, 7 very strong, 9 extreme; "
                                  "1/3 … 1/9 the inverse. Edit the upper triangle; the lower one is the reciprocal."))
        v3.addWidget(self.matrix)
        self.cr_label = QLabel("")
        self.cr_label.setWordWrap(True)
        v3.addWidget(self.cr_label)
        self.body.addWidget(g3)
        self._prev_matrix = dict(last.get("pairwise", {}))

        g4 = QGroupBox(tr("Options"))
        f4 = QGridLayout(g4)
        self.missing = QComboBox()
        for k, v in MISSING_LABELS.items():
            self.missing.addItem(tr(v), k)
        self.missing.setCurrentIndex(list(MISSING_LABELS).index(last.get("missing", "renormalise")))
        self.o_scores = QCheckBox(tr("score maps"))
        self.o_scores.setChecked(last.get("out_scores", False))
        self.prefix = QLineEdit(last.get("prefix", self.PREFIX))
        f4.addWidget(QLabel(tr("Missing evidence")), 0, 0)
        f4.addWidget(self.missing, 0, 1, 1, 3)
        f4.addWidget(QLabel(tr("Outputs")), 1, 0)
        o = QHBoxLayout()
        o.addWidget(QLabel(tr("index +")))
        o.addWidget(self.o_scores)
        o.addStretch(1)
        f4.addLayout(o, 1, 1, 1, 3)
        f4.addWidget(QLabel(tr("Output name prefix")), 2, 0)
        f4.addWidget(self.prefix, 2, 1)
        self._extra_options(f4, 3)
        self.body.addWidget(g4)
        self.weighting.currentIndexChanged.connect(self._sync)
        self.delta.valueChanged.connect(self._update_weights)
        self.matrix.itemChanged.connect(self._matrix_edited)
        self._sync()
        self.resize(1040, 940)

    def _extra_options(self, grid: QGridLayout, row: int):
        """Hook for subclasses: more option rows starting at *row*."""

    # ------------------------------------------------------------------ evidence rows
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
        rule = QComboBox()
        for k, v in RULE_LABELS.items():
            rule.addItem(tr(v), k)
        default = prev.get("rule") if prev else ("classes" if lyr.categorical else
                                                 "negative" if lyr.name.lower().startswith("dist") else "positive")
        rule.setCurrentIndex(list(RULE_LABELS).index(default))
        self.table.setCellWidget(r, 2, rule)
        spec = QLineEdit(prev.get("spec") if prev else default_spec(default, vals))
        self.table.setCellWidget(r, 3, spec)
        wt = QDoubleSpinBox()
        wt.setRange(0, 100)
        wt.setDecimals(2)
        wt.setValue(prev.get("weight", 1.0) if prev else 1.0)
        self.table.setCellWidget(r, 4, wt)

        def changed(_=None, rr=rule, s=spec, v=vals, st=[default]):
            new = rr.currentData()
            kinds = lambda k: "c" if k == "classes" else "r" if k in ("positive", "negative") else "t"  # noqa: E731
            if kinds(new) != kinds(st[0]):
                s.setText(default_spec(new, v))
            st[0] = new
        rule.currentIndexChanged.connect(changed)
        use.toggled.connect(lambda _=None: self._sync())
        self._rows.append(dict(layer=lyr, use=use, rule=rule, spec=spec, weight=wt))

    def _selected(self):
        return [r for r in self._rows if r["use"].isChecked()]

    # ------------------------------------------------------------------ AHP matrix
    def _sync(self):
        mode = self.weighting.currentData()
        for r in self._rows:
            r["weight"].setEnabled(mode == "custom")
        ahp = mode in ("ahp", "fuzzy_ahp")
        self.matrix.setVisible(ahp)
        self.cr_label.setVisible(ahp)
        self.delta.setEnabled(mode == "fuzzy_ahp")
        if ahp:
            self._build_matrix()

    def _build_matrix(self):
        names = [r["layer"].name for r in self._selected()]
        self.matrix.blockSignals(True)
        self.matrix.clear()
        self.matrix.setRowCount(len(names))
        self.matrix.setColumnCount(len(names))
        self.matrix.setHorizontalHeaderLabels(names)
        self.matrix.setVerticalHeaderLabels(names)
        for i, a in enumerate(names):
            for j, b in enumerate(names):
                if i == j:
                    it = QTableWidgetItem("1")
                    it.setFlags(it.flags() & ~Qt.ItemIsEditable)
                elif i < j:
                    it = QTableWidgetItem(self._prev_matrix.get(f"{a}|{b}", "1"))
                else:
                    it = QTableWidgetItem("")
                    it.setFlags(it.flags() & ~Qt.ItemIsEditable)
                    it.setForeground(theme.qcolor(theme.MUTED) if hasattr(theme, "qcolor") else it.foreground())
                it.setTextAlignment(Qt.AlignCenter)
                self.matrix.setItem(i, j, it)
        self.matrix.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.matrix.blockSignals(False)
        self._update_weights()

    def _matrix_edited(self, item):
        self._update_weights()

    def pairwise(self):
        """Full reciprocal matrix from the upper triangle (None if a judgement is invalid)."""
        n = self.matrix.rowCount()
        A = np.ones((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                txt = self.matrix.item(i, j).text() if self.matrix.item(i, j) else "1"
                try:
                    v = parse_judgement(txt)
                except (ValueError, ZeroDivisionError):
                    return None
                A[i, j], A[j, i] = v, 1.0 / v
        return A

    def _update_weights(self):
        if self.weighting.currentData() not in ("ahp", "fuzzy_ahp"):
            return
        A = self.pairwise()
        n = self.matrix.rowCount()
        self.matrix.blockSignals(True)
        if A is not None:
            for i in range(n):
                for j in range(i):
                    v = A[i, j]
                    self.matrix.item(i, j).setText(f"1/{1 / v:g}" if v < 1 else f"{v:g}")
        self.matrix.blockSignals(False)
        if n < 2:
            self.cr_label.setText(tr("Tick at least two evidence layers for pairwise comparisons."))
            return
        if A is None:
            self.cr_label.setText(tr("Invalid judgement — use numbers such as 3 or 1/5."))
            return
        res = ahp_weights(A) if self.weighting.currentData() == "ahp" else fuzzy_ahp_weights(A, self.delta.value())
        names = [self.matrix.verticalHeaderItem(i).text() for i in range(n)]
        ok = res.cr < 0.1 or n < 3
        self.cr_label.setText(tr("Weights: {w}.   λmax = {lam:.3f}, CI = {ci:.3f}, CR = {cr:.3f} {ok}",
                                 w=", ".join(f"{a} {b:.3f}" for a, b in zip(names, res.weights)), lam=res.lambda_max,
                                 ci=res.ci, cr=res.cr, ok="✓" if ok else tr("(≥ 0.10: revise the judgements)")))
        self.cr_label.setStyleSheet("" if ok else f"color:{theme.WARN}")

    # ------------------------------------------------------------------ validate / params
    def validate(self):
        sel = self._selected()
        if not sel:
            return tr("Tick at least one evidence layer.")
        for r in sel:
            try:
                _check_spec(r["rule"].currentData(), r["spec"].text())
            except ValueError as exc:
                return f"{r['layer'].name}: {exc}"
        mode = self.weighting.currentData()
        if mode in ("ahp", "fuzzy_ahp"):
            if len(sel) < 2:
                return tr("Tick at least two evidence layers for pairwise comparisons.")
            if self.matrix.rowCount() != len(sel) or self.pairwise() is None:
                return tr("Invalid judgement — use numbers such as 3 or 1/5.")
        if mode == "custom" and sum(r["weight"].value() for r in sel) <= 0:
            return tr("At least one layer needs a positive weight.")
        return None

    def params(self) -> dict:
        sel = self._selected()
        mode = self.weighting.currentData()
        P = self.pairwise() if mode in ("ahp", "fuzzy_ahp") else None
        if P is not None:
            names = [r["layer"].name for r in sel]
            for i in range(len(names)):
                for j in range(i + 1, len(names)):
                    self._prev_matrix[f"{names[i]}|{names[j]}"] = self.matrix.item(i, j).text()
        self.LAST.update(weighting=mode, delta=self.delta.value(), missing=self.missing.currentData(),
                     out_scores=self.o_scores.isChecked(), prefix=self.prefix.text(), pairwise=self._prev_matrix,
                     evidence={r["layer"].name: dict(use=r["use"].isChecked(), rule=r["rule"].currentData(),
                                                     spec=r["spec"].text(), weight=r["weight"].value())
                               for r in self._rows})
        return dict(
            evidence=[dict(name=r["layer"].name, data=r["layer"].data, grid=r["layer"].grid,
                           rule=r["rule"].currentData(), spec=r["spec"].text(), weight=r["weight"].value())
                      for r in sel],
            weighting=mode, pairwise=P.tolist() if P is not None else None, delta=self.delta.value(),
            missing=self.missing.currentData(),
            deposits=self.deposits.layer().gdf if self.deposits.layer() is not None else None,
            deposits_name=self.deposits.layer().name if self.deposits.layer() is not None else None,
            negatives=self.negatives.layer().gdf if self.negatives.layer() is not None else None,
            mask=(self.mask.layer().data, self.mask.layer().grid) if self.mask.layer() is not None else None,
            out_scores=self.o_scores.isChecked(), prefix=self.prefix.text().strip() or self.PREFIX)


class IndexOverlayResultsDialog(QDialog):
    TITLE = N_("Index Overlay (AHP / Fuzzy AHP weights)")
    WEIGHTINGS = WEIGHTING_LABELS
    RESULT_KEY = "io"
    EXPORT_NAME = "index_overlay_results.xlsx"

    def __init__(self, result, window=None, parent=None):
        super().__init__(parent)
        self.result = result
        self.main_window = window
        res = result.extra[self.RESULT_KEY]
        title = tr(self.TITLE)
        self.setWindowTitle(tr("{method} — results", method=title))
        lay = QVBoxLayout(self)
        lay.setContentsMargins(14, 12, 14, 12)
        t = QLabel(tr("{method} — results", method=title))
        t.setObjectName("title")
        lay.addWidget(t)
        ev = result.evaluation
        cards = QHBoxLayout()
        cards.addWidget(metric_card(f"{len(res.scores)}", tr("evidence layers")))
        self._method_cards(cards, res)
        cards.addWidget(metric_card(tr(self.WEIGHTINGS[result.extra["weighting"]]).split(" (")[0], tr("weights")))
        if res.ahp is not None:
            cards.addWidget(metric_card(f"{res.ahp.cr:.3f}", tr("consistency ratio CR")))
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
            w = hint("⚠ " + "\n⚠ ".join(result.messages))
            w.setStyleSheet(f"color:{theme.WARN}")
            lay.addWidget(w)
        self.tabs = QTabWidget()
        self.tabs.addTab(self._weights_tab(res), tr("Evidence and weights"))
        self._extra_tabs(res)
        if res.ahp_table is not None:
            w = QWidget()
            l = QVBoxLayout(w)
            lab = QLabel(tr("<b>Pairwise comparison matrix</b> — judgements a_ij (row versus column), the weights "
                            "(principal eigenvector; fuzzy AHP: centroid of the fuzzy weights l, m, u), λmax, the "
                            "consistency index CI = (λmax − n)/(n − 1) and ratio CR = CI/RI."))
            lab.setWordWrap(True)
            l.addWidget(lab)
            l.addWidget(make_table(res.ahp_table), 1)
            self.tabs.addTab(w, tr("AHP"))
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

    def _method_cards(self, cards, res):
        """Hook for subclasses: cards before the weighting card."""

    def _extra_tabs(self, res):
        """Hook for subclasses: tabs after "Evidence and weights"."""

    def _weights_tab(self, res):
        w = QWidget()
        l = QVBoxLayout(w)
        lab = QLabel(tr("<b>Evidence</b> — scoring rule, class scores / range / threshold, normalised weight, mean "
                        "score in the study area and at the deposits, share of the area with score ≥ 0.5."))
        lab.setWordWrap(True)
        l.addWidget(lab)
        row = QHBoxLayout()
        row.addWidget(make_table(res.table), 3)
        canvas = MplWidget(4.5, 3)
        canvas.setMinimumHeight(200)
        ax = canvas.fig.add_subplot(111)
        y = np.arange(len(res.table))
        ax.barh(y, res.table.weight, color=theme.SERIES[0])
        ax.set_yticks(y)
        ax.set_yticklabels(res.table.layer, fontsize=7)
        ax.invert_yaxis()
        ax.set_xlabel(tr("weight"), fontsize=7)
        ax.tick_params(axis="x", labelsize=7)
        canvas.fig.tight_layout()
        row.addWidget(canvas, 2)
        l.addLayout(row, 1)
        self.weights_canvas = canvas
        return w

    def show_evaluation(self):
        from gui.dialogs.evaluation_dialogs import EvaluationDialog
        if self.result.evaluation is not None:
            EvaluationDialog(self.result.evaluation, tr(self.TITLE), self).show()

    def export_excel(self):
        p, _ = QFileDialog.getSaveFileName(self, tr("Export tables"), self.EXPORT_NAME, "Excel (*.xlsx)")
        if not p:
            return
        with pd.ExcelWriter(p) as xw:
            for name, df in self.result.tables.items():
                df.to_excel(xw, sheet_name=name[:31], index=False)
            if self.result.evaluation is not None:
                pd.DataFrame(self.result.evaluation.summary_rows()).to_excel(xw, sheet_name="validation", index=False)
        QMessageBox.information(self, tr("Export"), tr("Saved {path}", path=p))
