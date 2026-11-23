"""Dialog parts shared by the hybrid methods: knowledge table (mineral-system processes, directions, expert weights),
training data group, generic option helpers and a generic results dialog."""
from __future__ import annotations

from typing import Dict, List, Optional

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

from .pipeline import PROCESSES, parse_scores

_LAST: Dict[str, dict] = {}            # remembered settings per method (session)
DIRECTIONS = {1: N_("high values favourable (+)"), -1: N_("low values favourable (−)")}
PROCESS_LABELS = {"source": N_("source"), "pathway": N_("pathway"), "trap": N_("trap"),
                  "deposition": N_("deposition"), "preservation": N_("preservation"), "footprint": N_("footprint")}


def _sample(lyr: RasterLayer, n: int = 400) -> np.ndarray:
    d = lyr.data
    return np.ma.compressed(d[:: max(1, d.shape[0] // n), :: max(1, d.shape[1] // n)]).astype(float)


def guess_process(name: str) -> str:
    n = name.lower()
    if any(t in n for t in ("fault", "struct", "lineament", "shear", "dist_f")):
        return "pathway"
    if any(t in n for t in ("geol", "host", "lith", "unit", "strat")):
        return "trap"
    if any(t in n for t in ("granite", "intrus", "rad_", "mag", "grav", "source")):
        return "source"
    if any(t in n for t in ("aster", "alter", "silica", "kaolin", "clay")):
        return "footprint"
    return "deposition"


def guess_direction(name: str) -> int:
    return -1 if name.lower().startswith("dist") or "distance" in name.lower() else 1


class KnowledgeTableWidget(QWidget):
    """Knowledge table: one row per evidence layer (use, layer, process, direction, class scores, expert weight)
    and the process weights."""
    COLS = [N_("Use"), N_("Evidence layer"), N_("Process"), N_("Direction / scores"), N_("Weight")]

    def __init__(self, store: LayerStore, prev: Optional[dict] = None, parent=None):
        super().__init__(parent)
        prev = prev or {}
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.table = QTableWidget(0, len(self.COLS))
        self.table.setHorizontalHeaderLabels([tr(c) for c in self.COLS])
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionMode(QAbstractItemView.NoSelection)
        self.table.setMinimumHeight(190)
        self.table.verticalHeader().setDefaultSectionSize(30)
        hh = self.table.horizontalHeader()
        hh.setSectionResizeMode(QHeaderView.Fixed)
        hh.setSectionResizeMode(1, QHeaderView.Stretch)
        hh.setSectionResizeMode(3, QHeaderView.Stretch)
        for col, w in ((0, 44), (2, 130), (4, 90)):
            self.table.setColumnWidth(col, w)
        self.rows: List[dict] = []
        for lyr in store.rasters():
            if "mask" in lyr.name.lower():
                continue
            self._add_row(lyr, prev.get("evidence", {}).get(lyr.name))
        lay.addWidget(self.table, 3)
        right = QVBoxLayout()
        right.addWidget(QLabel(tr("Process weights")))
        self.ptable = QTableWidget(0, 2)
        self.ptable.setHorizontalHeaderLabels([tr("Process"), tr("Weight")])
        self.ptable.verticalHeader().setVisible(False)
        self.ptable.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.ptable.setColumnWidth(1, 80)
        self.ptable.setMinimumWidth(220)
        right.addWidget(self.ptable, 1)
        lay.addLayout(right, 1)
        self._pw_prev = dict(prev.get("process_weights", {}))
        self.refresh_processes()

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
        proc = QComboBox()
        proc.setEditable(True)
        for p in PROCESSES:
            proc.addItem(tr(PROCESS_LABELS[p]), p)
        txt = prev.get("process") if prev else guess_process(lyr.name)
        i = proc.findData(txt)
        if i >= 0:
            proc.setCurrentIndex(i)
        else:
            proc.setEditText(txt)
        proc.setToolTip(tr("Mineral-system process the layer is a proxy of; type a new name for another process."))
        self.table.setCellWidget(r, 2, proc)
        if lyr.categorical:
            spec = QLineEdit(prev.get("scores") if prev and prev.get("scores") else
                             "; ".join(f"{c:g}=0.5" for c in np.unique(vals)[:40]))
            spec.setToolTip(tr("Expert favourability (0–1) of every class, e.g. '2=0.9; 1=0.1'; classes without "
                               "a score get 0.05."))
            self.table.setCellWidget(r, 3, spec)
            dirn = None
        else:
            dirn = QComboBox()
            for k, v in DIRECTIONS.items():
                dirn.addItem(tr(v), k)
            d = prev.get("direction", guess_direction(lyr.name)) if prev else guess_direction(lyr.name)
            dirn.setCurrentIndex(0 if d > 0 else 1)
            self.table.setCellWidget(r, 3, dirn)
            spec = None
        wt = QDoubleSpinBox()
        wt.setRange(0, 100)
        wt.setDecimals(2)
        wt.setValue(prev.get("weight", 1.0) if prev else 1.0)
        wt.setToolTip(tr("Relative expert weight of the layer inside its process (normalised per process)."))
        self.table.setCellWidget(r, 4, wt)
        row = dict(layer=lyr, use=use, process=proc, direction=dirn, scores=spec, weight=wt)
        self.rows.append(row)
        use.toggled.connect(lambda _=None: self.refresh_processes())
        proc.currentTextChanged.connect(lambda _=None: self.refresh_processes())

    @staticmethod
    def process_of(row) -> str:
        cb = row["process"]
        i = cb.findText(cb.currentText())
        return (cb.itemData(i) if i >= 0 else cb.currentText()).strip() or "process"

    def selected(self):
        return [r for r in self.rows if r["use"].isChecked()]

    def process_weights(self) -> Dict[str, float]:
        out = {}
        for i in range(self.ptable.rowCount()):
            out[self.ptable.item(i, 0).data(Qt.UserRole)] = self.ptable.cellWidget(i, 1).value()
        return out

    def refresh_processes(self):
        if self.ptable.rowCount():
            self._pw_prev.update(self.process_weights())
        procs = list(dict.fromkeys(self.process_of(r) for r in self.selected()))
        self.ptable.setRowCount(0)
        for p in procs:
            i = self.ptable.rowCount()
            self.ptable.insertRow(i)
            it = QTableWidgetItem(tr(PROCESS_LABELS[p]) if p in PROCESS_LABELS else p)
            it.setData(Qt.UserRole, p)
            it.setFlags(it.flags() & ~Qt.ItemIsEditable)
            self.ptable.setItem(i, 0, it)
            sp = QDoubleSpinBox()
            sp.setRange(0, 100)
            sp.setDecimals(2)
            sp.setValue(self._pw_prev.get(p, 1.0))
            self.ptable.setCellWidget(i, 1, sp)

    def validate(self) -> Optional[str]:
        sel = self.selected()
        if not sel:
            return tr("Tick at least one evidence layer.")
        for r in sel:
            if r["scores"] is not None:
                try:
                    parse_scores(r["scores"].text())
                except ValueError as exc:
                    return f"{r['layer'].name}: {exc}"
        if sum(self.process_weights().values()) <= 0:
            return tr("At least one process needs a positive weight.")
        return None

    def evidence(self) -> List[dict]:
        out = []
        for r in self.selected():
            lyr = r["layer"]
            out.append(dict(name=lyr.name, data=lyr.data, grid=lyr.grid, labels=lyr.labels(),
                            kind="categorical" if r["scores"] is not None else "continuous",
                            process=self.process_of(r),
                            direction=r["direction"].currentData() if r["direction"] is not None else 1,
                            scores=r["scores"].text() if r["scores"] is not None else None,
                            weight=r["weight"].value()))
        return out

    def state(self) -> dict:
        return dict(process_weights=self.process_weights(),
                    evidence={r["layer"].name: dict(use=r["use"].isChecked(), process=self.process_of(r),
                                                    direction=r["direction"].currentData() if r["direction"] else 1,
                                                    scores=r["scores"].text() if r["scores"] else None,
                                                    weight=r["weight"].value()) for r in self.rows})


class HybridDialog(BaseDialog):
    """Parameter dialog of a hybrid method. Subclasses set KEY / TITLE / SUBTITLE / PREFIX and implement
    :meth:`build_options` with :meth:`opt_spin`, :meth:`opt_check`, :meth:`opt_combo` (collected automatically)."""
    KEY = "hybrid"
    TITLE = ""
    SUBTITLE = ""
    PREFIX = "hybrid"
    NEED_DEPOSITS = True
    REGRESSION = False         # continuous target: deposits only validate (no split / non-deposit samples)
    USES_NEGATIVES = True      # False: the method counts all cells (WofE family) — no non-deposit sampling
    TEST_DEFAULT = 20
    WIDTH, HEIGHT = 1080, 940

    def __init__(self, store: LayerStore, parent=None):
        super().__init__(tr(self.TITLE), tr(self.SUBTITLE), parent, ok_text=tr("Run model"))
        self.store = store
        self.last = _LAST.setdefault(self.KEY, {})
        self.opts: Dict[str, QWidget] = {}
        cell = next((l.grid.res[0] for l in store.rasters()), 100.0)
        self.cell = cell

        g1 = QGroupBox(tr("Training data"))
        f1 = QFormLayout(g1)
        self.deposits = LayerCombo(store, "point", allow_none=not self.NEED_DEPOSITS,
                                   none_text=tr("— no deposits —"))
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
        self.test_pct.setValue(self.last.get("test_pct", self.TEST_DEFAULT))
        self.seed = QSpinBox()
        self.seed.setRange(0, 10 ** 6)
        self.seed.setValue(self.last.get("seed", 42))
        self.split = split_method_combo(self.last.get("split", "spatial"))
        self.thin = thinning_spin(self.last.get("thin", 0.0))
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
        self.min_dist = QDoubleSpinBox()
        self.min_dist.setRange(0, 1e7)
        self.min_dist.setDecimals(0)
        self.min_dist.setSingleStep(100)
        self.min_dist.setValue(self.last.get("min_dist", 3 * cell))
        self.min_dist.setToolTip(tr("Non-deposit samples keep at least this distance (map units) from every known "
                                    "deposit (buffer)."))
        if self.REGRESSION:
            f1.addRow(tr("Deposits (validation only)"), self.deposits)
            f1.addRow(tr("Study area mask"), self.mask)
            f1.addRow(tr("Absence points (ROC)"), self.negatives)
            self._unused = QWidget(self)               # classification-only settings: kept, but hidden
            hold = QVBoxLayout(self._unused)
            hold.addLayout(row)
            hold.addWidget(self.thin)
            hold.addWidget(self.min_dist)
            self._unused.hide()
            self.test_pct.setValue(0)
        else:
            f1.addRow(tr("Deposits (points)"), self.deposits)
            f1.addRow(tr("Study area mask"), self.mask)
            f1.addRow(tr("Hold out for testing"), row)
            f1.addRow(tr("Thin deposits (min. distance)"), self.thin)
            if self.USES_NEGATIVES:
                f1.addRow(tr("Min. distance of non-deposits"), self.min_dist)
            else:                                      # kept (and passed on), but not shown
                self._unused = QWidget(self)
                QVBoxLayout(self._unused).addWidget(self.min_dist)
                self._unused.hide()
            f1.addRow(tr("Absence points (ROC)"), self.negatives)
        self.body.addWidget(g1)

        g2 = QGroupBox(tr("Mineral-system knowledge table"))
        v2 = QVBoxLayout(g2)
        self.knowledge = KnowledgeTableWidget(store, self.last.get("knowledge"))
        v2.addWidget(self.knowledge, 1)
        v2.addWidget(hint(tr("Every layer is a proxy of one mineral-system process. Inside a process the proxies are "
                             "combined OR-like (power mean, p = 3), the processes AND-like (weighted geometric mean): "
                             "a cell is favourable only when every process is present. Continuous layers are scaled "
                             "robustly (median / IQR) in the expert direction; categorical layers get an expert score "
                             "per class.")))
        self.body.addWidget(g2, 1)

        g3 = QGroupBox(tr("Model options"))
        self.grid_opts = QGridLayout(g3)
        self._n_opts = 0
        self.build_options()
        self.align = QCheckBox(tr("Resample evidence to the first layer's grid if grids differ"))
        self.align.setChecked(True)
        self.prefix = QLineEdit(self.last.get("prefix", self.PREFIX))
        r = (self._n_opts + 1) // 2
        self.grid_opts.addWidget(self.align, r, 0, 1, 4)
        self.grid_opts.addWidget(QLabel(tr("Output name prefix")), r + 1, 0)
        self.grid_opts.addWidget(self.prefix, r + 1, 1)
        self.body.addWidget(g3)
        self.resize(self.WIDTH, self.HEIGHT)

    # ------------------------------------------------------------------ option helpers
    def build_options(self):
        pass

    def _place(self, name, label, widget, tip=""):
        if tip:
            widget.setToolTip(tr(tip))
        r, c = divmod(self._n_opts, 2)
        if isinstance(widget, QCheckBox):
            self.grid_opts.addWidget(widget, r, 2 * c, 1, 2)
        else:
            self.grid_opts.addWidget(QLabel(tr(label)), r, 2 * c)
            self.grid_opts.addWidget(widget, r, 2 * c + 1)
        self._n_opts += 1
        self.opts[name] = widget
        return widget

    def opt_spin(self, name, label, lo, hi, value, decimals=2, step=None, tip="", integer=False, special=None):
        w = QSpinBox() if integer else QDoubleSpinBox()
        w.setRange(lo, hi)
        if not integer:
            w.setDecimals(decimals)
        if step:
            w.setSingleStep(step)
        if special:
            w.setSpecialValueText(tr(special))
        w.setValue(self.last.get("opts", {}).get(name, value))
        return self._place(name, label, w, tip)

    def opt_check(self, name, label, value, tip=""):
        w = QCheckBox(tr(label))
        w.setChecked(self.last.get("opts", {}).get(name, value))
        return self._place(name, label, w, tip)

    def opt_combo(self, name, label, items, value, tip=""):
        w = QComboBox()
        for k, v in items:
            w.addItem(tr(v), k)
        cur = self.last.get("opts", {}).get(name, value)
        idx = [k for k, _ in items].index(cur) if cur in [k for k, _ in items] else 0
        w.setCurrentIndex(idx)
        w.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        w.setMinimumContentsLength(14)
        return self._place(name, label, w, tip)

    def option_values(self) -> dict:
        out = {}
        for k, w in self.opts.items():
            if isinstance(w, QCheckBox):
                out[k] = w.isChecked()
            elif isinstance(w, QComboBox):
                out[k] = w.currentData()
            else:
                out[k] = w.value()
        return out

    # ------------------------------------------------------------------ validate / params
    def extra_validate(self) -> Optional[str]:
        return None

    def validate(self):
        if self.NEED_DEPOSITS and self.deposits.layer() is None:
            return tr("Add a point layer with known deposits (training points).")
        msg = self.knowledge.validate()
        if msg:
            return msg
        sel = self.knowledge.selected()
        g0 = sel[0]["layer"].grid
        if not self.align.isChecked():
            for r in sel[1:]:
                if not r["layer"].grid.same_as(g0):
                    return tr("'{layer}' is not on the grid of '{ref}'. Enable resampling "
                              "or use Data ▸ Align raster to grid.", layer=r['layer'].name, ref=sel[0]['layer'].name)
        if g0.is_geographic:
            return tr("The evidence grid is in geographic degrees. Reproject it to a metric CRS first.")
        return self.extra_validate()

    def extra_params(self) -> dict:
        return {}

    def params(self) -> dict:
        opts = self.option_values()
        self.last.update(test_pct=self.test_pct.value(), seed=self.seed.value(), split=self.split.currentData(),
                         thin=self.thin.value(), min_dist=self.min_dist.value(), prefix=self.prefix.text(),
                         knowledge=self.knowledge.state(), opts=opts)
        dep = self.deposits.layer()
        p = dict(evidence=self.knowledge.evidence(), process_weights=self.knowledge.process_weights(),
                 deposits=dep.gdf if dep is not None else None, deposits_name=dep.name if dep is not None else None,
                 negatives=self.negatives.layer().gdf if self.negatives.layer() is not None else None,
                 mask=(self.mask.layer().data, self.mask.layer().grid) if self.mask.layer() is not None else None,
                 test_fraction=self.test_pct.value() / 100.0, seed=self.seed.value(),
                 split_method=self.split.currentData(), thin_distance=self.thin.value(),
                 min_distance=self.min_dist.value(), auto_align=self.align.isChecked(),
                 prefix=self.prefix.text().strip() or self.PREFIX)
        p.update(opts)
        p.update(self.extra_params())
        return p


# ============================================================================== results
class HybridResultsDialog(QDialog):
    """Generic results window: metric cards, messages, one tab per table (``extra['tabs']`` = [(table key, tab
    title, caption)]), expert-versus-fitted weight charts and importance chart, ROC / P–A, Excel export."""

    def __init__(self, result, title: str, window=None, parent=None):
        super().__init__(parent)
        self.result = result
        self.main_window = window
        self.title = title
        self.setWindowTitle(tr("{method} — results", method=title))
        lay = QVBoxLayout(self)
        lay.setContentsMargins(14, 12, 14, 12)
        t = QLabel(tr("{method} — results", method=title))
        t.setObjectName("title")
        lay.addWidget(t)
        ev = result.evaluation
        cards = QHBoxLayout()
        for val, cap in result.extra.get("cards", []):
            cards.addWidget(metric_card(val, tr(cap)))
        if ev is not None:
            cards.addWidget(metric_card(f"{ev.curves['training'].auc:.3f}",
                                        tr(result.extra.get("auc_caption", N_("ROC AUC — training")))))
            if "test" in ev.curves:
                cards.addWidget(metric_card(f"{ev.curves['test'].auc:.3f}", tr("ROC AUC — test")))
        cards.addStretch(1)
        lay.addLayout(cards)
        if result.messages:
            w = hint("⚠ " + "\n⚠ ".join(result.messages))
            w.setStyleSheet(f"color:{theme.WARN}")
            lay.addWidget(w)
        self.tabs = QTabWidget()
        self.canvases = []
        for key, tab, cap in result.extra.get("tabs", []):
            df = result.tables.get(key)
            if df is None or not len(df):
                continue
            self.tabs.addTab(self._tab(key, df, tr(cap)), tr(tab))
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
        fit_to_screen(self, 1100, 760)

    def _tab(self, key, df: pd.DataFrame, text: str) -> QWidget:
        w = QWidget()
        l = QVBoxLayout(w)
        lab = QLabel(text)
        lab.setWordWrap(True)
        l.addWidget(lab)
        chart = self._chart(key, df)
        if chart is None:
            l.addWidget(make_table(df), 1)
        else:
            row = QHBoxLayout()
            row.addWidget(make_table(df), 3)
            row.addWidget(chart, 2)
            l.addLayout(row, 1)
        return w

    def _chart(self, key, df):
        """Bar chart for weight tables (expert vs fitted) and importance tables."""
        cols = set(df.columns)
        name = "layer" if "layer" in cols else "process" if "process" in cols else None
        if name is None:
            return None
        if {"expert_weight", "fitted_weight"} <= cols:
            canvas = MplWidget(4.5, 3)
            canvas.setMinimumHeight(200)
            ax = canvas.fig.add_subplot(111)
            y = np.arange(len(df))
            ax.barh(y - 0.2, df.expert_weight, 0.4, color=theme.MUTED, label=tr("expert"))
            err = df.fitted_weight_sd if "fitted_weight_sd" in cols else None
            ax.barh(y + 0.2, df.fitted_weight, 0.4, xerr=err, color=theme.SERIES[0], label=tr("fitted"))
            ax.set_yticks(y)
            ax.set_yticklabels(df[name].astype(str), fontsize=7)
            ax.invert_yaxis()
            ax.legend(fontsize=7)
            ax.tick_params(axis="x", labelsize=7)
        else:
            val = next((c for c in df.columns if "drop" in c or c in ("importance", "weight_contrast")), None)
            if val is None:
                return None
            canvas = MplWidget(4.5, 3)
            canvas.setMinimumHeight(200)
            ax = canvas.fig.add_subplot(111)
            y = np.arange(len(df))
            ax.barh(y, df[val], color=theme.SERIES[0])
            ax.set_yticks(y)
            ax.set_yticklabels(df[name].astype(str), fontsize=7)
            ax.invert_yaxis()
            ax.set_xlabel(val.replace("_", " "), fontsize=7)
            ax.axvline(0, color=theme.MUTED, lw=0.8)
            ax.tick_params(axis="x", labelsize=7)
        canvas.fig.tight_layout()
        self.canvases.append(canvas)
        return canvas

    def show_evaluation(self):
        from gui.dialogs.evaluation_dialogs import EvaluationDialog
        if self.result.evaluation is not None:
            EvaluationDialog(self.result.evaluation, self.title, self).show()

    def export_excel(self):
        p, _ = QFileDialog.getSaveFileName(self, tr("Export tables"), f"{self.result.method}_results.xlsx",
                                           "Excel (*.xlsx)")
        if not p:
            return
        with pd.ExcelWriter(p) as xw:
            for name, df in self.result.tables.items():
                if isinstance(df, pd.DataFrame) and len(df):
                    df.to_excel(xw, sheet_name=name[:31], index=False)
            if self.result.evaluation is not None:
                pd.DataFrame(self.result.evaluation.summary_rows()).to_excel(xw, sheet_name="validation", index=False)
        QMessageBox.information(self, tr("Export"), tr("Saved {path}", path=p))


def show_hybrid_result(window, result, title: str):
    for out in reversed(result.outputs):
        window.add_output_raster(out)
    for vec in result.vectors:
        window.add_output_vector(vec)
    dlg = HybridResultsDialog(result, title, window, window)
    dlg.show()
    window.keep_dialog(dlg)
