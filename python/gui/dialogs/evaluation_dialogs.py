"""Validation result dialog (ROC/AUC, success-rate, prediction-area) + validation input dialog."""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
from PyQt5.QtWidgets import (QDialog, QFileDialog, QFormLayout, QHBoxLayout, QMessageBox, QPushButton,
                             QTabWidget, QVBoxLayout, QWidget, QLabel)

from evaluation.metrics import Evaluation, evaluate, negative_cells
from gui import theme
from i18n import N_, tr
from gui.widgets import BaseDialog, LayerCombo, MplWidget, fit_to_screen, hint, make_table, metric_card
from utils.raster_tools import points_to_cells

SET_COLORS = {"training": theme.SERIES[0], "test": theme.SERIES[1]}


def plot_roc(ax, ev: Evaluation):
    ax.plot([0, 1], [0, 1], ls="--", lw=1, color=theme.MUTED, label=tr("random (AUC = 0.5)"))
    for k, c in ev.curves.items():
        ax.plot(c.fpr, c.tpr, lw=2, color=SET_COLORS.get(k),
                label=tr("{set}: AUC = {auc:.3f} [{lo:.3f}–{hi:.3f}]  (n = {n})", set=tr(k), auc=c.auc,
                         lo=c.auc_ci[0], hi=c.auc_ci[1], n=c.n_pos))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.01)
    ax.set_xlabel(tr("False positive rate (1 − specificity)"))
    ax.set_ylabel(tr("True positive rate (sensitivity)"))
    ax.set_title(tr("ROC curve — {score}", score=ev.score_name))
    ax.set_aspect("equal")
    ax.legend(loc="lower right")


def plot_success(ax, ev: Evaluation):
    ax.plot([0, 100], [0, 100], ls="--", lw=1, color=theme.MUTED, label=tr("random"))
    for k, c in ev.curves.items():
        ax.plot(c.area_frac * 100, c.capture_frac * 100, lw=2, color=SET_COLORS.get(k),
                label=tr("{set}: area under curve = {auc:.3f}", set=tr(k), auc=c.sr_auc))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 101)
    ax.set_xlabel(tr("Study area classified as prospective (%) — highest scores first"))
    ax.set_ylabel(tr("Deposits captured (%)"))
    ax.set_title(tr("Success-rate / prediction-rate curve"))
    ax.legend(loc="lower right")


def plot_pa(ax, ev: Evaluation, which: str = "training"):
    c = ev.curves.get(which) or next(iter(ev.curves.values()))
    ax.plot(c.pa_threshold, c.pa_prediction, lw=2, color=theme.SERIES[0], label=tr("prediction rate (deposits)"))
    ax.set_xlabel(tr("{score} threshold", score=ev.score_name))
    ax.set_ylabel(tr("Prediction rate (%)"), color=theme.SERIES[0])
    ax.set_ylim(0, 100)
    ax2 = ax.twinx()
    ax2.plot(c.pa_threshold, c.pa_area, lw=2, color=theme.SERIES[1], label=tr("occupied area"))
    ax2.set_ylim(100, 0)
    ax2.set_ylabel(tr("Percentage of study area (%)"), color=theme.SERIES[1])
    ax2.grid(False)
    ax2.spines["right"].set_visible(True)
    ax2.spines["right"].set_color(theme.BORDER)
    if c.pa_intersection:
        x, p = c.pa_intersection["threshold"], c.pa_intersection["prediction"]
        ax.axvline(x, color=theme.WARN, lw=1, ls=":")
        ax.plot([x], [p], "o", color=theme.WARN, ms=7)
        ax.annotate(tr("{p:.0f}% of deposits in {a:.0f}% of area", p=p, a=c.pa_intersection['area']), (x, p),
                    xytext=(-10, -22), textcoords="offset points", ha="right", color=theme.TEXT, fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.3", fc=theme.PANEL, ec=theme.BORDER, alpha=0.9))
    ax.set_title(tr("Prediction–area plot ({set} deposits)", set=tr(c.name)))


def summary_table(ev: Evaluation, translate: bool = False) -> pd.DataFrame:
    """Validation summary as two columns (Index, Value): one row per statistic, training rows then test rows.

    ``translate=True`` gives the GUI version (current language); files are always written in English."""
    t = tr if translate else (lambda x: x)
    rows = [(t(N_("Score layer")), ev.score_name)]
    for r in ev.summary_rows():
        s = f" ({t(r['set'])})"
        rows += [
            (t(N_("Deposit cells")) + s, f"{r['deposits']:,}"),
            ((t(N_("Background cells")) if r.get("negatives", "background") == "background"
              else t(N_("Absence-point cells"))) + s, f"{r['background_cells']:,}"),
            (t(N_("ROC AUC")) + s, f"{r['roc_auc']:.4f}"),
            (t(N_("ROC AUC 95 % CI (DeLong)")) + s, f"{r['roc_auc_ci_low']:.4f} – {r['roc_auc_ci_high']:.4f}"),
            (t(N_("Success-rate AUC")) + s, f"{r['success_rate_auc']:.4f}"),
            (t(N_("P-A intersection: threshold")) + s,
             "—" if pd.isna(r["pa_threshold"]) else f"{r['pa_threshold']:.6g}"),
            (t(N_("P-A intersection: deposits predicted (%)")) + s,
             "—" if pd.isna(r["pa_prediction_rate"]) else f"{r['pa_prediction_rate']:.2f}"),
            (t(N_("P-A intersection: area occupied (%)")) + s,
             "—" if pd.isna(r["pa_area"]) else f"{r['pa_area']:.2f}"),
        ]
    return pd.DataFrame(rows, columns=[t(N_("Index")), t(N_("Value"))])


class EvaluationDialog(QDialog):
    def __init__(self, ev: Evaluation, title: str = "", parent=None):
        super().__init__(parent)
        self.ev = ev
        title = title or tr("Validation")
        self.setWindowTitle(tr("{title} — ROC / AUC", title=title))
        lay = QVBoxLayout(self)
        lay.setContentsMargins(14, 12, 14, 12)
        t = QLabel(f"{title}")
        t.setObjectName("title")
        lay.addWidget(t)
        lay.addWidget(hint(tr("Score: {score}. Positives = deposit cells, negatives = all other cells of the "
                              "study area. Test deposits (if any) were not used to fit the model.",
                              score=ev.score_name)))
        cards = QHBoxLayout()
        trn = ev.curves["training"]
        cards.addWidget(metric_card(f"{trn.auc:.3f}", tr("ROC AUC — training ({n} deposits)", n=trn.n_pos)))
        if "test" in ev.curves:
            te = ev.curves["test"]
            cards.addWidget(metric_card(f"{te.auc:.3f}", tr("ROC AUC — test ({n} deposits)", n=te.n_pos)))
        cards.addWidget(metric_card(f"{trn.sr_auc:.3f}", tr("Success-rate AUC — training")))
        if trn.pa_intersection:
            cards.addWidget(metric_card(f"{trn.pa_intersection['prediction']:.0f}% / {trn.pa_intersection['area']:.0f}%",
                                        tr("P-A: deposits predicted / area occupied")))
        cards.addStretch(1)
        lay.addLayout(cards)

        self.tabs = QTabWidget()
        self.figs = {}                  # English name (export file name) -> chart; tab order = insertion order
        for name, fn in ((N_("ROC curve"), plot_roc), (N_("Success rate"), plot_success),
                         (N_("Prediction–area"), plot_pa)):
            w = MplWidget(7, 5)
            ax = w.fig.add_subplot(111)
            fn(ax, ev)
            self.figs[name] = w
            self.tabs.addTab(w, tr(name))
        self.summary = summary_table(ev)
        table = make_table(summary_table(ev, translate=True))
        table.horizontalHeader().setStretchLastSection(True)
        self.tabs.addTab(table, tr("Summary"))
        lay.addWidget(self.tabs, 1)

        btns = QHBoxLayout()
        b_png = QPushButton(theme.icon("mdi6.image-outline"), tr("Export figure…"))
        b_png.clicked.connect(self.export_png)
        b_csv = QPushButton(theme.icon("mdi6.file-delimited-outline"), tr("Export curves (CSV)…"))
        b_csv.clicked.connect(self.export_csv)
        close = QPushButton(tr("Close"))
        close.setProperty("primary", True)
        close.clicked.connect(self.accept)
        btns.addWidget(b_png)
        btns.addWidget(b_csv)
        btns.addStretch(1)
        btns.addWidget(close)
        lay.addLayout(btns)
        fit_to_screen(self, 900, 760)

    def export_png(self):
        names = list(self.figs)
        i = self.tabs.currentIndex()
        name = names[i] if i < len(names) else names[0]
        w = self.figs[name]
        p, _ = QFileDialog.getSaveFileName(self, tr("Export figure"), f"{name.replace(' ', '_').lower()}.png",
                                           "PNG (*.png);;SVG (*.svg);;PDF (*.pdf)")
        if p:
            w.fig.savefig(p, dpi=200)

    def export_csv(self):
        p, _ = QFileDialog.getSaveFileName(self, tr("Export curves"), "evaluation_curves.csv", "CSV (*.csv)")
        if not p:
            return
        frames = []
        for k, c in self.ev.curves.items():
            frames.append(pd.DataFrame(dict(set=k, curve="roc", x=c.fpr, y=c.tpr)))
            frames.append(pd.DataFrame(dict(set=k, curve="success_rate", x=c.area_frac, y=c.capture_frac)))
            frames.append(pd.DataFrame(dict(set=k, curve="pa_prediction", x=c.pa_threshold, y=c.pa_prediction)))
            frames.append(pd.DataFrame(dict(set=k, curve="pa_area", x=c.pa_threshold, y=c.pa_area)))
        pd.concat(frames).to_csv(p, index=False)
        self.summary.to_csv(os.path.splitext(p)[0] + "_summary.csv", index=False)


class ValidationDialog(BaseDialog):
    """Pick any prospectivity raster + deposits -> EvaluationDialog."""

    def __init__(self, store, parent=None):
        super().__init__(tr("Validate prospectivity map"),
                         tr("ROC / AUC, success-rate and prediction-area analysis of any raster against known "
                            "deposits. Higher raster values must mean higher prospectivity."), parent,
                         ok_text=tr("Evaluate"))
        self.store = store
        form = QFormLayout()
        self.score = LayerCombo(store, "raster", predicate=lambda l: not l.categorical or True)
        self.train = LayerCombo(store, "point")
        self.test = LayerCombo(store, "point", allow_none=True, none_text=tr("— no separate test set —"))
        self.test.setCurrentIndex(0)
        self.mask = LayerCombo(store, "raster", allow_none=True, none_text=tr("— valid cells of the score raster —"))
        self.mask.setCurrentIndex(0)
        self.negatives = LayerCombo(store, "point", allow_none=True, none_text=tr("— all background cells —"))
        self.negatives.setCurrentIndex(0)
        self.negatives.setToolTip(tr("Optional known non-deposit points used as ROC negatives instead of all "
                                     "background cells."))
        form.addRow(tr("Prospectivity raster"), self.score)
        form.addRow(tr("Deposits (training)"), self.train)
        form.addRow(tr("Deposits (test)"), self.test)
        form.addRow(tr("Study-area mask"), self.mask)
        form.addRow(tr("Absence points (ROC)"), self.negatives)
        self.body.addLayout(form)
        self.resize(560, 0)

    def validate(self):
        if self.score.layer() is None or self.train.layer() is None:
            return tr("Choose a raster and a point layer.")
        return None

    def run(self, progress=None) -> Evaluation:
        s = self.score.layer()
        valid = ~np.ma.getmaskarray(s.data)
        m = self.mask.layer()
        if m is not None:
            if not m.grid.same_as(s.grid):
                raise ValueError(tr("The mask is not on the same grid as the prospectivity raster."))
            valid &= ~np.ma.getmaskarray(m.data)
        r, c = points_to_cells(self.train.layer().gdf, s.grid, valid_mask=valid)
        tr_r = te_r = te_c = None
        if self.test.layer() is not None:
            te_r, te_c = points_to_cells(self.test.layer().gdf, s.grid, valid_mask=valid)
        neg = self.negatives.layer()
        neg = negative_cells(neg.gdf if neg is not None else None, s.grid, valid)
        return evaluate(s.data.astype(float), r, c, te_r, te_c, valid_mask=valid, score_name=s.name, negatives=neg)
