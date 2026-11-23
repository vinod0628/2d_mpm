"""Statistics tab of the layer properties dialog: histogram, cumulative probability curve and descriptive statistics
of the cell values of a raster or of a numeric attribute of a vector layer (similar to the ArcMap histogram)."""
from __future__ import annotations

import os
from typing import List, Optional

import numpy as np
from PyQt5.QtWidgets import (QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QHBoxLayout, QLabel,
                             QMessageBox, QPushButton, QSpinBox, QVBoxLayout, QWidget)

from core import symbology as sym
from gui import theme
from gui.widgets import MplWidget, hint
from i18n import N_, tr
from utils import statistics as ustats

STAT_ROWS = [("count", N_("Count")), ("nulls", N_("Nulls")), ("min", N_("Minimum")), ("max", N_("Maximum")),
             ("range", N_("Range")), ("sum", N_("Sum")), ("mean", N_("Mean")), ("std", N_("Std. deviation")),
             ("cv", N_("Coef. of variation")), ("skewness", N_("Skewness")), ("kurtosis", N_("Kurtosis")),
             ("q1", N_("1st quartile")), ("median", N_("Median")), ("q3", N_("3rd quartile"))]


def fmt_stat(key: str, v) -> str:
    if v is None:
        return "—"
    if key in ("count", "nulls"):
        return f"{int(v):,}"
    return f"{v:,.3f}"


class StatisticsPage(QWidget):
    """Histogram (by number of intervals or by interval width), empirical cumulative probability curve on a second
    axis and a statistics box in the top-right corner. Values are read when the tab is first shown."""

    def __init__(self, layer, is_raster: bool, fields: Optional[List[str]] = None, start_dir: str = "",
                 parent=None):
        super().__init__(parent)
        self.start_dir = start_dir                    # first folder offered by Save histogram
        self.hist = None
        self.layer, self.is_raster = layer, is_raster
        self._cache = {}
        self._loaded = False
        self.values = np.array([])
        self.stats = {}
        lay = QVBoxLayout(self)
        top = QHBoxLayout()
        self.field = QComboBox()
        if is_raster:
            top.addWidget(QLabel(tr("Values")))
            top.addWidget(QLabel(tr("cell values (NoData excluded)")), 1)
        else:
            for f in fields or []:
                self.field.addItem(f, f)
            top.addWidget(QLabel(tr("Field")))
            top.addWidget(self.field, 1)

        row = QHBoxLayout()
        self.mode = QComboBox()
        self.mode.addItem(tr("Number of intervals"), "count")
        self.mode.addItem(tr("Interval width"), "width")
        self.mode.setToolTip(tr("Number of intervals: equal intervals from the minimum to the maximum. Interval "
                                "width: intervals of this size with limits at whole multiples of the width."))
        self.count = QSpinBox()
        self.count.setMinimumWidth(80)
        self.count.setRange(1, ustats.MAX_BINS)
        self.count.setValue(10)
        self.width = QDoubleSpinBox()
        self.width.setMinimumWidth(110)
        self.width.setRange(0.001, 1e12)
        self.width.setDecimals(3)
        self.width.setValue(1.0)
        self.width.setVisible(False)
        self.cumulative = QCheckBox(tr("Cumulative probability"))
        self.cumulative.setChecked(True)
        self.show_stats = QCheckBox(tr("Statistics box"))
        self.show_stats.setChecked(True)
        self.show_stats.setToolTip(tr("Show the statistics in the top-right corner of the histogram"))
        self.copy = QPushButton(theme.icon("mdi6.content-copy"), tr("Copy statistics"))
        self.copy.setToolTip(tr("Copy the statistics to the clipboard (tab-separated)"))
        self.save = QPushButton(theme.icon("mdi6.content-save-outline"), tr("Save histogram"))
        self.save.setToolTip(tr("Save the histogram as shown (intervals, cumulative curve, statistics box) as a PNG "
                                "file — 300 dpi, white background"))
        self.save.setEnabled(False)
        row.addWidget(QLabel(tr("Histogram")))
        row.addWidget(self.mode)
        row.addWidget(self.count)
        row.addWidget(self.width)
        row.addWidget(self.cumulative)
        row.addWidget(self.show_stats)
        row.addStretch(1)
        top.addWidget(self.copy)
        top.addWidget(self.save)
        lay.addLayout(top)
        lay.addLayout(row)
        self.chart = MplWidget(6, 3.6)
        self.chart.setMinimumHeight(240)
        lay.addWidget(self.chart, 1)
        self.message = hint("")
        self.message.setVisible(False)
        lay.addWidget(self.message)

        self.field.currentIndexChanged.connect(lambda _i: self._load_values())
        self.mode.currentIndexChanged.connect(self._mode_changed)
        self.count.valueChanged.connect(lambda _v: self.draw())
        self.width.valueChanged.connect(lambda _v: self.draw())
        self.cumulative.toggled.connect(lambda _v: self.draw())
        self.show_stats.toggled.connect(lambda _v: self.draw())
        self.copy.clicked.connect(self.copy_statistics)
        self.save.clicked.connect(lambda: self.save_histogram())
        if not is_raster and not fields:
            for w in (self.field, self.mode, self.count, self.width, self.cumulative, self.show_stats, self.copy):
                w.setEnabled(False)

    # ------------------------------------------------------------------ data
    def showEvent(self, e):
        super().showEvent(e)
        if not self._loaded:
            self.refresh()

    def refresh(self):
        self._loaded = True
        self._load_values()

    def _read(self, key):
        if key not in self._cache:
            if self.is_raster:
                d = self.layer.data
                v = np.ma.compressed(d).astype(float)
                nulls = int(d.size - v.size)
            else:
                import pandas as pd
                v = pd.to_numeric(self.layer.gdf[key], errors="coerce").to_numpy(dtype=float)
                nulls = 0
            st = ustats.describe(v, nulls)
            v = np.sort(v[np.isfinite(v)])
            self._cache[key] = (v, st)
        return self._cache[key]

    def _load_values(self):
        if not self._loaded:
            return
        if not self.is_raster and self.field.count() == 0:
            self._empty(tr("The layer has no numeric attributes."))
            return
        key = None if self.is_raster else self.field.currentData()
        self.values, self.stats = self._read(key)
        n = self.values.size
        span = (self.values[-1] - self.values[0]) if n else 0.0
        for w in (self.count, self.width):
            w.blockSignals(True)
        self.count.setValue(ustats.sturges_bins(n))
        self.width.setValue(ustats.nice_width(span, self.count.value()))
        for w in (self.count, self.width):
            w.blockSignals(False)
        self.draw()

    def _mode_changed(self, _i=None):
        by_width = self.mode.currentData() == "width"
        if by_width and self.values.size:                      # start from the current number of intervals
            self.width.blockSignals(True)
            self.width.setValue(ustats.nice_width(self.values[-1] - self.values[0], self.count.value()))
            self.width.blockSignals(False)
        self.count.setVisible(not by_width)
        self.width.setVisible(by_width)
        self.draw()

    def edges(self) -> np.ndarray:
        v = self.values
        if self.mode.currentData() == "width":
            return ustats.histogram_edges(v[0], v[-1], width=self.width.value())
        return ustats.histogram_edges(v[0], v[-1], count=self.count.value())

    # ------------------------------------------------------------------ drawing
    def _empty(self, text: str):
        self.hist = None
        self.save.setEnabled(False)
        self.chart.fig.clear()
        ax = self.chart.fig.add_subplot(111)
        ax.set_axis_off()
        ax.text(0.5, 0.5, text, ha="center", va="center", transform=ax.transAxes, color=theme.MUTED)
        self.chart.draw_idle()

    @staticmethod
    def _palette(print_style: bool = False) -> dict:
        """Chart colours: the current theme on screen; light colours on white for a saved PNG (reports)."""
        if print_style:
            L = theme.LIGHT
            return dict(bar=L["ACCENT"], bg="#ffffff", curve=L["SERIES"][1], muted=L["MUTED"], text=L["TEXT"],
                        border=L["BORDER"])
        return dict(bar=theme.ACCENT, bg=theme.PANEL, curve=theme.WARN, muted=theme.MUTED, text=theme.TEXT,
                    border=theme.BORDER)

    def draw(self):
        if not self._loaded:
            return
        self.message.setVisible(False)
        self.save.setEnabled(False)
        if self.values.size == 0:
            self._empty(tr("No valid values."))
            return
        try:
            edges = self.edges()
        except ValueError as exc:
            self.message.setText(str(exc))
            self.message.setVisible(True)
            return
        counts, _ = np.histogram(self.values, edges)
        self.hist = (counts, edges)
        self.chart.fig.clear()
        self.stats_artist = self._plot(self.chart.fig, self._palette())
        self.chart.draw_idle()
        self.save.setEnabled(True)

    def _plot(self, fig, c: dict):
        """Histogram, cumulative curve and statistics box into ``fig``; returns the statistics box (or None)."""
        counts, edges = self.hist
        ax = fig.add_subplot(111)
        ax.bar(edges[:-1], counts, width=np.diff(edges), align="edge", color=c["bar"], alpha=0.85,
               edgecolor=c["bg"] if len(counts) <= 150 else c["bar"], linewidth=0.5, zorder=2)
        ax.set_xlim(edges[0], edges[-1])
        ax.set_ylim(0, max(1, counts.max()) * 1.05)
        ax.set_xlabel(self.layer.name if self.is_raster else self.field.currentText())
        ax.set_ylabel(tr("Frequency"))
        top = ax
        if self.cumulative.isChecked():
            x, F = ustats.ecdf(self.values)
            ax2 = ax.twinx()
            ax2.step(np.r_[edges[0], x], np.r_[0.0, F], where="post", color=c["curve"], linewidth=1.6, zorder=3)
            ax2.set_ylim(0, 1.05)
            ax2.set_ylabel(tr("Cumulative probability"))
            ax2.grid(False)
            ax2.spines["right"].set_visible(True)
            top = ax2
        return self._stats_box(top, c) if self.show_stats.isChecked() else None

    def _stats_box(self, ax, c: dict):
        from matplotlib.offsetbox import AnchoredOffsetbox, HPacker, TextArea
        rows = self.stat_rows()
        labels = TextArea("\n".join(r[0] for r in rows), textprops=dict(color=c["muted"], size=8))
        values = TextArea("\n".join(r[1] for r in rows), textprops=dict(color=c["text"], size=8,
                                                                        multialignment="right"))
        box = AnchoredOffsetbox(loc="upper right", child=HPacker(children=[labels, values], align="top", pad=0,
                                                                 sep=10),
                                pad=0.4, borderpad=0.5, frameon=True)
        box.patch.set(facecolor=c["bg"], edgecolor=c["border"], alpha=0.9)
        box.set_zorder(10)
        ax.add_artist(box)
        return box

    def default_png_name(self) -> str:
        name = self.layer.name if self.is_raster else f"{self.layer.name}_{self.field.currentText()}"
        return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in name) + "_histogram.png"

    def save_histogram(self, path: Optional[str] = None) -> Optional[str]:
        """Save the histogram as it is set up (intervals, cumulative curve, statistics box) — PNG, 300 dpi, white
        background and dark text for reports."""
        if not self._loaded:
            self.refresh()
        if self.hist is None:
            return None
        if path is None:
            path, _ = QFileDialog.getSaveFileName(self, tr("Save histogram"),
                                                  os.path.join(self.start_dir, self.default_png_name()),
                                                  "PNG (*.png)")
            if not path:
                return None
        if not path.lower().endswith(".png"):
            path += ".png"
        import matplotlib as mpl
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure
        L = theme.LIGHT
        rc = {"figure.facecolor": "#ffffff", "axes.facecolor": "#ffffff", "savefig.facecolor": "#ffffff",
              "axes.edgecolor": L["BORDER"], "axes.labelcolor": L["TEXT"], "text.color": L["TEXT"],
              "xtick.color": L["TEXT"], "ytick.color": L["TEXT"], "grid.color": L["GRID"]}
        try:
            with mpl.rc_context(rc):
                fig = Figure(figsize=(7.0, 4.2))
                FigureCanvasAgg(fig)
                self._plot(fig, self._palette(print_style=True))
                fig.tight_layout()
                fig.savefig(path, dpi=300, facecolor="#ffffff")
        except OSError as exc:
            QMessageBox.warning(self, tr("Save histogram"), str(exc))
            return None
        self.start_dir = os.path.dirname(path)
        return path

    def stat_rows(self):
        out = []
        for key, label in STAT_ROWS:
            name = tr("NoData") if key == "nulls" and self.is_raster else tr(label)
            out.append((name, fmt_stat(key, self.stats.get(key))))
        return out

    def copy_statistics(self):
        if not self._loaded:
            self.refresh()
        what = self.layer.name if self.is_raster else f"{self.layer.name} — {self.field.currentText()}"
        QApplication.clipboard().setText("\n".join([what] + [f"{a}\t{b}" for a, b in self.stat_rows()]))


def numeric_fields(gdf) -> List[str]:
    return [str(c) for c in gdf.columns if c != gdf.geometry.name and sym.is_numeric(gdf[c].to_numpy())]
