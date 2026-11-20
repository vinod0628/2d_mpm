"""Target delineation dialog: threshold any raster by P-A intersection or C-A fractal breakpoint -> binary 0/1 map.

The graph shows the automatic threshold as a vertical dashed line; click or drag on the graph (cross cursor) to
move it, or type a value. *Confirm* creates the binary layer (1 = target, red; 0 = transparent).
"""
from __future__ import annotations

import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QComboBox, QDoubleSpinBox, QFormLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton,
                             QSpinBox, QWidget)

from evaluation import targets as tg
from gui import theme
from i18n import N_, tr
from gui.widgets import BaseDialog, LayerCombo, MplWidget, hint
from utils.raster_tools import points_to_cells

METHODS = {"pa": N_("P-A intersection (prediction–area plot, needs deposits)"),
           "ca": N_("C-A fractal (concentration–area breakpoint)")}


class TargetDialog(BaseDialog):
    def __init__(self, store, parent=None):
        super().__init__(tr("Target delineation"),
                         tr("Turn any raster (prospectivity map or evidence layer) into a binary target map: "
                            "1 = value ≥ threshold, 0 = below. The threshold comes from the P-A intersection "
                            "(Yousefi & Carranza 2015) or a C-A fractal breakpoint (Cheng et al. 1994) — drag the "
                            "dashed line on the graph to adjust it."), parent, ok_text=tr("Confirm"))
        self.store = store
        self._pa = self._ca = None
        self._values = None
        self._dep = (np.array([], int), np.array([], int))
        self._threshold = None
        self._auto = None
        self._dragging = False
        self._vline = None

        form = QFormLayout()
        self.raster = LayerCombo(store, "raster")
        self.method = QComboBox()
        for k, v in METHODS.items():
            self.method.addItem(tr(v), k)
        self.deposits = LayerCombo(store, "point", allow_none=True, none_text=tr("— none —"))
        dep = next((l for l in store.vectors("point") if "deposit" in l.name.lower()), None)
        if dep is not None:
            self.deposits.set_layer(dep)
        self.segments = QSpinBox()
        self.segments.setRange(2, 5)
        self.segments.setValue(3)
        self.breakpoint = QComboBox()
        seg_row = QHBoxLayout()
        seg_row.addWidget(self.segments)
        seg_row.addWidget(QLabel(tr("threshold at breakpoint")))
        seg_row.addWidget(self.breakpoint, 1)
        self.seg_widget = QWidget()
        self.seg_widget.setLayout(seg_row)
        seg_row.setContentsMargins(0, 0, 0, 0)
        form.addRow(tr("Raster"), self.raster)
        form.addRow(tr("Method"), self.method)
        form.addRow(tr("Deposits"), self.deposits)
        self.seg_label = QLabel(tr("C-A segments"))
        form.addRow(self.seg_label, self.seg_widget)
        thr_row = QHBoxLayout()
        self.thr = QDoubleSpinBox()
        self.thr.setDecimals(6)
        self.thr.setRange(-1e12, 1e12)
        self.thr.setKeyboardTracking(False)
        reset = QPushButton(theme.icon("mdi6.restore"), tr("Automatic"))
        reset.setToolTip(tr("Back to the P-A intersection / selected C-A breakpoint"))
        reset.clicked.connect(self._reset_threshold)
        thr_row.addWidget(self.thr, 1)
        thr_row.addWidget(reset)
        form.addRow(tr("Threshold"), thr_row)
        self.name = QLineEdit()
        form.addRow(tr("Output name"), self.name)
        self.body.addLayout(form)

        self.chart = MplWidget(8, 4.2)
        self.chart.setMinimumHeight(240)
        self.chart.setCursor(Qt.CrossCursor)                      # cross cursor on the graph
        self.body.addWidget(self.chart, 1)
        self.readout = hint("")
        self.stats = QLabel("")
        self.stats.setTextFormat(Qt.RichText)
        self.body.addWidget(self.readout)
        self.body.addWidget(self.stats)

        self.chart.mpl_connect("button_press_event", self._on_press)
        self.chart.mpl_connect("motion_notify_event", self._on_motion)
        self.chart.mpl_connect("button_release_event", self._on_release)
        self.raster.currentIndexChanged.connect(lambda _i: self._recompute())
        self.method.currentIndexChanged.connect(lambda _i: self._recompute())
        self.deposits.currentIndexChanged.connect(lambda _i: self._recompute())
        self.segments.valueChanged.connect(lambda _v: self._recompute())
        self.breakpoint.currentIndexChanged.connect(self._breakpoint_chosen)
        self.thr.valueChanged.connect(lambda v: self._set_threshold(v, from_spin=True))
        self.resize(900, 820)
        self._recompute()

    # ------------------------------------------------------------------ data
    def _layer(self):
        return self.raster.layer()

    def _x_of(self, v):
        """Graph x coordinate of a raster value."""
        if self.method.currentData() == "ca":
            return float(np.log10(v + self._ca.shift)) if v + self._ca.shift > 0 else float(self._ca.log_v[0])
        return float(v)

    def _v_of(self, x):
        if self.method.currentData() == "ca":
            return float(10 ** x - self._ca.shift)
        return float(x)

    def _recompute(self):
        lyr = self._layer()
        method = self.method.currentData()
        pa = method == "pa"
        self.seg_label.setVisible(not pa)
        self.seg_widget.setVisible(not pa)
        self.deposits.setToolTip(tr("required for P-A") if pa else tr("optional: deposits captured by the target"))
        self._pa = self._ca = None
        self._auto = None
        self._dragging = False
        self.readout.setText("")
        self.chart.fig.clear()
        if lyr is None:
            return self._message(tr("Choose a raster."))
        self._values = tg.valid_values(lyr.data)
        if self._values.size < 10:
            return self._message(tr("The raster has too few valid cells."))
        self.name.setText(f"target_{method}")
        dep_layer = self.deposits.layer()
        if dep_layer is not None:
            valid = ~np.ma.getmaskarray(lyr.data)
            self._dep = points_to_cells(dep_layer.gdf, lyr.grid, valid_mask=valid)
        else:
            self._dep = (np.array([], int), np.array([], int))
        try:
            if pa:
                if dep_layer is None:
                    return self._message(tr("P-A needs a deposit (point) layer."))
                self._pa = tg.pa_curve(lyr.data, *self._dep)
                self._auto = self._pa.intersection.get("threshold")
                if self._auto is None:
                    self._auto = float(np.median(self._values))
                    self.readout.setText(tr("No P-A intersection found — the median is used; adjust the threshold."))
            else:
                cell_km2 = lyr.grid.cell_area / 1e6 if not lyr.grid.is_geographic else 1.0
                self._ca = tg.ca_curve(lyr.data, cell_area=cell_km2, n_segments=self.segments.value())
                self.breakpoint.blockSignals(True)
                self.breakpoint.clear()
                for i, b in enumerate(self._ca.breakpoints, start=1):
                    self.breakpoint.addItem(f"{i}: {b:.6g}", b)
                self.breakpoint.setCurrentIndex(self.breakpoint.count() - 1)        # highest = target
                self.breakpoint.blockSignals(False)
                self._auto = self._ca.target_threshold
        except Exception as exc:  # noqa: BLE001
            return self._message(str(exc))
        lo, hi = float(self._values.min()), float(self._values.max())
        self.thr.blockSignals(True)
        self.thr.setRange(lo, hi)
        self.thr.setSingleStep(max((hi - lo) / 200, 1e-9))
        self.thr.blockSignals(False)
        self._draw()
        self._set_threshold(self._auto)

    def _message(self, text):
        self.chart.fig.clear()
        ax = self.chart.fig.add_subplot(111)
        ax.axis("off")
        ax.text(0.5, 0.5, text, ha="center", va="center", color=theme.MUTED, transform=ax.transAxes)
        self.chart.draw_idle()
        self.stats.setText("")
        self._vline = None

    # ------------------------------------------------------------------ plot
    def _draw(self):
        fig = self.chart.fig
        fig.clear()
        ax = fig.add_subplot(111)
        self._axes = [ax]
        if self._pa is not None:
            pa = self._pa
            ax.plot(pa.thresholds, pa.prediction, color=theme.SERIES[0], lw=2, label=tr("prediction rate (deposits)"))
            ax.set_ylim(0, 100)
            ax.set_xlabel(tr("{layer} value (threshold)", layer=self._layer().name))
            ax.set_ylabel(tr("Prediction rate (%)"), color=theme.SERIES[0])
            ax2 = ax.twinx()
            ax2.plot(pa.thresholds, pa.area, color=theme.SERIES[1], lw=2, label=tr("occupied area"))
            ax2.set_ylim(100, 0)
            ax2.set_ylabel(tr("Percentage of study area (%)"), color=theme.SERIES[1])
            ax2.grid(False)
            self._axes.append(ax2)
            if pa.intersection:
                ax.plot([pa.intersection["threshold"]], [pa.intersection["prediction"]], "o", color=theme.WARN, ms=7,
                        zorder=5, label=tr("P-A intersection"))
            ax.set_title(tr("Prediction–area plot ({n} deposits)", n=pa.n_deposits))
            h1, l1 = ax.get_legend_handles_labels()
            h2, l2 = ax2.get_legend_handles_labels()
            ax.legend(h1 + h2, l1 + l2, loc="center left", fontsize=8)
        else:
            ca = self._ca
            ax.scatter(ca.log_v, ca.log_a, s=10, color=theme.MUTED, zorder=2, label="log A(≥v)")
            for k, (i0, i1, slope, icpt) in enumerate(ca.segments):
                xs = np.array([ca.log_v[i0], ca.log_v[i1]])
                ax.plot(xs, slope * xs + icpt, lw=2.2, color=theme.SERIES[k % len(theme.SERIES)],
                        label=tr("segment {k} (slope {slope:.2f})", k=k + 1, slope=slope), zorder=3)
            for b in ca.breakpoints:
                ax.axvline(self._x_of(b), color=theme.BORDER, lw=1, ls=":", zorder=1)
            shift = f" + {ca.shift:.4g}" if ca.shift else ""
            ax.set_xlabel(tr("log10({layer} value{shift})", layer=self._layer().name, shift=shift))
            ax.set_ylabel(tr("log10(area ≥ value, km²)") if not self._layer().grid.is_geographic else tr("log10(cells ≥ value)"))
            ax.set_title(tr("Concentration–area fractal plot ({n} segments)", n=len(ca.segments)))
            ax.legend(loc="lower left", fontsize=8)
        self._vline = ax.axvline(self._x_of(self._auto), color=theme.TEXT, lw=1.6, ls="--", zorder=6)
        self.chart.draw_idle()

    # ------------------------------------------------------------------ threshold
    def _reset_threshold(self):
        if self._auto is not None:
            self._set_threshold(self._auto)

    def _breakpoint_chosen(self, _i):
        b = self.breakpoint.currentData()
        if b is not None:
            self._auto = float(b)
            self._set_threshold(self._auto)

    def _set_threshold(self, value, from_spin: bool = False):
        if value is None or self._values is None or self._vline is None:
            return
        lo, hi = float(self._values.min()), float(self._values.max())
        value = float(min(max(value, lo), hi))
        self._threshold = value
        if not from_spin:
            self.thr.blockSignals(True)
            self.thr.setValue(value)
            self.thr.blockSignals(False)
        x = self._x_of(value)
        self._vline.set_xdata([x, x])
        self.chart.draw_idle()
        self._update_stats()

    def _update_stats(self):
        lyr = self._layer()
        cell_km2 = None if lyr.grid.is_geographic else lyr.grid.cell_area / 1e6
        st = tg.target_stats(lyr.data, self._threshold, cell_km2, *self._dep) if len(self._dep[0]) else \
            tg.target_stats(lyr.data, self._threshold, cell_km2)
        parts = [tr("<b>Threshold</b> {value:.6g}", value=st['threshold']),
                 tr("<b>Target</b> {cells:,} cells = {pct:.2f} % of the area", cells=st['target_cells'],
                    pct=st['target_area_pct'])
                 + (f" ({st['target_area_km2']:,.1f} km²)" if "target_area_km2" in st else "")]
        if "deposits" in st:
            parts.append(tr("<b>Deposits in target</b> {n} / {total} ({pct:.1f} %)", n=st['deposits_in_target'],
                            total=st['deposits'], pct=st['deposits_in_target_pct']))
        auto = "" if self._auto is None or abs(self._threshold - self._auto) > 1e-12 else tr(" (automatic)")
        self.stats.setText("   ·   ".join(parts) + auto)
        self._last_stats = st

    # ------------------------------------------------------------------ mouse (cross cursor, drag the line)
    def _event_x(self, ev):
        if ev.inaxes in getattr(self, "_axes", []) and ev.xdata is not None:
            return ev.xdata
        return None

    def _on_press(self, ev):
        x = self._event_x(ev)
        if ev.button == 1 and x is not None and self._vline is not None:
            self._dragging = True
            self._set_threshold(self._v_of(x))

    def _on_motion(self, ev):
        x = self._event_x(ev)
        if x is None:
            self.readout.setText("")
            return
        v = self._v_of(x)
        self.readout.setText(tr("cursor: value {v:.6g}", v=v)
                             + (f"  (log10 = {x:.4f})" if self.method.currentData() == "ca" else "")
                             + (tr("   — release to set the threshold") if self._dragging else
                                tr("   — click or drag to move the threshold line")))
        if self._dragging:
            self._set_threshold(v)

    def _on_release(self, ev):
        if self._dragging:
            self._dragging = False
            self.readout.setText(tr("threshold set to {v:.6g} — drag again or press Automatic to reset", v=self._threshold))

    # ------------------------------------------------------------------ result
    def validate(self):
        if self._layer() is None or self._threshold is None or self._vline is None:
            return tr("Choose a raster (and deposits for P-A) so that a threshold can be computed.")
        return None

    def output(self) -> dict:
        lyr = self._layer()
        method = self.method.currentData()
        return dict(label=self.name.text().strip() or f"target_{method}", method=method, source=lyr.name,
                    grid=lyr.grid, threshold=self._threshold, automatic=self._auto,
                    binary=tg.binary_target(lyr.data, self._threshold), stats=dict(self._last_stats),
                    figure=self.chart.fig,
                    breakpoints=list(self._ca.breakpoints) if self._ca is not None else [],
                    pa_intersection=dict(self._pa.intersection) if self._pa is not None else {})
