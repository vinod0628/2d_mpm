"""Measure tool: distance (segment, total, geodetic azimuth) and area (area, perimeter) on the map.

Left-click adds points, the last segment follows the mouse in real time (blitted overlay), double-click or
right-click finishes, Esc / *Clear* starts over. The temporary graphics are removed as soon as the dialog closes
or another tool / command is used. The last active tab is remembered.
"""
from __future__ import annotations

import math

from PyQt5.QtCore import QSettings, Qt, pyqtSignal
from PyQt5.QtWidgets import (QDialog, QFormLayout, QHBoxLayout, QLabel, QPushButton, QTabWidget, QVBoxLayout,
                             QWidget)

from gui import theme
from i18n import tr
from gui.widgets import fit_to_screen, hint
from utils.measure import Measurer
from version import APP_NAME

SETTINGS_KEY = "measure/last_tab"


def _value_label(big: bool = False) -> QLabel:
    l = QLabel("—")
    l.setTextInteractionFlags(Qt.TextSelectableByMouse)
    if big:
        l.setStyleSheet("font-size: 12pt; font-weight: 600;")
    return l


class MeasureDialog(QDialog):
    closed = pyqtSignal()

    def __init__(self, store, canvas, parent=None):
        super().__init__(parent, Qt.Tool)
        self.store, self.canvas = store, canvas
        self.setWindowTitle(tr("Measure"))
        self.setAttribute(Qt.WA_ShowWithoutActivating)
        self.pts, self.cursor_xy, self.finished = [], None, False
        self._measurer, self._measurer_crs = None, object()

        lay = QVBoxLayout(self)
        lay.setContentsMargins(10, 10, 10, 10)
        self.tabs = QTabWidget()
        # distance tab
        d = QWidget()
        f = QFormLayout(d)
        self.d_segment, self.d_total = _value_label(True), _value_label(True)
        self.d_azimuth, self.d_grid, self.d_points = _value_label(True), _value_label(), _value_label()
        self.d_grid.setObjectName("hint")
        f.addRow(tr("Segment"), self.d_segment)
        f.addRow(tr("Total"), self.d_total)
        f.addRow(tr("Azimuth"), self.d_azimuth)
        f.addRow("", self.d_grid)
        f.addRow(tr("Points"), self.d_points)
        self.tabs.addTab(d, theme.icon("mdi6.ruler"), tr("Distance"))
        # area tab
        a = QWidget()
        g = QFormLayout(a)
        self.a_area, self.a_ha, self.a_m2 = _value_label(True), _value_label(), _value_label()
        self.a_perim, self.a_vertices = _value_label(True), _value_label()
        g.addRow(tr("Area"), self.a_area)
        g.addRow("", self.a_ha)
        g.addRow("", self.a_m2)
        g.addRow(tr("Perimeter"), self.a_perim)
        g.addRow(tr("Vertices"), self.a_vertices)
        self.tabs.addTab(a, theme.icon("mdi6.vector-square"), tr("Area"))
        lay.addWidget(self.tabs)
        lay.addWidget(hint(tr("Click: add point · double-click / right-click: finish · Esc: clear")))
        row = QHBoxLayout()
        clear = QPushButton(theme.icon("mdi6.eraser"), tr("Clear"))
        clear.clicked.connect(self.reset)
        close = QPushButton(tr("Close"))
        close.clicked.connect(self.close)
        row.addWidget(clear)
        row.addStretch(1)
        row.addWidget(close)
        lay.addLayout(row)

        last = QSettings("MPM", APP_NAME).value(SETTINGS_KEY, 0)            # default: Distance
        self.tabs.setCurrentIndex(int(last) if str(last).isdigit() and int(last) < self.tabs.count() else 0)
        self.tabs.currentChanged.connect(self._tab_changed)
        canvas.measure_clicked.connect(self._on_click)
        canvas.coords_changed.connect(self._on_move)
        canvas.measure_cancel.connect(self.reset)
        fit_to_screen(self, 340, 0)
        self._show_values()

    # ------------------------------------------------------------------ state
    @property
    def kind(self) -> str:
        return "area" if self.tabs.currentIndex() == 1 else "distance"

    def _active(self) -> bool:
        return self.isVisible() and self.canvas.mode == "measure"

    def _m(self) -> Measurer:
        if self._measurer is None or self._measurer_crs is not self.store.crs:
            self._measurer, self._measurer_crs = Measurer(self.store.crs), self.store.crs
        return self._measurer

    def _tab_changed(self, i):
        QSettings("MPM", APP_NAME).setValue(SETTINGS_KEY, int(i))          # remember the last tab
        self.reset()

    def reset(self):
        self.pts, self.cursor_xy, self.finished = [], None, False
        self.canvas.clear_overlay()
        self._show_values()

    def _on_click(self, x, y, finish):
        if not self._active():
            return
        if finish:
            if self.pts:
                self.finished, self.cursor_xy = True, None
                self._redraw()
            return
        if self.finished:                       # a click after finishing starts a new measurement
            self.pts, self.finished = [], False
        if not self.pts or (x, y) != self.pts[-1]:
            self.pts.append((x, y))
        self.cursor_xy = (x, y)
        self._redraw()

    def _on_move(self, x, y):
        if not self._active() or self.finished or not self.pts:
            return
        self.cursor_xy = (x, y)
        self._redraw()

    def _redraw(self):
        self.canvas.set_overlay(self.pts, None if self.finished else self.cursor_xy, polygon=self.kind == "area",
                                finished=self.finished)
        self._show_values()

    # ------------------------------------------------------------------ values
    def _show_values(self):
        m = self._m()
        live = self.cursor_xy is not None and not self.finished and self.pts
        pts = list(self.pts) + ([self.cursor_xy] if live and self.cursor_xy != self.pts[-1] else [])
        if self.kind == "distance":
            seg = az = grid = None
            if len(pts) >= 2:
                p, q = pts[-2], pts[-1]
                seg, az, grid = m.segment(p, q), m.azimuth(p, q), m.grid_azimuth(p, q)
            self.d_segment.setText(m.fmt_length(seg) if seg is not None else "—")
            self.d_total.setText(m.fmt_length(m.path_length(pts)) if len(pts) >= 2 else "—")
            self.d_azimuth.setText(m.fmt_angle(az) if az is not None else "—")
            self.d_azimuth.setToolTip(tr("Geodetic azimuth from the last point to the current point: degrees clockwise "
                                         "from true north (0 = N, 90 = E, 180 = S, 270 = W)"))
            self.d_grid.setText(tr("grid azimuth {a:.2f}° (map up = 0°)", a=grid) if grid is not None and m.has_units
                                and abs(((grid - az + 180) % 360) - 180) > 0.005 else "")
            self.d_points.setText(str(len(self.pts)) + ("" if self.finished or not self.pts else "  " + tr("(measuring…)")))
        else:
            if len(pts) >= 3:
                area, per = m.area_perimeter(pts)
                parts = m.fmt_area(area)
                self.a_area.setText(parts[0])
                self.a_ha.setText(parts[1] if len(parts) > 1 else "")
                self.a_m2.setText(parts[2] if len(parts) > 2 else "")
                self.a_perim.setText(m.fmt_length(per))
            else:
                for l in (self.a_area, self.a_perim):
                    l.setText("—")
                self.a_ha.setText("")
                self.a_m2.setText("")
            self.a_vertices.setText(str(len(self.pts)) + ("" if self.finished or not self.pts else "  " + tr("(measuring…)")))

    # ------------------------------------------------------------------ closing clears the temporary graphics
    def hideEvent(self, ev):
        self.reset()
        super().hideEvent(ev)

    def closeEvent(self, ev):
        self.reset()
        self.closed.emit()
        super().closeEvent(ev)
