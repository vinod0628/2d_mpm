"""Reusable widgets for dialogs."""
from __future__ import annotations

import os
from typing import Callable, List, Optional

import numpy as np
import pandas as pd
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PyQt5.QtCore import QAbstractTableModel, QModelIndex, QRect, QSize, Qt
from PyQt5.QtGui import QColor, QBrush
from PyQt5.QtWidgets import (QApplication, QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox, QFileDialog, QFrame,
                             QHBoxLayout, QLabel,
                             QLineEdit, QPushButton, QSizePolicy, QTableView, QToolButton, QVBoxLayout, QWidget,
                             QHeaderView, QScrollArea)

from core.layers import LayerStore, RasterLayer, VectorLayer
from gui import theme
from i18n import tr


class LayerCombo(QComboBox):
    """Combo of layers from the store, filtered by ``kind`` ('raster', 'vector', 'point', 'line', 'polygon')."""

    def __init__(self, store: LayerStore, kind: str = "raster", allow_none: bool = False,
                 none_text: Optional[str] = None, predicate: Optional[Callable] = None, parent=None):
        super().__init__(parent)
        none_text = tr("— none —") if none_text is None else none_text
        self.store, self.kind, self.allow_none, self.none_text, self.predicate = store, kind, allow_none, none_text, predicate
        self.refresh()

    def _layers(self):
        if self.kind == "raster":
            ls = self.store.rasters()
        elif self.kind == "vector":
            ls = self.store.vectors()
        else:
            ls = self.store.vectors(self.kind)
        return [l for l in ls if self.predicate is None or self.predicate(l)]

    def refresh(self):
        cur = self.layer()
        self.clear()
        if self.allow_none:
            self.addItem(self.none_text, None)
        from gui.toc import layer_icon
        for l in self._layers():
            self.addItem(layer_icon(l), l.name, l.id)
        if cur is not None:
            self.set_layer(cur)
        elif self.store.active is not None:
            self.set_layer(self.store.active)

    def layer(self):
        lid = self.currentData()
        return self.store.by_id(lid) if lid is not None else None

    def set_layer(self, layer):
        if layer is None:
            return
        i = self.findData(layer.id)
        if i >= 0:
            self.setCurrentIndex(i)


class PathEdit(QWidget):
    """Line edit + browse button. mode: 'open', 'save', 'dir', 'open_or_dir'."""

    def __init__(self, mode: str = "save", filter: str = "GeoTIFF (*.tif)", caption: Optional[str] = None,
                 parent=None):
        super().__init__(parent)
        self.mode, self.filter, self.caption = mode, filter, tr("Select file") if caption is None else caption
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.edit = QLineEdit()
        lay.addWidget(self.edit, 1)
        b = QToolButton()
        b.setIcon(theme.icon("mdi6.folder-open-outline"))
        b.setToolTip(tr("Browse…"))
        b.clicked.connect(self.browse)
        lay.addWidget(b)
        if mode == "open_or_dir":
            d = QToolButton()
            d.setIcon(theme.icon("mdi6.folder-outline"))
            d.setToolTip(tr("Choose a folder (e.g. ArcInfo binary grid)"))
            d.clicked.connect(lambda: self._dir())
            lay.addWidget(d)

    def text(self) -> str:
        return self.edit.text().strip()

    def setText(self, t: str):
        self.edit.setText(t)

    def setPlaceholderText(self, t: str):
        self.edit.setPlaceholderText(t)

    def _dir(self):
        p = QFileDialog.getExistingDirectory(self, self.caption, self.text() or "")
        if p:
            self.edit.setText(p)

    def browse(self):
        start = self.text() or ""
        if self.mode == "dir":
            return self._dir()
        if self.mode in ("open", "open_or_dir"):
            p, _ = QFileDialog.getOpenFileName(self, self.caption, start, self.filter)
        else:
            p, _ = QFileDialog.getSaveFileName(self, self.caption, start, self.filter)
        if p:
            self.edit.setText(p)


class DataFrameModel(QAbstractTableModel):
    def __init__(self, df: pd.DataFrame, highlight: Optional[Callable[[pd.Series], Optional[str]]] = None,
                 float_fmt: str = "{:.4f}", parent=None):
        super().__init__(parent)
        self.df = df.reset_index(drop=True)
        self.highlight = highlight
        self.float_fmt = float_fmt

    def rowCount(self, parent=QModelIndex()):
        return len(self.df)

    def columnCount(self, parent=QModelIndex()):
        return self.df.shape[1]

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid():
            return None
        v = self.df.iat[index.row(), index.column()]
        if role == Qt.DisplayRole:
            if isinstance(v, (float, np.floating)):
                if not np.isfinite(v):
                    return "—"
                return self.float_fmt.format(v) if abs(v) < 1e6 else f"{v:,.0f}"
            if isinstance(v, (bool, np.bool_)):
                return "✓" if v else ""
            return str(v)
        if role == Qt.TextAlignmentRole:
            return int(Qt.AlignRight | Qt.AlignVCenter) if isinstance(v, (int, float, np.number)) and not isinstance(v, bool) \
                else int(Qt.AlignLeft | Qt.AlignVCenter)
        if role == Qt.BackgroundRole and self.highlight is not None:
            col = self.highlight(self.df.iloc[index.row()])
            if col:
                return QBrush(QColor(col))
        return None

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role != Qt.DisplayRole:
            return None
        return str(self.df.columns[section]) if orientation == Qt.Horizontal else str(section + 1)


def make_table(df: pd.DataFrame, highlight=None, float_fmt="{:.4f}") -> QTableView:
    tv = QTableView()
    tv.setModel(DataFrameModel(df, highlight, float_fmt))
    tv.setAlternatingRowColors(True)
    tv.verticalHeader().setVisible(False)
    tv.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
    tv.horizontalHeader().setStretchLastSection(True)
    tv.setSelectionBehavior(QTableView.SelectRows)
    tv.setEditTriggers(QTableView.NoEditTriggers)
    return tv


class MplWidget(FigureCanvasQTAgg):
    def __init__(self, width=5, height=4, parent=None):
        self.fig = Figure(figsize=(width, height), facecolor=theme.PANEL, tight_layout=True)
        super().__init__(self.fig)
        self.setParent(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)


def metric_card(value: str, caption: str) -> QFrame:
    f = QFrame()
    f.setObjectName("card")
    lay = QVBoxLayout(f)
    lay.setContentsMargins(14, 10, 14, 10)
    v = QLabel(value)
    v.setObjectName("metric")
    c = QLabel(caption)
    c.setObjectName("metricCaption")
    lay.addWidget(v)
    lay.addWidget(c)
    return f


def raster_value_kind(layer, sample=None):
    """(number of distinct values in a sample, kind) of a raster; kind = "categorical" (class raster, e.g. with a
    class table), "integer" or "real" (floating-point values)."""
    import numpy as np
    data = layer.data
    if sample is None:
        sample = np.ma.compressed(data[:: max(1, data.shape[0] // 400), :: max(1, data.shape[1] // 400)])
    n = int(np.unique(sample).size)
    if layer.categorical:
        return n, "categorical"
    return n, ("integer" if np.dtype(data.dtype).kind in "iub" else "real")


def split_method_combo(current: str = "spatial") -> QComboBox:
    """Train/test split of the deposits (logic.common.training.split_deposits)."""
    cb = QComboBox()
    cb.addItem(tr("spatially stratified (even coverage)"), "spatial")
    cb.addItem(tr("random"), "random")
    cb.setCurrentIndex(0 if current != "random" else 1)
    cb.setToolTip(tr("Spatially stratified: test deposits are drawn one by one from consecutive groups along a "
                     "space-filling curve, so test and training deposits both cover the whole study area."))
    return cb


def thinning_spin(value: float = 0.0) -> QDoubleSpinBox:
    """Minimum distance between training deposits (thinning); 0 = off."""
    sp = QDoubleSpinBox()
    sp.setRange(0, 1e7)
    sp.setDecimals(0)
    sp.setSingleStep(100)
    sp.setSpecialValueText(tr("off"))
    sp.setValue(value)
    sp.setToolTip(tr("Thinning: a deposit is kept only if it is at least this distance (map units) from every "
                     "deposit already kept; e.g. √(unit cell area / π) = 564 m for 1 km²."))
    return sp


def hint(text: str) -> QLabel:
    l = QLabel(text)
    l.setObjectName("hint")
    l.setWordWrap(True)
    return l


FRAME_ALLOWANCE = 48        # title bar + window frame (px) not included in a widget's height
SIDE_ALLOWANCE = 24


def available_geometry(widget=None) -> QRect:
    """Usable area (screen minus taskbar) of the screen showing ``widget`` (else the primary screen)."""
    scr = None
    if widget is not None:
        win = widget.window() or widget
        try:
            scr = win.screen()
        except AttributeError:
            scr = None
        if scr is None:
            scr = QApplication.screenAt(win.frameGeometry().center())
    scr = scr or QApplication.primaryScreen()
    return scr.availableGeometry()


def fit_size(width: int, height: int, widget=None) -> QSize:
    """Requested size limited to the available screen area (height <= 0 is passed through)."""
    avail = available_geometry(widget)
    w = min(int(width), avail.width() - SIDE_ALLOWANCE)
    h = int(height) if height <= 0 else min(int(height), avail.height() - FRAME_ALLOWANCE)
    return QSize(max(w, 200), h)


def fit_to_screen(win, width: int, height: int):
    """Resize a window to the requested size, but never larger than the screen it is on."""
    s = fit_size(width, height, win.parentWidget() or win)
    QWidget.resize(win, s.width(), max(s.height(), 1))


def center_on(win, ref=None):
    """Centre a top-level window on ``ref`` (or the screen), keeping it fully inside the available area."""
    avail = available_geometry(ref or win)
    fg = win.frameGeometry()
    c = (ref.window().frameGeometry().center() if ref is not None else avail.center())
    x = min(max(c.x() - fg.width() // 2, avail.left()), avail.right() - fg.width() + 1)
    y = min(max(c.y() - fg.height() // 2, avail.top()), avail.bottom() - fg.height() + 1)
    win.move(x, y)


class BaseDialog(QDialog):
    """Dialog with a title/hint header and OK/Cancel footer (OK = primary button).

    Size adapts to the display: ``resize()`` is limited to the available screen area, and on first show the dialog
    takes the larger of the requested height and its content height, capped by the screen height; the body scrolls
    when the screen is too small (e.g. 1366 x 768 laptops).
    """

    def __init__(self, title: str, subtitle: str = "", parent=None, ok_text: Optional[str] = None):
        super().__init__(parent)
        self._requested = QSize(0, 0)
        self._fitted = False
        self.setWindowTitle(title)
        self.root = QVBoxLayout(self)
        self.root.setContentsMargins(16, 14, 16, 12)
        self.root.setSpacing(10)
        t = QLabel(title)
        t.setObjectName("title")
        self.root.addWidget(t)
        if subtitle:
            self.root.addWidget(hint(subtitle))
        # body in a scroll area: on small displays (e.g. 1366 x 768) the content scrolls while the header and
        # the OK / Cancel buttons stay visible
        holder = QWidget()
        holder.setObjectName("dialogBody")
        self.body = QVBoxLayout(holder)
        self.body.setContentsMargins(0, 0, 0, 0)
        self.body.setSpacing(8)
        scroll = QScrollArea()
        scroll.setObjectName("dialogScroll")
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setWidget(holder)
        self.root.addWidget(scroll, 1)
        self._holder, self._scroll = holder, scroll
        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        ok = self.buttons.button(QDialogButtonBox.Ok)
        ok.setText(tr("OK") if ok_text is None else ok_text)
        self.buttons.button(QDialogButtonBox.Cancel).setText(tr("Cancel"))
        ok.setProperty("primary", True)
        self.buttons.accepted.connect(self._accept)
        self.buttons.rejected.connect(self.reject)
        self.root.addWidget(self.buttons)

    def resize(self, *args):
        """resize(w, h) / resize(QSize): remembered and limited to the screen (h <= 0 = fit the content)."""
        w, h = (args[0].width(), args[0].height()) if len(args) == 1 else args
        self._requested = QSize(int(w), int(h))
        fit_to_screen(self, w, h if h > 0 else self.sizeHint().height())

    def showEvent(self, ev):
        super().showEvent(ev)
        if not self._fitted:
            self._fitted = True
            self._fit_to_content()

    def _fit_to_content(self):
        """Height = max(requested, content) but never taller than the screen; then re-centre on the parent."""
        avail = available_geometry(self.parentWidget() or self)
        content = self.sizeHint().height() - self._scroll.sizeHint().height() + self._holder.sizeHint().height() + 6
        want_h = max(self._requested.height(), content)
        want_w = max(self._requested.width(), self.sizeHint().width())
        QWidget.resize(self, min(want_w, avail.width() - SIDE_ALLOWANCE), min(want_h, avail.height() - FRAME_ALLOWANCE))
        center_on(self, self.parentWidget())

    def validate(self) -> Optional[str]:
        """Return an error message to block accept()."""
        return None

    def _accept(self):
        err = self.validate()
        if err:
            from PyQt5.QtWidgets import QMessageBox
            QMessageBox.warning(self, self.windowTitle(), err)
            return
        self.accept()
