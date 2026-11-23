"""Layer properties (General / Statistics / Symbology / Labels) and attribute table dialogs.

Symbology follows ArcMap: a list of renderers on the left (Single symbol, Unique values, Stretched — rasters only,
Categorized) and the settings of the selected one on the right. The renderer selected when OK is pressed is used.
"""
from __future__ import annotations

from typing import Callable, List, Optional, Tuple

import numpy as np
from PyQt5.QtCore import QAbstractTableModel, QModelIndex, QPointF, QRectF, QSize, Qt, pyqtSignal
from PyQt5.QtGui import QBrush, QColor, QFont, QIcon, QLinearGradient, QPainter, QPen, QPixmap, QPolygonF
from PyQt5.QtWidgets import (QAbstractItemView, QCheckBox, QColorDialog, QComboBox, QDialog, QDoubleSpinBox,
                             QFontComboBox, QFormLayout, QHBoxLayout, QHeaderView, QLabel, QLineEdit, QListWidget,
                             QListWidgetItem, QPushButton, QSlider, QSpinBox, QStackedWidget, QTableView, QTableWidget,
                             QTableWidgetItem, QTabWidget, QVBoxLayout, QWidget)
from matplotlib import colors as mcolors

from core import symbology as sym
from core.layers import RasterLayer, VectorLayer, describe_crs, format_length
from gui import theme
from i18n import N_, tr
from gui.widgets import BaseDialog, fit_to_screen, hint, make_table
from utils.raster_tools import percentile_stretch

CONTINUOUS_CMAPS = ["viridis", "magma", "inferno", "plasma", "cividis", "turbo", "Spectral_r", "RdYlBu_r",
                    "RdYlGn", "coolwarm", "YlOrRd", "YlGnBu", "Blues", "Greens", "Reds", "hot", "terrain",
                    "gist_earth", "gray", "bone"]
CATEGORICAL_CMAPS = ["tab20", "tab10", "Set3", "Paired", "Set2", "Set1", "Accent", "Pastel1", "Dark2"]
RAMPS = CATEGORICAL_CMAPS + CONTINUOUS_CMAPS
MARKERS = {N_("circle"): "o", N_("square"): "s", N_("triangle"): "^", N_("diamond"): "D", N_("star"): "*",
           N_("plus"): "P", N_("cross"): "X"}
DEFAULT_FILL = "#ffcc80"                         # used when "no fill" is switched off on a polygon without a fill
RENDERER_NAMES = {"single": N_("Single symbol"), "unique": N_("Unique values"), "stretch": N_("Stretched"),
                  "categorized": N_("Categorized")}
METHOD_NAMES = {"unique": N_("Unique values"), "equal": N_("Equal interval"), "quantile": N_("Quantile"),
                "jenks": N_("Natural breaks (Jenks)")}
GEOM_NAMES = {"point": N_("points"), "line": N_("lines"), "polygon": N_("polygons")}
MAX_ROWS = 20000
DEFAULT_LABEL_FONT = "Arial"                                 # unique values table of a vector layer
SWATCH = QSize(30, 18)


# ================================================================================================ small widgets
class ColorButton(QPushButton):
    picked = pyqtSignal(str)                     # a colour was chosen in the colour dialog

    def __init__(self, color: str, allow_none: bool = False, parent=None):
        super().__init__(parent)
        self.setObjectName("colorButton")
        self.allow_none = allow_none
        self.set_color(color)
        self.clicked.connect(self._pick)

    def set_color(self, color):
        self.color = color
        if color in (None, "none", ""):
            self.setText(tr("no fill"))
            self.setStyleSheet("")
        else:
            hexc = mcolors.to_hex(color)
            self.setText(hexc)
            fg = "#000000" if QColor(hexc).lightness() > 140 else "#ffffff"
            # selector: the style must not cascade to child widgets
            self.setStyleSheet(f"QPushButton#colorButton {{ background:{hexc}; color:{fg}; "
                               f"border:1px solid {theme.BORDER}; }}")

    def _pick(self):
        c = pick_color(self, self.color)
        if c:
            self.set_color(c)
            self.picked.emit(c)


def pick_color(widget, start) -> Optional[str]:
    """Colour dialog parented to the dialog window (never inherits a button's style); None = cancelled."""
    q = QColor(mcolors.to_hex(start)) if start not in (None, "none", "") else QColor("#888888")
    c = QColorDialog.getColor(q, widget.window(), tr("Choose colour"))
    return c.name() if c.isValid() else None


def ramp_icon(name: str, w: int = 110, h: int = 14) -> QIcon:
    pm = QPixmap(w, h)
    pm.fill(Qt.transparent)
    p = QPainter(pm)
    cm = sym.get_cmap(name)
    if getattr(cm, "N", 256) <= 20:                       # qualitative: blocks
        n = min(cm.N, 12)
        for i in range(n):
            p.fillRect(QRectF(i * w / n, 0, w / n + 1, h), QColor(mcolors.to_hex(cm(i))))
    else:
        g = QLinearGradient(0, 0, w, 0)
        for i in range(11):
            g.setColorAt(i / 10, QColor(mcolors.to_hex(cm(i / 10))))
        p.fillRect(0, 0, w, h, QBrush(g))
    p.end()
    return QIcon(pm)


class RampCombo(QComboBox):
    """Colour ramp chooser with gradient previews."""

    def __init__(self, names: List[str], current: Optional[str] = None, parent=None):
        super().__init__(parent)
        self.setIconSize(QSize(110, 14))
        for n in names:
            self.addItem(ramp_icon(n), n, n)
        if current:
            self.set_ramp(current)

    def set_ramp(self, name: str):
        i = self.findData(name)
        if i < 0:
            self.insertItem(0, ramp_icon(name), name, name)
            i = 0
        self.setCurrentIndex(i)

    def ramp(self) -> str:
        return self.currentData() or "viridis"


def _qcolor(c, alpha_none=True) -> QColor:
    if c in (None, "none", ""):
        return QColor(0, 0, 0, 0)
    return QColor(mcolors.to_hex(c))


def _marker_polygon(marker: str, cx: float, cy: float, r: float) -> Optional[QPolygonF]:
    if marker == "s":
        pts = [(-r, -r), (r, -r), (r, r), (-r, r)]
    elif marker == "^":
        pts = [(0, -r), (r, r * 0.8), (-r, r * 0.8)]
    elif marker == "D":
        pts = [(0, -r), (r, 0), (0, r), (-r, 0)]
    elif marker == "*":
        pts = []
        for i in range(10):
            a = np.pi / 2 + i * np.pi / 5
            rr = r if i % 2 == 0 else r * 0.45
            pts.append((rr * np.cos(a), -rr * np.sin(a)))
    elif marker in ("P", "X"):
        t = r * 0.35
        pts = [(-t, -r), (t, -r), (t, -t), (r, -t), (r, t), (t, t), (t, r), (-t, r), (-t, t), (-r, t), (-r, -t), (-t, -t)]
        if marker == "X":
            c, s = np.cos(np.pi / 4), np.sin(np.pi / 4)
            pts = [(x * c - y * s, x * s + y * c) for x, y in pts]
    else:
        return None
    return QPolygonF([QPointF(cx + x, cy + y) for x, y in pts])


_PIXMAPS: dict = {}


def symbol_pixmap(kind: str, st: dict, size: QSize = SWATCH) -> QPixmap:
    """Swatch of a symbol: kind = point / line / polygon / raster (st['color']); cached per look."""
    keys = {"raster": ("color",), "polygon": ("facecolor", "edgecolor", "linewidth"), "line": ("color", "linewidth"),
            "point": ("color", "edgecolor", "marker", "markersize")}.get(kind, ())
    key = (kind, size.width(), size.height()) + tuple(str(st.get(k)) for k in keys)
    pm = _PIXMAPS.get(key)
    if pm is None:
        if len(_PIXMAPS) > 4000:
            _PIXMAPS.clear()
        pm = _PIXMAPS[key] = _draw_symbol(kind, st, size)
    return pm


def _draw_symbol(kind: str, st: dict, size: QSize) -> QPixmap:
    w, h = size.width(), size.height()
    pm = QPixmap(w, h)
    pm.fill(Qt.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.Antialiasing)
    if kind == "raster":
        p.setPen(QPen(QColor(theme.BORDER), 1))
        p.setBrush(_qcolor(st.get("color")))
        p.drawRect(QRectF(2, 2, w - 4, h - 4))
    elif kind == "polygon":
        p.setPen(QPen(_qcolor(st.get("edgecolor", "#b0bec5")), max(1.0, float(st.get("linewidth", 0.8)) * 1.3)))
        p.setBrush(_qcolor(st.get("facecolor", "none")))
        p.drawRect(QRectF(3, 3, w - 6, h - 6))
    elif kind == "line":
        p.setPen(QPen(_qcolor(st.get("color", "#ff8a65")), max(1.0, float(st.get("linewidth", 0.8)) * 1.5)))
        p.drawLine(QPointF(3, h / 2), QPointF(w - 3, h / 2))
    else:
        r = min(h / 2 - 2, 3 + np.sqrt(float(st.get("markersize", 24))) * 0.9)
        p.setPen(QPen(_qcolor(st.get("edgecolor", "#202020")), 1))
        p.setBrush(_qcolor(st.get("color", "#ffd54f")))
        poly = _marker_polygon(st.get("marker", "o"), w / 2, h / 2, r)
        if poly is None:
            p.drawEllipse(QPointF(w / 2, h / 2), r, r)
        else:
            p.drawPolygon(poly)
    p.end()
    return pm


def _table(headers: List[str], stretch: Optional[int] = None) -> QTableWidget:
    t = QTableWidget(0, len(headers))
    t.setHorizontalHeaderLabels(headers)
    t.setIconSize(SWATCH)
    t.verticalHeader().setVisible(False)
    t.verticalHeader().setDefaultSectionSize(24)
    t.setSelectionBehavior(QAbstractItemView.SelectRows)
    t.setEditTriggers(QAbstractItemView.NoEditTriggers)
    hh = t.horizontalHeader()
    hh.setSectionResizeMode(QHeaderView.ResizeToContents)
    if stretch is None:
        hh.setStretchLastSection(True)
    else:
        hh.setSectionResizeMode(stretch, QHeaderView.Stretch)
    return t


def _item(text, editable=False, icon: Optional[QPixmap] = None) -> QTableWidgetItem:
    it = QTableWidgetItem("" if text is None else str(text))
    flags = Qt.ItemIsSelectable | Qt.ItemIsEnabled
    if editable:
        flags |= Qt.ItemIsEditable
    it.setFlags(flags)
    if icon is not None:
        it.setIcon(QIcon(icon))
    return it


class FeatureSymbolModel(QAbstractTableModel):
    """Symbol / number / label of every feature — Qt asks only for the visible rows, so thousands of features
    open instantly (a QTableWidget item and swatch per feature took seconds)."""

    def __init__(self, kind: str, parent=None):
        super().__init__(parent)
        self.kind = kind
        self.styles: List[dict] = []
        self.labels = None

    def set_data(self, styles: List[dict], labels):
        self.beginResetModel()
        self.styles, self.labels = styles, labels
        self.endResetModel()

    def set_labels(self, labels):
        self.labels = labels
        if self.styles:
            self.dataChanged.emit(self.index(0, 2), self.index(len(self.styles) - 1, 2))

    def set_style(self, row: int, st: dict):
        self.styles[row] = st
        self.dataChanged.emit(self.index(row, 0), self.index(row, 0))

    def rowCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else min(len(self.styles), MAX_ROWS)

    def columnCount(self, parent=QModelIndex()):
        return 3

    def data(self, index, role=Qt.DisplayRole):
        r, c = index.row(), index.column()
        if role == Qt.DecorationRole and c == 0:
            return QIcon(symbol_pixmap(self.kind, self.styles[r]))
        if role == Qt.DisplayRole:
            if c == 1:
                return str(r + 1)
            if c == 2 and self.labels is not None:
                v = self.labels[r]
                return "" if v is None else str(v)
        return None

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role == Qt.DisplayRole and orientation == Qt.Horizontal:
            return [tr("Symbol"), "#", tr("Label")][section]
        return None


# ================================================================================================ symbol editor
class SymbolEditor(QWidget):
    """Fill / outline / marker / width of one vector symbol (point, line or polygon)."""
    changed = pyqtSignal()

    def __init__(self, geom_type: str, st: dict, parent=None):
        super().__init__(parent)
        self.geom_type = geom_type
        self._base = dict(st)
        form = QFormLayout(self)
        form.setLabelAlignment(Qt.AlignRight)
        form.setContentsMargins(0, 0, 0, 0)
        if geom_type == "point":
            self.color = ColorButton(st.get("color", "#ffd54f"))
            self.edge = ColorButton(st.get("edgecolor", "#202020"))
            self.marker_size = QDoubleSpinBox()
            self.marker_size.setRange(1, 500)
            self.marker_size.setValue(float(st.get("markersize", 24)))
            self.marker = QComboBox()
            for k, m in MARKERS.items():
                self.marker.addItem(QIcon(symbol_pixmap("point", dict(marker=m, color="#9e9e9e"))), tr(k), k)
            inv = {v: k for k, v in MARKERS.items()}
            self.marker.setCurrentIndex(max(0, self.marker.findData(inv.get(st.get("marker", "o"), "circle"))))
            form.addRow(tr("Marker"), self.marker)
            form.addRow(tr("Fill colour"), self.color)
            form.addRow(tr("Border colour"), self.edge)
            form.addRow(tr("Size"), self.marker_size)
            sigs = (self.color.picked, self.edge.picked, self.marker_size.valueChanged, self.marker.currentIndexChanged)
        elif geom_type == "line":
            self.color = ColorButton(st.get("color", "#ff8a65"))
            self.line_width = QDoubleSpinBox()
            self.line_width.setRange(0.1, 10)
            self.line_width.setSingleStep(0.2)
            self.line_width.setValue(float(st.get("linewidth", 0.8)))
            form.addRow(tr("Colour"), self.color)
            form.addRow(tr("Line width"), self.line_width)
            sigs = (self.color.picked, self.line_width.valueChanged)
        else:
            self.fill = ColorButton(st.get("facecolor", "none"))
            self.nofill = QCheckBox(tr("No fill (outline only)"))
            self.nofill.setChecked(st.get("facecolor", "none") in (None, "none", ""))
            self.edge = ColorButton(st.get("edgecolor", "#b0bec5"))
            self.line_width = QDoubleSpinBox()
            self.line_width.setRange(0.0, 10)
            self.line_width.setSingleStep(0.2)
            self.line_width.setValue(float(st.get("linewidth", 0.8)))
            # choosing a fill colour switches "no fill" off; unticking "no fill" without a colour picks one
            self.fill.picked.connect(lambda _c: self.nofill.setChecked(False))
            self.nofill.toggled.connect(self._nofill_toggled)
            form.addRow(tr("Fill colour"), self.fill)
            form.addRow("", self.nofill)
            form.addRow(tr("Border colour"), self.edge)
            form.addRow(tr("Border width"), self.line_width)
            sigs = (self.fill.picked, self.nofill.toggled, self.edge.picked, self.line_width.valueChanged)
        self.preview = QLabel()
        form.addRow(tr("Preview"), self.preview)
        for s in sigs:
            s.connect(lambda *_: self._changed())
        self._changed()

    def _nofill_toggled(self, checked: bool):
        if not checked and self.fill.color in (None, "none", ""):
            self.fill.set_color(DEFAULT_FILL)

    def _changed(self):
        self.preview.setPixmap(symbol_pixmap(self.geom_type, self.symbol(), QSize(44, 26)))
        self.changed.emit()

    def symbol(self) -> dict:
        st = dict(self._base)
        if self.geom_type == "point":
            st.update(color=self.color.color, edgecolor=self.edge.color, markersize=self.marker_size.value(),
                      marker=MARKERS[self.marker.currentData()])
        elif self.geom_type == "line":
            st.update(color=self.color.color, linewidth=self.line_width.value())
        else:
            st.update(facecolor="none" if self.nofill.isChecked() else self.fill.color,
                      edgecolor=self.edge.color, linewidth=self.line_width.value())
        return st


class SymbolDialog(BaseDialog):
    """Symbol of one feature / one category (double-click in a symbology table)."""

    def __init__(self, geom_type: str, st: dict, title: str, parent=None):
        super().__init__(tr("Symbol"), title, parent)
        self.editor = SymbolEditor(geom_type, st)
        self.body.addWidget(self.editor)
        self.resize(360, 0)

    def symbol(self) -> dict:
        return self.editor.symbol()


def _symbol_keys(geom_type: str) -> Tuple[str, ...]:
    return {"point": ("color", "edgecolor", "marker", "markersize"), "line": ("color", "linewidth"),
            "polygon": ("facecolor", "edgecolor", "linewidth")}.get(geom_type, ("color",))


# ================================================================================================ categorized page
class CategorizedPage(QWidget):
    """Categories of an attribute (vector field / raster value or class table column), ArcMap 'Categories'."""

    def __init__(self, kind: str, fields: List[Tuple[str, str, bool]], values_for: Callable, base_style: dict,
                 state: dict, parent=None, count_for: Optional[Callable] = None):
        """kind: point / line / polygon / raster. fields: (label, key, numeric). values_for(key) -> (values,
        weights or None, labels or None)."""
        super().__init__(parent)
        self.kind, self.values_for, self.base_style = kind, values_for, dict(base_style)
        self.count_for = count_for                      # rasters: count_for(field, classes) -> cells per class
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignRight)
        self.field = QComboBox()
        for label, key, numeric in fields:
            self.field.addItem(label, (key, numeric))
        self.method = QComboBox()
        for k, v in METHOD_NAMES.items():
            self.method.addItem(tr(v), k)
        self.n = QSpinBox()
        self.n.setRange(2, 32)
        self.n.setValue(int(state.get("n", 5)))
        self.ramp = RampCombo(RAMPS, state.get("cmap") or ("tab20" if kind != "raster" else "viridis"))
        row = QHBoxLayout()
        row.addWidget(self.method, 1)
        row.addWidget(QLabel(tr("Classes")))
        row.addWidget(self.n)
        form.addRow(tr("Field"), self.field)
        form.addRow(tr("Method"), row)
        form.addRow(tr("Colour ramp"), self.ramp)
        lay.addLayout(form)
        btns = QHBoxLayout()
        self.classify_btn = QPushButton(tr("Classify"))
        self.classify_btn.setProperty("primary", True)
        self.classify_btn.clicked.connect(self.classify)
        self.remove_btn = QPushButton(tr("Remove"))
        self.remove_btn.clicked.connect(self._remove)
        btns.addWidget(self.classify_btn)
        btns.addWidget(self.remove_btn)
        btns.addStretch(1)
        lay.addLayout(btns)
        self.msg = hint("")
        self.msg.setVisible(False)
        lay.addWidget(self.msg)
        self.table = _table([tr("Symbol"), tr("Value"), tr("Label"), tr("Count")], stretch=2)
        self.table.setEditTriggers(QAbstractItemView.DoubleClicked | QAbstractItemView.EditKeyPressed)
        self.table.cellDoubleClicked.connect(self._edit_symbol)
        self.table.itemChanged.connect(self._label_edited)
        lay.addWidget(self.table, 1)
        lay.addWidget(hint(tr("Double-click a symbol to change it; double-click a label to rename it.")))

        self.classes: List[dict] = [dict(c) for c in (state.get("classes") or [])]
        key = state.get("field")
        for i in range(self.field.count()):
            if self.field.itemData(i)[0] == key:
                self.field.setCurrentIndex(i)
                break
        mi = self.method.findData(state.get("method") or self._default_method())
        self.method.setCurrentIndex(max(0, mi))
        self.field.currentIndexChanged.connect(self._field_changed)
        self.method.currentIndexChanged.connect(self._method_changed)
        self._field_changed(reclassify=False)
        self._fill_table()

    # -------------------------------------------------------------- state
    def _field_changed(self, *_a, reclassify=True):
        data = self.field.currentData()
        numeric = bool(data and data[1])
        for i in range(1, self.method.count()):                  # ranges only for numbers
            self.method.model().item(i).setEnabled(numeric)
        if not numeric:
            self.method.setCurrentIndex(0)
        elif reclassify:                                         # many different numbers: ranges
            self.method.setCurrentIndex(max(0, self.method.findData(self._default_method())))
        if reclassify:                                           # ranges: sequential ramp, values: qualitative
            ranges = self.method.currentData() != "unique"
            if ranges and self.ramp.ramp() in CATEGORICAL_CMAPS:
                self.ramp.set_ramp("viridis")
            elif not ranges and self.ramp.ramp() in CONTINUOUS_CMAPS:
                self.ramp.set_ramp("tab20")
        self._method_changed()
        if reclassify and self.isVisible():
            self.classify()

    def _default_method(self) -> str:
        data = self.field.currentData()
        if not data or not data[1]:
            return "unique"
        values, _w, _l = self.values_for(data[0], "sample")
        try:
            vals = np.array([np.nan if v is None else v for v in values], dtype=float)
        except (TypeError, ValueError):
            return "unique"
        n = np.unique(vals[np.isfinite(vals)]).size
        return "unique" if n <= 32 else "jenks"

    def _method_changed(self, *_a):
        self.n.setEnabled(self.method.currentData() != "unique")

    def state(self) -> dict:
        key = (self.field.currentData() or (None,))[0]
        return dict(field=key, method=self.method.currentData(), n=self.n.value(), cmap=self.ramp.ramp(),
                    classes=[dict(c) for c in self.classes])

    def classify(self):
        data = self.field.currentData()
        if not data:
            return
        values, weights, labels = self.values_for(data[0], self.method.currentData())
        try:
            self.classes = sym.build_categories(values, self.method.currentData(), self.n.value(), self.ramp.ramp(),
                                                labels=labels, weights=weights)
            self._show_msg("")
        except ValueError as exc:
            self.classes = []
            self._show_msg(tr("{n} — choose Equal interval, Quantile or Natural breaks.", n=str(exc).split(";")[0]))
        if self.kind == "polygon":
            for c in self.classes:
                c["facecolor"] = c.pop("color")
        if self.kind == "raster" and self.classes and self.count_for is not None:   # exact cell counts
            counts = self.count_for(data[0], self.classes)
            for c, n in zip(self.classes, counts):
                c["count"] = int(n)
        self._fill_table()

    def _show_msg(self, text):
        self.msg.setText(text)
        self.msg.setVisible(bool(text))

    def _symbol(self, c: dict) -> dict:
        if self.kind == "raster":
            return dict(color=c.get("color"))
        st = dict(self.base_style)
        for k in ("color", "facecolor", "edgecolor", "marker", "markersize", "linewidth"):
            if k in c:
                st[k] = c[k]
        return st

    def _fill_table(self):
        t = self.table
        t.blockSignals(True)
        t.setRowCount(len(self.classes))
        for r, c in enumerate(self.classes):
            val = (f"{sym.fmt_number(c['lo'])} – {sym.fmt_number(c['hi'])}" if "lo" in c
                   else (sym.fmt_number(c["value"]) if isinstance(c.get("value"), (int, float)) else str(c.get("value"))))
            t.setItem(r, 0, _item("", icon=symbol_pixmap(self.kind, self._symbol(c))))
            t.setItem(r, 1, _item(val))
            t.setItem(r, 2, _item(c.get("label", ""), editable=True))
            t.setItem(r, 3, _item(f"{int(c.get('count', 0)):,}"))
        t.blockSignals(False)

    def _label_edited(self, it: QTableWidgetItem):
        if it.column() == 2 and 0 <= it.row() < len(self.classes):
            self.classes[it.row()]["label"] = it.text()

    def _edit_symbol(self, row: int, col: int):
        if col == 2 or not (0 <= row < len(self.classes)):
            return
        c = self.classes[row]
        if self.kind == "raster":
            col_ = pick_color(self, c.get("color"))
            if col_:
                c["color"] = col_
        else:
            dlg = SymbolDialog(self.kind, self._symbol(c), c.get("label", ""), self)
            if not dlg.exec_():
                return
            st = dlg.symbol()
            for k in _symbol_keys(self.kind):
                c[k] = st[k]
            if self.kind == "polygon":
                c.pop("color", None)
        self.table.item(row, 0).setIcon(QIcon(symbol_pixmap(self.kind, self._symbol(c))))

    def _remove(self):
        rows = sorted({i.row() for i in self.table.selectedIndexes()}, reverse=True)
        for r in rows:
            del self.classes[r]
        self._fill_table()


# ================================================================================================ properties dialog
def _results_dir(store) -> str:
    """The project's results folder (else the project folder, else empty) — first folder of file dialogs."""
    import os
    from core import project
    folder = project.project_dir(store) if store is not None else None
    if folder and os.path.isdir(os.path.join(folder, "results")):
        return os.path.join(folder, "results")
    return folder or ""


class LayerPropertiesDialog(BaseDialog):
    def __init__(self, layer, store, parent=None, tab: Optional[str] = None):
        super().__init__(tr("Layer properties"), layer.name, parent)
        self.layer, self.store = layer, store
        self.is_raster = isinstance(layer, RasterLayer)
        self.kind = "raster" if self.is_raster else layer.geom_type
        self.tabs = QTabWidget()
        self.tabs.addTab(self._general_tab(), theme.icon("mdi6.information-outline"), tr("General"))
        from gui.dialogs.layer_statistics import StatisticsPage, numeric_fields
        self.statistics = StatisticsPage(layer, self.is_raster, None if self.is_raster else numeric_fields(layer.gdf),
                                         start_dir=_results_dir(store))
        self.tabs.addTab(self.statistics, theme.icon("mdi6.chart-histogram"), tr("Statistics"))
        self.tabs.addTab(self._symbology_tab(), theme.icon("mdi6.palette-outline"), tr("Symbology"))
        if not self.is_raster:
            self.tabs.addTab(self._labels_tab(), theme.icon("mdi6.label-outline"), tr("Labels"))
        self.body.addWidget(self.tabs)
        if tab:
            names = ["general", "statistics", "symbology", "labels"]
            if tab in names and names.index(tab) < self.tabs.count():
                self.tabs.setCurrentIndex(names.index(tab))
        self.resize(760, 600)

    # ------------------------------------------------------------------ general
    def _general_tab(self) -> QWidget:
        w = QWidget()
        form = QFormLayout(w)
        form.setLabelAlignment(Qt.AlignRight)
        l = self.layer
        self.name = QLineEdit(l.name)
        form.addRow(tr("Name"), self.name)

        def ro(text):
            e = QLineEdit(text)
            e.setReadOnly(True)
            e.setCursorPosition(0)
            return e
        form.addRow(tr("Source"), ro(l.source or tr("in-memory layer — written to the project data folder on save")))
        if self.is_raster:
            g = l.grid
            form.addRow(tr("Type"), ro(tr("Raster — {dtype}, {cols:,} columns × {rows:,} rows",
                                          dtype=str(l.data.dtype), cols=g.width, rows=g.height)))
        else:
            form.addRow(tr("Type"), ro(tr("Vector — {geom}, {n:,} features", geom=tr(GEOM_NAMES.get(l.geom_type, l.geom_type)), n=len(l.gdf))))
        b = l.bounds()
        geo = l.crs is not None and l.crs.is_geographic
        f = (lambda v: f"{v:.6f}") if geo else (lambda v: f"{v:,.2f}")
        ext = QFormLayout()
        ext.addRow(tr("X min"), ro(f(b[0])))
        ext.addRow(tr("Y min"), ro(f(b[1])))
        ext.addRow(tr("X max"), ro(f(b[2])))
        ext.addRow(tr("Y max"), ro(f(b[3])))
        holder = QWidget()
        holder.setLayout(ext)
        ext.setContentsMargins(0, 0, 0, 0)
        form.addRow(tr("Extent"), holder)
        form.addRow(tr("Coordinate system"), ro(describe_crs(l.crs)[1]))
        if self.is_raster:
            rx, ry = l.grid.res
            cell = format_length(rx, l.crs) if abs(rx - ry) <= 1e-9 * rx else \
                f"{format_length(rx, l.crs)} × {format_length(ry, l.crs)}"
            form.addRow(tr("Cell size"), ro(cell))
            form.addRow(tr("NoData"), ro("—" if l.nodata is None else f"{l.nodata:g}"))
            self.categorical = QCheckBox(tr("Values are classes (categorical)"))
            self.categorical.setToolTip(tr("Class rasters are resampled with nearest neighbour and listed by class "
                                           "in the methods"))
            self.categorical.setChecked(l.categorical)
            form.addRow("", self.categorical)
        return w

    # ------------------------------------------------------------------ symbology
    def _symbology_tab(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        orow = QHBoxLayout()
        self.opacity = QSlider(Qt.Horizontal)
        self.opacity.setRange(5, 100)
        self.opacity.setValue(int(round(self.layer.opacity * 100)))
        olab = QLabel(f"{self.opacity.value()} %")
        olab.setMinimumWidth(44)
        self.opacity.valueChanged.connect(lambda v: olab.setText(f"{v} %"))
        orow.addWidget(QLabel(tr("Opacity")))
        orow.addWidget(self.opacity, 1)
        orow.addWidget(olab)
        lay.addLayout(orow)

        row = QHBoxLayout()
        self.renderers = QListWidget()
        self.renderers.setFixedWidth(150)
        self.pages = QStackedWidget()
        modes = sym.RENDERERS_RASTER if self.is_raster else sym.RENDERERS_VECTOR
        self._modes = list(modes)
        icons = {"single": "mdi6.square-rounded", "unique": "mdi6.palette-swatch-variant",
                 "stretch": "mdi6.gradient-horizontal", "categorized": "mdi6.format-list-bulleted-type"}
        for m in modes:
            QListWidgetItem(theme.icon(icons[m]), tr(RENDERER_NAMES[m]), self.renderers)
            self.pages.addWidget(self._page(m))
        self.renderers.currentRowChanged.connect(self.pages.setCurrentIndex)
        cur = self.layer.renderer if self.layer.renderer in modes else modes[0]
        self.renderers.setCurrentRow(self._modes.index(cur))
        row.addWidget(self.renderers)
        row.addWidget(self.pages, 1)
        lay.addLayout(row, 1)
        return w

    def _page(self, mode: str) -> QWidget:
        if self.is_raster:
            return {"single": self._raster_single, "unique": self._raster_unique, "stretch": self._raster_stretch,
                    "categorized": self._raster_categorized}[mode]()
        return {"single": self._vector_single, "unique": self._vector_unique,
                "categorized": self._vector_categorized}[mode]()

    @staticmethod
    def _page_box(title: str, text: str) -> Tuple[QWidget, QVBoxLayout]:
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(8, 0, 0, 0)
        t = QLabel(title)
        t.setStyleSheet("font-weight:600;")
        lay.addWidget(t)
        lay.addWidget(hint(text))
        return w, lay

    # ---------------------------------------------------------------- vector pages
    def _vector_single(self) -> QWidget:
        w, lay = self._page_box(tr("Single symbol"), tr("Draw all features with the same symbol."))
        self.single = SymbolEditor(self.layer.geom_type, self.layer.style)
        for a in ("fill", "nofill", "edge"):           # shortcuts (never width / size: QWidget methods)
            if hasattr(self.single, a):
                setattr(self, a, getattr(self.single, a))
        lay.addWidget(self.single)
        lay.addStretch(1)
        return w

    def _fields(self, numeric_only=False) -> List[Tuple[str, str, bool]]:
        gdf = self.layer.gdf
        out = []
        for c in gdf.columns:
            if c == gdf.geometry.name:
                continue
            num = sym.is_numeric(gdf[c].to_numpy())
            if numeric_only and not num:
                continue
            out.append((str(c), str(c), num))
        return out

    def _vector_unique(self) -> QWidget:
        w, lay = self._page_box(tr("Unique values"), tr("Draw every feature with its own symbol. Double-click a "
                                                        "feature to change its marker, fill and border."))
        l = self.layer
        u = l.unique or {}
        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignRight)
        self.u_field = QComboBox()
        self.u_field.addItem(tr("(feature number)"), None)
        for label, key, _n in self._fields():
            self.u_field.addItem(label, key)
        if u.get("field") is not None:
            self.u_field.setCurrentIndex(max(0, self.u_field.findData(u.get("field"))))
        else:
            text = [k for _l, k, num in self._fields() if not num]
            if text:
                self.u_field.setCurrentIndex(self.u_field.findData(text[0]))
        self.u_ramp = RampCombo(RAMPS, u.get("cmap", "tab20"))
        ramp_btn = QPushButton(tr("Apply ramp"))
        ramp_btn.clicked.connect(self._u_apply_ramp)
        rr = QHBoxLayout()
        rr.addWidget(self.u_ramp, 1)
        rr.addWidget(ramp_btn)
        form.addRow(tr("Label field"), self.u_field)
        form.addRow(tr("Colour ramp"), rr)
        lay.addLayout(form)
        self.u_over = {str(k): dict(v) for k, v in (u.get("styles") or {}).items()}
        self.u_cmap = u.get("cmap", "tab20")
        self.u_model = FeatureSymbolModel(self.kind, self)
        self.u_table = QTableView()
        self.u_table.setModel(self.u_model)
        self.u_table.setIconSize(SWATCH)
        self.u_table.verticalHeader().setVisible(False)
        self.u_table.verticalHeader().setDefaultSectionSize(24)
        self.u_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.u_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        hh = self.u_table.horizontalHeader()
        hh.resizeSection(0, 60)
        hh.resizeSection(1, 60)
        hh.setStretchLastSection(True)
        self.u_table.doubleClicked.connect(lambda ix: self._u_edit(ix.row(), ix.column()))
        lay.addWidget(self.u_table, 1)
        sel = QPushButton(tr("Change symbol of selected features…"))
        sel.clicked.connect(lambda: self._u_edit(-1, 0))
        lay.addWidget(sel)
        self.u_field.currentIndexChanged.connect(lambda _i: self.u_model.set_labels(self._u_labels()))
        self._u_fill()
        return w

    def _u_labels(self):
        field = self.u_field.currentData()
        return self.layer.gdf[field].to_numpy() if field is not None else None

    def _u_styles(self) -> List[dict]:
        n = len(self.layer.gdf)
        fill = "facecolor" if self.layer.geom_type == "polygon" else "color"
        cols = sym.ramp_colors(self.u_cmap, n)
        base = self.single.symbol() if hasattr(self, "single") else dict(self.layer.style)
        out = []
        for i in range(n):
            st = dict(base)
            st[fill] = cols[i]
            over = self.u_over.get(str(i))
            if over:
                st.update(over)
            out.append(st)
        return out

    def _u_fill(self, *_a):
        self.u_model.set_data(self._u_styles(), self._u_labels())

    def _u_apply_ramp(self):
        self.u_cmap = self.u_ramp.ramp()
        fill = "facecolor" if self.layer.geom_type == "polygon" else "color"
        for st in self.u_over.values():
            st.pop(fill, None)
        self._u_fill()

    def _u_edit(self, row: int, _col: int):
        rows = [row] if row >= 0 else sorted({i.row() for i in self.u_table.selectionModel().selectedRows()})
        if not rows:
            return
        styles = self.u_model.styles
        title = tr("Feature {n}", n=rows[0] + 1) if len(rows) == 1 else tr("{n} features", n=len(rows))
        dlg = SymbolDialog(self.kind, styles[rows[0]], title, self)
        if not dlg.exec_():
            return
        st = dlg.symbol()
        for r in rows:
            self.u_over[str(r)] = {k: st[k] for k in _symbol_keys(self.kind)}
            self.u_model.set_style(r, {**styles[r], **self.u_over[str(r)]})

    def _vector_categorized(self) -> QWidget:
        w, lay = self._page_box(tr("Categorized"), tr("Draw features by the value of an attribute: one symbol per "
                                                      "value, or per range of values (e.g. contour elevations)."))
        gdf = self.layer.gdf

        def values_for(key, method="unique"):
            return gdf[key].to_numpy(), None, None
        fields = self._fields()
        if not fields:
            lay.addWidget(hint(tr("The layer has no attributes.")))
            self.cat = None
            lay.addStretch(1)
            return w
        self.cat = CategorizedPage(self.kind, fields, values_for, self.layer.style, self.layer.categorized or {})
        lay.addWidget(self.cat, 1)
        return w

    # ---------------------------------------------------------------- raster pages
    def _raster_single(self) -> QWidget:
        w, lay = self._page_box(tr("Single symbol"), tr("Draw all valid cells with one colour (NoData stays empty)."))
        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignRight)
        self.r_color = ColorButton(self.layer.single_color)
        form.addRow(tr("Colour"), self.r_color)
        lay.addLayout(form)
        lay.addStretch(1)
        return w

    def _raster_sample(self, n: int = 200000) -> np.ndarray:
        """Up to ``n`` evenly spread valid cells — for class breaks (no full pass over large rasters)."""
        cache = self.__dict__.setdefault("_samples", {})
        if n not in cache:
            v = np.ma.compressed(self.layer.data)
            if v.size > n:                                    # keep the true min / max: ranges cover every cell
                v = np.concatenate([v[:: int(np.ceil(v.size / n))], [v.min(), v.max()]])
            cache[n] = v
        return cache[n]

    def _raster_values(self) -> Optional[np.ndarray]:
        """Distinct cell values (cached on the layer); None when there are more than MAX_UNIQUE."""
        l = self.layer
        if not l.categorical and np.issubdtype(l.data.dtype, np.floating) and \
                np.unique(self._raster_sample(20000)).size > sym.MAX_UNIQUE:
            return None                                       # continuous: do not list millions of values
        vals = l.classes()
        return vals if vals.size <= sym.MAX_UNIQUE else None

    def _raster_unique(self) -> QWidget:
        w, lay = self._page_box(tr("Unique values"), tr("Draw every cell value with its own colour. Choose a colour "
                                                        "ramp, or double-click a row to change its colour."))
        l = self.layer
        u = l.unique or {}
        self.ru_ramp = RampCombo(RAMPS, u.get("cmap") or l.cmap)
        ramp_btn = QPushButton(tr("Apply ramp"))
        ramp_btn.clicked.connect(self._ru_apply_ramp)
        rr = QHBoxLayout()
        rr.addWidget(QLabel(tr("Colour ramp")))
        rr.addWidget(self.ru_ramp, 1)
        rr.addWidget(ramp_btn)
        lay.addLayout(rr)
        self.ru_table = _table([tr("Symbol"), tr("Value"), tr("Label"), tr("Count")], stretch=2)
        self.ru_table.setEditTriggers(QAbstractItemView.DoubleClicked | QAbstractItemView.EditKeyPressed)
        self.ru_table.cellDoubleClicked.connect(self._ru_edit)
        self.ru_dirty = False
        self.ru_cmap = u.get("cmap")
        vals = self._raster_values()
        if vals is None:
            lay.addWidget(hint(tr("More than {n:,} different values — too many for this renderer; use Stretched or "
                                  "Categorized.", n=sym.MAX_UNIQUE)))
            self.ru_values, self.ru_colors, self.ru_labels, self.ru_counts = np.array([]), [], [], []
        else:
            self.ru_values = vals
            self.ru_colors = sym.raster_unique_colors(l, vals)
            labels = sym.raster_unique_labels(l)
            self.ru_labels = [labels.get(float(v), sym.fmt_number(v)) for v in vals]
            self.ru_counts = l.value_counts()[1]                  # cached on the layer
        self._ru_fill()
        self.ru_table.itemChanged.connect(self._ru_label_edited)
        lay.addWidget(self.ru_table, 1)
        return w

    def _ru_fill(self):
        t = self.ru_table
        t.blockSignals(True)
        t.setRowCount(len(self.ru_values))
        for r, v in enumerate(self.ru_values):
            t.setItem(r, 0, _item("", icon=symbol_pixmap("raster", dict(color=self.ru_colors[r]))))
            t.setItem(r, 1, _item(sym.fmt_number(v)))
            t.setItem(r, 2, _item(self.ru_labels[r], editable=True))
            t.setItem(r, 3, _item(f"{int(self.ru_counts[r]):,}"))
        t.blockSignals(False)

    def _ru_apply_ramp(self):
        self.ru_cmap = self.ru_ramp.ramp()
        self.ru_colors = sym.ramp_colors(self.ru_cmap, len(self.ru_values))
        self.ru_dirty = True
        self._ru_fill()

    def _ru_edit(self, row: int, col: int):
        if col == 2:
            return
        c = pick_color(self, self.ru_colors[row])
        if c:
            self.ru_colors[row] = c
            self.ru_dirty = True
            self.ru_table.item(row, 0).setIcon(QIcon(symbol_pixmap("raster", dict(color=c))))

    def _ru_label_edited(self, it: QTableWidgetItem):
        if it.column() == 2:
            self.ru_labels[it.row()] = it.text()
            self.ru_dirty = True

    def _raster_stretch(self) -> QWidget:
        w, lay = self._page_box(tr("Stretched"), tr("Draw continuous values with a colour ramp stretched between a "
                                                    "minimum and a maximum."))
        l = self.layer
        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignRight)
        name = l.cmap or "viridis"
        inverted = name.endswith("_r") and name[:-2] in CONTINUOUS_CMAPS
        self.cmap = RampCombo(CONTINUOUS_CMAPS + CATEGORICAL_CMAPS, name[:-2] if inverted else name)
        self.invert = QCheckBox(tr("Invert"))
        self.invert.setChecked(inverted)
        rr = QHBoxLayout()
        rr.addWidget(self.cmap, 1)
        rr.addWidget(self.invert)
        form.addRow(tr("Colour ramp"), rr)
        self.vmin, self.vmax = QDoubleSpinBox(), QDoubleSpinBox()
        for sb, v in ((self.vmin, l.vmin), (self.vmax, l.vmax)):
            sb.setRange(-1e12, 1e12)
            sb.setDecimals(6)
            sb.setValue(v)
        form.addRow(tr("Minimum"), self.vmin)
        form.addRow(tr("Maximum"), self.vmax)
        b1, b2 = QPushButton(tr("Min–max")), QPushButton("2–98 %")
        b1.clicked.connect(lambda: self._stretch(0, 100))
        b2.clicked.connect(lambda: self._stretch(2, 98))
        br = QHBoxLayout()
        br.addWidget(b1)
        br.addWidget(b2)
        br.addStretch(1)
        form.addRow(tr("Stretch"), br)
        st = l.stats()
        if st.get("valid"):
            form.addRow(tr("Statistics"), QLabel(tr("min {a:.6g}   max {b:.6g}   mean {m:.6g}   std {s:.6g}",
                                                    a=st["min"], b=st["max"], m=st["mean"], s=st["std"])))
        lay.addLayout(form)
        lay.addStretch(1)
        return w

    def _stretch(self, lo, hi):
        a, b = percentile_stretch(self.layer.data, lo, hi)
        self.vmin.setValue(a)
        self.vmax.setValue(b)

    def _raster_categorized(self) -> QWidget:
        w, lay = self._page_box(tr("Categorized"), tr("Draw cells by classes of their value — or of a class table "
                                                      "column — as unique values or ranges."))
        l = self.layer
        fields = [(tr("Value"), sym.VALUE_FIELD, True)]
        if l.class_table is not None:
            for c in l.class_table.columns:
                if c in ("value", "color", "count"):
                    continue
                fields.append((str(c), str(c), sym.is_numeric(l.class_table[c].to_numpy())))

        def values_for(key, method="unique"):
            """Unique values: the distinct cell values; ranges: a sample of cells (no cell counting)."""
            labels = sym.raster_unique_labels(l) if key == sym.VALUE_FIELD else None
            vals = self._raster_values() if method == "unique" else None
            if vals is None:
                vals = self._raster_sample()
            return sym.raster_field_values(l, key, vals), None, labels
        state = dict(self.layer.categorized or {})
        state.setdefault("method", "unique" if l.categorical else "jenks")
        state.setdefault("cmap", "tab20" if l.categorical else "viridis")
        def count_for(key, cats):
            """Cells per class: ranges of the value by one sorted lookup over the cells; otherwise from the
            cached value counts of the layer (class rasters)."""
            if key == sym.VALUE_FIELD and "lo" in cats[0]:
                v = np.ma.compressed(l.data)
                return [int(n) for n in np.bincount(sym.assign_categories(v, cats) + 1, minlength=len(cats) + 1)[1:]]
            vals, counts = l.value_counts()
            idx = sym.assign_categories(sym.raster_field_values(l, key, vals), cats)
            return [int(counts[idx == i].sum()) for i in range(len(cats))]
        self.cat = CategorizedPage("raster", fields, values_for, {}, state, count_for=count_for)
        lay.addWidget(self.cat, 1)
        return w

    # ------------------------------------------------------------------ labels
    def _labels_tab(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        lab = self.layer.labels or {}
        self.lbl_on = QCheckBox(tr("Label features on the map"))
        self.lbl_on.setChecked(bool(lab.get("enabled")))
        lay.addWidget(self.lbl_on)
        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignRight)
        self.lbl_field = QComboBox()
        for label, key, _n in self._fields():
            self.lbl_field.addItem(label, key)
        if lab.get("field") is not None:
            self.lbl_field.setCurrentIndex(max(0, self.lbl_field.findData(lab["field"])))
        else:
            text = [k for _l, k, num in self._fields() if not num]
            if text:
                self.lbl_field.setCurrentIndex(self.lbl_field.findData(text[0]))
        self.lbl_font = QFontComboBox()
        self.lbl_font.setCurrentFont(QFont(lab.get("family") or DEFAULT_LABEL_FONT))
        self.lbl_size = QDoubleSpinBox()
        self.lbl_size.setRange(4, 72)
        self.lbl_size.setValue(float(lab.get("size", 8)))
        self.lbl_color = ColorButton(lab.get("color", "#202020"))
        form.addRow(tr("Label field"), self.lbl_field)
        form.addRow(tr("Font"), self.lbl_font)
        form.addRow(tr("Font size"), self.lbl_size)
        form.addRow(tr("Font colour"), self.lbl_color)
        lay.addLayout(form)
        if not self._fields():
            lay.addWidget(hint(tr("The layer has no attributes.")))
            self.lbl_on.setEnabled(False)
        lay.addStretch(1)
        return w

    # ------------------------------------------------------------------ apply
    def current_renderer(self) -> str:
        return self._modes[max(0, self.renderers.currentRow())]

    def apply(self):
        l = self.layer
        l.name = self.name.text().strip() or l.name
        l.opacity = self.opacity.value() / 100
        mode = self.current_renderer()
        if self.is_raster:
            l.categorical = self.categorical.isChecked()
            l.single_color = self.r_color.color
            l.cmap = self.cmap.ramp() + ("_r" if self.invert.isChecked() else "")
            l.vmin, l.vmax = self.vmin.value(), self.vmax.value()
            if self.ru_dirty:
                base = l.labels()                           # class table labels
                l.unique = dict(cmap=self.ru_cmap,
                                colors={str(float(v)): c for v, c in zip(self.ru_values, self.ru_colors)},
                                labels={str(float(v)): t for v, t in zip(self.ru_values, self.ru_labels)
                                        if t != base.get(float(v), sym.fmt_number(v))})
                l.unique = {k: v for k, v in l.unique.items() if v}
            if mode == "categorized" and not self.cat.classes:
                self.cat.classify()
            l.categorized = self.cat.state()
            l.renderer = mode
        else:
            l.style.update(self.single.symbol())
            l.unique = dict(cmap=self.u_cmap, field=self.u_field.currentData(), styles=self.u_over)
            if self.cat is not None:
                if mode == "categorized" and not self.cat.classes:
                    self.cat.classify()
                l.categorized = self.cat.state()
            elif mode == "categorized":
                mode = "single"
            l.renderer = mode
            l.labels = dict(enabled=self.lbl_on.isChecked(), field=self.lbl_field.currentData(),
                            size=self.lbl_size.value(), color=self.lbl_color.color,
                            family=self.lbl_font.currentFont().family())
        self.store.notify(l)


class AttributeTableDialog(QDialog):
    def __init__(self, layer: VectorLayer, parent=None):
        super().__init__(parent)
        self.setWindowTitle(tr("Attributes — {name}  ({n} features)", name=layer.name, n=len(layer.gdf)))
        lay = QVBoxLayout(self)
        df = layer.gdf.drop(columns="geometry")
        lay.addWidget(make_table(df, float_fmt="{:.6g}"))
        fit_to_screen(self, 900, 520)
