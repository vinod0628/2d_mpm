"""Identify tool dialog: identified layers (tree, top) + attributes of the selected item (table, bottom)."""
from __future__ import annotations

import numpy as np
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (QAbstractItemView, QComboBox, QDialog, QHBoxLayout, QHeaderView, QLabel, QSplitter,
                             QTableWidget, QTableWidgetItem, QTreeWidget, QTreeWidgetItem, QVBoxLayout)

from core.layers import LayerStore, RasterLayer, VectorLayer
from gui import theme
from i18n import N_, tr
from gui.toc import layer_icon
from gui.widgets import fit_to_screen

SCOPES = {"all": N_("All layers"), "visible": N_("Visible layers"), "active": N_("Selected layer (TOC)")}
PICK_TOLERANCE_PX = 6
MAX_FEATURES = 50


def _fmt(v) -> str:
    if v is None:
        return ""
    if isinstance(v, (float, np.floating)):
        return "NaN" if not np.isfinite(v) else f"{v:.6g}"
    return str(v)


class IdentifyDialog(QDialog):
    """Non-modal tool window. Call :meth:`identify` with map coordinates."""

    closed = pyqtSignal()

    def __init__(self, store: LayerStore, canvas, parent=None):
        super().__init__(parent, Qt.Tool)
        self.store, self.canvas = store, canvas
        self.setWindowTitle(tr("Identify"))
        self.setAttribute(Qt.WA_ShowWithoutActivating)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(10, 10, 10, 10)
        lay.setSpacing(6)

        top = QHBoxLayout()
        top.addWidget(QLabel(tr("Layers")))
        self.scope = QComboBox()
        for k, v in SCOPES.items():
            self.scope.addItem(tr(v), k)
        self.scope.setCurrentIndex(1)
        self.scope.currentIndexChanged.connect(lambda _i: self.refresh())
        top.addWidget(self.scope, 1)
        lay.addLayout(top)
        self.where = QLabel(tr("Click on the map to identify features."))
        self.where.setObjectName("hint")
        lay.addWidget(self.where)

        self.split = QSplitter(Qt.Vertical)
        self.split.setChildrenCollapsible(False)
        self.split.setHandleWidth(6)
        self.tree = QTreeWidget()
        self.tree.setColumnCount(2)
        self.tree.setHeaderLabels([tr("Layer / feature"), tr("Value")])
        self.tree.setSelectionMode(QAbstractItemView.SingleSelection)
        self.tree.header().setSectionResizeMode(0, QHeaderView.Stretch)
        self.tree.header().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.tree.currentItemChanged.connect(self._show_attributes)
        self.attrs = QTableWidget(0, 2)
        self.attrs.setHorizontalHeaderLabels([tr("Field"), tr("Value")])
        self.attrs.verticalHeader().setVisible(False)
        self.attrs.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.attrs.horizontalHeader().setStretchLastSection(True)
        self.attrs.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.attrs.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.attrs.setAlternatingRowColors(True)
        self.split.addWidget(self.tree)
        self.split.addWidget(self.attrs)
        self.split.setSizes([260, 260])
        lay.addWidget(self.split, 1)
        self._xy = None
        self._records = {}
        fit_to_screen(self, 420, 560)

    # ------------------------------------------------------------------ public
    def identify(self, x: float, y: float):
        self._xy = (x, y)
        self.canvas.show_marker(x, y)
        self.refresh()

    def refresh(self):
        self.tree.clear()
        self.attrs.setRowCount(0)
        self._records = {}
        if self._xy is None:
            return
        x, y = self._xy
        self.where.setText(f"X {x:,.2f}   Y {y:,.2f}")
        first = None
        n = 0
        for layer in self._layers():
            item = self._identify_layer(layer, x, y)
            if item is not None:
                self.tree.addTopLevelItem(item)
                item.setExpanded(True)
                n += 1
                if first is None:
                    first = item.child(0) if item.childCount() else item
        if first is None:
            empty = QTreeWidgetItem([tr("No identifiable layers at this location"), ""])
            empty.setForeground(0, Qt.gray)
            empty.setFlags(Qt.NoItemFlags)
            self.tree.addTopLevelItem(empty)
        else:
            self.tree.setCurrentItem(first)
        self.where.setText(tr("X {x:,.2f}   Y {y:,.2f}   —   {n} layer(s)", x=x, y=y, n=n))

    # ------------------------------------------------------------------ helpers
    def _layers(self):
        scope = self.scope.currentData()
        if scope == "active":
            return [self.store.active] if self.store.active is not None else []
        if scope == "visible":
            return [l for l in self.store.layers if l.visible]
        return list(self.store.layers)

    def _identify_layer(self, layer, x, y):
        if isinstance(layer, RasterLayer):
            r, c = layer.grid.xy_to_rowcol([x], [y])
            r, c = int(r[0]), int(c[0])
            if not (0 <= r < layer.grid.height and 0 <= c < layer.grid.width):
                return None
            v = layer.data[r, c]
            if v is np.ma.masked:
                return None
            val = v.item() if hasattr(v, "item") else v
            text = _fmt(val)
            if layer.categorical and layer.labels():
                text = f"{text}  ({layer.label_for(val)})"
            item = QTreeWidgetItem([layer.name, text])
            item.setIcon(0, layer_icon(layer))
            self._store_record(item, self._raster_record(layer, val, r, c, x, y))
            return item
        if isinstance(layer, VectorLayer):
            hits = self._vector_hits(layer, x, y)
            if hits is None or len(hits) == 0:
                return None
            item = QTreeWidgetItem([layer.name, tr("{n} features", n=len(hits)) if len(hits) > 1 else tr("1 feature")])
            item.setIcon(0, layer_icon(layer))
            label_col = self._label_field(layer)
            for idx, row in list(hits.iterrows())[:MAX_FEATURES]:
                lab = _fmt(row[label_col]) if label_col else ""
                child = QTreeWidgetItem([f"#{idx}" + (f"  {lab}" if lab else ""), row.geometry.geom_type])
                item.addChild(child)
                self._store_record(child, self._vector_record(layer, idx, row))
            if len(hits) > MAX_FEATURES:
                more = QTreeWidgetItem([tr("… {n} more", n=len(hits) - MAX_FEATURES), ""])
                more.setFlags(Qt.NoItemFlags)
                item.addChild(more)
            first = item.child(0)
            item.setData(0, Qt.UserRole, first.data(0, Qt.UserRole))
            return item
        return None

    def _store_record(self, item, rec):
        key = len(self._records) + 1
        self._records[key] = rec
        item.setData(0, Qt.UserRole, key)

    def _vector_hits(self, layer: VectorLayer, x, y):
        from shapely.geometry import Point
        tol = PICK_TOLERANCE_PX * self.canvas.map_units_per_pixel()
        gdf = layer.gdf
        probe = Point(x, y).buffer(tol) if layer.geom_type != "polygon" else Point(x, y)
        try:
            cand = gdf.iloc[list(gdf.sindex.query(probe))]
        except Exception:  # noqa: BLE001
            cand = gdf
        hits = cand[cand.intersects(probe)]
        if layer.geom_type != "polygon" and len(hits) > 1:      # nearest first
            hits = hits.assign(_d=hits.distance(Point(x, y))).sort_values("_d").drop(columns="_d")
        return hits

    @staticmethod
    def _label_field(layer: VectorLayer):
        cols = [c for c in layer.gdf.columns if c != "geometry"]
        for pref in ("name", "site", "dep_name", "label", "id", "sampleid", "stratname"):
            for c in cols:
                if c.lower() == pref:
                    return c
        return next((c for c in cols if layer.gdf[c].dtype == object), None)

    def _raster_record(self, layer: RasterLayer, val, r, c, x, y):
        rec = [(tr("Layer"), layer.name),
               (tr("Type"), tr("raster, categorical") if layer.categorical else tr("raster, continuous")),
               (tr("Value"), _fmt(val))]
        if layer.categorical and layer.class_table is not None:
            t = layer.class_table
            match = t[t["value"].astype(float) == float(val)]
            if len(match):
                for col, v in match.iloc[0].items():
                    if col != "value":
                        rec.append((tr("class · {col}", col=col), _fmt(v)))
        rec += [(tr("Row, column"), f"{r}, {c}"), (tr("Cell centre X"), _fmt(layer.grid.transform.c + (c + 0.5) * layer.grid.transform.a)),
                (tr("Cell centre Y"), _fmt(layer.grid.transform.f + (r + 0.5) * layer.grid.transform.e)),
                (tr("Clicked X, Y"), f"{x:,.2f}, {y:,.2f}"), (tr("Source"), layer.source or tr("in memory"))]
        return rec

    def _vector_record(self, layer: VectorLayer, idx, row):
        g = row.geometry
        rec = [(tr("Layer"), layer.name), (tr("Feature id"), str(idx)), (tr("Geometry"), g.geom_type)]
        for col, v in row.items():
            if col != "geometry":
                rec.append((str(col), _fmt(v)))
        if layer.geom_type == "point":
            rec.append(("X, Y", f"{g.centroid.x:,.2f}, {g.centroid.y:,.2f}"))
        elif layer.geom_type == "line":
            rec.append((tr("Length"), _fmt(g.length)))
        else:
            rec += [(tr("Area"), _fmt(g.area)), (tr("Perimeter"), _fmt(g.length))]
        return rec

    def _show_attributes(self, cur, _prev):
        rec = self._records.get(cur.data(0, Qt.UserRole)) if cur is not None else None
        self.attrs.setRowCount(0)
        if not rec:
            return
        self.attrs.setRowCount(len(rec))
        for i, (k, v) in enumerate(rec):
            ki = QTableWidgetItem(k)
            if k.startswith(tr("class · {col}", col="")):
                ki.setForeground(Qt.gray)
            self.attrs.setItem(i, 0, ki)
            self.attrs.setItem(i, 1, QTableWidgetItem(v))

    def closeEvent(self, ev):
        self.canvas.clear_marker()
        self.closed.emit()
        super().closeEvent(ev)

    def hideEvent(self, ev):
        self.canvas.clear_marker()
        super().hideEvent(ev)
