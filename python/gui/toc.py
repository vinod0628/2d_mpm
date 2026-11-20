"""Table of contents (layer list) + layer information panel."""
from __future__ import annotations

import html

import numpy as np
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (QAbstractItemView, QMenu, QSplitter, QTextBrowser, QTreeWidget, QTreeWidgetItem,
                             QVBoxLayout, QWidget)

from core.layers import LayerStore, RasterLayer, VectorLayer
from gui import theme
from i18n import N_, tr

_GEOM_NAMES = (N_("point"), N_("line"), N_("polygon"))     # layer.geom_type values, translated with tr() on display

ICONS = {"raster": "mdi6.grid", "categorical": "mdi6.shape-outline", "point": "mdi6.map-marker",
         "line": "mdi6.vector-polyline", "polygon": "mdi6.vector-square"}
ICON_COLORS = {"raster": "#4fc1c9", "categorical": "#b18cf2", "point": "#e5c07b", "line": "#f0a35e",
               "polygon": "#9fb7c8"}


def layer_icon(layer):
    if isinstance(layer, RasterLayer):
        key = "categorical" if layer.categorical else "raster"
    else:
        key = layer.geom_type
    return theme.icon(ICONS[key], ICON_COLORS[key])


class LayerTree(QTreeWidget):
    """Flat, drag-reorderable, checkable layer list. Top item = drawn on top."""

    zoom_to_layer = pyqtSignal(object)
    properties_requested = pyqtSignal(object)
    attributes_requested = pyqtSignal(object)
    save_requested = pyqtSignal(object)
    export_legend_requested = pyqtSignal(object)

    def __init__(self, store: LayerStore, parent=None):
        super().__init__(parent)
        self.store = store
        self.setHeaderHidden(True)
        self.setRootIsDecorated(False)
        self.setIndentation(4)
        self.setDragDropMode(QAbstractItemView.InternalMove)
        self.setSelectionMode(QAbstractItemView.SingleSelection)
        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self._menu)
        self.itemChanged.connect(self._item_changed)
        self.currentItemChanged.connect(self._current_changed)
        self.itemDoubleClicked.connect(lambda it, _c: self.properties_requested.emit(self._layer(it)))
        self._updating = False
        store.changed.connect(self.rebuild)
        store.layer_changed.connect(self._refresh_item)
        store.active_changed.connect(self._select)
        self.rebuild()

    def _layer(self, item):
        return self.store.by_id(item.data(0, Qt.UserRole)) if item is not None else None

    def rebuild(self):
        self._updating = True
        self.clear()
        for layer in self.store.layers:
            it = QTreeWidgetItem([layer.name])
            it.setData(0, Qt.UserRole, layer.id)
            it.setFlags(Qt.ItemIsSelectable | Qt.ItemIsEnabled | Qt.ItemIsUserCheckable | Qt.ItemIsDragEnabled)
            it.setCheckState(0, Qt.Checked if layer.visible else Qt.Unchecked)
            it.setIcon(0, layer_icon(layer))
            it.setToolTip(0, f"{layer.name}\n{layer.source or tr('(in memory — not saved)')}"
                             + (f"\n{layer.description}" if layer.description else ""))
            if not layer.source:
                f = it.font(0)
                f.setItalic(True)
                it.setFont(0, f)
            self.addTopLevelItem(it)
        self._updating = False
        self._select(self.store.active)

    def _refresh_item(self, layer):
        for i in range(self.topLevelItemCount()):
            it = self.topLevelItem(i)
            if it.data(0, Qt.UserRole) == layer.id:
                self._updating = True
                it.setText(0, layer.name)
                it.setCheckState(0, Qt.Checked if layer.visible else Qt.Unchecked)
                it.setIcon(0, layer_icon(layer))
                self._updating = False

    def _select(self, layer):
        if layer is None:
            return
        for i in range(self.topLevelItemCount()):
            it = self.topLevelItem(i)
            if it.data(0, Qt.UserRole) == layer.id and self.currentItem() is not it:
                self._updating = True
                self.setCurrentItem(it)
                self._updating = False

    def _item_changed(self, item, _col):
        if self._updating:
            return
        layer = self._layer(item)
        if layer is not None:
            self.store.set_visible(layer, item.checkState(0) == Qt.Checked)

    def _current_changed(self, cur, _prev):
        if not self._updating and cur is not None:
            self.store.set_active(self._layer(cur))

    def dropEvent(self, event):
        super().dropEvent(event)
        ids = [self.topLevelItem(i).data(0, Qt.UserRole) for i in range(self.topLevelItemCount())]
        self.store.reorder(ids)

    def _menu(self, pos):
        layer = self._layer(self.itemAt(pos))
        if layer is None:
            return
        m = QMenu(self)
        m.addAction(theme.icon("mdi6.magnify-scan"), tr("Zoom to layer"), lambda: self.zoom_to_layer.emit(layer))
        m.addAction(theme.icon("mdi6.palette-outline"), tr("Properties…"), lambda: self.properties_requested.emit(layer))
        if isinstance(layer, VectorLayer):
            m.addAction(theme.icon("mdi6.table"), tr("Attribute table…"), lambda: self.attributes_requested.emit(layer))
        m.addSeparator()
        m.addAction(theme.icon("mdi6.arrow-up"), tr("Move up"), lambda: self.store.move(layer, -1))
        m.addAction(theme.icon("mdi6.arrow-down"), tr("Move down"), lambda: self.store.move(layer, 1))
        m.addSeparator()
        m.addAction(theme.icon("mdi6.content-save-outline"), tr("Save as…"), lambda: self.save_requested.emit(layer))
        m.addAction(theme.icon("mdi6.image-outline"), tr("Export legend to PNG…"),
                    lambda: self.export_legend_requested.emit(layer))
        m.addAction(theme.icon("mdi6.delete-outline", theme.BAD), tr("Remove"), lambda: self.store.remove(layer))
        m.exec_(self.viewport().mapToGlobal(pos))


class LayerInfo(QTextBrowser):
    """HTML summary of the active layer."""

    def __init__(self, store: LayerStore, parent=None):
        super().__init__(parent)
        self.store = store
        self.setOpenExternalLinks(False)
        store.active_changed.connect(self.show_layer)
        store.layer_changed.connect(lambda l: self.show_layer(l) if l is store.active else None)
        store.data_loaded.connect(lambda l: self.show_layer(l) if l is store.active else None)
        self.show_layer(None)

    def show_layer(self, layer):
        if layer is None:
            self.setHtml(f"<p style='color:{theme.MUTED}'>{tr('Select a layer to see its details.')}</p>")
            return
        rows = []
        if isinstance(layer, RasterLayer):
            g = layer.grid
            rows += [(tr("Type"), tr("Raster — categorical") if layer.categorical else tr("Raster — continuous")),
                     (tr("Size"), tr("{w} × {h} cells", w=g.width, h=g.height)),
                     (tr("Cell size"), f"{g.res[0]:.6g} × {g.res[1]:.6g}"),
                     (tr("CRS"), _crs_name(g.crs))]
            if not layer.loaded:                       # statistics once the cells are read (background)
                self.store.request_data(layer)
                rows.append((tr("Statistics"), f"<span style='color:{theme.MUTED}'>{tr('reading…')}</span>"))
                st = {}
            else:
                st = layer.stats()
                rows += [(tr("dtype"), str(layer.data.dtype)), (tr("Valid cells"), f"{st.get('valid', 0):,}")]
            if st.get("valid"):
                rows += [(tr("Min / Max"), f"{st['min']:.6g} / {st['max']:.6g}"),
                         (tr("Mean ± std"), f"{st['mean']:.6g} ± {st['std']:.4g}")]
            if layer.categorical and layer.loaded:
                cls = layer.classes()
                labels = layer.labels()
                txt = ", ".join(f"{c:g}: {html.escape(labels.get(float(c), ''))}" if labels else f"{c:g}"
                                for c in cls[:20])
                rows.append((tr("Classes"), f"{cls.size} — {txt}{' …' if cls.size > 20 else ''}"))
        else:
            rows += [(tr("Type"), tr("Vector — {geom}", geom=tr(layer.geom_type))),
                     (tr("Features"), f"{len(layer.gdf):,}"),
                     (tr("CRS"), _crs_name(layer.crs)),
                     (tr("Fields"), ", ".join(html.escape(str(c)) for c in layer.gdf.columns if c != "geometry")[:400])]
        rows.append((tr("Source"), html.escape(layer.source or tr("in memory (not saved)"))))
        if layer.description:
            rows.append((tr("Notes"), html.escape(layer.description)))
        body = "".join(f"<tr><td style='color:{theme.MUTED};padding:2px 8px 2px 0;vertical-align:top'>{k}</td>"
                       f"<td style='padding:2px 0'>{v}</td></tr>" for k, v in rows)
        self.setHtml(f"<div style='font-weight:600;font-size:10pt;margin-bottom:4px'>{html.escape(layer.name)}</div>"
                     f"<table cellspacing=0>{body}</table>")


def _crs_name(crs) -> str:
    if crs is None:
        return f"<span style='color:{theme.WARN}'>{tr('none')}</span>"
    from core.layers import describe_crs
    return html.escape(describe_crs(crs)[1])


class TocPanel(QWidget):
    def __init__(self, store: LayerStore, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(6, 6, 6, 6)
        split = QSplitter(Qt.Vertical)
        self.tree = LayerTree(store)
        self.info = LayerInfo(store)
        split.addWidget(self.tree)
        split.addWidget(self.info)
        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 2)
        split.setHandleWidth(1)
        lay.addWidget(split)
