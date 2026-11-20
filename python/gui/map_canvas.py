"""Map view: matplotlib canvas with GIS-style navigation (wheel zoom, pan, box zoom, identify)."""
from __future__ import annotations

from functools import lru_cache
from typing import Optional

import numpy as np
import matplotlib as mpl
from matplotlib import cm, colors as mcolors
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib import transforms as mtransforms
from matplotlib.collections import LineCollection, PathCollection
from matplotlib.path import Path
from matplotlib.figure import Figure
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib.transforms import Bbox
from PyQt5.QtCore import QPoint, QRect, QSize, Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QColor, QImage, QPainter, QPalette, QPixmap
from matplotlib.backend_bases import MouseButton
from PyQt5.QtWidgets import QFrame, QHBoxLayout, QLabel, QRubberBand, QSizePolicy, QVBoxLayout, QWidget

from core import symbology as sym
from core.layers import LayerStore, RasterLayer, VectorLayer
from gui import theme
from i18n import tr

MAX_DISPLAY = 2400          # raster display decimation threshold (pixels per side)
MAX_LABELS_DRAWN = 600     # labels on the map at once (all layers; overlapping labels are skipped)


@lru_cache(maxsize=64)
def _mpl_font(family):
    """The font family if matplotlib knows it (Qt-only names such as 'MS Shell Dlg 2' -> None: default font)."""
    if not family:
        return None
    from matplotlib import font_manager
    names = {f.name.lower() for f in font_manager.fontManager.ttflist}
    return family if family.lower() in names else None


def _rgba(c, alpha: float = 1.0):
    """RGBA with the layer opacity applied per colour — never as a collection alpha, which would turn the
    transparent 'no fill' colour into opaque black."""
    if c in (None, "none", ""):
        return (0.0, 0.0, 0.0, 0.0)
    r, g, b, a = mcolors.to_rgba(c)
    return (r, g, b, a * alpha)


def _vector_handle(geom_type: str, st: dict, label: str):
    """Legend handle of a vector symbol."""
    if geom_type == "point":
        return Line2D([], [], marker=st.get("marker", "o"), linestyle="", markersize=8, label=label,
                      markerfacecolor=st.get("color"), markeredgecolor=st.get("edgecolor", "#202020"))
    if geom_type == "line":
        return Line2D([], [], color=st.get("color"), linewidth=max(1.5, st.get("linewidth", 0.8)), label=label)
    face = st.get("facecolor")
    return Patch(facecolor=face if face not in (None, "none", "") else "none", edgecolor=st.get("edgecolor"),
                 label=label)


def _polygon_path(poly) -> Path:
    """Matplotlib path of a polygon with its holes."""
    verts, codes = [], []
    for ring in [poly.exterior, *poly.interiors]:
        xy = np.asarray(ring.coords)[:, :2]
        if len(xy) < 3:
            continue
        verts.append(xy)
        c = np.full(len(xy), Path.LINETO, np.uint8)
        c[0], c[-1] = Path.MOVETO, Path.CLOSEPOLY
        codes.append(c)
    if not verts:
        return Path(np.zeros((1, 2)))
    return Path(np.concatenate(verts), np.concatenate(codes))


def get_cmap(name: str):
    try:
        return mpl.colormaps[name]
    except KeyError:
        return mpl.colormaps["viridis"]


def category_colors(layer: RasterLayer, classes: np.ndarray):
    """Colours of the unique values renderer: user colours, class table 'color' column, else the colour map."""
    return sym.raster_unique_colors(layer, classes)


def legend_spec(layer, limit: Optional[int] = None) -> dict:
    """What the legend of a layer shows, following its renderer (map legend and exported legend).

    ``{"kind": "colorbar", "vmin", "vmax", "cmap"}`` for stretched rasters, else ``{"kind": "items", "handles"}``
    (``limit`` entries at most, then a "… n more" line)."""
    if isinstance(layer, RasterLayer) and (layer.renderer == "stretch" or (
            layer.renderer == "unique" and layer.classes().size > sym.MAX_UNIQUE)):
        return dict(kind="colorbar", vmin=layer.vmin, vmax=layer.vmax, cmap=layer.cmap)
    handles = []
    if isinstance(layer, RasterLayer):
        if layer.renderer == "single":
            entries = [(layer.single_color, layer.name)]
        elif layer.renderer == "categorized" and (layer.categorized or {}).get("classes"):
            entries = [(c.get("color", "#888888"), c.get("label", "")) for c in layer.categorized["classes"]]
        else:
            classes = layer.classes()
            labels = sym.raster_unique_labels(layer)
            entries = [(col, labels.get(float(c), f"{c:g}")) for c, col in zip(classes, category_colors(layer, classes))]
        make = lambda e: Patch(facecolor=e[0], edgecolor="none", label=e[1])          # noqa: E731
    else:
        cats = (layer.categorized or {}).get("classes") or []
        if layer.renderer == "categorized" and cats:
            entries = [(sym.category_style(layer, c), c.get("label", "")) for c in cats]
        elif layer.renderer == "unique":
            entries = [(st, tr("{n} features, each with its own symbol", n=len(layer.gdf)))
                       for st in sym.feature_styles(layer)[:1]]
        else:
            entries = [(layer.style, tr("{n} features", n=len(layer.gdf)))]
        make = lambda e: _vector_handle(layer.geom_type, e[0], e[1])                   # noqa: E731
    shown = entries if limit is None else entries[:limit]
    handles = [make(e) for e in shown]
    if len(entries) > len(shown):
        handles.append(Patch(facecolor="none", edgecolor="none", label=tr("… {n} more", n=len(entries) - len(shown))))
    return dict(kind="items", handles=handles)


def export_legend(layer, path: str, dpi: int = 300, fontsize: float = 10.0) -> str:
    """Legend of ``layer`` as an image for documents: all entries, black text of ``fontsize`` pt on white, saved at
    ``dpi`` with the resolution written into the file — the text is ``fontsize`` pt in Word at 100 %. The image is
    cropped to the legend, so its width follows the longest text."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    family = _mpl_font("Arial")
    font = dict(family=family) if family else {}
    spec = legend_spec(layer)
    n = len(spec.get("handles", []))
    fig = Figure(figsize=(8, max(4.0, 0.3 * (n + 3))), facecolor="white")   # large enough; cropped on save
    FigureCanvasAgg(fig)
    if spec["kind"] == "colorbar":
        h = 2.0                                                      # inches: colour bar 2" high, 0.18" wide
        cax = fig.add_axes([0.1, 0.1, 0.18 / 8, h / fig.get_figheight()])
        sm = cm.ScalarMappable(norm=mcolors.Normalize(spec["vmin"], spec["vmax"]), cmap=get_cmap(spec["cmap"]))
        cb = fig.colorbar(sm, cax=cax)
        cb.outline.set_edgecolor("#000000")
        cb.outline.set_linewidth(0.5)
        cax.tick_params(colors="#000000", labelsize=fontsize, length=3, width=0.5)
        for t in cax.get_yticklabels():
            t.set_fontfamily(font.get("family", t.get_fontfamily()))
        cax.set_title(layer.name, loc="left", fontsize=fontsize, fontweight="bold", color="#000000", pad=8, **font)
    else:
        leg = fig.legend(handles=spec["handles"], title=layer.name, loc="upper left", frameon=False,
                         prop=dict(size=fontsize, **font), title_fontproperties=dict(size=fontsize, weight="bold", **font),
                         labelcolor="#000000", alignment="left", borderaxespad=0.2)
        leg.get_title().set_color("#000000")
    fig.savefig(path, dpi=dpi, facecolor="white", bbox_inches="tight", pad_inches=0.06)
    return path


class MapCanvas(FigureCanvasQTAgg):
    coords_changed = pyqtSignal(float, float)
    coords_cleared = pyqtSignal()                # mouse left the map
    identify_requested = pyqtSignal(float, float)
    measure_clicked = pyqtSignal(float, float, bool)   # x, y, finish (double-click / right-click)
    measure_cancel = pyqtSignal()                      # Esc in measure mode
    mode_changed = pyqtSignal(str)

    def __init__(self, store: LayerStore, parent=None):
        self.fig = Figure(facecolor=theme.BG)
        super().__init__(self.fig)
        self.setParent(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setMouseTracking(True)
        self.store = store
        self.ax = self.fig.add_axes([0, 0, 1, 1])
        self._style_axes()
        self.mode = "pan"
        self.show_legend = True
        self._has_view = False
        self._pan = None
        self._pan_pixmap = None
        self._pan_legend = None
        self._pan_offset = QPoint(0, 0)
        self._band = QRubberBand(QRubberBand.Rectangle, self)
        self._band_origin = None
        self._legend_artists = []
        self._layer_artists = []
        self._label_artists = []
        self._label_layers = []         # (layer, z) with labels on; placed per view in _layout_labels()
        self._label_view = None
        self._label_cache = {}
        self._geom_cache = {}           # layer id -> drawing geometry (see _geometry)
        self._marker_xy = None
        self._marker_artists = []
        self._ov = None                 # temporary measure overlay: dict(pts, cursor, polygon, finished)
        self._ov_artists = []
        self._bg = None                 # map background for blitting the overlay
        self._bg_fixed = None           # map + fixed part of the overlay (mouse-move fast path)
        self._rubber_box = None

        store.changed.connect(self.redraw)
        store.layer_changed.connect(lambda _l: self.redraw())
        store.active_changed.connect(lambda _l: self.update_legend(draw=True))
        store.data_loaded.connect(self._data_loaded)
        store.data_failed.connect(lambda l, msg: __import__("logging").getLogger("mpm").warning(
            "could not read %s: %s", l.name, msg))
        self._redraw_timer = QTimer(self)                     # several rasters arriving: one redraw
        self._redraw_timer.setSingleShot(True)
        self._redraw_timer.setInterval(60)
        self._redraw_timer.timeout.connect(self.redraw)
        self.mpl_connect("scroll_event", self._on_scroll)
        self.mpl_connect("button_press_event", self._on_press)
        self.mpl_connect("button_release_event", self._on_release)
        self.mpl_connect("motion_notify_event", self._on_motion)
        self.mpl_connect("resize_event", self._on_resize)
        self.mpl_connect("draw_event", self._remember_scale)
        self.mpl_connect("draw_event", self._on_draw_overlay)
        self.redraw()

    def sizeHint(self):
        return QSize(900, 700)

    # ------------------------------------------------------------------ drawing
    def _data_loaded(self, layer):
        if layer in self.store.layers and layer.visible:
            self._redraw_timer.start()

    def _map_bg(self) -> str:
        """Map background: the theme's window colour while there are no layers (start-up), white with layers."""
        return theme.MAP_BG if self.store.layers else theme.BG

    def _apply_bg(self):
        bg = self._map_bg()
        self.fig.set_facecolor(bg)
        self.ax.set_facecolor(bg)
        self.setPalette(QPalette(QColor(bg)))     # FigureCanvasQT sets a white palette (shows at buffer edges)

    def _style_axes(self):
        ax = self.ax
        ax.set_facecolor(self._map_bg())      # also re-applied after geopandas plots
        ax.set_aspect("auto")   # 1:1 scale is kept by _set_limits_fitted()
        ax.set_anchor("C")
        ax.grid(False)
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        for s in ax.spines.values():
            s.set_visible(False)

    def redraw(self):
        lims = (self.ax.get_xlim(), self.ax.get_ylim()) if self._has_view else None
        for a in list(self.ax.collections) + list(self.ax.images) + list(self.ax.patches) + list(self.ax.lines):
            try:
                a.remove()
            except Exception:  # noqa: BLE001
                pass
        for t in self._label_artists:
            try:
                t.remove()
            except Exception:  # noqa: BLE001
                pass
        self._layer_artists, self._label_artists, self._label_layers = [], [], []
        self._label_view = None
        self._apply_bg()

        layers = list(reversed(self.store.layers))      # bottom first
        ids = {l.id for l in layers}                    # forget caches of removed layers
        self._geom_cache = {k: v for k, v in self._geom_cache.items() if k in ids}
        self._label_cache = {k: v for k, v in self._label_cache.items() if k in ids}
        for z, layer in enumerate(layers, start=1):
            if not layer.visible:
                continue
            try:
                if isinstance(layer, RasterLayer):
                    self._draw_raster(layer, z)
                elif isinstance(layer, VectorLayer):
                    self._draw_vector(layer, z)
            except Exception as exc:  # noqa: BLE001
                import logging
                logging.getLogger("mpm").warning("could not draw %s: %s", layer.name, exc)

        if not self.store.layers:                       # plain theme-coloured map, no start-up text
            self._has_view = False
            self.ax.set_xlim(0, 1)
            self.ax.set_ylim(0, 1)
        elif lims is None:
            self.zoom_full(draw=False)
        else:
            self.ax.set_xlim(*lims[0])
            self.ax.set_ylim(*lims[1])
        # imshow / geopandas switch the axes to a fixed-aspect box that shrinks the drawing area to the data's
        # shape (everything outside it is clipped): keep the axes filling the whole canvas
        self._style_axes()
        self._draw_marker()
        self._ov_artists = []           # removed together with the other lines above
        self._update_overlay_artists()
        self.update_legend(draw=False)
        self.draw_idle()

    # ------------------------------------------------------------------ identify marker
    def show_marker(self, x: float, y: float):
        self._marker_xy = (x, y)
        self._draw_marker()
        self.draw_idle()

    def clear_marker(self):
        self._marker_xy = None
        self._draw_marker()
        self.draw_idle()

    def _draw_marker(self):
        for a in self._marker_artists:
            try:
                a.remove()
            except Exception:  # noqa: BLE001
                pass
        self._marker_artists = []
        if self._marker_xy is None:
            return
        x, y = self._marker_xy
        halo = self.ax.plot([x], [y], marker="+", ms=22, mew=5, color=self._map_bg(), zorder=9000)[0]
        mark = self.ax.plot([x], [y], marker="+", ms=20, mew=2, color=theme.ACCENT, zorder=9001)[0]
        ring = self.ax.plot([x], [y], marker="o", ms=9, mfc="none", mew=1.5, mec=theme.ACCENT, zorder=9001)[0]
        self._marker_artists = [halo, mark, ring]

    # ------------------------------------------------------------------ temporary measure overlay (blitted)
    MEASURE_COLOR = "#ff00ff"   # magenta: best worst-case contrast (CIEDE2000) over all app colour maps + dark bg
    MEASURE_LW = 1.0            # line width (points)
    MEASURE_DOT = 2.8           # vertex marker diameter (points)

    def set_overlay(self, pts, cursor=None, polygon: bool = False, finished: bool = False):
        """Temporary measurement graphics: line through ``pts`` + dashed rubber band to the cursor (and back to the
        first point for polygons).

        Fast path for mouse moves: the fixed part is drawn only when the points change and is stored together with
        the map background; a move redraws just the single rubber-band line and repaints only its neighbourhood.
        """
        old = self._ov
        self._ov = dict(pts=list(pts), cursor=cursor, polygon=polygon, finished=finished)
        static_changed = (old is None or old["pts"] != self._ov["pts"] or old["polygon"] != polygon
                          or old["finished"] != finished)
        self._update_overlay_artists()
        if static_changed:
            self._refresh_static_overlay()
        else:
            self._blit_rubber()

    def clear_overlay(self):
        self._ov = None
        self._update_overlay_artists()
        self._refresh_static_overlay()

    def _update_overlay_artists(self):
        ov = self._ov
        if not ov or not ov["pts"]:
            for a in self._ov_artists:
                try:
                    a.remove()
                except Exception:  # noqa: BLE001
                    pass
            self._ov_artists = []
            return
        if not self._ov_artists or self._ov_artists[0].axes is None:
            # [0] fixed line (+ vertices), [1] rubber band (single colour, dashed)
            fixed = Line2D([], [], color=self.MEASURE_COLOR, lw=self.MEASURE_LW, marker="o", ms=self.MEASURE_DOT,
                           mfc=self.MEASURE_COLOR, mec=self.MEASURE_COLOR, mew=0, animated=True, zorder=9501)
            rubber = Line2D([], [], color=self.MEASURE_COLOR, lw=self.MEASURE_LW, ls=(0, (4, 3)), animated=True,
                            zorder=9501)
            for art in (fixed, rubber):
                self.ax.add_line(art)
            self._ov_artists = [fixed, rubber]
        pts, cur = ov["pts"], ov["cursor"]
        closed = ov["polygon"] and ov["finished"] and len(pts) > 2
        fixed_pts = list(pts) + ([pts[0]] if closed else [])
        dyn = []
        if cur is not None and not ov["finished"]:
            dyn = [pts[-1], cur] + ([pts[0]] if ov["polygon"] and len(pts) >= 2 else [])
        self._ov_artists[0].set_data([p[0] for p in fixed_pts], [p[1] for p in fixed_pts])
        self._ov_artists[1].set_data([p[0] for p in dyn], [p[1] for p in dyn])

    def _static_overlay(self):
        a = self._ov_artists
        return [a[0]] if len(a) == 2 else []

    def _rubber(self):
        return self._ov_artists[1] if len(self._ov_artists) == 2 else None

    def _capture_static(self):
        """Buffer currently holds the clean map: add the fixed overlay and remember the result."""
        for a in self._static_overlay():
            self.ax.draw_artist(a)
        self._bg_fixed = self.copy_from_bbox(self.fig.bbox)

    def _on_draw_overlay(self, _ev=None):
        """After every full render: remember the map (without the animated overlay) and paint the overlay on top."""
        self._bg = self.copy_from_bbox(self.fig.bbox)
        self._capture_static()
        r = self._rubber()
        if r is not None:
            self.ax.draw_artist(r)
        self._rubber_box = self._rubber_bbox()

    def _refresh_static_overlay(self):
        if self._bg is None or self._pan is not None:
            self.draw_idle()
            return
        self.restore_region(self._bg)
        self._capture_static()
        r = self._rubber()
        if r is not None:
            self.ax.draw_artist(r)
        self._rubber_box = self._rubber_bbox()
        self.blit(self.fig.bbox)

    def _rubber_bbox(self):
        r = self._rubber()
        if r is None or not len(r.get_xdata()):
            return None
        xy = self.ax.transData.transform(np.column_stack([r.get_xdata(), r.get_ydata()]))
        pad = 4 * self._dpr()
        return Bbox.from_extents(xy[:, 0].min() - pad, xy[:, 1].min() - pad, xy[:, 0].max() + pad, xy[:, 1].max() + pad)

    def _blit_rubber(self):
        """Mouse-move fast path: restore map + fixed overlay, draw the rubber band, repaint only where it changed."""
        if getattr(self, "_bg_fixed", None) is None or self._pan is not None:
            self._refresh_static_overlay()
            return
        new = self._rubber_bbox()
        boxes = [b for b in (getattr(self, "_rubber_box", None), new) if b is not None]
        self._rubber_box = new
        if not boxes:
            return
        dirty = Bbox.intersection(Bbox.union(boxes), self.fig.bbox)
        if dirty is None or dirty.width <= 0 or dirty.height <= 0:
            return
        # restore only the dirty rectangle (Agg region coords: origin top-left, offset of the saved region = 0,0)
        h = self.fig.bbox.height
        x0, y0, x1, y1 = dirty.extents
        self.restore_region(self._bg_fixed, bbox=(int(x0), int(h - y1), int(np.ceil(x1)), int(np.ceil(h - y0))),
                            xy=(0, 0))
        r = self._rubber()
        if r is not None:
            self.ax.draw_artist(r)
        self.blit(dirty)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape and self.mode == "measure":
            self.measure_cancel.emit()
            return
        super().keyPressEvent(event)

    def leaveEvent(self, event):
        self.coords_cleared.emit()
        super().leaveEvent(event)

    def map_units_per_pixel(self) -> float:
        x0, x1 = self.ax.get_xlim()
        return (x1 - x0) / max(1.0, self.ax.bbox.width / self._dpr())

    def _draw_raster(self, layer: RasterLayer, z: int):
        if not layer.loaded:                                   # read in the background; drawn when it arrives
            self.store.request_data(layer)
            return
        data = layer.data
        step = int(np.ceil(max(data.shape) / MAX_DISPLAY))
        if step > 1:
            data = data[::step, ::step]
        extent = layer.grid.extent
        mode = layer.renderer
        kw = dict(extent=extent, origin="upper", aspect="auto", interpolation="nearest", alpha=layer.opacity, zorder=z)
        if mode == "single":
            idx = np.ma.array(np.zeros(data.shape, np.uint8), mask=np.ma.getmaskarray(data))
            cmap = mcolors.ListedColormap([layer.single_color]).with_extremes(bad=(0, 0, 0, 0))
            im = self.ax.imshow(idx, cmap=cmap, norm=mcolors.Normalize(-0.5, 0.5), **kw)
        elif mode in ("unique", "categorized"):
            if mode == "unique":
                classes = layer.classes()
                if classes.size == 0 or classes.size > sym.MAX_UNIQUE:
                    return self._draw_stretch(layer, data, kw)
                cols = category_colors(layer, classes)
                idx = np.searchsorted(classes, np.ma.getdata(data))
                idx = np.ma.array(np.clip(idx, 0, classes.size - 1), mask=np.ma.getmaskarray(data))
            else:
                cats = (layer.categorized or {}).get("classes") or []
                if not cats:
                    return self._draw_stretch(layer, data, kw)
                cols = [c.get("color", "#888888") for c in cats]
                idx = sym.raster_category_index(layer, data)
            cmap = mcolors.ListedColormap(cols).with_extremes(bad=(0, 0, 0, 0))
            im = self.ax.imshow(idx, cmap=cmap, norm=mcolors.Normalize(-0.5, len(cols) - 0.5), **kw)
        else:
            return self._draw_stretch(layer, data, kw)
        self._layer_artists.append(im)

    def _draw_stretch(self, layer: RasterLayer, data, kw):
        cmap = get_cmap(layer.cmap).with_extremes(bad=(0, 0, 0, 0))
        im = self.ax.imshow(data, cmap=cmap, vmin=layer.vmin, vmax=layer.vmax, **kw)
        self._layer_artists.append(im)

    def _geometry(self, layer: VectorLayer):
        """Drawing geometry of a layer, cached until its data changes (per-feature Shapely 1.8 calls are slow):
        points -> (index, x, y); lines -> [(index, xy)]; polygons -> [(index, Path)]."""
        key = (id(layer.gdf), len(layer.gdf))
        hit = self._geom_cache.get(layer.id)
        if hit is not None and hit[0] == key:
            return hit[1]
        out = []
        for i, g in enumerate(layer.gdf.geometry):
            if g is None or g.is_empty:
                continue
            if layer.geom_type == "point":
                c = g if g.geom_type == "Point" else g.centroid
                out.append((i, c.x, c.y))
            elif layer.geom_type == "line":
                out += [(i, np.asarray(part.coords)[:, :2]) for part in getattr(g, "geoms", [g])]
            else:
                out += [(i, _polygon_path(poly)) for poly in getattr(g, "geoms", [g]) if poly.geom_type == "Polygon"]
        if layer.geom_type == "point":
            arr = np.array(out, float).reshape(-1, 3)
            out = (arr[:, 0].astype(int), arr[:, 1], arr[:, 2])
        self._geom_cache[layer.id] = (key, out)
        return out

    def _draw_vector(self, layer: VectorLayer, z: int):
        styles = sym.feature_styles(layer)
        geo = self._geometry(layer)
        op = layer.opacity
        if layer.geom_type == "point":
            idx, xs, ys = geo
            groups = {}
            for j, i in enumerate(idx):                         # one scatter per marker shape
                st = styles[i]
                if st is not None:
                    groups.setdefault(st.get("marker", "o"), []).append(j)
            for marker, js in groups.items():
                sts = [styles[idx[j]] for j in js]
                a = self.ax.scatter(xs[js], ys[js], s=[st.get("markersize", 24) for st in sts],
                                    c=[_rgba(st.get("color", "#ffd54f"), op) for st in sts], marker=marker,
                                    edgecolors=[_rgba(st.get("edgecolor", "#202020"), op) for st in sts],
                                    linewidths=0.6, zorder=z + 1000)
                self._layer_artists.append(a)
        elif layer.geom_type == "line":
            parts = [(xy, styles[i]) for i, xy in geo if styles[i] is not None]
            if parts:
                a = LineCollection([xy for xy, _st in parts],
                                   colors=[_rgba(st.get("color", "#ff8a65"), op) for _xy, st in parts],
                                   linewidths=[st.get("linewidth", 0.8) for _xy, st in parts], zorder=z + 500)
                self.ax.add_collection(a)
                self._layer_artists.append(a)
        else:
            parts = [(path, styles[i]) for i, path in geo if styles[i] is not None]
            if parts:
                # same level as lines: vectors always draw above rasters
                a = PathCollection([path for path, _st in parts],
                                   facecolors=[_rgba(st.get("facecolor", "none"), op) for _p, st in parts],
                                   edgecolors=[_rgba(st.get("edgecolor", "#b0bec5"), op) for _p, st in parts],
                                   linewidths=[st.get("linewidth", 0.8) for _p, st in parts], zorder=z + 500)
                self.ax.add_collection(a)
                self._layer_artists.append(a)
        self._label_layers.append((layer, z))                     # placed for the current view in draw()

    # ------------------------------------------------------------------ labels
    def _label_data(self, layer: VectorLayer):
        """(x, y, texts) of the labels of a layer — cached until the data or the label field changes."""
        field = layer.labels.get("field")
        key = (id(layer.gdf), len(layer.gdf), field)
        hit = self._label_cache.get(layer.id)
        if hit is not None and hit[0] == key:
            return hit[1:]
        if layer.geom_type == "point":                         # the cached drawing coordinates
            idx, xs, ys = self._geometry(layer)
        else:
            idx, pts = [], []
            for i, g in enumerate(layer.gdf.geometry):
                if g is not None and not g.is_empty:
                    p = g.representative_point()
                    idx.append(i)
                    pts.append((p.x, p.y))
            idx = np.asarray(idx, int)
            xy = np.asarray(pts, float).reshape(-1, 2)
            xs, ys = xy[:, 0], xy[:, 1]
        vals = layer.gdf[field].to_numpy()[idx]
        ok = np.array([not (v is None or (isinstance(v, float) and np.isnan(v))) for v in vals], bool)
        xs, ys, vals = xs[ok], ys[ok], vals[ok]
        texts = [sym.fmt_number(v) if isinstance(v, (int, float, np.number)) and not isinstance(v, bool) else str(v)
                 for v in vals]
        self._label_cache[layer.id] = (key, xs, ys, texts)
        return xs, ys, texts

    def _layout_labels(self):
        """Labels for the current view only, without overlaps (first come, first placed; top layers first) and at
        most MAX_LABELS_DRAWN — placed again only when the view or the label settings change. Drawing one text per
        feature made a layer with thousands of features take ~15 s per render."""
        layers = [(l, z) for l, z in self._label_layers if l.visible and (l.labels or {}).get("enabled")
                  and l.labels.get("field") in l.gdf.columns]
        ax = self.ax
        key = (ax.get_xlim(), ax.get_ylim(), round(ax.bbox.width), round(ax.bbox.height), self.fig.dpi,
               tuple((l.id, z, id(l.gdf), tuple(sorted((k, str(v)) for k, v in l.labels.items()))) for l, z in layers))
        if key == self._label_view:
            return
        self._label_view = key
        for t in self._label_artists:
            try:
                t.remove()
            except Exception:  # noqa: BLE001
                pass
        self._label_artists = []
        if not layers:
            return
        (x0, x1), (y0, y1) = ax.get_xlim(), ax.get_ylim()
        dpi = self.fig.dpi
        occupied = set()
        budget = MAX_LABELS_DRAWN
        for layer, z in reversed(layers):                       # top layer first
            lab = layer.labels
            xs, ys, texts = self._label_data(layer)
            inview = np.nonzero((xs >= min(x0, x1)) & (xs <= max(x0, x1)) & (ys >= min(y0, y1)) & (ys <= max(y0, y1)))[0]
            if inview.size == 0:
                continue
            size = float(lab.get("size", 8))
            h = size * 1.3 * dpi / 72                           # label box in pixels (estimated from the text length)
            cw = size * 0.62 * dpi / 72
            cell = max(4.0, h)
            dy = 0 if layer.geom_type == "polygon" else 4
            off = dy * dpi / 72
            font = dict(fontsize=size, color=lab.get("color", "#202020"))
            family = _mpl_font(lab.get("family"))
            if family:
                font["fontfamily"] = family
            trans = mtransforms.offset_copy(ax.transData, fig=self.fig, y=dy, units="points")
            pix = ax.transData.transform(np.c_[xs[inview], ys[inview]])
            for j, (px, py) in zip(inview, pix):
                if budget <= 0:
                    return
                w = len(texts[j]) * cw
                top = py + off + (h if dy else h / 2)
                bottom = top - h
                c0, c1 = int((px - w / 2) // cell), int((px + w / 2) // cell)
                r0, r1 = int(bottom // cell), int(top // cell)
                cells = [(c, r) for c in range(c0, c1 + 1) for r in range(r0, r1 + 1)]
                if any(c in occupied for c in cells):
                    continue
                occupied.update(cells)
                t = ax.text(xs[j], ys[j], texts[j], transform=trans, ha="center", va="center" if dy == 0 else "bottom",
                            zorder=z + 2000, clip_on=True, **font)
                self._label_artists.append(t)
                budget -= 1

    def draw(self):
        try:
            self._layout_labels()
        except Exception as exc:  # noqa: BLE001
            import logging
            logging.getLogger("mpm").warning("could not place labels: %s", exc)
        super().draw()

    # ------------------------------------------------------------------ legend
    def update_legend(self, draw: bool = True):
        for a in self._legend_artists:
            try:
                a.remove()
            except Exception:  # noqa: BLE001
                pass
        self._legend_artists = []
        layer = self.store.active
        if not (self.show_legend and layer is not None and layer.visible):
            if draw:
                self.draw_idle()
            return
        if isinstance(layer, RasterLayer) and not layer.loaded:   # legend follows when the cells are read
            self.store.request_data(layer)
            if draw:
                self.draw_idle()
            return
        title = layer.name if len(layer.name) <= 32 else layer.name[:30] + "…"
        spec = legend_spec(layer, limit=14)
        if spec["kind"] == "colorbar":
            cax = self.fig.add_axes([0.93, 0.05, 0.014, 0.26])
            sm = cm.ScalarMappable(norm=mcolors.Normalize(spec["vmin"], spec["vmax"]), cmap=get_cmap(spec["cmap"]))
            cb = self.fig.colorbar(sm, cax=cax)
            cb.outline.set_edgecolor(theme.BORDER)
            cax.yaxis.set_ticks_position("left")
            cax.tick_params(colors=theme.TEXT, labelsize=7, length=2)
            t = self.fig.text(0.937, 0.325, title, ha="right", va="bottom", fontsize=8, color=theme.TEXT,
                              bbox=dict(boxstyle="round,pad=0.3", fc=theme.PANEL, ec=theme.BORDER, alpha=1.0))
            self._legend_artists += [cax, t]
        elif spec["handles"]:
            leg = self.ax.legend(handles=spec["handles"], title=title, loc="lower right", fontsize=7, title_fontsize=8,
                                 frameon=True, framealpha=1.0, facecolor=theme.PANEL, edgecolor=theme.BORDER,
                                 labelcolor=theme.TEXT, borderpad=0.7)
            leg.get_title().set_color(theme.TEXT)
            leg.set_zorder(5000)
            self._legend_artists.append(leg)
        if draw:
            self.draw_idle()

    # ------------------------------------------------------------------ navigation
    def zoom_full(self, draw: bool = True):
        bounds = [l.bounds() for l in self.store.layers]
        if not bounds:
            return
        b = np.array(bounds, float)
        self.set_extent(b[:, 0].min(), b[:, 1].min(), b[:, 2].max(), b[:, 3].max(), draw=draw)

    def zoom_to_layer(self, layer):
        self.set_extent(*layer.bounds())

    def set_extent(self, left, bottom, right, top, margin: float = 0.03, draw: bool = True):
        w, h = right - left, top - bottom
        if w <= 0 or h <= 0:
            w = h = max(w, h, 1.0)
        self._set_limits_fitted(left - w * margin, right + w * margin, bottom - h * margin, top + h * margin)
        self._has_view = True
        if draw:
            self.draw_idle()

    def _set_limits_fitted(self, x0, x1, y0, y1):
        """Set limits that contain the box and give 1:1 map units per pixel (expands the short side)."""
        bbox = self.ax.bbox
        pw, ph = max(bbox.width, 1.0), max(bbox.height, 1.0)
        w, h = x1 - x0, y1 - y0
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        scale = max(w / pw, h / ph)
        self.ax.set_xlim(cx - scale * pw / 2, cx + scale * pw / 2)
        self.ax.set_ylim(cy - scale * ph / 2, cy + scale * ph / 2)

    def _remember_scale(self, _ev=None):
        x0, x1 = self.ax.get_xlim()
        if self.ax.bbox.width > 1:
            self._upp = (x1 - x0) / self.ax.bbox.width      # map units per pixel at last draw

    def _on_resize(self, _ev=None):
        """Keep the centre and the scale (map units per pixel) when the window is resized."""
        upp = getattr(self, "_upp", None)
        if not self._has_view or not upp:
            return
        x0, x1 = self.ax.get_xlim()
        y0, y1 = self.ax.get_ylim()
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        pw, ph = self.ax.bbox.width, self.ax.bbox.height
        self.ax.set_xlim(cx - upp * pw / 2, cx + upp * pw / 2)
        self.ax.set_ylim(cy - upp * ph / 2, cy + upp * ph / 2)
        self.draw_idle()

    def zoom(self, factor: float, x=None, y=None):
        x0, x1 = self.ax.get_xlim()
        y0, y1 = self.ax.get_ylim()
        x = (x0 + x1) / 2 if x is None else x
        y = (y0 + y1) / 2 if y is None else y
        self.ax.set_xlim(x - (x - x0) / factor, x + (x1 - x) / factor)
        self.ax.set_ylim(y - (y - y0) / factor, y + (y1 - y) / factor)
        self._has_view = True
        self.draw_idle()

    def set_mode(self, mode: str):
        self.mode = mode
        self.setCursor({"pan": Qt.OpenHandCursor, "zoom_in": Qt.CrossCursor, "zoom_out": Qt.CrossCursor,
                        "identify": Qt.WhatsThisCursor, "measure": Qt.CrossCursor}.get(mode, Qt.ArrowCursor))
        self.mode_changed.emit(mode)

    def _on_scroll(self, ev):
        if ev.xdata is None or not self.store.layers:
            return
        self.zoom(1.25 if ev.button == "up" else 1 / 1.25, ev.xdata, ev.ydata)

    def _on_press(self, ev):
        if not self.store.layers or ev.x is None:
            return
        # middle button pans in every tool mode; left button pans in Pan mode
        if ev.button == MouseButton.MIDDLE or (ev.button == MouseButton.LEFT and self.mode == "pan"):
            self._pan = (ev.x, ev.y, self.ax.get_xlim(), self.ax.get_ylim(), ev.xdata, ev.ydata)
            self._start_snapshot_pan()
            self.setCursor(Qt.ClosedHandCursor)
        elif ev.button == MouseButton.LEFT and self.mode == "zoom_in":
            self._band_origin = self._qt_point(ev)
            self._band.setGeometry(QRect(self._band_origin, QSize()))
            self._band.show()
        elif ev.button == MouseButton.LEFT and self.mode == "zoom_out" and ev.xdata is not None:
            self.zoom(0.5, ev.xdata, ev.ydata)
        elif ev.button == MouseButton.LEFT and self.mode == "identify" and ev.xdata is not None:
            self.identify_requested.emit(ev.xdata, ev.ydata)
        elif self.mode == "measure":
            if ev.button == MouseButton.LEFT and ev.xdata is not None:
                self.measure_clicked.emit(ev.xdata, ev.ydata, bool(ev.dblclick))
            elif ev.button == MouseButton.RIGHT:
                self.measure_clicked.emit(float("nan"), float("nan"), True)

    # ------------------------------------------------------------------ snapshot panning
    # Like ArcMap / QGIS: while dragging, a rendered image of the map slides with the cursor and the map
    # is re-rendered once on release. The legend is rendered into a separate transparent overlay that is
    # painted at a fixed position, so it never moves with the map.
    def _render_pixmap(self) -> QPixmap:
        self.draw()                                           # synchronous Agg render
        buf = np.asarray(self.buffer_rgba())
        h, w = buf.shape[:2]
        img = QImage(buf.tobytes(), w, h, 4 * w, QImage.Format_RGBA8888).copy()
        img.setDevicePixelRatio(self._dpr())
        return QPixmap.fromImage(img)

    def _start_snapshot_pan(self):
        legend = [a for a in self._legend_artists if a is not None]
        # 1) map without legend
        for a in legend:
            a.set_visible(False)
        self._pan_pixmap = self._render_pixmap()
        # 2) legend only, on a transparent background
        self._pan_legend = None
        if legend:
            hidden = [a for a in self.ax.get_children() if a.get_visible() and a not in legend]
            hidden += [self.fig.patch]
            for a in hidden:
                a.set_visible(False)
            for a in legend:
                a.set_visible(True)
            try:
                self._pan_legend = self._render_pixmap()
            finally:
                for a in hidden:
                    a.set_visible(True)
        for a in legend:
            a.set_visible(True)
        self._pan_offset = QPoint(0, 0)
        self.update()

    def paintEvent(self, event):
        if getattr(self, "_pan_pixmap", None) is not None:
            p = QPainter(self)
            p.fillRect(self.rect(), QColor(self._map_bg()))
            p.drawPixmap(self._pan_offset, self._pan_pixmap)        # map follows the cursor
            if self._pan_legend is not None:
                p.drawPixmap(QPoint(0, 0), self._pan_legend)         # legend stays put
            p.end()
            return
        super().paintEvent(event)

    def _finish_pan(self, ev):
        x0, y0, xl, yl = self._pan[:4]
        self._pan = None
        self._pan_pixmap = None
        self._pan_legend = None
        if ev is not None and ev.x is not None and (ev.x != x0 or ev.y != y0):
            bbox = self.ax.bbox
            dx = (ev.x - x0) * (xl[1] - xl[0]) / bbox.width
            dy = (ev.y - y0) * (yl[1] - yl[0]) / bbox.height
            self.ax.set_xlim(xl[0] - dx, xl[1] - dx)
            self.ax.set_ylim(yl[0] - dy, yl[1] - dy)
            self._has_view = True
        self.draw_idle()                                       # one full render at the new position
        self.set_mode(self.mode)

    def _on_motion(self, ev):
        if self._pan is not None and ev.x is not None:
            x0, y0 = self._pan[0], self._pan[1]
            r = self._dpr()
            self._pan_offset = QPoint(int(round((ev.x - x0) / r)), int(round(-(ev.y - y0) / r)))
            self.update()
            if self._pan[4] is not None:                      # the grabbed map point stays under the cursor
                self.coords_changed.emit(self._pan[4], self._pan[5])
            return
        if ev.xdata is not None:
            self.coords_changed.emit(ev.xdata, ev.ydata)
        if self._band_origin is not None and ev.x is not None:
            self._band.setGeometry(QRect(self._band_origin, self._qt_point(ev)).normalized())

    def _on_release(self, ev):
        if self._pan is not None:
            self._finish_pan(ev)
            return
        if self._band_origin is not None:
            rect = self._band.geometry()
            self._band.hide()
            self._band_origin = None
            if rect.width() > 4 and rect.height() > 4:
                inv = self.ax.transData.inverted()
                r, h = self._dpr(), self.height()
                (xa, ya) = inv.transform((rect.left() * r, (h - rect.bottom()) * r))
                (xb, yb) = inv.transform((rect.right() * r, (h - rect.top()) * r))
                self.set_extent(min(xa, xb), min(ya, yb), max(xa, xb), max(ya, yb), margin=0)
            elif ev.xdata is not None:
                self.zoom(2.0, ev.xdata, ev.ydata)

    # matplotlib event coords: physical pixels from bottom-left; Qt widget coords: logical px from top-left
    def _dpr(self) -> float:
        return float(getattr(self, "device_pixel_ratio", None) or self.devicePixelRatioF() or 1.0)

    def _qt_point(self, ev) -> QPoint:
        r = self._dpr()
        return QPoint(int(ev.x / r), int(self.height() - ev.y / r))

    def export_image(self, path: str, dpi: int = 200):
        self.fig.savefig(path, dpi=dpi, facecolor=self._map_bg())


class MapPanel(QWidget):
    """Map canvas + a footer bar below it: cursor coordinates (bottom-left) and map CRS (bottom-right)."""

    def __init__(self, store: LayerStore, parent=None):
        super().__init__(parent)
        self.store = store
        self.canvas = MapCanvas(store, self)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        lay.addWidget(self.canvas, 1)
        footer = QFrame()
        footer.setObjectName("mapFooter")
        footer.setFixedHeight(26)          # constant height: text changes must never resize the map
        fl = QHBoxLayout(footer)
        fl.setContentsMargins(10, 3, 10, 3)
        fl.setSpacing(16)
        self.xy_icon = QLabel()
        self.xy_icon.setPixmap(theme.icon("mdi6.crosshairs-gps", theme.MUTED).pixmap(14, 14))
        self.xy_label = QLabel()
        self.xy_label.setObjectName("coords")
        self.cell_label = QLabel()                    # project raster cell size (Project ▸ Project properties)
        self.cell_label.setObjectName("coordsMuted")
        self.crs_label = QLabel()
        self.crs_label.setObjectName("coordsMuted")
        # fixed width: updating the text on every mouse move then needs no re-layout (keeps live tools fast)
        fm = self.xy_label.fontMetrics()
        self.xy_label.setFixedWidth(fm.horizontalAdvance("X -99,999,999.99   Y -99,999,999.99") + 24)
        fl.addWidget(self.xy_icon)
        fl.addWidget(self.xy_label)
        fl.addStretch(1)
        fl.addWidget(self.cell_label)
        fl.addWidget(self.crs_label)
        lay.addWidget(footer)
        self._geographic = False
        self.canvas.coords_changed.connect(self.show_coords)
        self.canvas.coords_cleared.connect(self.clear_coords)
        store.changed.connect(self._crs_changed)
        self._crs_changed()
        self.clear_coords()

    def theme_changed(self):
        """Light / dark switch: static pixmaps do not follow the palette by themselves."""
        self.xy_icon.setPixmap(theme.icon("mdi6.crosshairs-gps", "MUTED").pixmap(14, 14))

    def _crs_changed(self):
        from core.layers import describe_crs, format_length
        crs = self.store.crs
        short, long = describe_crs(crs)
        self.crs_label.setText(short if crs is not None else tr("no CRS"))
        self.crs_label.setToolTip(long)
        cell = self.store.cell_size
        self.cell_label.setText(tr("Cell {size}", size=format_length(cell, crs)) if cell else "")
        self.cell_label.setToolTip(tr("Project raster cell size (Project ▸ Project properties)"))
        self._geographic = bool(crs is not None and crs.is_geographic)

    def show_coords(self, x: float, y: float):
        """Cursor position in the map CRS: decimal degrees for a geographic CRS, map units otherwise."""
        if self._geographic:
            self.xy_label.setText(f"X {x:.6f}   Y {y:.6f}")
        else:
            self.xy_label.setText(f"X {x:,.2f}   Y {y:,.2f}")

    def clear_coords(self):
        self.xy_label.setText("X —   Y —")
