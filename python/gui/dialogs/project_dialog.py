"""Project ▸ Project properties: General, Extent (with the raster cell size) and Coordinate system tabs.

The coordinate system and the raster cell size are editable while the project has no layers; once a layer is
loaded they are fixed (unset values are taken from the first layer / first raster layer, see LayerStore.add).
"""
from __future__ import annotations

import datetime as _dt
import math
import os
from typing import Optional

import numpy as np
import pandas as pd
from PyQt5.QtCore import QUrl, Qt, pyqtSignal
from PyQt5.QtGui import QDesktopServices, QFontDatabase
from PyQt5.QtWidgets import (QComboBox, QFileDialog, QFormLayout, QHBoxLayout, QLabel, QLineEdit, QPlainTextEdit,
                             QPushButton, QTabWidget, QVBoxLayout, QWidget)
from rasterio.crs import CRS

from core import project
from core.layers import LayerStore, RasterLayer, crs_unit, describe_crs, format_length
from gui import theme
from gui.dialogs.settings_dialog import _section
from gui.widgets import BaseDialog, hint, make_table
from i18n import N_, tr
from utils import crs_catalog as cat

CRS_TYPES = [("none", N_("Not set — taken from the first layer")),
             ("geographic", N_("Geographic (GCS) — latitude / longitude")),
             ("utm", N_("Projected — UTM")),
             ("projected", N_("Projected — other (PCS)")),
             ("custom", N_("Custom — EPSG code, WKT or PROJ string"))]


# ============================================================================================ helpers
def folder_size(folder: str):
    """(bytes, number of files) of a folder tree."""
    total = n = 0
    for root, _dirs, files in os.walk(folder):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
                n += 1
            except OSError:
                pass
    return total, n


def human_size(b: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if b < 1024 or unit == "GB":
            return f"{b:,.0f} {unit}" if unit == "B" else f"{b:,.1f} {unit}"
        b /= 1024
    return f"{b:,.1f} TB"


def project_extent(store: LayerStore):
    """(left, bottom, right, top) of all layers in the project CRS, None without layers."""
    bounds = [l.bounds() for l in store.layers]
    if not bounds:
        return None
    return (min(b[0] for b in bounds), min(b[1] for b in bounds), max(b[2] for b in bounds),
            max(b[3] for b in bounds))


def _coord(v: float, geographic: bool) -> str:
    return f"{v:.6f}" if geographic else f"{v:,.2f}"


def _value_label(text: str = "") -> QLabel:
    lab = QLabel(text)
    lab.setTextInteractionFlags(Qt.TextSelectableByMouse)
    lab.setWordWrap(True)
    return lab


def parse_crs_text(text: str) -> Optional[CRS]:
    """EPSG code ('32650', 'EPSG:32650'), WKT or PROJ string; '' -> None. Raises ValueError when invalid."""
    t = text.strip()
    if not t:
        return None
    if t.isdigit():
        t = f"EPSG:{t}"
    try:
        return CRS.from_user_input(t)
    except Exception as exc:  # noqa: BLE001
        raise ValueError(tr("Not a valid coordinate system: {error}", error=exc)) from None


def crs_from_file(path: str) -> Optional[CRS]:
    """CRS of a raster (GeoTIFF, ArcInfo grid folder …) or vector file."""
    import rasterio
    try:
        with rasterio.open(path) as ds:
            if ds.crs is not None:
                return ds.crs
    except Exception:  # noqa: BLE001
        pass
    import geopandas as gpd
    gdf = gpd.read_file(path, rows=1)
    return CRS.from_user_input(gdf.crs.to_wkt()) if gdf.crs is not None else None


# ============================================================================================ CRS picker
class CrsPicker(QWidget):
    """Type ▸ datum ▸ UTM zone / projection / custom definition. ``crs()`` returns the chosen CRS (or None)."""

    changed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        form = QFormLayout(self)
        form.setContentsMargins(0, 0, 0, 0)
        self.form = form
        self.kind = QComboBox()
        for k, label in CRS_TYPES:
            self.kind.addItem(tr(label), k)
        self.datum = QComboBox()
        for label, epsg in cat.DATUMS:
            self.datum.addItem(cat.datum_label(label, epsg), epsg)
        self.zone = QComboBox()                       # all 120 zones: −60 … −1 (60S … 1S), +1 … +60 (1N … 60N)
        self.zone.setMaxVisibleItems(20)
        for z, h, label in cat.utm_zones():
            self.zone.addItem(label, f"{z}{h}")
        self.zone.setCurrentIndex(self.zone.findData("50N"))
        self.zone.setToolTip(tr("−60 … −1 = south zones 60S … 1S, +1 … +60 = north zones 1N … 60N; "
                                "each zone is 6° of longitude wide"))
        self.zone_w = self.zone
        self.proj = QComboBox()
        for code in cat.PROJECTED:
            self.proj.addItem(cat.projected_label(code), code)
        self.custom = QLineEdit()
        self.custom.setPlaceholderText(tr("e.g. 32650, EPSG:28354, a WKT or a PROJ string"))
        self.from_file = QPushButton(theme.icon("mdi6.file-search-outline"), tr("From a data file…"))
        self.from_file.setToolTip(tr("Use the coordinate system of a raster or vector file"))
        self.from_file.clicked.connect(self._pick_file)

        self._rows = {}
        for key, label, w in (("kind", N_("Type"), self.kind), ("datum", N_("Datum"), self.datum),
                              ("zone", N_("UTM zone"), self.zone_w), ("proj", N_("Projection"), self.proj),
                              ("custom", N_("Definition"), self.custom)):
            lab = QLabel(tr(label))
            form.addRow(lab, w)
            self._rows[key] = (lab, w)
        form.addRow("", self.from_file)

        self.kind.currentIndexChanged.connect(self._kind_changed)
        for w in (self.datum, self.zone, self.proj):
            w.currentIndexChanged.connect(lambda _i: self.changed.emit())
        self.custom.textChanged.connect(lambda _t: self.changed.emit())
        self._kind_changed()

    def _show(self, key: str, on: bool):
        for w in self._rows[key]:
            w.setVisible(on)

    def _kind_changed(self):
        k = self.kind.currentData()
        self._show("datum", k in ("geographic", "utm"))
        self._show("zone", k == "utm")
        self._show("proj", k == "projected")
        self._show("custom", k == "custom")
        self.changed.emit()

    def set_editable(self, on: bool):
        for w in (self.kind, self.datum, self.zone, self.proj, self.custom, self.from_file):
            w.setEnabled(on)

    def crs(self) -> Optional[CRS]:
        k = self.kind.currentData()
        if k == "none":
            return None
        if k == "geographic":
            return CRS.from_epsg(self.datum.currentData())
        if k == "utm":
            zh = self.zone.currentData()
            return cat.utm_crs(self.datum.currentData(), int(zh[:-1]), zh[-1])
        if k == "projected":
            return CRS.from_user_input(self.proj.currentData())
        t = self.custom.text().strip()
        if not t:
            raise ValueError(tr("Enter an EPSG code, a WKT or a PROJ string."))
        return parse_crs_text(t)

    def set_crs(self, crs: Optional[CRS]):
        widgets = (self.kind, self.datum, self.zone, self.proj, self.custom)
        for w in widgets:
            w.blockSignals(True)
        try:
            c = cat.classify(crs)
            if c.get("gcs") is not None:
                self.datum.setCurrentIndex(max(0, self.datum.findData(c["gcs"])))
            if c["kind"] == "utm":
                self.zone.setCurrentIndex(self.zone.findData(f"{c['zone']}{c['hemisphere']}"))
            elif c["kind"] == "projected":
                self.proj.setCurrentIndex(self.proj.findData(c["code"]))
            elif c["kind"] == "custom":
                self.custom.setText(c.get("code") or crs.to_wkt())
            self.kind.setCurrentIndex(self.kind.findData(c["kind"]))
        finally:
            for w in widgets:
                w.blockSignals(False)
        self._kind_changed()

    def _pick_file(self):
        path, _ = QFileDialog.getOpenFileName(self, tr("Coordinate system from a data file"), "",
                                              tr("GIS data") + " (*.tif *.tiff *.img *.asc *.vrt *.gpkg *.shp *.geojson "
                                              "*.json hdr.adf);;" + tr("All files") + " (*)")
        if not path:
            return
        if os.path.basename(path).lower() == "hdr.adf":
            path = os.path.dirname(path)
        try:
            crs = crs_from_file(path)
        except Exception as exc:  # noqa: BLE001
            crs, err = None, str(exc)
        else:
            err = tr("The file has no coordinate system.") if crs is None else ""
        if crs is None:
            from PyQt5.QtWidgets import QMessageBox
            QMessageBox.warning(self, tr("Coordinate system"), f"{path}\n\n{err}")
            return
        self.set_crs(crs)


# ============================================================================================ dialog
def _fmt_nodata(v) -> str:
    return "—" if v is None else (str(int(v)) if float(v).is_integer() else f"{v:g}")


class ProjectPropertiesDialog(BaseDialog):
    TABS = ("general", "extent", "crs")

    def __init__(self, store: LayerStore, parent=None, tab: Optional[str] = None):
        folder = project.project_dir(store) or ""
        super().__init__(tr("Project properties"), folder, parent, ok_text=tr("OK"))
        self.store = store
        self.locked = store.settings_locked
        self.tabs = QTabWidget()
        self.tabs.addTab(self._general_tab(), theme.icon("mdi6.information-outline"), tr("General"))
        self.tabs.addTab(self._extent_tab(), theme.icon("mdi6.border-outside"), tr("Extent"))
        self.tabs.addTab(self._crs_tab(), theme.icon("mdi6.earth"), tr("Coordinate system"))
        self.body.addWidget(self.tabs)
        self.picker.set_crs(store.crs)
        self.cell.setText(f"{store.cell_size:.10g}" if store.cell_size else "")
        self._set_extent_fields(store.extent)
        self.picker.set_editable(not self.locked)
        for w in [self.cell, self.ext_from_file] + [e for _k, e in self.ext_edits]:
            w.setEnabled(not self.locked)                   # greyed out: fixed once the project has layers
        self._update_crs_info()
        if tab in self.TABS:
            self.tabs.setCurrentIndex(self.TABS.index(tab))
        self.resize(720, 560)

    # ------------------------------------------------------------ tabs
    def _lock_hint(self) -> QLabel:
        if self.locked:
            text = tr("The project has layers, so the coordinate system, extent and raster cell size are fixed. "
                      "Remove all layers to change them.")
        else:
            text = tr("The project has no layers yet: set the coordinate system, extent and raster cell size now, or "
                      "leave them unset to take them from the first layer loaded (extent and cell size: first raster). "
                      "Rasters added later are converted onto this grid.")
        lab = hint(text)
        lab.setObjectName("hint")
        return lab

    def _general_tab(self) -> QWidget:
        s = self.store
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.addWidget(_section(tr("Project")))
        f = QFormLayout()
        folder = project.project_dir(s)
        loc = QHBoxLayout()
        loc.addWidget(_value_label(folder or "—"), 1)
        open_btn = QPushButton(theme.icon("mdi6.folder-open-outline"), tr("Open folder"))
        open_btn.clicked.connect(lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(folder)))
        open_btn.setEnabled(bool(folder))
        loc.addWidget(open_btn)
        f.addRow(tr("Name"), _value_label(os.path.basename(folder) if folder else "—"))
        f.addRow(tr("Location"), loc)
        f.addRow(tr("Project file"), _value_label(os.path.basename(s.project_path) if s.project_path else "—"))
        if s.project_path and os.path.isfile(s.project_path):
            saved = _dt.datetime.fromtimestamp(os.path.getmtime(s.project_path)).strftime("%Y-%m-%d %H:%M")
            f.addRow(tr("Last saved"), _value_label(saved + ("   " + tr("(unsaved changes)") if s.dirty else "")))
        if folder and os.path.isdir(folder):
            size, nfiles = folder_size(folder)
            f.addRow(tr("Size on disk"), _value_label(tr("{size}  ({n} files)", size=human_size(size), n=nfiles)))
            res = os.path.join(folder, project.RESULTS_DIR)
            runs = len([d for d in os.listdir(res) if os.path.isdir(os.path.join(res, d))]) if os.path.isdir(res) else 0
            f.addRow(tr("Method runs"), _value_label(tr("{n} in {folder}", n=runs, folder=project.RESULTS_DIR + os.sep)))
        lay.addLayout(f)

        lay.addWidget(_section(tr("Contents")))
        f = QFormLayout()
        nr, nv = len(s.rasters()), len(s.vectors())
        f.addRow(tr("Layers"), _value_label(tr("{n}  ({r} raster, {v} vector)", n=len(s.layers), r=nr, v=nv)))
        mem = len(project.unsaved_layers(s))
        if mem:
            f.addRow(tr("In memory"), _value_label(tr("{n} layers not yet saved (Project ▸ Save project writes "
                                                      "them to data{sep})", n=mem, sep=os.sep)))
        f.addRow(tr("Coordinate system"), _value_label(describe_crs(s.crs)[1] if s.crs is not None else tr("not set")))
        f.addRow(tr("Raster cell size"), _value_label(format_length(s.cell_size, s.crs) if s.cell_size else tr("not set")))
        f.addRow(tr("NoData value"), _value_label(_fmt_nodata(s.nodata)))
        g = s.project_grid()
        f.addRow(tr("Project grid"), _value_label(tr("{cols:,} columns × {rows:,} rows", cols=g.width, rows=g.height)
                                                  if g is not None else tr("not set")))
        lay.addLayout(f)
        lay.addStretch(1)
        return w

    def _extent_tab(self) -> QWidget:
        s = self.store
        w = QWidget()
        lay = QVBoxLayout(w)
        unit = crs_unit(s.crs)
        lay.addWidget(_section(tr("Project grid")))
        f = QFormLayout()
        self.ext_edits = []
        for label, keys in ((tr("X min / X max"), (0, 2)), (tr("Y min / Y max"), (1, 3))):
            row = QHBoxLayout()
            for k in keys:
                e = QLineEdit()
                e.setPlaceholderText(tr("not set — first raster"))
                e.textChanged.connect(lambda _t: self._update_grid_info())
                row.addWidget(e, 1)
                self.ext_edits.append((k, e))
            f.addRow(label, row)
        self.ext_edits.sort(key=lambda ke: ke[0])                 # left, bottom, right, top
        self.ext_from_file = QPushButton(theme.icon("mdi6.file-search-outline"), tr("Extent from a raster file…"))
        self.ext_from_file.clicked.connect(self._extent_from_file)
        f.addRow("", self.ext_from_file)
        row = QHBoxLayout()
        self.cell = QLineEdit()
        self.cell.setPlaceholderText(tr("not set — taken from the first raster layer"))
        self.cell_unit = QLabel(unit)
        row.addWidget(self.cell, 1)
        row.addWidget(self.cell_unit)
        f.addRow(tr("Raster cell size"), row)
        self.grid_info = _value_label()
        f.addRow(tr("Grid"), self.grid_info)
        self.nodata = QLineEdit(_fmt_nodata(s.nodata))
        self.nodata.setToolTip(tr("Written into the rasters created by the MPM methods and tools for cells without "
                                  "a value (outside the study area). Can be changed at any time; existing rasters "
                                  "keep their value."))
        f.addRow(tr("NoData value"), self.nodata)
        ext = project_extent(s)
        if ext is not None:
            geo = bool(s.crs is not None and s.crs.is_geographic)
            l, b, r, t = ext
            f.addRow(tr("All layers"), _value_label(f"X {_coord(l, geo)} … {_coord(r, geo)}   ·   "
                                                    f"Y {_coord(b, geo)} … {_coord(t, geo)}"))
        lay.addLayout(f)
        self.cell.textChanged.connect(lambda _t: self._update_grid_info())
        lay.addWidget(self._lock_hint())

        if s.rasters():
            g = s.project_grid()
            rows = []
            for lyr in s.rasters():
                rx, ry = lyr.grid.res
                on = "—" if g is None else (tr("yes") if lyr.grid.same_as(g) else tr("no"))
                rows.append({tr("Raster layer"): lyr.name,
                             tr("Cell size"): format_length(rx, s.crs) + ("" if abs(rx - ry) < 1e-9 * rx
                                                                           else f" × {format_length(ry, s.crs)}"),
                             tr("Columns × rows"): f"{lyr.grid.width} × {lyr.grid.height}",
                             tr("On the project grid"): on})
            lay.addWidget(make_table(pd.DataFrame(rows)), 1)
            lay.addWidget(hint(tr("Rasters added with Data ▸ Add raster are converted onto the project grid; others "
                                  "can be put on it with Data ▸ Align raster to grid…")))
        else:
            lay.addStretch(1)
        return w

    def _extent_values(self) -> Optional[tuple]:
        """(left, bottom, right, top) from the four fields; None when all are empty. Raises ValueError."""
        texts = [e.text().strip().replace(",", "") for _k, e in self.ext_edits]
        if not any(texts):
            return None
        if not all(texts):
            raise ValueError(tr("Enter all four extent values (or none)."))
        try:
            l, b, r, t = (float(x) for x in texts)
        except ValueError:
            raise ValueError(tr("The extent values must be numbers.")) from None
        if not (r > l and t > b):
            raise ValueError(tr("X max must be larger than X min and Y max larger than Y min."))
        return l, b, r, t

    def _set_extent_fields(self, ext):
        for (_k, e), v in zip(self.ext_edits, ext or (None,) * 4):
            e.setText("" if v is None else f"{v:.10g}")

    def _extent_from_file(self):
        import rasterio
        from rasterio.warp import transform_bounds
        path, _ = QFileDialog.getOpenFileName(self, tr("Extent from a raster file"), "",
                                              tr("Rasters") + " (*.tif *.tiff *.img *.asc *.vrt hdr.adf);;"
                                              + tr("All files") + " (*)")
        if not path:
            return
        if os.path.basename(path).lower() == "hdr.adf":
            path = os.path.dirname(path)
        with rasterio.open(path) as ds:
            b, src = tuple(ds.bounds), ds.crs
        dst, _err = self._picked_crs()
        if src is not None and dst is not None and src != dst:
            b = transform_bounds(src, dst, *b)
        self._set_extent_fields(b)

    def _crs_tab(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.addWidget(_section(tr("Project coordinate system")))
        self.picker = CrsPicker()
        self.picker.changed.connect(self._update_crs_info)
        lay.addWidget(self.picker)
        lay.addWidget(self._lock_hint())
        lay.addWidget(_section(tr("Details")))
        f = QFormLayout()
        self.crs_name = _value_label()
        self.crs_units = _value_label()
        f.addRow(tr("Name"), self.crs_name)
        f.addRow(tr("Units"), self.crs_units)
        lay.addLayout(f)
        self.wkt = QPlainTextEdit()
        self.wkt.setReadOnly(True)
        self.wkt.setFont(QFontDatabase.systemFont(QFontDatabase.FixedFont))
        self.wkt.setMinimumHeight(110)
        lay.addWidget(self.wkt, 1)
        return w

    # ------------------------------------------------------------ live updates
    def _picked_crs(self):
        try:
            return self.picker.crs(), None
        except Exception as exc:  # noqa: BLE001
            return None, str(exc)

    def _update_crs_info(self):
        crs, err = self._picked_crs()
        if err:
            self.crs_name.setText(f"<span style='color:{theme.BAD}'>{err}</span>")
            self.crs_units.setText("")
            self.wkt.setPlainText("")
        elif crs is None:
            self.crs_name.setText(tr("not set — taken from the first layer loaded"))
            self.crs_units.setText("")
            self.wkt.setPlainText("")
        else:
            self.crs_name.setText(describe_crs(crs)[1])
            u = crs_unit(crs)
            kind = tr("geographic (degrees)") if crs.is_geographic else tr("projected")
            self.crs_units.setText(f"{u or '?'}  ·  {kind}")
            try:
                from pyproj import CRS as PCRS
                self.wkt.setPlainText(PCRS.from_user_input(crs.to_wkt()).to_wkt(pretty=True))
            except Exception:  # noqa: BLE001
                self.wkt.setPlainText(crs.to_wkt())
        self.cell_unit.setText(crs_unit(crs) if crs is not None else "")
        self._update_grid_info()

    def _cell_value(self) -> Optional[float]:
        t = self.cell.text().strip().replace(",", "")
        if not t:
            return None
        v = float(t)
        if not math.isfinite(v) or v <= 0:
            raise ValueError
        return v

    def _update_grid_info(self):
        try:
            ext = self._extent_values() or project_extent(self.store)
            cell = self._cell_value()
        except ValueError as exc:
            msg = str(exc) or tr("The cell size must be a positive number.")
            self.grid_info.setText(f"<span style='color:{theme.BAD}'>{msg}</span>")
            return
        if cell is None:
            self.grid_info.setText("—")
            return
        crs = self.store.crs if self.locked else self._picked_crs()[0]
        area = format_length(cell, crs) + " × " + format_length(cell, crs)
        if crs_unit(crs) == "m":
            area += f"  =  {cell * cell / 1e6:.6g} km²"
        if ext is None:
            self.grid_info.setText(tr("cell area {area}", area=area))
            return
        cols = math.ceil((ext[2] - ext[0]) / cell - 1e-9)
        rows = math.ceil((ext[3] - ext[1]) / cell - 1e-9)
        self.grid_info.setText(tr("{cols:,} columns × {rows:,} rows = {n:,} cells;  cell area {area}",
                                  cols=cols, rows=rows, n=cols * rows, area=area))

    # ------------------------------------------------------------ OK
    def _nodata_value(self) -> float:
        v = float(self.nodata.text().strip().replace(",", ""))
        if not np.isfinite(v):
            raise ValueError
        return v

    def validate(self):
        try:
            self._nodata_value()
        except ValueError:
            self.tabs.setCurrentIndex(self.TABS.index("extent"))
            return tr("The NoData value must be a number.")
        if self.locked:
            return None
        crs, err = self._picked_crs()
        if err:
            self.tabs.setCurrentIndex(self.TABS.index("crs"))
            return err
        try:
            self._cell_value()
        except ValueError:
            self.tabs.setCurrentIndex(self.TABS.index("extent"))
            return tr("The cell size must be a positive number.")
        try:
            self._extent_values()
        except ValueError as exc:
            self.tabs.setCurrentIndex(self.TABS.index("extent"))
            return str(exc)
        return None

    def apply(self) -> bool:
        """Write the NoData value and (no layers yet) coordinate system / extent / cell size to the project.
        True when something changed."""
        nod = self._nodata_value()
        changed = nod != self.store.nodata
        if changed:
            self.store.nodata = nod
            self.store.dirty = True
        if self.locked:
            return changed
        crs, cell, ext = self.picker.crs(), self._cell_value(), self._extent_values()
        same_crs = (crs is None and self.store.crs is None) or (
            crs is not None and self.store.crs is not None and crs == self.store.crs)
        if same_crs and cell == self.store.cell_size and ext == self.store.extent:
            return changed
        self.store.set_project_settings(crs, cell, ext)
        return True
