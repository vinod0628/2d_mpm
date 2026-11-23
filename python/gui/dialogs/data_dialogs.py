"""Data-preparation dialogs: convert, reclassify, align to grid, clip, polygon to raster, distance to features."""
from __future__ import annotations

import os
import re

import numpy as np
import pandas as pd
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QCheckBox, QComboBox, QDoubleSpinBox, QFormLayout, QGroupBox, QHBoxLayout, QLineEdit,
                             QPushButton, QRadioButton, QSpinBox, QSplitter, QVBoxLayout, QWidget)
from rasterio.crs import CRS

from core.layers import LayerStore, RasterLayer, VectorLayer
from gui import theme
from i18n import N_, tr
from gui.widgets import BaseDialog, LayerCombo, MplWidget, PathEdit, hint, make_table
from utils import convert as uconv
from utils import reclass as urec
from utils.raster_tools import Grid, align_to_grid, distance_to_features

KEEP_CRS, MAP_CRS = N_("(keep source CRS)"), N_("(map CRS)")
CRS_CHOICES = [KEEP_CRS, MAP_CRS, "EPSG:4326", "EPSG:4283", "EPSG:28354", "EPSG:32611", "EPSG:26911"]


def parse_crs(text: str, store: LayerStore):
    t = text.strip()
    if not t or t.startswith("(keep") or t == tr(KEEP_CRS):
        return None
    if t.startswith("(map") or t == tr(MAP_CRS):
        return store.crs
    return CRS.from_user_input(t)


def _out_name_row(form: QFormLayout, default: str, filt: str = "GeoTIFF (*.tif)"):
    name = QLineEdit(default)
    path = PathEdit("save", filt, tr("Save output as"))
    path.setPlaceholderText(tr("optional — leave empty to keep the result in memory"))
    form.addRow(tr("Output layer name"), name)
    form.addRow(tr("Save to file"), path)
    return name, path


# ================================================================================ convert
class ConvertDialog(BaseDialog):
    """Convert any GDAL/OGR dataset (incl. ArcInfo grids, shapefiles) to GeoTIFF / GeoPackage."""

    def __init__(self, store: LayerStore, parent=None):
        super().__init__(tr("Convert to GeoTIFF / GeoPackage"),
                         tr("Rasters (ArcInfo grid folders, .img, .asc, .tif …) → compressed GeoTIFF with a "
                            "class-label sidecar; vectors (.shp, .geojson …) → GeoPackage or Shapefile (by the "
                            "output extension). Assign a CRS when the source has none."), parent, ok_text=tr("Convert"))
        self.store = store
        form = QFormLayout()
        kind = QHBoxLayout()
        self.r_raster, self.r_vector = QRadioButton(tr("Raster")), QRadioButton(tr("Vector"))
        self.r_raster.setChecked(True)
        kind.addWidget(self.r_raster)
        kind.addWidget(self.r_vector)
        kind.addStretch(1)
        form.addRow(tr("Data type"), kind)
        self.src = PathEdit("open_or_dir", tr("All files (*)"), tr("Select input"))
        self.src.edit.textChanged.connect(self._suggest)
        form.addRow(tr("Input"), self.src)
        self.dst = PathEdit("save", "GeoTIFF (*.tif);;GeoPackage (*.gpkg);;ESRI Shapefile (*.shp)", tr("Output file"))
        form.addRow(tr("Output"), self.dst)
        self.crs = QComboBox()
        self.crs.setEditable(True)
        self.crs.addItems([tr(c) for c in CRS_CHOICES])
        form.addRow(tr("Assign CRS"), self.crs)
        self.override = QCheckBox(tr("Override the source CRS (raster) / reproject to it (vector)"))
        form.addRow("", self.override)
        self.nodata = QLineEdit()
        self.nodata.setPlaceholderText(tr("keep source value"))
        form.addRow(tr("NoData (raster)"), self.nodata)
        self.add = QCheckBox(tr("Add result to the map"))
        self.add.setChecked(True)
        form.addRow("", self.add)
        self.body.addLayout(form)
        self.r_vector.toggled.connect(self._suggest)
        self.resize(620, 0)

    def _suggest(self):
        s = self.src.text()
        if not s:
            return
        base = os.path.splitext(s.rstrip("\\/"))[0] if not os.path.isdir(s) else s.rstrip("\\/")
        if os.path.isfile(os.path.join(s, "hdr.adf")) or os.path.isfile(os.path.join(s, "HDR.ADF")):
            self.r_raster.setChecked(True)
        if s.lower().endswith(uconv.VECTOR_EXT):
            self.r_vector.blockSignals(True)
            self.r_vector.setChecked(True)
            self.r_vector.blockSignals(False)
        self.dst.setText(base + (".gpkg" if self.r_vector.isChecked() else ".tif"))

    def validate(self):
        if not self.src.text() or not os.path.exists(self.src.text()):
            return tr("Choose an existing input file or folder.")
        if not self.dst.text():
            return tr("Choose an output file.")
        try:
            parse_crs(self.crs.currentText(), self.store)
        except Exception as exc:  # noqa: BLE001
            return tr("Invalid CRS: {error}", error=exc)
        if self.nodata.text().strip():
            try:
                float(self.nodata.text())
            except ValueError:
                return tr("NoData must be a number.")
        return None

    def run(self, progress=None):
        crs = parse_crs(self.crs.currentText(), self.store)
        src, dst = self.src.text(), self.dst.text()
        if self.r_raster.isChecked():
            nod = float(self.nodata.text()) if self.nodata.text().strip() else None
            out = uconv.raster_to_geotiff(src, dst, crs=crs, override_crs=self.override.isChecked(), nodata=nod)
            return ("raster", out)
        to_crs = crs if self.override.isChecked() else None
        if os.path.splitext(dst)[1].lower() not in (".gpkg", ".shp", ".geojson", ".json"):
            dst = os.path.splitext(dst)[0] + ".gpkg"
        out = uconv.vector_convert(src, dst, crs=crs, to_crs=to_crs)
        return ("vector", out)


# ================================================================================ reclassify
class ReclassDialog(BaseDialog):
    def __init__(self, store: LayerStore, parent=None):
        super().__init__(tr("Reclassify raster"),
                         tr("Turn a continuous raster into classes (needed for Weights of Evidence and other "
                            "class-based methods). Class codes start at 1; labels are written to the class table."),
                         parent, ok_text=tr("Create layer"))
        self.store = store
        split = QSplitter(Qt.Horizontal)
        left = QWidget()
        form = QFormLayout(left)
        self.src = LayerCombo(store, "raster")
        self.src.currentIndexChanged.connect(self._src_changed)
        form.addRow(tr("Input raster"), self.src)
        self.method = QComboBox()
        for k, v in urec.METHOD_LABELS.items():
            self.method.addItem(tr(v), k)
        self.method.setCurrentIndex(1)
        form.addRow(tr("Method"), self.method)
        self.n = QSpinBox()
        self.n.setRange(2, 50)
        self.n.setValue(8)
        form.addRow(tr("Number of classes"), self.n)
        self.breaks = QLineEdit()
        self.breaks.setPlaceholderText(tr("e.g. 500, 1000, 2000, 4000"))
        form.addRow(tr("Manual breaks"), self.breaks)
        prev = QPushButton(theme.icon("mdi6.eye-outline"), tr("Preview"))
        prev.clicked.connect(self.preview)
        form.addRow("", prev)
        self.name, self.path = _out_name_row(form, "")
        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(0, 0, 0, 0)
        self.chart = MplWidget(4.5, 2.4)
        rl.addWidget(self.chart, 2)
        self.table_holder = QVBoxLayout()
        rl.addLayout(self.table_holder, 3)
        split.addWidget(left)
        split.addWidget(right)
        split.setSizes([380, 520])
        self.body.addWidget(split)
        self.method.currentIndexChanged.connect(lambda _i: self.breaks.setEnabled(self.method.currentData() == "manual"))
        self.breaks.setEnabled(False)
        self._table = None
        self._result = None
        self._src_changed()
        self.resize(980, 560)

    def _src_changed(self):
        l = self.src.layer()
        if l is not None:
            self.name.setText(f"{l.name}_cls")
            self.preview()

    def _breaks(self):
        l = self.src.layer()
        vals = np.ma.compressed(l.data).astype(float)
        manual = [float(x) for x in re.split(r"[,\s;]+", self.breaks.text().strip()) if x] \
            if self.method.currentData() == "manual" else None
        return vals, urec.compute_breaks(vals, self.method.currentData(), self.n.value(), manual)

    def preview(self):
        l = self.src.layer()
        if l is None:
            return
        try:
            vals, b = self._breaks()
            cls, table = urec.reclass_by_breaks(l.data, b)
        except Exception as exc:  # noqa: BLE001
            self.chart.fig.clear()
            ax = self.chart.fig.add_subplot(111)
            ax.text(0.5, 0.5, str(exc), ha="center", va="center", transform=ax.transAxes, color=theme.BAD, wrap=True)
            self.chart.draw_idle()
            return
        table["percent"] = 100 * table["count"] / max(1, table["count"].sum())
        self._result = (cls, table)
        self.chart.fig.clear()
        ax = self.chart.fig.add_subplot(111)
        sample = vals if vals.size <= 400000 else vals[:: vals.size // 400000]
        ax.hist(sample, bins=80, color=theme.ACCENT, alpha=0.85)
        for x in b:
            ax.axvline(x, color=theme.WARN, lw=1)
        ax.set_title(tr("{layer}: {n} classes", layer=l.name, n=len(b) + 1))
        ax.set_ylabel(tr("cells"))
        self.chart.draw_idle()
        if self._table is not None:
            self._table.setParent(None)
        self._table = make_table(table[["value", "label", "count", "percent"]].rename(columns={
            "value": tr("Value"), "label": tr("Label"), "count": tr("Count"), "percent": tr("Percent")}),
            float_fmt="{:.2f}")
        self.table_holder.addWidget(self._table)

    def validate(self):
        if self.src.layer() is None:
            return tr("Choose an input raster.")
        self.preview()
        if self._result is None:
            return tr("Could not classify with these settings.")
        return None

    def run(self, progress=None):
        l = self.src.layer()
        cls, table = self._result
        lyr = RasterLayer(self.name.text().strip() or f"{l.name}_cls", cls, l.grid, categorical=True,
                          class_table=table.drop(columns=["percent"]), nodata=0)
        lyr.cmap = "viridis"
        lyr.description = f"Reclassified from {l.name} ({self.method.currentText()})"
        if self.path.text():
            lyr.save(self.path.text())
        return lyr


# ================================================================================ align
class AlignDialog(BaseDialog):
    def __init__(self, store: LayerStore, parent=None):
        super().__init__(tr("Align raster to grid"),
                         tr("Resample / reproject a raster onto the grid (CRS, cell size, extent) of a template "
                            "raster. Multi-layer methods need all evidence on one grid."), parent,
                         ok_text=tr("Create layer"))
        self.store = store
        form = QFormLayout()
        self.src = LayerCombo(store, "raster")
        self.tpl = LayerCombo(store, "raster")
        self.res = QComboBox()
        self.res.addItems(["nearest", "bilinear", "cubic", "average", "mode"])
        self.src.currentIndexChanged.connect(self._defaults)
        form.addRow(tr("Input raster"), self.src)
        form.addRow(tr("Template grid"), self.tpl)
        form.addRow(tr("Resampling"), self.res)
        self.mask = QCheckBox(tr("Mask with template nodata"))
        self.mask.setChecked(True)
        form.addRow("", self.mask)
        self.name, self.path = _out_name_row(form, "")
        self.body.addLayout(form)
        self.body.addWidget(hint(tr("Use 'nearest' (or 'mode') for categorical rasters, 'bilinear' for continuous data.")))
        self._defaults()
        self.resize(560, 0)

    def _defaults(self):
        l = self.src.layer()
        if l is not None:
            self.name.setText(f"{l.name}_aligned")
            self.res.setCurrentText("nearest" if l.categorical else "bilinear")

    def validate(self):
        if self.src.layer() is None or self.tpl.layer() is None:
            return tr("Choose input and template rasters.")
        return None

    def run(self, progress=None):
        s, t = self.src.layer(), self.tpl.layer()
        out = align_to_grid(s.data, s.grid, t.grid, self.res.currentText())
        if self.mask.isChecked():
            out = np.ma.array(out, mask=np.ma.getmaskarray(out) | np.ma.getmaskarray(t.data))
        lyr = RasterLayer(self.name.text().strip(), out, t.grid, categorical=s.categorical,
                          class_table=s.class_table, cmap=s.cmap, nodata=s.nodata)
        lyr.description = f"{s.name} aligned to {t.name} ({self.res.currentText()})"
        if self.path.text():
            lyr.save(self.path.text())
        return lyr


# ================================================================================ clip
ALL_FEATURES = N_("(all features)")
DEFAULT_NODATA = "-9999"


def format_nodata(v) -> str:
    """Project NoData value as text (-9999, not -9999.0)."""
    if v is None:
        return DEFAULT_NODATA
    return str(int(v)) if float(v).is_integer() else f"{v:g}"


from utils.raster_tools import nodata_dtype  # noqa: E402,F401  (shared helper, kept importable here)


class ClipDialog(BaseDialog):
    """Clip raster by mask layer (cf. QGIS 'Clip raster by mask layer', ArcGIS 'Extract by Mask')."""

    def __init__(self, store: LayerStore, parent=None):
        super().__init__(tr("Clip raster by mask"),
                         tr("Keep the raster cells inside (or outside) the polygons of a mask layer. Cells stay on "
                            "the input grid — nothing is resampled. The mask is reprojected to the raster CRS when "
                            "needed."), parent, ok_text=tr("Clip"))
        self.store = store
        self._auto_path = self._auto_name = ""

        g_in = QGroupBox(tr("Input"))
        f = QFormLayout(g_in)
        self.src = LayerCombo(store, "raster")
        self.mask_layer = LayerCombo(store, "polygon")
        f.addRow(tr("Input raster"), self.src)
        f.addRow(tr("Mask layer (polygons)"), self.mask_layer)
        self.field = QComboBox()
        self.value = QComboBox()
        self.value.setEditable(True)
        fr = QHBoxLayout()
        fr.addWidget(self.field, 1)
        fr.addWidget(self.value, 1)
        f.addRow(tr("Use features where"), fr)

        g_opt = QGroupBox(tr("Options"))
        f = QFormLayout(g_opt)
        self.area = QComboBox()
        self.area.addItem(tr("Inside the polygons"), "inside")
        self.area.addItem(tr("Outside the polygons"), "outside")
        f.addRow(tr("Keep cells"), self.area)
        self.touched = QComboBox()
        self.touched.addItem(tr("Cell centre inside a polygon"), False)
        self.touched.addItem(tr("Every cell touched by a polygon"), True)
        f.addRow(tr("A cell is inside when"), self.touched)
        self.crop = QCheckBox(tr("Crop the extent to the mask layer"))
        self.crop.setChecked(False)          # off: the result stays on the project grid (same extent for all rasters)
        self.crop.setToolTip(tr("Off keeps the input grid, so the result can be combined cell by cell with the other "
                                "project rasters"))
        f.addRow("", self.crop)
        self.nodata = QLineEdit(format_nodata(getattr(store, "nodata", None)))   # project NoData value
        self.nodata.setToolTip(tr("Integer rasters that cannot hold this value (e.g. 8-bit classes) are written "
                                  "with a wider integer type."))
        f.addRow(tr("Output NoData value"), self.nodata)
        self.compress = QComboBox()
        for c in ("LZW", "DEFLATE"):
            self.compress.addItem(c, c)
        self.compress.addItem(tr("none"), None)
        f.addRow(tr("Compression"), self.compress)

        g_out = QGroupBox(tr("Output"))
        f = QFormLayout(g_out)
        self.path = PathEdit("save", "GeoTIFF (*.tif)", tr("Save clipped raster as"))
        f.addRow(tr("Output file"), self.path)
        self.name = QLineEdit()
        f.addRow(tr("Output layer name"), self.name)
        self.add = QCheckBox(tr("Add result to the map"))
        self.add.setChecked(True)
        f.addRow("", self.add)

        labels = [g.layout().itemAt(i, QFormLayout.LabelRole) for g in (g_in, g_opt, g_out)
                  for i in range(g.layout().rowCount())]
        labels = [it.widget() for it in labels if it is not None and it.widget() is not None]
        lw = max(l.sizeHint().width() for l in labels)
        for l in labels:                                    # one label column across the three groups
            l.setMinimumWidth(lw)
        for g in (g_in, g_opt, g_out):
            self.body.addWidget(g)
        self.summary = hint("")
        self.body.addWidget(self.summary)

        self.src.currentIndexChanged.connect(self._src_changed)
        self.mask_layer.currentIndexChanged.connect(self._mask_changed)
        self.field.currentIndexChanged.connect(self._field_changed)
        self.value.currentTextChanged.connect(lambda _t: self._update_summary())
        self.area.currentIndexChanged.connect(self._area_changed)
        self.crop.toggled.connect(lambda _b: self._update_summary())
        self.path.edit.textChanged.connect(self._path_changed)
        self._mask_changed()
        self._src_changed()
        self.resize(600, 0)

    # ------------------------------------------------------------ defaults
    def default_path(self) -> str:
        """<project>/results/<input file name>_clip.tif (next to the input file when there is no project)."""
        from core import project
        l = self.src.layer()
        if l is None:
            return ""
        stem = os.path.splitext(os.path.basename(l.source.rstrip("\\/")))[0] if l.source else project.safe_name(l.name)
        pdir = project.project_dir(self.store)
        if pdir:
            folder = os.path.join(pdir, project.RESULTS_DIR)
        elif l.source:
            folder = os.path.dirname(os.path.abspath(l.source.rstrip("\\/")))
        else:
            return ""
        return project.unique_path(os.path.join(folder, f"{stem}_clip.tif"))

    def _src_changed(self):
        l = self.src.layer()
        if l is None:
            return
        if self.path.text() in ("", self._auto_path):
            self._auto_path = self.default_path()
            self.path.setText(self._auto_path)
        self._update_summary()

    def _path_changed(self, text: str):
        stem = os.path.splitext(os.path.basename(text.strip()))[0]
        if self.name.text() in ("", self._auto_name):
            self._auto_name = stem
            self.name.setText(stem)

    def _mask_changed(self):
        m = self.mask_layer.layer()
        self.field.blockSignals(True)
        self.field.clear()
        self.field.addItem(tr(ALL_FEATURES), None)
        if m is not None:
            for c in m.gdf.columns:
                if c != m.gdf.geometry.name:
                    self.field.addItem(str(c), c)
        self.field.blockSignals(False)
        self._field_changed()

    def _field_changed(self):
        m, fld = self.mask_layer.layer(), self.field.currentData()
        self.value.blockSignals(True)
        self.value.clear()
        if m is not None and fld is not None:
            vals = pd.unique(m.gdf[fld].dropna())
            try:
                vals = sorted(vals)
            except TypeError:
                pass
            self.value.addItems([str(v) for v in vals[:1000]])
        self.value.setEnabled(fld is not None)
        self.value.blockSignals(False)
        self._update_summary()

    def _area_changed(self):
        outside = self.area.currentData() == "outside"
        self.crop.setEnabled(not outside)
        self._update_summary()

    # ------------------------------------------------------------ mask features
    def mask_features(self):
        m = self.mask_layer.layer()
        if m is None:
            return None
        gdf = m.gdf
        fld = self.field.currentData()
        if fld is not None:
            gdf = gdf[gdf[fld].astype(str) == self.value.currentText()]
        return gdf

    def _update_summary(self):
        l, gdf = self.src.layer(), self.mask_features()
        if l is None or gdf is None:
            self.summary.setText("")
            return
        text = tr("{n} of {total} mask features used.", n=len(gdf), total=len(self.mask_layer.layer().gdf))
        if len(gdf) and self.crop.isChecked() and self.crop.isEnabled():
            try:
                from rasterio.warp import transform_bounds
                b = gdf.total_bounds
                if gdf.crs is not None and l.grid.crs is not None and gdf.crs != l.grid.crs:
                    b = transform_bounds(gdf.crs, l.grid.crs, *b)
                gl, gb, gr, gt = l.grid.bounds
                rx, ry = l.grid.res
                w = max(0, int(np.ceil((min(b[2], gr) - max(b[0], gl)) / rx)))
                h = max(0, int(np.ceil((min(b[3], gt) - max(b[1], gb)) / ry)))
                text += " " + (tr("Output about {w} × {h} cells (input {iw} × {ih}).", w=w, h=h,
                                  iw=l.grid.width, ih=l.grid.height) if w and h
                               else tr("The mask does not overlap the raster."))
            except Exception:  # noqa: BLE001
                pass
        self.summary.setText(text)

    def _nodata_value(self):
        """Output NoData value (int when it is a whole number); raises ValueError when it is not a number."""
        try:
            v = float(self.nodata.text().strip())
        except ValueError:
            raise ValueError(tr("NoData must be a number.")) from None
        if not np.isfinite(v):
            raise ValueError(tr("NoData must be a number."))
        return int(v) if v.is_integer() else v

    # ------------------------------------------------------------ run
    def validate(self):
        if self.src.layer() is None:
            return tr("Choose an input raster.")
        if self.mask_layer.layer() is None:
            return tr("Choose a polygon mask layer (add one with Data ▸ Add vector…).")
        if self.field.currentData() is not None and not len(self.mask_features()):
            return tr("No mask features match {field} = {value}.", field=self.field.currentText(),
                      value=self.value.currentText())
        if not self.path.text():
            return tr("Choose an output file.")
        if not self.name.text().strip():
            return tr("Enter an output layer name.")
        try:
            self._nodata_value()
        except ValueError as exc:
            return str(exc)
        return None

    def run(self, progress=None):
        from utils.raster_tools import clip_by_mask, write_raster
        l = self.src.layer()
        data, grid = clip_by_mask(l.data, l.grid, self.mask_features(), crop=self.crop.isChecked(),
                                  invert=self.area.currentData() == "outside",
                                  all_touched=bool(self.touched.currentData()))
        nodata = self._nodata_value()
        data = data.astype(nodata_dtype(data.dtype, nodata))
        lyr = RasterLayer(self.name.text().strip(), data, grid, categorical=l.categorical,
                          class_table=l.class_table, cmap=l.cmap, nodata=nodata)
        lyr.description = f"{l.name} clipped by {self.mask_layer.layer().name}"
        path = self.path.text()
        if not path.lower().endswith((".tif", ".tiff")):
            path += ".tif"
        write_raster(path, lyr.data, lyr.grid, nodata=nodata, compress=self.compress.currentData(),
                     class_table=lyr.class_table if lyr.categorical else None)
        lyr.source = path
        return lyr


# ================================================================================ polygon to raster
NO_LABEL_FIELD = N_("(none — one class for all polygons)")
MAX_OUTPUT_CELLS = 200_000_000


class PolygonToRasterDialog(BaseDialog):
    """Polygons (e.g. a geological map) to a categorical raster with a class table (cf. ArcGIS Polygon to Raster)."""

    def __init__(self, store: LayerStore, parent=None):
        super().__init__(tr("Polygon to raster"),
                         tr("Convert polygons (e.g. a geological map) into a categorical raster for the class-based "
                            "methods: every value of the label field becomes a class, written with a class table. "
                            "Cells of the study area outside every polygon form a gap class."),
                         parent, ok_text=tr("Create raster"))
        self.store = store
        self._auto_path = self._auto_name = ""

        g_in = QGroupBox(tr("Input"))
        f = QFormLayout(g_in)
        self.src = LayerCombo(store, "polygon")
        f.addRow(tr("Polygon layer"), self.src)
        self.field = QComboBox()
        f.addRow(tr("Class label field"), self.field)
        self.codes = QComboBox()
        self.codes.addItem(tr("sequential 1, 2, 3 … (labels sorted)"), "sequential")
        self.codes.addItem(tr("values of the label field (whole numbers)"), "field")
        f.addRow(tr("Class codes"), self.codes)
        self.copy_attr = QCheckBox(tr("Copy class attributes to the class table"))
        self.copy_attr.setChecked(True)
        self.copy_attr.setToolTip(tr("Attributes with one value per class (e.g. the unit name or a host-rock flag of "
                                     "every map unit) are added to the class table — then usable as class-table "
                                     "columns in the methods"))
        f.addRow("", self.copy_attr)

        g_grid = QGroupBox(tr("Output grid"))
        f = QFormLayout(g_grid)
        self.grid_from = QComboBox()
        self.grid_from.addItem(tr("reference raster (cell size, extent, alignment)"), "raster")
        self.grid_from.addItem(tr("study area polygon and cell size"), "polygon")
        f.addRow(tr("Grid from"), self.grid_from)
        self.ref = LayerCombo(store, "raster")
        f.addRow(tr("Reference raster"), self.ref)
        self.cell = QDoubleSpinBox()
        self.cell.setRange(1e-6, 1e9)
        self.cell.setDecimals(6 if store.crs is not None and store.crs.is_geographic else 3)
        self.cell.setValue(store.cell_size or 100.0)
        self.cell.setToolTip(tr("Default: the project cell size"))
        f.addRow(tr("Cell size"), self.cell)
        self.extent = QComboBox()
        self.extent.addItem(tr("project extent (same grid as the project rasters)"), "project")
        self.extent.addItem(tr("study area polygon extent (aligned to the project grid)"), "study")
        f.addRow(tr("Extent"), self.extent)
        self.assign = QComboBox()
        self.assign.addItem(tr("cell centre — polygon containing the cell centre"), "center")
        self.assign.addItem(tr("maximum area — polygon covering most of the cell"), "max_area")
        self.assign.setToolTip(tr("Cell centre is exact and fast (overlapping polygons: the last feature wins). "
                                  "Maximum area keeps narrow units and small polygons that miss every cell centre "
                                  "(estimated on 4 × 4 points per cell)."))
        f.addRow(tr("Cell value"), self.assign)

        g_area = QGroupBox(tr("Study area"))
        f = QFormLayout(g_area)
        self.study = QComboBox()
        self.study.addItem(tr("(whole output grid)"), "none")
        self.study.addItem(tr("valid cells of the reference raster"), "ref")
        for l in store.vectors("polygon"):
            self.study.addItem(l.name, l.id)
        self.study.setToolTip(tr("Cells outside the study area are NoData. A polygon study area uses the cells "
                                 "whose centre is inside."))
        f.addRow(tr("Study area"), self.study)
        self.gap = QComboBox()
        self.gap.addItem(tr("a class labelled"), "class")
        self.gap.addItem(tr("NoData"), "nodata")
        self.gap_label = QLineEdit("Other")
        gr = QHBoxLayout()
        gr.addWidget(self.gap)
        gr.addWidget(self.gap_label, 1)
        f.addRow(tr("Gaps (no polygon)"), gr)

        g_out = QGroupBox(tr("Output"))
        f = QFormLayout(g_out)
        self.path = PathEdit("save", "GeoTIFF (*.tif)", tr("Save raster as"))
        f.addRow(tr("Output file"), self.path)
        self.name = QLineEdit()
        f.addRow(tr("Output layer name"), self.name)
        self.add = QCheckBox(tr("Add result to the map"))
        self.add.setChecked(True)
        f.addRow("", self.add)

        for cb in (self.src, self.field, self.codes, self.grid_from, self.ref, self.extent, self.assign, self.study,
                   self.gap):                              # long item texts must not widen the columns
            cb.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
            cb.setMinimumContentsLength(12)
        cols = QHBoxLayout()                               # two columns: fits a 1366 × 768 display unscrolled
        for groups in ((g_in, g_area), (g_grid, g_out)):
            labels = [g.layout().itemAt(i, QFormLayout.LabelRole) for g in groups
                      for i in range(g.layout().rowCount())]
            labels = [it.widget() for it in labels if it is not None and it.widget() is not None]
            lw = max(l.sizeHint().width() for l in labels)
            for l in labels:                                # one label column per column of groups
                l.setMinimumWidth(lw)
            col = QVBoxLayout()
            for g in groups:
                col.addWidget(g)
            col.addStretch(1)
            cols.addLayout(col, 1)
        self.body.addLayout(cols)
        self.summary = hint("")
        self.summary.setWordWrap(True)
        self.body.addWidget(self.summary)

        self._default_study()
        self.src.currentIndexChanged.connect(self._src_changed)
        self.field.currentIndexChanged.connect(self._field_changed)
        for w in (self.codes, self.grid_from, self.ref, self.extent, self.study, self.gap, self.assign):
            w.currentIndexChanged.connect(lambda _i: self._sync())
        self.cell.valueChanged.connect(lambda _v: self._update_summary())
        self.path.edit.textChanged.connect(self._path_changed)
        self._src_changed()
        self.resize(980, 0)

    # ------------------------------------------------------------ defaults / state
    def _default_study(self):
        """A polygon layer called like 'study area' if there is one, else the reference raster's valid cells
        (a study-mask raster when the project has one)."""
        for i in range(self.study.count()):
            if re.search(r"study|aoi|boundary", self.study.itemText(i), re.I) and self.study.itemData(i) not in \
                    ("none", "ref"):
                self.study.setCurrentIndex(i)
                break
        else:
            self.study.setCurrentIndex(1)
        masks = [r for r in self.store.rasters() if re.search(r"study|mask", r.name, re.I)]
        if masks:
            self.ref.set_layer(masks[0])

    def _src_changed(self):
        l = self.src.layer()
        self.field.blockSignals(True)
        self.field.clear()
        self.field.addItem(tr(NO_LABEL_FIELD), None)
        if l is not None:
            cols = [c for c in l.gdf.columns if c != l.gdf.geometry.name]
            for c in cols:
                self.field.addItem(str(c), c)
            text = [c for c in cols if not pd.api.types.is_numeric_dtype(l.gdf[c])]
            pick = text[0] if text else (cols[0] if cols else None)
            if pick is not None:
                self.field.setCurrentIndex(self.field.findData(pick))
        self.field.blockSignals(False)
        self._field_changed()

    def _field_whole_numbers(self) -> bool:
        l, fld = self.src.layer(), self.field.currentData()
        if l is None or fld is None or not pd.api.types.is_numeric_dtype(l.gdf[fld]):
            return False
        v = l.gdf[fld].dropna().to_numpy(float)
        return bool(v.size) and bool(np.all(v == np.round(v)))

    def _field_changed(self):
        if self.path.text() in ("", self._auto_path):
            self._auto_path = self.default_path()
            self.path.setText(self._auto_path)
        self._sync()

    def _sync(self):
        whole = self._field_whole_numbers()
        self.codes.model().item(1).setEnabled(whole)
        if not whole and self.codes.currentData() == "field":
            self.codes.setCurrentIndex(0)
        self.copy_attr.setEnabled(self.field.currentData() is not None)
        by_raster = self.grid_from.currentData() == "raster"
        self.ref.setEnabled(by_raster or self.study.currentData() == "ref")
        self.cell.setEnabled(not by_raster)
        self.extent.setEnabled(not by_raster)
        self.extent.model().item(0).setEnabled(bool(self.store.extent))
        if not self.store.extent and self.extent.currentData() == "project":
            self.extent.setCurrentIndex(1)
        if by_raster and self.ref.layer() is not None:
            self.cell.blockSignals(True)
            self.cell.setValue(self.ref.layer().grid.res[0])
            self.cell.blockSignals(False)
        self.study.model().item(1).setEnabled(by_raster)
        if not by_raster and self.study.currentData() == "ref":
            polys = [i for i in range(2, self.study.count())]
            self.study.setCurrentIndex(polys[0] if polys else 0)
        self.gap_label.setEnabled(self.gap.currentData() == "class")
        self._update_summary()

    def default_path(self) -> str:
        """<project>/results/<polygon layer>_<field>.tif"""
        from core import project
        l = self.src.layer()
        pdir = project.project_dir(self.store)
        if l is None or not pdir:
            return ""
        fld = self.field.currentData()
        stem = project.safe_name(l.name + (f"_{fld}" if fld is not None else ""))
        return project.unique_path(os.path.join(pdir, project.RESULTS_DIR, f"{stem}.tif"))

    def _path_changed(self, text: str):
        stem = os.path.splitext(os.path.basename(text.strip()))[0]
        if self.name.text() in ("", self._auto_name):
            self._auto_name = stem
            self.name.setText(stem)

    # ------------------------------------------------------------ grid / study area
    def study_layer(self):
        d = self.study.currentData()
        return None if d in ("none", "ref") else self.store.by_id(d)

    def output_grid(self) -> Grid:
        from utils.polygon_raster import snapped_grid
        if self.grid_from.currentData() == "raster":
            r = self.ref.layer()
            if r is None:
                raise ValueError(tr("Choose a reference raster."))
            return r.grid
        cell = float(self.cell.value())
        src = self.src.layer()
        crs = self.store.crs or (src.crs if src is not None else None)
        ext = self.store.extent
        origin = (ext[0], ext[3]) if ext else None
        if self.extent.currentData() == "project" and ext:
            pg = self.store.project_grid()
            if pg is not None and abs(pg.res[0] - cell) <= 1e-9 * cell:
                return pg
            return snapped_grid(ext, cell, crs, origin)
        st = self.study_layer()
        if st is None:
            raise ValueError(tr("Choose a study area polygon layer for the extent."))
        gdf = st.gdf
        if gdf.crs is not None and crs is not None and gdf.crs != crs:
            gdf = gdf.to_crs(crs)
        return snapped_grid(tuple(gdf.total_bounds), cell, crs, origin)

    def study_mask(self, grid: Grid):
        from utils.raster_tools import rasterize_vector
        d = self.study.currentData()
        if d == "none":
            return None
        if d == "ref":
            r = self.ref.layer()
            if r is None or not r.grid.same_as(grid):
                raise ValueError(tr("The reference raster is not on the output grid."))
            m = ~np.ma.getmaskarray(r.data)
            return m
        return rasterize_vector(self.study_layer().gdf, grid, dtype="uint8") == 1

    def _update_summary(self):
        l = self.src.layer()
        if l is None:
            self.summary.setText("")
            return
        fld = self.field.currentData()
        k = 1 if fld is None else int(l.gdf[fld].nunique())
        text = tr("{n} polygons, {k} classes.", n=len(l.gdf), k=k)
        try:
            g = self.output_grid()
            from core.layers import format_length
            text += " " + tr("Output {w} × {h} cells of {cell}.", w=f"{g.width:,}", h=f"{g.height:,}",
                             cell=format_length(g.res[0], g.crs))
            pg = self.store.project_grid()
            if pg is not None and not g.same_as(pg):
                text += " " + tr("Not the project grid — align it before combining it with the project rasters.")
        except Exception as exc:  # noqa: BLE001
            text += " " + str(exc)
        self.summary.setText(text)

    # ------------------------------------------------------------ run
    def validate(self):
        from utils.polygon_raster import class_codes
        l = self.src.layer()
        if l is None:
            return tr("Choose a polygon layer.")
        if self.field.currentData() is not None and self.codes.currentData() == "field":
            try:
                class_codes(l.gdf[self.field.currentData()], "field")
            except ValueError as exc:
                return str(exc)
        try:
            g = self.output_grid()
            if g.width * g.height > MAX_OUTPUT_CELLS:
                return tr("{n:,} cells — too many; use a larger cell size.", n=g.width * g.height)
            if self.study.currentData() == "ref":
                self.study_mask(g)                      # reference raster on the output grid
        except ValueError as exc:
            return str(exc)
        if self.gap.currentData() == "class" and not self.gap_label.text().strip():
            return tr("Enter a label for the gap class.")
        if not self.path.text():
            return tr("Choose an output file.")
        if not self.name.text().strip():
            return tr("Enter an output layer name.")
        return None

    def run(self, progress=None):
        from utils.polygon_raster import polygons_to_raster
        from utils.raster_tools import write_raster
        l = self.src.layer()
        grid = self.output_grid()
        fld = self.field.currentData()
        res = polygons_to_raster(l.gdf, grid, fld, codes=self.codes.currentData(),
                                 assignment=self.assign.currentData(), study=self.study_mask(grid),
                                 gap_label=self.gap_label.text().strip() if self.gap.currentData() == "class" else None,
                                 single_label=l.name, copy_attributes=self.copy_attr.isChecked())
        self.result_info = res
        lyr = RasterLayer(self.name.text().strip(), res.data, grid, categorical=True, class_table=res.table,
                          cmap="tab20", nodata=res.nodata)
        lyr.description = f"{l.name} ({fld or 'polygons'}) rasterised"
        path = self.path.text()
        if not path.lower().endswith((".tif", ".tiff")):
            path += ".tif"
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        write_raster(path, lyr.data, grid, nodata=res.nodata, class_table=res.table)
        lyr.source = path
        return lyr


# ================================================================================ distance
class DistanceDialog(BaseDialog):
    def __init__(self, store: LayerStore, parent=None):
        super().__init__(tr("Distance to features"),
                         tr("Euclidean distance (map units) from every cell to the nearest vector feature, e.g. "
                            "distance to faults or intrusive contacts."), parent, ok_text=tr("Create layer"))
        self.store = store
        form = QFormLayout()
        self.src = LayerCombo(store, "vector")
        self.tpl = LayerCombo(store, "raster")
        self.mode = QComboBox()
        self.mode.addItem(tr("to features (0 inside polygons)"), "feature")
        self.mode.addItem(tr("to lines / polygon boundaries"), "boundary")
        self.maxd = QDoubleSpinBox()
        self.maxd.setRange(0, 1e9)
        self.maxd.setSpecialValueText(tr("no limit"))
        self.mask = QCheckBox(tr("Mask with template nodata"))
        self.mask.setChecked(True)
        form.addRow(tr("Features"), self.src)
        form.addRow(tr("Grid template"), self.tpl)
        form.addRow(tr("Distance"), self.mode)
        form.addRow(tr("Cap distance at"), self.maxd)
        form.addRow("", self.mask)
        self.src.currentIndexChanged.connect(lambda _i: self.name.setText(
            f"dist_{self.src.layer().name}" if self.src.layer() else ""))
        self.name, self.path = _out_name_row(form, "")
        if self.src.layer():
            self.name.setText(f"dist_{self.src.layer().name}")
        self.body.addLayout(form)
        self.resize(560, 0)

    def validate(self):
        if self.src.layer() is None or self.tpl.layer() is None:
            return tr("Choose a vector layer and a template raster.")
        if self.tpl.layer().grid.is_geographic:
            return tr("The template grid is in geographic degrees — align/reproject it to a metric CRS first.")
        return None

    def run(self, progress=None):
        v, t = self.src.layer(), self.tpl.layer()
        d = distance_to_features(v.gdf, t.grid, self.mode.currentData(), self.maxd.value() or None)
        out = np.ma.array(d, mask=np.ma.getmaskarray(t.data) if self.mask.isChecked() else np.zeros(d.shape, bool))
        lyr = RasterLayer(self.name.text().strip(), out, t.grid, categorical=False, cmap="magma_r",
                          nodata=getattr(self.store, "nodata", -9999.0))
        lyr.description = f"Distance to {v.name}"
        if self.path.text():
            lyr.save(self.path.text())
        return lyr
