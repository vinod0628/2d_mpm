"""Main application window."""
from __future__ import annotations

import logging
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
from PyQt5.QtCore import QObject, QSettings, Qt, pyqtSignal
from PyQt5.QtGui import QIcon, QPixmap
from PyQt5.QtWidgets import (QAction, QActionGroup, QApplication, QDockWidget, QFileDialog, QLabel, QMainWindow,
                             QMenu, QMessageBox, QPlainTextEdit, QProgressBar,
                             QTabWidget, QToolBar)

from core import methods as reg
from core import project
from core.layers import LayerStore, RasterLayer, VectorLayer
from core.methods import OutputRaster, OutputVector

VECTOR_RESULT_EXT = ".shp"      # vector method results (training / test points …) as Shapefiles for ArcMap
from core.worker import Task
from gui import theme
from i18n import N_, tr
from gui.widgets import available_geometry, center_on, fit_to_screen
from gui.map_canvas import MapPanel
from gui.toc import TocPanel

from version import APP_NAME, VERSION

# project root: the repository (source run) or the folder that contains "MPM Studio.exe" (PyInstaller build)
ROOT = Path(sys.executable).resolve().parent if getattr(sys, "frozen", False) else Path(__file__).resolve().parents[2]
DEMO_PROJECTS = {
    N_("Carlin-type Au, Nevada (USGS)"): ROOT / "demo_data/nevada_carlin_au/processed/carlin_au.mpm.json",
    N_("REE, Curnamona, South Australia"): ROOT / "demo_data/curnamona_ree/processed/curnamona_ree.mpm.json",
}
log = logging.getLogger("mpm")
# menu titles of the method categories (core.methods.CATEGORIES) with their keyboard mnemonics
CATEGORY_MENUS = {"Knowledge-driven": N_("&Knowledge-driven"), "Data-driven": N_("Da&ta-driven"),
                  "Hybrid": N_("H&ybrid")}


class _LogEmitter(QObject):
    message = pyqtSignal(str, int)


class QtLogHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.emitter = _LogEmitter()

    def emit(self, record):
        self.emitter.message.emit(self.format(record), record.levelno)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"{APP_NAME} {VERSION}")
        self.setWindowIcon(QIcon(str(Path(__file__).resolve().parent / "resources" / "mpm_studio.ico")))
        self.store = LayerStore()
        self._dialogs = []
        self._tasks = []
        self.map_panel = MapPanel(self.store, self)
        self.canvas = self.map_panel.canvas
        self.setCentralWidget(self.map_panel)
        self.identify_dlg = None
        self.measure_dlg = None
        self._identify_placed = False
        self._build_docks()
        self._build_actions()
        self._build_menus()
        self._build_toolbar()
        self._build_statusbar()
        self._setup_logging()
        self.store.changed.connect(self._update_title)
        self.store.layer_changed.connect(lambda _l: self._update_title())
        self.canvas.identify_requested.connect(self.identify)
        self.canvas.mode_changed.connect(self._mode_changed)
        fit_to_screen(self, 1500, 920)          # never larger than the display (e.g. 1366 x 768 laptops)
        center_on(self)
        s = QSettings("MPM", APP_NAME)
        if s.value("geometry"):
            self.restoreGeometry(s.value("geometry"))
            self._keep_on_screen()               # geometry saved on a bigger monitor
        if s.value("state"):
            self.restoreState(s.value("state"))
        self._update_project_state()            # no project yet: only New / Open / demo / view / help
        log.info(tr("Ready. Create a project (Project ▸ New project…), open one (Project ▸ Open project…) "
                    "or start from a demo (Project ▸ Open demo project)."))

    # ================================================================== UI construction
    def _build_docks(self):
        self.toc = TocPanel(self.store)
        d = QDockWidget(tr("Layers"), self)
        d.setObjectName("dock_layers")
        d.setWidget(self.toc)
        d.setFeatures(QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable)
        self.addDockWidget(Qt.LeftDockWidgetArea, d)
        self.dock_layers = d
        t = self.toc.tree
        t.zoom_to_layer.connect(self.canvas.zoom_to_layer)
        t.properties_requested.connect(self.layer_properties)
        t.attributes_requested.connect(self.attribute_table)
        t.save_requested.connect(self.save_layer)
        t.export_legend_requested.connect(self.export_legend)

        self.bottom = QTabWidget()
        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(5000)
        self.bottom.addTab(self.log_view, theme.icon("mdi6.console-line"), tr("Log"))
        b = QDockWidget(tr("Output"), self)
        b.setObjectName("dock_output")
        b.setWidget(self.bottom)
        b.setFeatures(QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable | QDockWidget.DockWidgetClosable)
        self.addDockWidget(Qt.BottomDockWidgetArea, b)
        self.resizeDocks([d], [330], Qt.Horizontal)
        self.resizeDocks([b], [170], Qt.Vertical)
        self.dock_output = b

    def _act(self, text, slot, icon=None, shortcut=None, tip=None, checkable=False):
        a = QAction(theme.icon(icon) if icon else theme.icon("mdi6.circle-small"), text, self)
        if shortcut:
            a.setShortcut(shortcut)
        a.setToolTip(tip or text)
        a.setStatusTip(tip or text)
        a.setCheckable(checkable)
        if slot:
            a.triggered.connect(slot)
        return a

    def _build_actions(self):
        A = self._act
        self.a_new = A(tr("New project…"), self.new_project, "mdi6.folder-plus-outline", "Ctrl+N",
                       tr("Create a project in a folder"))
        self.a_open = A(tr("Open project…"), lambda: self.open_project(), "mdi6.folder-open-outline", "Ctrl+O",
                        tr("Open a project folder"))
        self.a_save = A(tr("Save project"), self.save_project, "mdi6.content-save-outline", "Ctrl+S",
                       tr("Save the project (in-memory layers are written to the project data folder)"))
        self.a_save_as = A(tr("Save project as…"), self.save_project_as, "mdi6.content-save-edit-outline",
                           tip=tr("Copy the project folder to another folder and continue there"))
        self.a_add_r = A(tr("Add raster…"), self.add_raster, "mdi6.grid-large", "Ctrl+R",
                         tr("Add raster layer (GeoTIFF, ArcInfo grid, …)"))
        self.a_add_v = A(tr("Add vector…"), self.add_vector, "mdi6.vector-polygon", "Ctrl+Shift+V",
                         tr("Add vector layer (GPKG, SHP, GeoJSON)"))
        self.a_proj_props = A(tr("Project properties…"), self.project_properties, "mdi6.folder-cog-outline",
                              "Ctrl+Shift+P", tr("Project location, extent, coordinate system and raster cell size"))
        self.a_export_map = A(tr("Export map image…"), self.export_map, "mdi6.image-outline")
        self.a_quit = A(tr("Exit"), self.close, "mdi6.exit-to-app", "Ctrl+Q")

        self.a_full = A(tr("Zoom to full extent"), lambda: self.canvas.zoom_full(), "mdi6.fit-to-screen-outline", "Home")
        self.a_zin_step = A(tr("Zoom in"), lambda: self.canvas.zoom(1.5), "mdi6.magnify-plus-outline", "Ctrl++")
        self.a_zout_step = A(tr("Zoom out"), lambda: self.canvas.zoom(1 / 1.5), "mdi6.magnify-minus-outline", "Ctrl+-")
        self.mode_group = QActionGroup(self)
        self.a_pan = A(tr("Pan"), lambda: self.canvas.set_mode("pan"), "mdi6.hand-back-right-outline", "P", checkable=True)
        self.a_measure = A(tr("Measure distance / area"), lambda: self.canvas.set_mode("measure"), "mdi6.ruler", "M",
                           tr("Measure distances, azimuths and areas on the map"), checkable=True)
        self.a_zbox = A(tr("Zoom to rectangle"), lambda: self.canvas.set_mode("zoom_in"), "mdi6.selection-search", "Z", checkable=True)
        self.a_ident = A(tr("Identify"), lambda: self.canvas.set_mode("identify"), "mdi6.information-outline", "I",
                         tr("Click the map to read values of all layers"), checkable=True)
        for a in (self.a_pan, self.a_measure, self.a_zbox, self.a_ident):
            self.mode_group.addAction(a)
        self.a_pan.setChecked(True)
        self.a_legend = A(tr("Show legend"), self._toggle_legend, "mdi6.format-list-bulleted-type", checkable=True)
        self.a_legend.setChecked(True)

        self.a_convert = A(tr("Convert to GeoTIFF / GeoPackage…"), self.tool_convert, "mdi6.swap-horizontal")
        self.a_reclass = A(tr("Reclassify raster…"), self.tool_reclass, "mdi6.chart-histogram")
        self.a_align = A(tr("Align raster to grid…"), self.tool_align, "mdi6.grid")
        self.a_clip = A(tr("Clip raster by mask…"), self.tool_clip, "mdi6.content-cut",
                        tip=tr("Keep the raster cells inside (or outside) the polygons of a mask layer"))
        self.a_poly2ras = A(tr("Polygon to raster…"), self.tool_polygon_to_raster, "mdi6.vector-square",
                            tip=tr("Polygons (e.g. a geological map) to a categorical raster with a class table"))
        self.a_dist = A(tr("Distance to features…"), self.tool_distance, "mdi6.map-marker-distance")
        self.a_validate = A(tr("ROC / AUC validation…"), self.tool_validate, "mdi6.chart-bell-curve-cumulative",
                            tip=tr("Validate any prospectivity raster against deposits (ROC, AUC, success-rate, P-A)"))
        self.a_targets = A(tr("Target delineation (P-A / C-A)…"), self.tool_targets, "mdi6.target",
                           tip=tr("Binary target map (1 / 0) from any raster: P-A intersection or C-A fractal threshold"))
        self.a_about = A(tr("About"), self.about, "mdi6.information-variant")
        self.a_settings = A(tr("General settings…"), self.general_settings, "mdi6.cog-outline", "Ctrl+,",
                            tr("Language and theme"))
        self.a_methods_doc = A(tr("Methods overview"), self.methods_doc, "mdi6.book-open-variant")

    def _build_menus(self):
        mb = self.menuBar()
        m = mb.addMenu(tr("&Project"))
        m.addActions([self.a_new, self.a_open])
        self.recent_menu = m.addMenu(theme.icon("mdi6.history"), tr("Recent projects"))
        self.recent_menu.aboutToShow.connect(self._fill_recent_menu)
        self._fill_recent_menu()
        demo = m.addMenu(theme.icon("mdi6.earth"), tr("Open demo project"))
        for name, path in DEMO_PROJECTS.items():
            demo.addAction(theme.icon("mdi6.map-outline"), tr(name), lambda _c=False, n=name, p=path: self.open_demo(n, p))
        demo.addSeparator()
        demo.addAction(theme.icon("mdi6.cog-refresh-outline"), tr("Rebuild demo data from raw files…"), self.rebuild_demo)
        m.addActions([self.a_save, self.a_save_as])
        m.addSeparator()
        m.addAction(self.a_proj_props)
        m.addSeparator()
        m.addAction(self.a_export_map)
        m.addSeparator()
        m.addAction(self.a_quit)

        m = mb.addMenu(tr("&View"))
        m.addActions([self.a_full, self.a_zin_step, self.a_zout_step])
        m.addSeparator()
        m.addActions([self.a_pan, self.a_measure, self.a_zbox, self.a_ident])
        m.addSeparator()
        m.addAction(self.a_legend)
        m.addAction(self.dock_layers.toggleViewAction())
        m.addAction(self.dock_output.toggleViewAction())

        m = mb.addMenu(tr("&Data"))
        m.addActions([self.a_add_r, self.a_add_v])
        m.addSeparator()
        m.addActions([self.a_convert, self.a_reclass, self.a_align, self.a_clip, self.a_poly2ras, self.a_dist])

        self.method_actions = {}
        for cat in reg.CATEGORIES:                      # Knowledge-driven, Data-driven, Hybrid
            mm = mb.addMenu(tr(CATEGORY_MENUS.get(cat, cat)))
            groups = defaultdict(list)
            for meth in reg.methods(cat):
                groups[meth.group].append(("impl", meth))
            for c, g, n in reg.PLANNED:
                if c == cat and not any(x.name == n for x in reg.methods(cat)):
                    groups[g].append(("planned", n))
            for g, items in groups.items():
                sub = mm.addMenu(tr(g)) if g else mm
                for kind, item in items:
                    if kind == "impl":
                        a = self._act(tr(item.name) + "…", lambda _c=False, mt=item: self.run_method(mt), item.icon,
                                      tip=tr(item.description))
                        self.method_actions[item.key] = a
                    else:
                        a = QAction(tr("{name}  (planned)", name=tr(item)), self)
                        a.setEnabled(False)
                    sub.addAction(a)

        m = mb.addMenu(tr("V&alidation"))
        m.addAction(self.a_validate)
        m.addAction(self.a_targets)
        m = mb.addMenu(tr("&Settings"))
        m.addAction(self.a_settings)
        m = mb.addMenu(tr("&Help"))
        m.addActions([self.a_methods_doc, self.a_about])

    def _build_toolbar(self):
        tb = QToolBar(tr("Main toolbar"), self)
        tb.setObjectName("tb_main")
        tb.setMovable(False)
        tb.setToolButtonStyle(Qt.ToolButtonIconOnly)
        tb.addActions([self.a_open, self.a_save])
        tb.addSeparator()
        tb.addActions([self.a_add_r, self.a_add_v])
        tb.addSeparator()
        tb.addActions([self.a_pan, self.a_measure, self.a_zbox, self.a_zin_step, self.a_zout_step, self.a_full,
                       self.a_ident])
        tb.addSeparator()
        tb.addActions([self.a_reclass, self.a_align, self.a_clip, self.a_dist])
        tb.addSeparator()
        if "wofe" in self.method_actions:
            tb.addAction(self.method_actions["wofe"])
        tb.addAction(self.a_validate)
        tb.addAction(self.a_targets)
        self.addToolBar(Qt.TopToolBarArea, tb)
        tb.actionTriggered.connect(self._on_command)          # any other command ends measuring
        self.menuBar().triggered.connect(self._on_command)

    def _build_statusbar(self):
        sb = self.statusBar()
        sb.setFixedHeight(24)              # constant height: progress bar / messages must not resize the map
        self.msg_label = QLabel("")
        self.progress = QProgressBar()
        self.progress.setFixedWidth(180)
        self.progress.setVisible(False)
        sb.addWidget(self.msg_label, 1)
        sb.addPermanentWidget(self.progress)

    def _setup_logging(self):
        h = QtLogHandler()
        h.setFormatter(logging.Formatter("%(asctime)s  %(levelname)-7s %(message)s", "%H:%M:%S"))
        h.emitter.message.connect(self._append_log)
        log.addHandler(h)
        log.setLevel(logging.INFO)
        self._log_handler = h

    def _append_log(self, text, level):
        safe = text.replace("&", "&amp;").replace("<", "&lt;")
        if level >= logging.WARNING:
            col = theme.BAD if level >= logging.ERROR else theme.WARN
            self.log_view.appendHtml(f"<span style='color:{col}; white-space:pre'>{safe}</span>")
        else:                                   # palette text colour: stays readable after a theme switch
            self.log_view.appendHtml(f"<span style='white-space:pre'>{safe}</span>")

    # ================================================================== helpers
    def keep_dialog(self, dlg):
        self._dialogs = [d for d in self._dialogs if d.isVisible()] + [dlg]

    def _update_title(self):
        folder = project.project_dir(self.store)
        title = f"{APP_NAME} {VERSION}" + (f" — {os.path.basename(folder)}  ({folder})[*]" if folder else "")
        self.setWindowTitle(title)
        self.setWindowModified(bool(folder) and self.store.dirty)

    def _toggle_legend(self, on):
        self.canvas.show_legend = on
        self.canvas.update_legend()

    def busy(self, on: bool, msg: str = ""):
        if on:
            QApplication.setOverrideCursor(Qt.WaitCursor)
            self.msg_label.setText(msg)
        else:
            QApplication.restoreOverrideCursor()
            self.msg_label.setText("")
        QApplication.processEvents()

    def error(self, title, exc, tb=None):
        log.error("%s: %s", title, exc)
        if tb:
            log.debug(tb)
        QMessageBox.critical(self, title, str(exc))

    def run_task(self, fn, on_done, title: str, *args, **kw):
        """Run ``fn(*args, progress=..., **kw)`` in a worker thread with a status-bar progress bar."""
        task = Task(fn, *args, parent=self, **kw)
        self.progress.setVisible(True)
        self.progress.setValue(0)
        self.msg_label.setText(title)

        def progressed(p, m):
            self.progress.setValue(p)
            if m:
                self.msg_label.setText(f"{title}: {m}")

        def finished():
            self.progress.setVisible(False)
            self.msg_label.setText("")
            if task in self._tasks:
                self._tasks.remove(task)

        def ok(res):
            finished()
            try:
                on_done(res)
            except Exception as exc:  # noqa: BLE001
                import traceback
                self.error(title, exc, traceback.format_exc())

        def fail(msg, tb):
            finished()
            log.debug(tb)
            self.error(title, msg)

        task.progressed.connect(progressed)
        task.succeeded.connect(ok)
        task.failed.connect(fail)
        self._tasks.append(task)
        task.start()
        return task

    # ================================================================== project (folder based)
    def _choose_project_folder(self, title: str, start: str = "") -> Optional[str]:
        """Folder picker; a non-empty folder needs confirmation ("Choose another folder" / "Continue")."""
        while True:
            folder = QFileDialog.getExistingDirectory(self, title, start)
            if not folder:
                return None
            if project.is_empty_folder(folder):
                return folder
            box = QMessageBox(self)
            box.setIcon(QMessageBox.Warning)
            box.setWindowTitle(title)
            box.setText(tr("The selected folder is not empty."))
            projs = [os.path.basename(f) for f in project.project_files(folder)]
            box.setInformativeText(f"{folder}\n\n" + (tr("Its project file(s) {files} will be replaced.",
                                                          files=", ".join(projs)) + " " if projs else "")
                                   + tr("Existing files with the same names may be overwritten."))
            choose = box.addButton(tr("Choose another folder"), QMessageBox.RejectRole)
            cont = box.addButton(tr("Continue"), QMessageBox.AcceptRole)
            box.setDefaultButton(choose)
            box.setEscapeButton(choose)
            box.exec_()
            if box.clickedButton() is cont:
                return folder
            start = folder

    def _maybe_save(self) -> bool:
        """Ask to save unsaved changes. False = the user cancelled."""
        if not self.store.project_path or not self.store.dirty:
            return True
        r = QMessageBox.question(self, APP_NAME, tr("The project has unsaved changes. Save them?"),
                                 QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel, QMessageBox.Save)
        if r == QMessageBox.Cancel:
            return False
        if r == QMessageBox.Save:
            return self.save_project()
        return True

    # ------------------------------------------------------------------ recent projects
    def _remember_project(self):
        from gui import app_settings
        folder = project.project_dir(self.store)
        if folder:
            app_settings.add_recent_project(folder)

    def _fill_recent_menu(self):
        """Project ▸ Recent projects: the last 5 project folders (rebuilt each time the menu opens)."""
        from gui import app_settings
        m = self.recent_menu
        m.clear()
        items = app_settings.recent_projects()
        fm = m.fontMetrics()
        for i, folder in enumerate(items, start=1):
            pfile = project.find_project_file(folder)
            exists = pfile is not None
            # name: the project file name (carlin_au.mpm.json -> carlin_au), the folder name for project.mpm.json
            stem = os.path.basename(pfile)[:-len(".mpm.json")] if exists else ""
            name = stem if stem and stem != "project" else (os.path.basename(folder.rstrip("\\/")) or folder)
            name = fm.elidedText(name, Qt.ElideRight, 220).replace("&", "&&")
            shown = fm.elidedText(folder, Qt.ElideMiddle, 420).replace("&", "&&")
            if not exists:
                shown += "  " + tr("(not found)")
            # trailing spaces: the styled menu leaves no gap between the name and the path column
            a = m.addAction(f"&{i}  {name}{' ' * 8}\t{shown}", lambda _c=False, f=folder: self.open_recent(f))
            a.setToolTip(folder)
            a.setStatusTip(folder)
            a.setEnabled(exists)
        if not items:
            m.addAction(tr("(no recent projects)")).setEnabled(False)
        m.addSeparator()
        clear = m.addAction(theme.icon("mdi6.delete-sweep-outline"), tr("Clear recent projects"), self.clear_recent)
        clear.setEnabled(bool(items))
        m.setToolTipsVisible(True)

    def open_recent(self, folder: str):
        from gui import app_settings
        if project.find_project_file(folder) is None:
            QMessageBox.warning(self, tr("Recent projects"), tr("The project folder {path} no longer contains a "
                                                                 "project. It is removed from the list.", path=folder))
            app_settings.remove_recent_project(folder)
            return
        if not self._maybe_save():
            return
        self.open_project(folder)

    def clear_recent(self):
        from gui import app_settings
        app_settings.clear_recent_projects()
        self._fill_recent_menu()

    def _project_actions(self):
        return [self.a_save, self.a_save_as, self.a_proj_props, self.a_add_r, self.a_add_v, self.a_export_map, self.a_convert,
                self.a_reclass, self.a_align, self.a_clip, self.a_poly2ras, self.a_dist, self.a_validate, self.a_targets] + list(self.method_actions.values())

    def _update_project_state(self):
        """Everything except New / Open / demo / view / help needs an open project."""
        has = self.store.project_path is not None
        for act in self._project_actions():
            act.setEnabled(has)
        self._update_title()

    def new_project(self):
        if not self._maybe_save():
            return
        folder = self._choose_project_folder(tr("New project — choose a project folder"))
        if not folder:
            return
        try:
            path = project.create_project(self.store, folder)
        except Exception as exc:  # noqa: BLE001
            return self.error(tr("New project"), exc)
        self._update_project_state()
        self._remember_project()
        log.info(tr("Created project {path}", path=path))
        self.project_properties("crs")          # coordinate system / cell size can be set before data is loaded

    def open_project(self, path: str = None):
        """Open a project folder (or a *.mpm.json file, e.g. given on the command line)."""
        if not path:
            if not self._maybe_save():
                return
            path = QFileDialog.getExistingDirectory(self, tr("Open project — choose a project folder"),
                                                    project.project_dir(self.store) or "")
            if not path:
                return
        try:
            project.resolve_project_file(path)          # exactly one *.mpm.json in the project folder
        except project.ProjectFileError as exc:
            QMessageBox.warning(self, tr("Open project"), str(exc))
            return
        self.busy(True, tr("Loading {name}…", name=os.path.basename(path)))
        try:
            errors = project.open_project(self.store, path)
        except Exception as exc:  # noqa: BLE001
            self.busy(False)
            return self.error(tr("Open project"), exc)
        self.busy(False)
        self.store.prefetch()                           # visible rasters: read in the background, drawn on arrival
        self.canvas.zoom_full()
        self._update_project_state()
        self._remember_project()
        log.info(tr("Opened project {path} ({n} layers)", path=self.store.project_path, n=len(self.store.layers)))
        for e in errors:
            log.warning(tr("could not load {what}", what=e))

    def open_demo(self, name: str, src_json: Path):
        """Copy a demo project into a new project folder and open it (the demo data stays untouched)."""
        if not src_json.is_file():
            if QMessageBox.question(self, tr("Demo data"), f"{src_json}\n\n" + tr(
                    "does not exist yet. Build the demo data now (about 1 minute)?")) == QMessageBox.Yes:
                self.rebuild_demo(then=lambda: self.open_demo(name, src_json))
            return
        if not self._maybe_save():
            return
        folder = self._choose_project_folder(tr("Demo project '{name}' — choose a folder for the new project", name=tr(name)))
        if not folder:
            return
        self.busy(True, tr("Copying demo project…"))
        try:
            path, dropped = project.copy_project(str(src_json), folder)
        except Exception as exc:  # noqa: BLE001
            self.busy(False)
            return self.error(tr("Demo project"), exc)
        self.busy(False)
        for d in dropped:
            log.warning(tr("demo layer {name} skipped (file not found)", name=d))
        self.open_project(path)

    def save_project(self) -> bool:
        """Save into the project folder: in-memory layers -> data/, then the project file (no file dialog)."""
        if not self.store.project_path:
            return False
        try:
            written = project.save_project(self.store)
        except Exception as exc:  # noqa: BLE001
            self.error(tr("Save project"), exc)
            return False
        self.toc.tree.rebuild()
        self._update_title()
        for w in written:
            log.info(tr("Saved layer {name}", name=w))
        log.info(tr("Saved project {path}", path=self.store.project_path))
        return True

    def save_project_as(self) -> bool:
        """Copy the whole project folder to another folder and continue working there."""
        if not self.store.project_path:
            return False
        folder = self._choose_project_folder(tr("Save project as — choose the new project folder"))
        if not folder:
            return False
        self.busy(True, tr("Copying project…"))
        try:
            project.save_project_as(self.store, folder)
        except Exception as exc:  # noqa: BLE001
            self.busy(False)
            self.error(tr("Save project as"), exc)
            return False
        self.busy(False)
        self.toc.tree.rebuild()
        self._update_title()
        self._remember_project()
        log.info(tr("Project copied to {path}", path=self.store.project_path))
        return True

    def add_data_file(self, kind: str, path: str, batch: Optional[dict] = None, remaining: int = 0):
        """Load a raster / vector file into the project and return the added layer (None = not loaded).

        ``batch`` / ``remaining`` (several files in one go) are kept for callers; no question is asked any more.

        * rasters are loaded exactly as stored — no check of coordinate system, cell size or extent against the
          project (a different CRS is only noted in the log); methods resample their inputs to a common grid.
        * vectors in another CRS are reprojected to the project CRS (lossless).
        * a missing / unrecognised CRS (or one that cannot be transformed) is taken to be the project CRS (warning).
        * unset project settings are taken from the first layer / first raster (Project ▸ Project properties).
        """
        from core.layers import conform_vector_crs, describe_crs, format_length
        from utils.geo_env import unrecognized
        s = self.store
        crs0, cell0, ext0 = s.crs, s.cell_size, s.extent
        name = os.path.basename(path.rstrip("\\/"))
        if kind == "vector":
            lyr = VectorLayer.from_file(path)
            if s.crs is not None:
                src = None if unrecognized(lyr.crs) else describe_crs(lyr.crs)[0]
                gdf, moved = conform_vector_crs(lyr.gdf, s.crs, lyr.name)  # reprojected, or unrecognised -> project CRS
                if moved:
                    log.info(tr("{name}: reprojected from {src} to the project coordinate system {dst}", name=lyr.name,
                                src=src, dst=describe_crs(s.crs)[0]))
                lyr = VectorLayer(lyr.name, gdf, source=path)
            s.add(lyr)
            log.info(tr("Added vector {path} ({n} features)", path=path, n=len(lyr.gdf)))
        else:
            lyr = RasterLayer.from_file(path, assume_crs=s.crs)    # as stored; unrecognised CRS -> project CRS
            src_crs = lyr.grid.crs
            if s.crs is not None and src_crs is not None and src_crs != s.crs:
                log.warning(tr("{name}: coordinate system {src} differs from the project ({dst}) — loaded as stored",
                               name=name, src=describe_crs(src_crs)[0], dst=describe_crs(s.crs)[0]))
            s.adopt_settings(lyr)
            s.add(lyr)
            log.info(tr("Added raster {path}", path=path))
        if crs0 is None and s.crs is not None:
            log.info(tr("Project coordinate system set from {name}: {crs}", name=lyr.name, crs=describe_crs(s.crs)[1]))
        if cell0 is None and s.cell_size is not None:
            log.info(tr("Project raster cell size set from {name}: {cell}", name=lyr.name,
                        cell=format_length(s.cell_size, s.crs)))
        if ext0 is None and s.extent is not None:
            log.info(tr("Project extent set from {name}", name=lyr.name))
        return lyr

    def project_properties(self, tab: str = None):
        from gui.dialogs.project_dialog import ProjectPropertiesDialog
        from core.layers import describe_crs, format_length
        if not self.store.project_path:
            return
        dlg = ProjectPropertiesDialog(self.store, self, tab)
        if dlg.exec_() and dlg.apply():
            s = self.store
            log.info(tr("Project coordinate system: {crs}; raster cell size: {cell}; NoData value: {nodata:g}",
                        crs=describe_crs(s.crs)[1] if s.crs is not None else tr("not set"),
                        cell=format_length(s.cell_size, s.crs) if s.cell_size else tr("not set"), nodata=s.nodata))
            self._update_title()

    def add_raster(self):
        paths, _ = QFileDialog.getOpenFileNames(self, tr("Add raster"), project.project_dir(self.store) or "",
                                                tr("Rasters") + " (*.tif *.tiff *.img *.asc *.vrt *.grd *.nc hdr.adf);;"
                                                + tr("All files") + " (*)")
        batch, n0 = {}, len(self.store.layers)
        for i, p in enumerate(paths):
            if os.path.basename(p).lower() == "hdr.adf":
                p = os.path.dirname(p)
            try:
                self.add_data_file("raster", p, batch, remaining=len(paths) - i - 1)
            except Exception as exc:  # noqa: BLE001
                self.error(tr("Add raster"), exc)
        if n0 == 0 and self.store.layers:
            self.canvas.zoom_full()

    def add_vector(self):
        paths, _ = QFileDialog.getOpenFileNames(self, tr("Add vector"), project.project_dir(self.store) or "",
                                                tr("Vectors") + " (*.gpkg *.shp *.geojson *.json *.kml);;"
                                                + tr("All files") + " (*)")
        batch, n0 = {}, len(self.store.layers)
        for i, p in enumerate(paths):
            try:
                self.add_data_file("vector", p, batch, remaining=len(paths) - i - 1)
            except Exception as exc:  # noqa: BLE001
                self.error(tr("Add vector"), exc)
        if n0 == 0 and self.store.layers:
            self.canvas.zoom_full()

    def save_layer(self, layer):
        start = os.path.join(project.project_dir(self.store) or "", project.DATA_DIR, project.safe_name(layer.name))
        if isinstance(layer, RasterLayer):
            p, _ = QFileDialog.getSaveFileName(self, tr("Save raster"), start + ".tif", "GeoTIFF (*.tif)")
        else:
            filters = {"ESRI Shapefile (*.shp)": ".shp", "GeoPackage (*.gpkg)": ".gpkg"}
            last = getattr(self, "_vector_filter", "ESRI Shapefile (*.shp)")
            p, chosen = QFileDialog.getSaveFileName(self, tr("Save vector"), start + filters[last],
                                                    ";;".join(filters), last)
            if p:
                ext = filters.get(chosen, ".shp")
                if os.path.splitext(p)[1].lower() not in (".shp", ".gpkg"):
                    p += ext                                         # the format chosen in the dialog
                self._vector_filter = chosen if chosen in filters else last
        if p:
            layer.save(p)
            self.store.notify(layer)
            self.toc.tree.rebuild()
            log.info(tr("Saved {name} -> {path}", name=layer.name, path=p))

    def export_legend(self, layer, path: Optional[str] = None) -> Optional[str]:
        """Legend of a layer as a 300 dpi PNG with 10 pt text (true size in Word at 100 %)."""
        from gui.map_canvas import export_legend
        if path is None:
            folder = project.project_dir(self.store)
            start = os.path.join(folder, "results") if folder and os.path.isdir(os.path.join(folder, "results")) \
                else (folder or "")
            safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in layer.name)
            path, _ = QFileDialog.getSaveFileName(self, tr("Export legend to PNG"),
                                                  os.path.join(start, f"{safe}_legend.png"), "PNG (*.png)")
            if not path:
                return None
        if not path.lower().endswith(".png"):
            path += ".png"
        try:
            export_legend(layer, path, dpi=300, fontsize=10)
        except Exception as exc:  # noqa: BLE001
            self.error(tr("Export legend to PNG"), exc)
            return None
        log.info(tr("Legend of {name} exported to {path} (300 dpi, 10 pt text)", name=layer.name, path=path))
        return path

    def export_map(self):
        p, _ = QFileDialog.getSaveFileName(self, tr("Export map image"),
                                           os.path.join(project.project_dir(self.store) or "", "map.png"),
                                           "PNG (*.png);;PDF (*.pdf);;SVG (*.svg)")
        if p:
            self.canvas.export_image(p)
            log.info(tr("Map exported to {path}", path=p))

    def rebuild_demo(self, then=None):
        from utils import prepare_demo_data as pdd
        raw = [ROOT / "demo_data/nevada_carlin_au/raw", ROOT / "demo_data/curnamona_ree/raw"]
        if not all(p.is_dir() for p in raw):
            QMessageBox.information(self, tr("Demo data"), tr("The raw demo data is not included in this installation, "
                                    "so the demo layers cannot be rebuilt.") + "\n\n" + tr("Expected:") + "\n"
                                    + "\n".join(map(str, raw)))
            return
        pdd.DEMO = ROOT / "demo_data"

        def job(progress):
            progress(10, "Carlin")
            pdd.prepare_carlin(pdd.DEMO)
            progress(60, "Curnamona")
            pdd.prepare_curnamona(pdd.DEMO)
            return True

        def done(_r):
            log.info(tr("Demo data rebuilt."))
            if then:
                then()
        self.run_task(job, done, tr("Building demo data"))

    # ================================================================== layer dialogs
    def layer_properties(self, layer):
        from gui.dialogs.layer_dialogs import LayerPropertiesDialog
        dlg = LayerPropertiesDialog(layer, self.store, self)
        if dlg.exec_():
            dlg.apply()

    def attribute_table(self, layer):
        from gui.dialogs.layer_dialogs import AttributeTableDialog
        dlg = AttributeTableDialog(layer, self)
        dlg.show()
        self.keep_dialog(dlg)

    def add_output_raster(self, out: OutputRaster):
        """Add a method output. During a project run it is written to results/<run_id>/<name>.tif first."""
        run = getattr(self, "_run", None)
        name = f"{run[0]}_{out.name}" if run else out.name
        lyr = RasterLayer(name, out.data, out.grid, categorical=out.categorical, cmap=out.cmap,
                          nodata=self.store.nodata)              # the project NoData value
        lyr.description = out.description
        lyr.visible = out.visible
        if run:
            lyr.save(os.path.join(run[1], project.safe_name(out.name) + ".tif"))
        self.store.add(lyr)
        log.info(tr("Added result layer {name}", name=lyr.name) + (f" ({lyr.source})" if lyr.source else " " + tr("(in memory)")))
        return lyr

    def add_output_vector(self, out: OutputVector):
        """Add a vector method output (e.g. training / test points); during a project run it is written to
        results/<run_id>/<name>.gpkg first."""
        run = getattr(self, "_run", None)
        name = f"{run[0]}_{out.name}" if run else out.name
        lyr = VectorLayer(name, out.gdf, style=out.style or None)
        lyr.description = out.description
        lyr.visible = out.visible
        if run and len(out.gdf):                                    # Shapefile: results are shared with ArcMap
            lyr.save(os.path.join(run[1], project.safe_name(out.name) + VECTOR_RESULT_EXT))
        self.store.add(lyr)
        log.info(tr("Added result layer {name}", name=lyr.name) + (f" ({lyr.source})" if lyr.source else " " + tr("(in memory)")))
        return lyr

    # ================================================================== data tools
    def _run_dialog_tool(self, dlg_cls, title):
        if dlg_cls.__name__ != "ConvertDialog" and not self.store.layers:
            QMessageBox.information(self, title, tr("Add some layers first."))
            return
        dlg = dlg_cls(self.store, self)
        if not dlg.exec_():
            return
        self.busy(True, title + "…")
        try:
            out = dlg.run()
        except Exception as exc:  # noqa: BLE001
            self.busy(False)
            return self.error(title, exc)
        self.busy(False)
        return dlg, out

    def tool_convert(self):
        from gui.dialogs.data_dialogs import ConvertDialog
        r = self._run_dialog_tool(ConvertDialog, tr("Convert"))
        if not r:
            return
        dlg, (kind, path) = r
        log.info(tr("Converted -> {path}", path=path))
        if dlg.add.isChecked():
            n0 = len(self.store.layers)
            self.add_data_file(kind, path)          # checked against the project CRS / cell size
            if n0 == 0 and self.store.layers:
                self.canvas.zoom_full()

    def tool_reclass(self):
        from gui.dialogs.data_dialogs import ReclassDialog
        r = self._run_dialog_tool(ReclassDialog, tr("Reclassify"))
        if r:
            self.store.add(r[1])
            log.info("Created %s", r[1].name)

    def tool_align(self):
        from gui.dialogs.data_dialogs import AlignDialog
        r = self._run_dialog_tool(AlignDialog, tr("Align raster"))
        if r:
            self.store.add(r[1])
            log.info("Created %s", r[1].name)

    def tool_clip(self):
        from gui.dialogs.data_dialogs import ClipDialog
        title = tr("Clip raster by mask")
        if not self.store.vectors("polygon"):
            QMessageBox.information(self, title, tr("Add a polygon layer to use as the mask first (Data ▸ Add vector…)."))
            return
        r = self._run_dialog_tool(ClipDialog, title)
        if not r:
            return
        dlg, lyr = r
        if dlg.add.isChecked():
            self.store.add(lyr)
        log.info(tr("Clipped {src} -> {path}", src=dlg.src.layer().name, path=lyr.source))

    def tool_polygon_to_raster(self):
        from gui.dialogs.data_dialogs import PolygonToRasterDialog
        title = tr("Polygon to raster")
        if not self.store.vectors("polygon"):
            QMessageBox.information(self, title, tr("Add a polygon layer first (Data ▸ Add vector…)."))
            return
        r = self._run_dialog_tool(PolygonToRasterDialog, title)
        if not r:
            return
        dlg, lyr = r
        if dlg.add.isChecked():
            self.store.add(lyr)
        res = dlg.result_info
        log.info(tr("Rasterised {src} -> {path} ({k} classes)", src=dlg.src.layer().name, path=lyr.source,
                    k=len(res.table)))
        if res.empty:
            log.warning(tr("Classes without cells (polygons smaller than a cell or outside the grid): {names} — "
                           "use the maximum-area rule or a smaller cell size", names=", ".join(res.empty[:20])))
        if res.skipped:
            log.warning(tr("{n} features without a label or polygon geometry were skipped", n=res.skipped))

    def tool_distance(self):
        from gui.dialogs.data_dialogs import DistanceDialog
        r = self._run_dialog_tool(DistanceDialog, tr("Distance to features"))
        if r:
            self.store.add(r[1])
            log.info("Created %s", r[1].name)

    def tool_targets(self):
        from gui.dialogs.target_dialog import TargetDialog
        if not self.store.rasters():
            QMessageBox.information(self, tr("Target delineation"), tr("Add a raster first."))
            return
        dlg = TargetDialog(self.store, self)
        if not dlg.exec_():
            return
        try:
            self.add_target_layer(dlg.output())
        except Exception as exc:  # noqa: BLE001
            self.error(tr("Target delineation"), exc)

    def add_target_layer(self, out: dict):
        """Binary target layer (1 = red, 0 = transparent), written to results/<label>_<time>/target.tif."""
        import pandas as pd
        from evaluation.targets import target_class_table
        run = project.new_run(self.store, out["label"]) if self.store.project_path else None
        name = run[0] if run else out["label"]
        lyr = RasterLayer(name, out["binary"], out["grid"], categorical=True, class_table=target_class_table(),
                          nodata=255)
        method = "P-A intersection" if out["method"] == "pa" else "C-A fractal breakpoint"
        lyr.description = f"Target map of {out['source']}: value >= {out['threshold']:.6g} ({method})"
        if run:
            folder = run[1]
            lyr.save(os.path.join(folder, "target.tif"))
            out["figure"].savefig(os.path.join(folder, "plot.png"), dpi=150)
            st = out["stats"]
            rows = [("source raster", out["source"]), ("method", method), ("threshold", out["threshold"]),
                    ("automatic threshold", out["automatic"])]
            rows += [(k, v) for k, v in st.items() if k != "threshold"]
            if out["breakpoints"]:
                rows.append(("C-A breakpoints", ", ".join(f"{b:.6g}" for b in out["breakpoints"])))
            if out["pa_intersection"]:
                rows += [(f"P-A intersection {k}", v) for k, v in out["pa_intersection"].items()]
            pd.DataFrame(rows, columns=["Index", "Value"]).to_csv(os.path.join(folder, "target_report.csv"), index=False)
        self.store.add(lyr)
        if run:
            self.store.save_project(self.store.project_path, name=os.path.basename(project.project_dir(self.store)))
            self.store.dirty = bool(project.unsaved_layers(self.store))
            self._update_title()
        st = out["stats"]
        log.info(tr("Target map {name}: threshold {t:.6g}, {pct:.2f} % of the area", name=lyr.name,
                    t=out["threshold"], pct=st["target_area_pct"])
                 + (tr(", {k}/{n} deposits", k=st["deposits_in_target"], n=st["deposits"]) if "deposits" in st else ""))
        return lyr

    def tool_validate(self):
        from gui.dialogs.evaluation_dialogs import EvaluationDialog, ValidationDialog
        r = self._run_dialog_tool(ValidationDialog, tr("Validation"))
        if r:
            dlg = EvaluationDialog(r[1], r[0].score.layer().name, self)
            dlg.show()
            self.keep_dialog(dlg)

    # ================================================================== methods
    def run_method(self, method):
        if not self.store.rasters():
            QMessageBox.information(self, tr(method.name), tr("Add evidence rasters first (e.g. Project ▸ Open demo project)."))
            return
        try:
            dlg = method.create_dialog(self, self.store)
        except Exception as exc:  # noqa: BLE001
            return self.error(tr(method.name), exc)
        if not dlg.exec_():
            return
        params = dlg.params()
        if isinstance(params, dict):                    # every method can use the project NoData value
            params.setdefault("nodata", self.store.nodata)
        log.info(tr("Running {name} …", name=tr(method.name)))

        self.run_task(method.run, lambda result: self.method_finished(method, result), tr(method.name), params)

    def method_finished(self, method, result):
        """GUI-thread end of a method run: run folder, tables, result display, project file update."""
        for m in result.messages:
            log.warning("%s: %s", tr(method.name), m)
        run = None
        if self.store.project_path:
            run = project.new_run(self.store, result.label or method.key)
            self._write_run_files(method, result, run)
            if result.evaluation is not None:
                result.evaluation.score_name = f"{run[0]}_{result.evaluation.score_name}"
        if result.evaluation is not None:
            c = result.evaluation.curves
            log.info(tr("{name} finished — AUC training {auc:.3f}", name=tr(method.name), auc=c["training"].auc)
                     + (tr(", test {auc:.3f}", auc=c["test"].auc) if "test" in c else ""))
        self._run = run
        try:
            method.show_result(self, result)
        finally:
            self._run = None
        if run:
            self.store.save_project(self.store.project_path, name=os.path.basename(project.project_dir(self.store)))
            self.store.dirty = bool(project.unsaved_layers(self.store))
            self._update_title()
            log.info(tr("Results saved in {path}", path=run[1]))
        return run

    @staticmethod
    def _write_run_files(method, result, run):
        import datetime
        import pandas as pd
        run_id, folder = run
        for name, df in result.tables.items():
            if isinstance(df, pd.DataFrame):
                df.to_csv(os.path.join(folder, project.safe_name(name) + ".csv"), index=False)
        if result.evaluation is not None:
            pd.DataFrame(result.evaluation.summary_rows()).to_csv(os.path.join(folder, "validation.csv"), index=False)
        lines = [f"run:     {run_id}", f"method:  {method.name} ({method.key}, {method.category})",
                 f"date:    {datetime.datetime.now():%Y-%m-%d %H:%M:%S}",
                 "outputs: " + ", ".join([f"{project.safe_name(o.name)}.tif" for o in result.outputs]
                                         + [f"{project.safe_name(v.name)}{VECTOR_RESULT_EXT}" for v in result.vectors
                                            if len(v.gdf)])]
        lines += [f"note:    {m}" for m in result.messages]
        with open(os.path.join(folder, "run_info.txt"), "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines) + "\n")

    # ================================================================== identify
    def identify(self, x, y):
        self._ensure_identify_dialog()
        if not self.identify_dlg.isVisible():
            self.identify_dlg.show()
            if not self._identify_placed:
                self._place_identify_dialog()
        self.identify_dlg.identify(x, y)

    def _ensure_identify_dialog(self):
        if self.identify_dlg is None:
            from gui.dialogs.identify_dialog import IdentifyDialog
            self.identify_dlg = IdentifyDialog(self.store, self.canvas, self)
            self.identify_dlg.closed.connect(self._identify_closed)
            self._identify_placed = False

    IDENTIFY_RIGHT_GAP = 80                    # px between the identify dialog and the right edge of the display

    def _place_identify_dialog(self):
        self._place_tool_dialog(self.identify_dlg)
        self._identify_placed = True

    def _place_tool_dialog(self, dlg):
        """Right side of the display, right border IDENTIFY_RIGHT_GAP px from the screen edge, top at the map."""
        avail = available_geometry(self)
        fg = dlg.frameGeometry()                      # includes the title bar / frame (valid once shown)
        x = avail.right() + 1 - self.IDENTIFY_RIGHT_GAP - fg.width()
        y = self.map_panel.mapToGlobal(self.map_panel.rect().topLeft()).y() + 16
        y = max(avail.top(), min(y, avail.bottom() + 1 - fg.height()))
        dlg.move(max(avail.left(), x), y)

    def _mode_changed(self, mode):
        """The identify dialog lives as long as the identify tool is active."""
        if mode == "identify":
            self._ensure_identify_dialog()
            self.identify_dlg.show()
            if not self._identify_placed:             # first time: place it; afterwards keep where the user put it
                self._place_identify_dialog()
            self.identify_dlg.raise_()
        elif self.identify_dlg is not None and self.identify_dlg.isVisible():
            self.identify_dlg.hide()
        if mode == "measure":
            if self.measure_dlg is None:
                from gui.dialogs.measure_dialog import MeasureDialog
                self.measure_dlg = MeasureDialog(self.store, self.canvas, self)
                self.measure_dlg.closed.connect(self._measure_closed)
                self.measure_dlg.show()
                self._place_tool_dialog(self.measure_dlg)
            self.measure_dlg.show()
            self.measure_dlg.raise_()
        elif self.measure_dlg is not None and self.measure_dlg.isVisible():
            self.measure_dlg.hide()                          # hide -> temporary lines / shapes are cleared

    def _measure_closed(self):
        if self.canvas.mode == "measure":
            self.a_pan.setChecked(True)
            self.canvas.set_mode("pan")

    def _on_command(self, action):
        """A toolbar / menu command other than the map tools closes the measure tool (and clears its graphics)."""
        if self.canvas.mode == "measure" and action not in self.mode_group.actions():
            self.a_pan.setChecked(True)
            self.canvas.set_mode("pan")

    def _identify_closed(self):
        if self.canvas.mode == "identify":
            self.a_pan.setChecked(True)
            self.canvas.set_mode("pan")

    # ================================================================== help
    def methods_doc(self):
        p = ROOT / "docs" / "methods_survey.md"
        from PyQt5.QtWidgets import QDialog, QTextBrowser, QVBoxLayout
        dlg = QDialog(self)
        dlg.setWindowTitle(tr("Methods overview"))
        tb = QTextBrowser()
        tb.setMarkdown(p.read_text(encoding="utf-8") if p.exists() else tr("{file} not found", file="docs/methods_survey.md"))
        lay = QVBoxLayout(dlg)
        lay.addWidget(tb)
        fit_to_screen(dlg, 1000, 700)
        dlg.show()
        self.keep_dialog(dlg)

    def about(self):
        impl = ", ".join(tr(m.name) for m in reg.methods()) or tr("none")
        box = QMessageBox(self)
        box.setWindowTitle(tr("About {app}", app=APP_NAME))
        box.setIconPixmap(QPixmap(str(Path(__file__).resolve().parent / "resources" / "mpm_studio.png")).scaled(
            96, 96, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        box.setText(f"<b style='font-size:13pt'>{APP_NAME}</b>&nbsp;&nbsp;" + tr("version {v}", v=VERSION) + "<br>"
                    + tr("2D mineral prospectivity mapping"))
        box.setInformativeText(tr("Implemented methods: {impl}", impl=impl) + "<br>"
                               + tr("Planned: {n} more (Help ▸ Methods overview).", n=len(reg.PLANNED))
                               + "<br><br>Python 3.10 · PyQt5 · matplotlib · rasterio · geopandas · scikit-learn")
        box.exec_()

    # ================================================================== settings
    def general_settings(self, page: str = None):
        from gui.dialogs.settings_dialog import GeneralSettingsDialog
        dlg = GeneralSettingsDialog(self)
        if page:
            dlg.show_page(page)
        dlg.theme_applied.connect(self.apply_theme)
        dlg.language_applied.connect(self._language_applied)
        dlg.exec_()

    def apply_theme(self, mode: str):
        """Re-style the running application (palette, stylesheet, icons, map, charts opened from now on)."""
        theme.apply(QApplication.instance(), mode)
        self.map_panel.theme_changed()
        self.toc.info.show_layer(self.store.active)      # HTML with theme colours
        self.canvas.redraw()
        log.info(tr("Theme: {mode}", mode=tr("Dark") if theme.MODE == "dark" else tr("Light")))

    def _language_applied(self, code: str):
        import i18n
        if code == i18n.language():
            return
        name = i18n.available().get(code, {}).get("name", code)
        r = QMessageBox.question(self, tr("Language"),
                                 tr("The language changes to {name} after a restart. Restart MPM Studio now?",
                                    name=name), QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
        if r == QMessageBox.Yes:
            self.restart()

    def restart(self) -> bool:
        """Close (asking to save) and start a new instance that reopens the current project."""
        if not self._maybe_save():
            return False
        from PyQt5.QtCore import QProcess
        folder = project.project_dir(self.store)
        args = [] if getattr(sys, "frozen", False) else [str(Path(__file__).resolve().parents[1] / "main.py")]
        if folder:
            args.append(folder)
        self._restarting = True                 # unsaved changes were handled above: no second question
        if not self.close():
            self._restarting = False
            return False
        QProcess.startDetached(sys.executable, args)
        QApplication.instance().quit()
        return True

    def _keep_on_screen(self):
        """Shrink / move the window if a restored geometry does not fit the current display."""
        avail = available_geometry(self)
        fg = self.frameGeometry()
        if fg.width() > avail.width() or fg.height() > avail.height() or not avail.intersects(fg):
            if self.isMaximized():
                return
            fit_to_screen(self, min(self.width(), avail.width()), min(self.height(), avail.height()))
            center_on(self)

    def closeEvent(self, ev):
        if not getattr(self, "_restarting", False) and not self._maybe_save():
            ev.ignore()
            return
        s = QSettings("MPM", APP_NAME)
        s.setValue("geometry", self.saveGeometry())
        s.setValue("state", self.saveState())
        self.store.shutdown()                           # stop background raster reads
        super().closeEvent(ev)
