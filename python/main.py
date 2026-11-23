"""MPM Studio — 2D mineral prospectivity mapping.

    .venv\\Scripts\\python.exe python\\main.py [project folder | project.mpm.json]
    MPM Studio.exe [project folder | project.mpm.json]

Options:
    --version            print the version and exit
    --selftest REPORT    headless check: open the Carlin demo, run WofE, write REPORT, exit (0 = ok)
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
os.environ.setdefault("USE_PYGEOS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
# fully offline (e.g. underground mines): coordinate systems come from the local EPSG database (proj.db shipped
# with rasterio / fiona / pyproj); PROJ never downloads anything (PROJ_NETWORK=OFF). PROJ_LIB / PROJ_DATA / GDAL_DATA
# set by other GIS software would hide that database -> removed before any geo library is imported.
from utils import geo_env  # noqa: E402  (no geo imports in this module)

GEO_ENV_NOTES = geo_env.isolate()


def log_dir() -> str:
    d = os.path.join(os.environ.get("LOCALAPPDATA", os.path.expanduser("~")), "MPM Studio")
    os.makedirs(d, exist_ok=True)
    return d


def _setup_streams_and_crash_log():
    """A windowed (PyInstaller, console=False) app has no stdout/stderr: anything printed by a library would
    raise, and an unhandled exception would only show the bootloader's error box. Route both to a log file
    and record crashes there (and show them in a message box when the GUI is up)."""
    import faulthandler
    import traceback
    log = open(os.path.join(log_dir(), "mpm_studio.log"), "a", encoding="utf-8", buffering=1)
    if sys.stdout is None:
        sys.stdout = log
    if sys.stderr is None:
        sys.stderr = log
    try:
        faulthandler.enable(log)
    except Exception:  # noqa: BLE001
        pass

    def hook(etype, value, tb):
        text = "".join(traceback.format_exception(etype, value, tb))
        log.write("\n=== unhandled exception\n" + text)
        try:
            from PyQt5.QtWidgets import QApplication, QMessageBox
            if QApplication.instance() is not None:
                QMessageBox.critical(None, "MPM Studio — unexpected error",
                                     f"{value}\n\nDetails were written to\n{log.name}")
        except Exception:  # noqa: BLE001
            pass
    sys.excepthook = hook


def resource_path(*parts) -> str:
    """Path of a bundled resource (works from source and from the PyInstaller build)."""
    base = getattr(sys, "_MEIPASS", HERE)
    return os.path.join(base, *parts)


def _set_windows_app_id():
    """Own taskbar identity so Windows shows the MPM icon instead of python.exe's."""
    if sys.platform == "win32":
        try:
            import ctypes
            from version import APP_ID
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(APP_ID)
        except Exception:  # noqa: BLE001
            pass


def _selftest(win, app, report_path: str) -> int:
    """Copy the Carlin demo into a temporary project, run WofE, save / save-as, write a short report.

    Works on temporary copies only, so the demo data (and a release folder) are never modified.
    """
    import shutil
    import tempfile
    import time
    import traceback
    from version import APP_NAME, VERSION
    lines, code, t0 = [f"{APP_NAME} {VERSION}  frozen={getattr(sys, 'frozen', False)}"], 1, time.time()
    tmp = tempfile.mkdtemp(prefix="mpm_selftest_")
    try:
        from core import project
        from core.methods import REGISTRY
        from gui.main_window import DEMO_PROJECTS
        src = str(list(DEMO_PROJECTS.values())[0])
        path, _dropped = project.copy_project(src, os.path.join(tmp, "carlin"))
        errors = project.open_project(win.store, path)
        win._update_project_state()
        lines.append(f"project: demo copied to {path} ({len(win.store.layers)} layers, errors={errors})")
        s = win.store
        ev = []
        for n, t in (("geology_host", "categorical"), ("dist_faults_class", "ascending"), ("as_q8", "descending")):
            l = s.by_name(n)
            ev.append(dict(name=n, data=l.data, grid=l.grid, labels=l.labels(), type=t, criterion="max_contrast",
                           min_stud=1.96))
        m = s.by_name("study_mask")
        params = dict(evidence=ev, deposits=s.by_name("deposits").gdf, mask=(m.data, m.grid), test_fraction=0.2,
                      seed=42, unit_cell_km2=1.0, missing="zero", zero_deposit="zero", prefix="selftest")
        res = REGISTRY["wofe"].run(params, lambda p, msg="": None)
        run_id, run_dir = win.method_finished(REGISTRY["wofe"], res)
        win.canvas.draw()
        app.processEvents()
        c = res.evaluation.curves
        lines.append(f"wofe: deposits={res.extra['wofe'].n_deposits} auc_train={c['training'].auc:.3f} "
                     f"auc_test={c['test'].auc:.3f} layers_now={len(win.store.layers)}")
        files = sorted(os.listdir(run_dir))
        assert {"post.tif", "post_std.tif", "conf.tif", "validation.csv", "run_info.txt"} <= set(files), files
        lines.append(f"results folder: results/{run_id}/ -> {', '.join(files)}")
        project.save_project(win.store)
        project.save_project_as(win.store, os.path.join(tmp, "carlin_copy"))
        path2 = win.store.project_path
        errs2 = project.open_project(win.store, path2)
        assert not errs2 and win.store.by_name(f"{run_id}_post") is not None, errs2
        lines.append(f"save / save as / reopen: ok ({len(win.store.layers)} layers in the copy)")
        from evaluation.metrics import evaluate  # noqa: F401
        from gui.dialogs.evaluation_dialogs import EvaluationDialog
        EvaluationDialog(res.evaluation, "selftest", win).close()
        from utils.reclass import compute_breaks
        compute_breaks(res.outputs[0].data.compressed(), "natural_breaks", 5)     # mapclassify
        import qtawesome  # noqa: F401
        lines.append("dialogs, reclass (natural breaks), icons: ok")
        from logic.data_driven.cnn.model import _tf                          # CNN tool: bundled TensorFlow
        import numpy as np
        tf = _tf()
        net = tf.keras.Sequential([tf.keras.layers.Input((5, 5, 2)), tf.keras.layers.Conv2D(2, 3),
                                   tf.keras.layers.Flatten(), tf.keras.layers.Dense(1, activation="sigmoid")])
        out = net.predict(np.zeros((3, 5, 5, 2), "float32"), verbose=0)
        assert out.shape == (3, 1)
        lines.append(f"TensorFlow {tf.__version__} / Keras (CNN tool): ok")
        import pyproj
        from rasterio.crs import CRS
        assert CRS.from_epsg(28354).to_epsg() == 28354
        assert "MGA zone 54" in pyproj.CRS.from_epsg(28354).name
        lines.append("PROJ database (rasterio + pyproj EPSG lookup): ok")
        cur, _d = project.copy_project(str(list(DEMO_PROJECTS.values())[1]), os.path.join(tmp, "curnamona"))
        errs = project.open_project(win.store, cur)
        lines.append(f"project: Curnamona demo copy ({len(win.store.layers)} layers, errors={errs}, "
                     f"crs=EPSG:{win.store.crs.to_epsg()})")
        if errs:
            raise RuntimeError(f"Curnamona project errors: {errs}")
        code = 0
    except Exception:  # noqa: BLE001
        lines.append("ERROR\n" + traceback.format_exc())
    finally:
        win.store.clear()
        shutil.rmtree(tmp, ignore_errors=True)
    try:
        from utils.gdal_rat import gdal_version
        ver = gdal_version()
        lines.append(f"attribute-table reader: GDAL {ver} (rasterio wheel)" if ver else "attribute-table reader: GDAL NOT FOUND")
        if not ver:
            code = 1
    except Exception as exc:  # noqa: BLE001
        lines.append(f"attribute-table reader: ERROR {exc}")
        code = 1
    lines.append(f"elapsed: {time.time() - t0:.1f}s")
    lines.append("SELFTEST " + ("OK" if code == 0 else "FAILED"))
    with open(report_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    return code


def main(argv=None):
    argv = list(sys.argv if argv is None else argv)
    _setup_streams_and_crash_log()
    from version import APP_NAME, VERSION
    if "--version" in argv:
        print(f"{APP_NAME} {VERSION}")
        return 0
    selftest = None
    if "--selftest" in argv:
        i = argv.index("--selftest")
        selftest = argv[i + 1] if i + 1 < len(argv) else os.path.join(os.getcwd(), "mpm_selftest.txt")
        del argv[i:i + 2]
        try:
            return _gui_main(argv, selftest)
        except BaseException:  # noqa: BLE001  - start-up failure: still write a report
            import traceback
            with open(selftest, "w", encoding="utf-8") as fh:
                fh.write(f"{APP_NAME} {VERSION} frozen={getattr(sys, 'frozen', False)}\nSTART-UP ERROR\n"
                         + traceback.format_exc() + "SELFTEST FAILED\n")
            return 1
    return _gui_main(argv, None)


def _gui_main(argv, selftest):
    from version import APP_NAME, VERSION
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning, module="geopandas")
    warnings.filterwarnings("ignore", category=FutureWarning)
    _set_windows_app_id()
    from PyQt5.QtCore import Qt
    from PyQt5.QtGui import QIcon
    from PyQt5.QtWidgets import QApplication

    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    app = QApplication(argv)
    app.setApplicationName(APP_NAME)
    app.setApplicationVersion(VERSION)
    app.setWindowIcon(QIcon(resource_path("gui", "resources", "mpm_studio.ico")))

    import i18n
    from gui import app_settings
    i18n.set_language(app_settings.language())   # before the GUI modules are imported (module-level texts)
    tr = i18n.tr
    splash = None
    if not selftest:                        # splash until the main window is ready (at least 2 s)
        from gui.splash import StartupSplash
        splash = StartupSplash.create(VERSION)

    def step(text):
        if splash is not None:
            splash.message(text)

    try:
        step(tr("Starting…"))
        import matplotlib
        matplotlib.use("Qt5Agg")
        from gui import theme
        app._qt_translators = i18n.install_qt_translators(app)
        theme.apply(app, app_settings.theme_mode())
        step(tr("Loading methods…"))
        from core.methods import load_builtin_methods
        load_builtin_methods()
        step(tr("Building the main window…"))
        from gui.main_window import MainWindow
        win = MainWindow()
        win.showMaximized()                 # start maximized (the restored normal geometry is kept for un-maximize)
        app.processEvents()                 # first paint of the main window before the splash goes
    except BaseException:
        if splash is not None:              # do not hide the error message behind the splash
            splash.close()
        raise
    import logging
    for note in GEO_ENV_NOTES:
        logging.getLogger("mpm").info(note)
    problem = geo_env.crs_database_problem()
    if problem:
        logging.getLogger("mpm").error(problem)
    if selftest:
        return _selftest(win, app, selftest)
    args = argv[1:]

    def open_arg():
        if args and os.path.exists(args[0]):     # project folder or *.mpm.json (also used by Settings ▸ restart)
            win.open_project(args[0])
    if splash is not None:
        splash.message(tr("Ready"))
        splash.finish_when_ready(win, then=open_arg)    # the project opens after the splash (its dialogs stay visible)
    else:
        open_arg()
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())
