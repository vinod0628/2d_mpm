"""Offscreen GUI smoke test. Set MPM_SCREENSHOT_DIR to save screenshots of every window."""
import os
import warnings

import numpy as np
import pytest

from conftest import CARLIN

warnings.filterwarnings("ignore")
need_data = pytest.mark.skipif(not os.path.isdir(CARLIN), reason="run utils.prepare_demo_data first")
SHOTS = os.environ.get("MPM_SCREENSHOT_DIR")


@pytest.fixture(scope="module")
def app():
    from PyQt5.QtWidgets import QApplication
    import matplotlib
    matplotlib.use("Qt5Agg")
    a = QApplication.instance() or QApplication([])
    from gui import theme
    theme.apply(a)
    from core.methods import load_builtin_methods
    load_builtin_methods()
    yield a


def shot(widget, name):
    if SHOTS:
        from PyQt5.QtWidgets import QApplication
        widget.show()
        for _ in range(5):
            QApplication.processEvents()
        if hasattr(widget, "draw"):
            pass
        widget.grab().save(os.path.join(SHOTS, name + ".png"))


def wait_tasks(win, app, timeout=120):
    import time
    t0 = time.time()
    while win._tasks and time.time() - t0 < timeout:
        app.processEvents()
        time.sleep(0.05)
    for _ in range(10):
        app.processEvents()
    assert not win._tasks, "task did not finish"


def demo_copy(tmp_path, index=0, name="proj"):
    """Copy a demo project into tmp_path (tests never write into demo_data)."""
    from core import project
    from gui.main_window import DEMO_PROJECTS
    path, _ = project.copy_project(str(list(DEMO_PROJECTS.values())[index]), str(tmp_path / name))
    return path


@need_data
def test_main_window_wofe_workflow(app, monkeypatch, tmp_path):
    from PyQt5.QtWidgets import QMessageBox
    from gui.main_window import DEMO_PROJECTS, MainWindow
    from logic.data_driven.wofe.dialogs import WofEDialog, WofEResultsDialog
    from gui.dialogs.evaluation_dialogs import EvaluationDialog
    from core.methods import REGISTRY

    errors = []
    monkeypatch.setattr(QMessageBox, "critical", lambda *a, **k: errors.append(a[2] if len(a) > 2 else a))
    win = MainWindow()
    win.resize(1500, 920)
    win.show()
    # no project yet: data / method actions are disabled
    assert not win.a_add_r.isEnabled() and not win.a_save.isEnabled() and not win.method_actions["wofe"].isEnabled()
    assert win.a_new.isEnabled() and win.a_open.isEnabled()
    win.open_project(demo_copy(tmp_path))
    assert win.a_add_r.isEnabled() and win.method_actions["wofe"].isEnabled()
    assert len(win.store.layers) >= 15
    assert win.store.by_name("deposits") is not None
    win.store.set_active(win.store.by_name("geology"))
    win.canvas.draw()
    shot(win, "01_main_window_carlin")

    # parameter dialog
    dlg = WofEDialog(win.store, win)
    chosen = {"geology_host": "categorical", "dist_faults_class": "ascending", "as_q8": "descending",
              "sb_q8": "descending"}
    for r in dlg._rows:
        if r["layer"].name in chosen:
            r["use"].setChecked(True)
            r["type"].setCurrentIndex(["categorical", "ascending", "descending"].index(chosen[r["layer"].name]))
    dlg.test_pct.setValue(20)
    assert dlg.validate() is None
    # WofE takes integer (class) rasters only: real-valued layers are disabled, without a class count
    real = next(r for r in dlg._rows if r["layer"].name == "sb_surface")
    assert not real["use"].isEnabled() and dlg.table.item(dlg._rows.index(real), 2).text() == "—"
    geo = next(r for r in dlg._rows if r["layer"].name == "geology_host")
    assert geo["use"].isEnabled() and dlg.table.item(dlg._rows.index(geo), 2).text() == "2"
    shot(dlg, "02_wofe_dialog")
    params = dlg.params()
    assert len(params["evidence"]) == 4

    method = REGISTRY["wofe"]
    runs = []
    win.run_task(method.run, lambda res: runs.append(win.method_finished(method, res)), "WofE", params)
    wait_tasks(win, app)
    assert not errors, errors
    run_id, run_dir = runs[0]
    post = win.store.by_name(f"{run_id}_post")
    assert post is not None and post.source == os.path.join(run_dir, "post.tif")     # written, not in memory
    assert {"post.tif", "post_std.tif", "conf.tif", "validation.csv", "run_info.txt",
            "training_points.shp", "training_points.dbf", "training_points.prj", "test_points.shp"} <= set(os.listdir(run_dir))
    trp, tep = win.store.by_name(f"{run_id}_training_points"), win.store.by_name(f"{run_id}_test_points")
    assert trp is not None and tep is not None and trp.geom_type == "point"
    assert trp.source == os.path.join(run_dir, "training_points.shp")       # Shapefile: shared with ArcMap
    assert len(tep.gdf) >= 1 and len(trp.gdf) > 3 * len(tep.gdf) / 2           # 20 % held out
    assert not set(trp.gdf.geometry.to_wkt()) & set(tep.gdf.geometry.to_wkt())    # a point is in one set only
    assert set(trp.gdf.split) == {"training"} and set(tep.gdf.split) == {"test"}
    assert not win.store.dirty                                 # project file updated after the run
    results = [d for d in win._dialogs if isinstance(d, WofEResultsDialog)]
    assert results, "results dialog not opened"
    rd = results[-1]
    ev = rd.result.evaluation
    assert ev.curves["training"].auc > 0.75 and "test" in ev.curves
    shot(rd, "03_wofe_results")
    rd.show_evaluation()                                       # the "ROC / AUC…" button of the results dialog
    app.processEvents()
    opened = rd.findChildren(EvaluationDialog)
    assert opened and opened[-1].isVisible()
    opened[-1].close()
    ed = EvaluationDialog(ev, "Weights of Evidence", win)
    shot(ed, "04_roc_auc")
    ed.tabs.setCurrentIndex(2)
    shot(ed, "05_prediction_area")
    win.store.set_active(post)
    win.canvas.draw()
    shot(win, "06_posterior_map")

    # data dialogs construct and run
    from gui.dialogs.data_dialogs import ReclassDialog, DistanceDialog, ConvertDialog, AlignDialog
    rc = ReclassDialog(win.store, win)
    rc.src.set_layer(win.store.by_name("dist_faults"))
    rc.method.setCurrentIndex(1)
    rc.preview()
    shot(rc, "07_reclass_dialog")
    lyr = rc.run()
    assert lyr.categorical and lyr.classes().size == 8
    for cls in (DistanceDialog, ConvertDialog, AlignDialog):
        d = cls(win.store, win)
        assert d is not None
    win.store.dirty = False                   # discard (no modal "save changes?" prompt in tests)
    win.close()


@need_data
def test_logistic_regression_workflow(app, monkeypatch, tmp_path):
    """LR: menu entry, dialog (design switch, validation), run in the worker, run files, results dialog."""
    from PyQt5.QtWidgets import QMessageBox
    from core.methods import PLANNED, REGISTRY
    from gui.main_window import MainWindow
    from logic.data_driven.logistic_regression.dialogs import (LogisticRegressionDialog,
                                                               LogisticRegressionResultsDialog)

    errors = []
    monkeypatch.setattr(QMessageBox, "critical", lambda *a, **k: errors.append(a[2] if len(a) > 2 else a))
    assert not any(p[2] == "Logistic regression" for p in PLANNED)          # implemented, no longer planned
    win = MainWindow()
    win.resize(1500, 920)
    win.show()
    win.open_project(demo_copy(tmp_path))
    assert win.method_actions["logistic_regression"].isEnabled()

    dlg = LogisticRegressionDialog(win.store, win)
    assert not dlg.neg_ratio.isEnabled() and not dlg.ridge.isEnabled()      # weighted LR on all cells by default
    dlg.design.setCurrentIndex(1)
    assert dlg.neg_ratio.isEnabled() and dlg.repeats.isEnabled()
    dlg.design.setCurrentIndex(0)
    assert dlg.validate() is not None                                       # nothing ticked yet
    chosen = {"geology_host": "categorical", "dist_faults": "continuous", "sb_log_idw": "continuous"}
    for r in dlg._rows:
        if r["layer"].name in chosen:
            r["use"].setChecked(True)
            r["kind"].setCurrentIndex(["continuous", "categorical"].index(chosen[r["layer"].name]))
    row = next(r for r in dlg._rows if r["layer"].name == "dist_faults")
    row["transform"].setCurrentIndex(1)                                     # log10 of a distance with zeros
    assert "log10" in dlg.validate()
    row["transform"].setCurrentIndex(0)
    dlg.test_pct.setValue(20)
    assert dlg.validate() is None
    shot(dlg, "07_lr_dialog")
    params = dlg.params()
    assert {e["name"]: e["kind"] for e in params["evidence"]} == chosen

    method = REGISTRY["logistic_regression"]
    runs = []
    win.run_task(method.run, lambda res: runs.append(win.method_finished(method, res)), "LR", params)
    wait_tasks(win, app, timeout=300)
    assert not errors, errors
    run_id, run_dir = runs[0]
    post = win.store.by_name(f"{run_id}_post")
    assert post is not None and post.source == os.path.join(run_dir, "post.tif")
    assert {"post.tif", "post_std.tif", "tvalue.tif", "coefficients.csv", "model_fit.csv", "variables.csv",
            "hosmer_lemeshow.csv", "validation.csv", "run_info.txt", "training_points.shp",
            "test_points.shp"} <= set(os.listdir(run_dir))
    trp, tep = win.store.by_name(f"{run_id}_training_points"), win.store.by_name(f"{run_id}_test_points")
    assert len(trp.gdf) > len(tep.gdf) >= 1 and set(tep.gdf.split) == {"test"}
    # Save as… offers ESRI Shapefile: the chosen format's extension is added when it is missing
    from PyQt5.QtWidgets import QFileDialog
    target = str(tmp_path / "shared" / "lr_train")
    monkeypatch.setattr(QFileDialog, "getSaveFileName",
                        staticmethod(lambda *a, **k: (target, "ESRI Shapefile (*.shp)")))
    os.makedirs(tmp_path / "shared", exist_ok=True)
    win.save_layer(trp)
    assert trp.source == target + ".shp" and os.path.exists(target + ".prj") and os.path.exists(target + ".cpg")
    from utils.vector_io import read_vector
    assert len(read_vector(target + ".shp")) == len(trp.gdf)
    results = [d for d in win._dialogs if isinstance(d, LogisticRegressionResultsDialog)]
    assert results, "results dialog not opened"
    rd = results[-1]
    ev = rd.result.evaluation
    assert ev.curves["training"].auc > 0.75 and "test" in ev.curves
    lr = rd.result.extra["lr"]
    assert set(lr.coefficients["layer"]) >= set(chosen)
    assert rd.tabs.count() >= 5
    shot(rd, "08_lr_results")
    from gui.dialogs.evaluation_dialogs import EvaluationDialog
    rd.show_evaluation()                                       # the "ROC / AUC…" button of the results dialog
    app.processEvents()
    opened = rd.findChildren(EvaluationDialog)
    assert opened and opened[-1].isVisible()
    opened[-1].close()
    win.store.dirty = False
    win.close()


@need_data
def test_random_forest_workflow(app, monkeypatch, tmp_path):
    """RF: menu entry, dialog, run in the worker, run files (score, std, training / test points), results dialog."""
    from PyQt5.QtWidgets import QMessageBox
    from core.methods import PLANNED, REGISTRY
    from gui.dialogs.evaluation_dialogs import EvaluationDialog
    from gui.main_window import MainWindow
    from logic.data_driven.random_forest.dialogs import RandomForestDialog, RandomForestResultsDialog

    errors = []
    monkeypatch.setattr(QMessageBox, "critical", lambda *a, **k: errors.append(a[2] if len(a) > 2 else a))
    assert not any(p[2] == "Random forest" for p in PLANNED)
    win = MainWindow()
    win.resize(1500, 920)
    win.show()
    win.open_project(demo_copy(tmp_path))
    assert win.method_actions["random_forest"].isEnabled()

    dlg = RandomForestDialog(win.store, win)
    assert dlg.test_pct.value() == 20 and dlg.trees.value() == 500
    assert dlg.validate() is not None                                       # nothing ticked yet
    chosen = {"geology_host": "categorical", "dist_faults": "continuous", "as_log_idw": "continuous",
              "sb_log_idw": "continuous"}
    for r in dlg._rows:
        if r["layer"].name in chosen:
            r["use"].setChecked(True)
            r["kind"].setCurrentIndex(["continuous", "categorical"].index(chosen[r["layer"].name]))
    dlg.trees.setValue(150)
    dlg.repeats.setValue(2)
    dlg.neg_ratio.setValue(3)
    assert dlg.validate() is None
    shot(dlg, "09_rf_dialog")
    params = dlg.params()
    assert {e["name"]: e["kind"] for e in params["evidence"]} == chosen

    method = REGISTRY["random_forest"]
    runs = []
    win.run_task(method.run, lambda res: runs.append(win.method_finished(method, res)), "RF", params)
    wait_tasks(win, app, timeout=300)
    assert not errors, errors
    run_id, run_dir = runs[0]
    assert {"score.tif", "score_std.tif", "importance.csv", "sample_validation.csv", "validation.csv",
            "training_points.shp", "test_points.shp", "run_info.txt"} <= set(os.listdir(run_dir))
    assert win.store.by_name(f"{run_id}_score").source == os.path.join(run_dir, "score.tif")
    rd = [d for d in win._dialogs if isinstance(d, RandomForestResultsDialog)][-1]
    ev = rd.result.evaluation
    assert ev.curves["training"].auc > 0.8 and "test" in ev.curves
    rf = rd.result.extra["rf"]
    assert set(rf.importance.layer) == set(chosen) and rd.tabs.count() == 4
    shot(rd, "10_rf_results")
    rd.tabs.setCurrentIndex(1)
    shot(rd, "11_rf_importance")
    rd.show_evaluation()
    app.processEvents()
    opened = rd.findChildren(EvaluationDialog)
    assert opened and opened[-1].isVisible()
    opened[-1].close()
    win.store.dirty = False
    win.close()


@need_data
def test_svm_workflow(app, monkeypatch, tmp_path):
    """SVM: menu entry, dialog, run in the worker, run files (score, std, decision, points), results dialog."""
    from PyQt5.QtWidgets import QMessageBox
    from core.methods import PLANNED, REGISTRY
    from gui.dialogs.evaluation_dialogs import EvaluationDialog
    from gui.main_window import MainWindow
    from logic.data_driven.svm.dialogs import SVMDialog, SVMResultsDialog

    errors = []
    monkeypatch.setattr(QMessageBox, "critical", lambda *a, **k: errors.append(a[2] if len(a) > 2 else a))
    assert not any(p[2] == "Support vector machine" for p in PLANNED)
    win = MainWindow()
    win.resize(1500, 920)
    win.show()
    win.open_project(demo_copy(tmp_path))
    assert win.method_actions["svm"].isEnabled()

    dlg = SVMDialog(win.store, win)
    assert dlg.test_pct.value() == 20 and dlg.tune.isChecked() and not dlg.C.isEnabled()
    dlg.kernel.setCurrentIndex(1)                                           # linear: no γ
    dlg.tune.setChecked(False)
    assert dlg.C.isEnabled() and not dlg.gamma.isEnabled() and not dlg.degree.isEnabled()
    dlg.kernel.setCurrentIndex(0)
    dlg.tune.setChecked(True)
    assert dlg.validate() is not None                                       # nothing ticked yet
    chosen = {"geology_host": "categorical", "dist_faults": "continuous", "as_log_idw": "continuous",
              "sb_log_idw": "continuous"}
    for r in dlg._rows:
        if r["layer"].name in chosen:
            r["use"].setChecked(True)
            r["kind"].setCurrentIndex(["continuous", "categorical"].index(chosen[r["layer"].name]))
    dlg.repeats.setValue(2)
    dlg.neg_ratio.setValue(3)
    dlg.o_dec.setChecked(True)
    assert dlg.validate() is None
    shot(dlg, "12_svm_dialog")
    params = dlg.params()
    assert {e["name"]: e["kind"] for e in params["evidence"]} == chosen and params["tune"]

    method = REGISTRY["svm"]
    runs = []
    win.run_task(method.run, lambda res: runs.append(win.method_finished(method, res)), "SVM", params)
    wait_tasks(win, app, timeout=300)
    assert not errors, errors
    run_id, run_dir = runs[0]
    assert {"score.tif", "score_std.tif", "decision.tif", "importance.csv", "tuning.csv", "sample_validation.csv",
            "validation.csv", "training_points.shp", "test_points.shp", "run_info.txt"} <= set(os.listdir(run_dir))
    assert win.store.by_name(f"{run_id}_score").source == os.path.join(run_dir, "score.tif")
    rd = [d for d in win._dialogs if isinstance(d, SVMResultsDialog)][-1]
    ev = rd.result.evaluation
    assert ev.curves["training"].auc > 0.8 and "test" in ev.curves
    res = rd.result.extra["svm"]
    assert set(res.importance.layer) == set(chosen) and rd.tabs.count() == 5
    shot(rd, "13_svm_results")
    rd.tabs.setCurrentIndex(1)
    shot(rd, "14_svm_tuning")
    rd.show_evaluation()
    app.processEvents()
    opened = rd.findChildren(EvaluationDialog)
    assert opened and opened[-1].isVisible()
    opened[-1].close()
    win.store.dirty = False
    win.close()


@need_data
def test_xgboost_workflow(app, monkeypatch, tmp_path):
    """XGBoost: menu entry, dialog (monotone column, device), run files (score, std, SHAP maps), results dialog."""
    from PyQt5.QtWidgets import QMessageBox
    from core.methods import PLANNED, REGISTRY
    from gui.dialogs.evaluation_dialogs import EvaluationDialog
    from gui.main_window import MainWindow
    from logic.data_driven.xgboost_mpm.dialogs import XGBoostDialog, XGBoostResultsDialog

    errors = []
    monkeypatch.setattr(QMessageBox, "critical", lambda *a, **k: errors.append(a[2] if len(a) > 2 else a))
    assert not any(p[2] == "Gradient boosting" for p in PLANNED)
    win = MainWindow()
    win.resize(1500, 920)
    win.show()
    win.open_project(demo_copy(tmp_path))
    assert win.method_actions["xgboost"].isEnabled()

    dlg = XGBoostDialog(win.store, win)
    assert dlg.test_pct.value() == 20 and dlg.device.currentData() == "auto" and dlg.device_info.text()
    assert dlg.validate() is not None
    chosen = {"geology_host": ("categorical", 0), "dist_faults": ("continuous", -1),
              "as_log_idw": ("continuous", 0), "sb_log_idw": ("continuous", 1)}
    for r in dlg._rows:
        if r["layer"].name in chosen:
            kind, mono = chosen[r["layer"].name]
            r["use"].setChecked(True)
            r["kind"].setCurrentIndex(["continuous", "categorical"].index(kind))
            r["mono"].setCurrentIndex([0, 1, -1].index(mono))
            assert r["mono"].isEnabled() == (kind == "continuous")
    dlg.repeats.setValue(2)
    dlg.neg_ratio.setValue(3)
    dlg.o_shap.setChecked(True)
    assert dlg.validate() is None
    shot(dlg, "15_xgb_dialog")
    params = dlg.params()
    assert {e["name"]: (e["kind"], e["monotone"]) for e in params["evidence"]} == chosen

    method = REGISTRY["xgboost"]
    runs = []
    win.run_task(method.run, lambda res: runs.append(win.method_finished(method, res)), "XGB", params)
    wait_tasks(win, app, timeout=300)
    assert not errors, errors
    run_id, run_dir = runs[0]
    files = set(os.listdir(run_dir))
    assert {"score.tif", "score_std.tif", "shap_sb_log_idw.tif", "shap_geology_host.tif", "importance.csv",
            "boosting_curve.csv", "validation.csv", "training_points.shp", "test_points.shp"} <= files
    rd = [d for d in win._dialogs if isinstance(d, XGBoostResultsDialog)][-1]
    ev = rd.result.evaluation
    assert ev.curves["training"].auc > 0.8 and "test" in ev.curves
    res = rd.result.extra["xgb"]
    assert set(res.importance.layer) == set(chosen) and rd.tabs.count() == 5
    shot(rd, "16_xgb_results")
    rd.tabs.setCurrentIndex(2)
    shot(rd, "17_xgb_importance")
    rd.show_evaluation()
    app.processEvents()
    opened = rd.findChildren(EvaluationDialog)
    assert opened and opened[-1].isVisible()
    opened[-1].close()
    win.store.dirty = False
    win.close()


@need_data
def test_rvm_workflow(app, monkeypatch, tmp_path):
    """RVM: menu entry, dialog, run in the worker, run files (score, predictive s.d., relevance vectors), results."""
    from PyQt5.QtWidgets import QMessageBox
    from core.methods import PLANNED, REGISTRY
    from gui.dialogs.evaluation_dialogs import EvaluationDialog
    from gui.main_window import MainWindow
    from logic.data_driven.rvm.dialogs import RVMDialog, RVMResultsDialog

    errors = []
    monkeypatch.setattr(QMessageBox, "critical", lambda *a, **k: errors.append(a[2] if len(a) > 2 else a))
    assert not any(p[2] == "Relevance vector machine" for p in PLANNED)
    win = MainWindow()
    win.resize(1500, 920)
    win.show()
    win.open_project(demo_copy(tmp_path))
    assert win.method_actions["rvm"].isEnabled()

    dlg = RVMDialog(win.store, win)
    assert dlg.test_pct.value() == 20 and dlg.tune.isChecked() and not dlg.gamma.isEnabled()
    dlg.kernel.setCurrentIndex(1)                                           # linear basis: no kernel width
    assert not dlg.tune.isEnabled() and not dlg.gamma.isEnabled()
    dlg.kernel.setCurrentIndex(0)
    assert dlg.validate() is not None
    chosen = {"geology_host": "categorical", "dist_faults": "continuous", "as_log_idw": "continuous",
              "sb_log_idw": "continuous"}
    for r in dlg._rows:
        if r["layer"].name in chosen:
            r["use"].setChecked(True)
            r["kind"].setCurrentIndex(["continuous", "categorical"].index(chosen[r["layer"].name]))
    dlg.repeats.setValue(2)
    dlg.neg_ratio.setValue(2)
    assert dlg.validate() is None
    shot(dlg, "18_rvm_dialog")
    params = dlg.params()
    assert {e["name"]: e["kind"] for e in params["evidence"]} == chosen and params["kernel"] == "rbf"

    method = REGISTRY["rvm"]
    runs = []
    win.run_task(method.run, lambda res: runs.append(win.method_finished(method, res)), "RVM", params)
    wait_tasks(win, app, timeout=600)
    assert not errors, errors
    run_id, run_dir = runs[0]
    files = set(os.listdir(run_dir))
    assert {"score.tif", "pred_sd.tif", "score_std.tif", "relevance_vectors.shp", "relevance_vectors.csv",
            "importance.csv", "tuning.csv", "validation.csv", "training_points.shp", "test_points.shp"} <= files
    assert win.store.by_name(f"{run_id}_relevance_vectors") is not None
    rd = [d for d in win._dialogs if isinstance(d, RVMResultsDialog)][-1]
    ev = rd.result.evaluation
    assert ev.curves["training"].auc > 0.8 and "test" in ev.curves
    res = rd.result.extra["rvm"]
    assert set(res.importance.layer) == set(chosen) and rd.tabs.count() == 6
    shot(rd, "19_rvm_results")
    rd.tabs.setCurrentIndex(1)
    shot(rd, "20_rvm_relevance")
    rd.show_evaluation()
    app.processEvents()
    opened = rd.findChildren(EvaluationDialog)
    assert opened and opened[-1].isVisible()
    opened[-1].close()
    win.store.dirty = False
    win.close()


@need_data
def test_mlp_workflow(app, monkeypatch, tmp_path):
    """MLP: menu entry, dialog, run in the worker, run files (score, std), results dialog."""
    from PyQt5.QtWidgets import QMessageBox
    from core.methods import PLANNED, REGISTRY
    from gui.dialogs.evaluation_dialogs import EvaluationDialog
    from gui.main_window import MainWindow
    from logic.data_driven.mlp.dialogs import MLPDialog, MLPResultsDialog

    errors = []
    monkeypatch.setattr(QMessageBox, "critical", lambda *a, **k: errors.append(a[2] if len(a) > 2 else a))
    assert not any(p[2] == "Neural network (MLP)" for p in PLANNED)
    win = MainWindow()
    win.resize(1500, 920)
    win.show()
    win.open_project(demo_copy(tmp_path))
    assert win.method_actions["mlp"].isEnabled()

    dlg = MLPDialog(win.store, win)
    assert dlg.test_pct.value() == 20 and dlg.hidden.text() == "32, 16" and dlg.activation.currentData() == "relu"
    assert dlg.validate() is not None
    chosen = {"geology_host": "categorical", "dist_faults": "continuous", "as_log_idw": "continuous",
              "sb_log_idw": "continuous"}
    for r in dlg._rows:
        if r["layer"].name in chosen:
            r["use"].setChecked(True)
            r["kind"].setCurrentIndex(["continuous", "categorical"].index(chosen[r["layer"].name]))
    dlg.repeats.setValue(2)
    dlg.networks.setValue(3)
    dlg.neg_ratio.setValue(3)
    dlg.hidden.setText("16, 8")
    assert dlg.validate() is None
    shot(dlg, "21_mlp_dialog")
    params = dlg.params()
    assert params["hidden"] == (16, 8) and {e["name"]: e["kind"] for e in params["evidence"]} == chosen

    method = REGISTRY["mlp"]
    runs = []
    win.run_task(method.run, lambda res: runs.append(win.method_finished(method, res)), "MLP", params)
    wait_tasks(win, app, timeout=600)
    assert not errors, errors
    run_id, run_dir = runs[0]
    files = set(os.listdir(run_dir))
    assert {"score.tif", "score_std.tif", "importance.csv", "learning_curve.csv", "validation.csv",
            "training_points.shp", "test_points.shp"} <= files
    rd = [d for d in win._dialogs if isinstance(d, MLPResultsDialog)][-1]
    ev = rd.result.evaluation
    assert ev.curves["training"].auc > 0.8 and "test" in ev.curves
    res = rd.result.extra["mlp"]
    assert set(res.importance.layer) == set(chosen) and rd.tabs.count() == 5 and res.stats["networks"] == 6
    shot(rd, "22_mlp_results")
    rd.tabs.setCurrentIndex(1)
    shot(rd, "23_mlp_curve")
    rd.show_evaluation()
    app.processEvents()
    opened = rd.findChildren(EvaluationDialog)
    assert opened and opened[-1].isVisible()
    opened[-1].close()
    win.store.dirty = False
    win.close()


@need_data
def test_elm_workflow(app, monkeypatch, tmp_path):
    """ELM: menu entry, dialog, run in the worker, run files (score, std, tuning), results dialog."""
    from PyQt5.QtWidgets import QMessageBox
    from core.methods import REGISTRY
    from gui.dialogs.evaluation_dialogs import EvaluationDialog
    from gui.main_window import MainWindow
    from logic.data_driven.elm.dialogs import ELMDialog, ELMResultsDialog

    errors = []
    monkeypatch.setattr(QMessageBox, "critical", lambda *a, **k: errors.append(a[2] if len(a) > 2 else a))
    win = MainWindow()
    win.resize(1500, 920)
    win.show()
    win.open_project(demo_copy(tmp_path))
    assert win.method_actions["elm"].isEnabled()

    dlg = ELMDialog(win.store, win)
    assert dlg.test_pct.value() == 20 and dlg.hidden.value() == 100 and dlg.tune.isChecked() and not dlg.C.isEnabled()
    assert dlg.validate() is not None
    chosen = {"geology_host": "categorical", "dist_faults": "continuous", "as_log_idw": "continuous",
              "sb_log_idw": "continuous"}
    for r in dlg._rows:
        if r["layer"].name in chosen:
            r["use"].setChecked(True)
            r["kind"].setCurrentIndex(["continuous", "categorical"].index(chosen[r["layer"].name]))
    dlg.repeats.setValue(2)
    dlg.networks.setValue(5)
    dlg.neg_ratio.setValue(3)
    assert dlg.validate() is None
    shot(dlg, "24_elm_dialog")
    params = dlg.params()
    assert params["hidden"] == 100 and {e["name"]: e["kind"] for e in params["evidence"]} == chosen

    method = REGISTRY["elm"]
    runs = []
    win.run_task(method.run, lambda res: runs.append(win.method_finished(method, res)), "ELM", params)
    wait_tasks(win, app, timeout=600)
    assert not errors, errors
    run_id, run_dir = runs[0]
    files = set(os.listdir(run_dir))
    assert {"score.tif", "score_std.tif", "importance.csv", "tuning.csv", "validation.csv", "training_points.shp",
            "test_points.shp"} <= files
    rd = [d for d in win._dialogs if isinstance(d, ELMResultsDialog)][-1]
    ev = rd.result.evaluation
    assert ev.curves["training"].auc > 0.8 and "test" in ev.curves
    res = rd.result.extra["elm"]
    assert set(res.importance.layer) == set(chosen) and rd.tabs.count() == 5 and res.stats["networks"] == 10
    shot(rd, "25_elm_results")
    rd.tabs.setCurrentIndex(1)
    shot(rd, "26_elm_regularisation")
    rd.show_evaluation()
    app.processEvents()
    opened = rd.findChildren(EvaluationDialog)
    assert opened and opened[-1].isVisible()
    opened[-1].close()
    win.store.dirty = False
    win.close()


@need_data
def test_cnn_workflow(app, monkeypatch, tmp_path):
    """CNN: menu entry, dialog, run in the worker (TensorFlow), run files, results dialog."""
    from PyQt5.QtWidgets import QMessageBox
    from core.methods import REGISTRY
    from gui.dialogs.evaluation_dialogs import EvaluationDialog
    from gui.main_window import MainWindow
    from logic.data_driven.cnn.dialogs import CNNDialog, CNNResultsDialog

    errors = []
    monkeypatch.setattr(QMessageBox, "critical", lambda *a, **k: errors.append(a[2] if len(a) > 2 else a))
    win = MainWindow()
    win.resize(1500, 920)
    win.show()
    win.open_project(demo_copy(tmp_path))
    assert win.method_actions["cnn"].isEnabled()

    dlg = CNNDialog(win.store, win)
    assert dlg.test_pct.value() == 20 and dlg.patch.value() == 9 and dlg.augment.isChecked()
    assert dlg.cv.currentData() == 3 and "map units" in dlg.patch_info.text()
    assert dlg.validate() is not None
    chosen = {"geology_host": "categorical", "dist_faults": "continuous", "as_log_idw": "continuous",
              "sb_log_idw": "continuous"}
    for r in dlg._rows:
        if r["layer"].name in chosen:
            r["use"].setChecked(True)
            r["kind"].setCurrentIndex(["continuous", "categorical"].index(chosen[r["layer"].name]))
    dlg.repeats.setValue(1)
    dlg.networks.setValue(1)
    dlg.neg_ratio.setValue(3)
    dlg.epochs.setValue(40)
    assert dlg.validate() is None
    shot(dlg, "27_cnn_dialog")
    params = dlg.params()
    assert params["filters"] == (8, 16) and {e["name"]: e["kind"] for e in params["evidence"]} == chosen

    method = REGISTRY["cnn"]
    runs = []
    win.run_task(method.run, lambda res: runs.append(win.method_finished(method, res)), "CNN", params)
    wait_tasks(win, app, timeout=900)
    assert not errors, errors
    run_id, run_dir = runs[0]
    files = set(os.listdir(run_dir))
    assert {"score.tif", "score_std.tif", "importance.csv", "learning_curve.csv", "validation.csv",
            "training_points.shp", "test_points.shp"} <= files
    rd = [d for d in win._dialogs if isinstance(d, CNNResultsDialog)][-1]
    ev = rd.result.evaluation
    assert ev.curves["training"].auc > 0.8 and "test" in ev.curves
    res = rd.result.extra["cnn"]
    assert set(res.importance.layer) == set(chosen) and rd.tabs.count() == 4
    shot(rd, "28_cnn_results")
    rd.tabs.setCurrentIndex(1)
    shot(rd, "29_cnn_curve")
    rd.show_evaluation()
    app.processEvents()
    opened = rd.findChildren(EvaluationDialog)
    assert opened and opened[-1].isVisible()
    opened[-1].close()
    win.store.dirty = False
    win.close()


@need_data
def test_bayesian_network_workflow(app, monkeypatch, tmp_path):
    """Bayesian network: menu entry, dialog, run, run files (posterior, unit, s.d., tables), results dialog."""
    from PyQt5.QtWidgets import QMessageBox
    from core.methods import REGISTRY
    from gui.dialogs.evaluation_dialogs import EvaluationDialog
    from gui.main_window import MainWindow
    from logic.data_driven.bayesian_network.dialogs import BayesianNetworkDialog, BayesianNetworkResultsDialog

    errors = []
    monkeypatch.setattr(QMessageBox, "critical", lambda *a, **k: errors.append(a[2] if len(a) > 2 else a))
    win = MainWindow()
    win.resize(1500, 920)
    win.show()
    win.open_project(demo_copy(tmp_path))
    assert win.method_actions["bayesian_network"].isEnabled()

    dlg = BayesianNetworkDialog(win.store, win)
    assert dlg.test_pct.value() == 20 and dlg.structure.currentData() == "tan" and dlg.design.currentData() == "all"
    assert not dlg.neg_ratio.isEnabled() and dlg.unit.isEnabled()
    dlg.design.setCurrentIndex(1)
    assert dlg.neg_ratio.isEnabled() and not dlg.unit.isEnabled()
    dlg.design.setCurrentIndex(0)
    assert dlg.validate() is not None
    chosen = {"geology_host": "categorical", "dist_faults": "continuous", "as_log_idw": "continuous",
              "sb_log_idw": "continuous"}
    for r in dlg._rows:
        if r["layer"].name in chosen:
            r["use"].setChecked(True)
            r["kind"].setCurrentIndex(["continuous", "categorical"].index(chosen[r["layer"].name]))
    dlg.draws.setValue(10)
    assert dlg.validate() is None
    shot(dlg, "30_bn_dialog")
    params = dlg.params()
    assert {e["name"]: e["kind"] for e in params["evidence"]} == chosen and params["structure"] == "tan"

    method = REGISTRY["bayesian_network"]
    runs = []
    win.run_task(method.run, lambda res: runs.append(win.method_finished(method, res)), "BN", params)
    wait_tasks(win, app, timeout=600)
    assert not errors, errors
    run_id, run_dir = runs[0]
    files = set(os.listdir(run_dir))
    assert {"post.tif", "post_unit.tif", "post_sd.tif", "network_edges.csv", "probability_tables.csv",
            "evidence_layers.csv", "structure_comparison.csv", "validation.csv", "training_points.shp"} <= files
    rd = [d for d in win._dialogs if isinstance(d, BayesianNetworkResultsDialog)][-1]
    ev = rd.result.evaluation
    assert ev.curves["training"].auc > 0.8 and "test" in ev.curves
    res = rd.result.extra["bn"]
    assert len(res.edges) == 3 and rd.tabs.count() == 5
    shot(rd, "31_bn_results")
    rd.tabs.setCurrentIndex(1)
    shot(rd, "32_bn_network")
    rd.show_evaluation()
    app.processEvents()
    opened = rd.findChildren(EvaluationDialog)
    assert opened and opened[-1].isVisible()
    opened[-1].close()
    win.store.dirty = False
    win.close()


@need_data
def test_ebf_workflow(app, monkeypatch, tmp_path):
    """Evidential belief functions: menu entry, dialog defaults, run, run files (Bel, Dis, Unc, Pls, tables), results."""
    from PyQt5.QtWidgets import QMessageBox
    from core.methods import REGISTRY
    from gui.main_window import MainWindow
    from logic.data_driven.ebf.dialogs import EBFDialog, EBFResultsDialog

    errors = []
    monkeypatch.setattr(QMessageBox, "critical", lambda *a, **k: errors.append(a[2] if len(a) > 2 else a))
    win = MainWindow()
    win.resize(1500, 920)
    win.show()
    win.open_project(demo_copy(tmp_path))
    assert win.method_actions["ebf"].isEnabled()

    dlg = EBFDialog(win.store, win)
    rows = {r["layer"].name: r for r in dlg._rows}
    assert rows["geology_host"]["classing"].currentData() == "values"          # class raster
    assert rows["as_log_idw"]["classing"].currentData() == "quantile"          # real values
    assert not rows["geology_host"]["n"].isEnabled() and rows["as_log_idw"]["n"].isEnabled()
    assert dlg.score.currentData() == "bel" and dlg.zero.currentData() == "zero"
    assert dlg.validate() is not None
    for n in ("geology_host", "dist_faults_class", "as_log_idw", "sb_log_idw"):
        rows[n]["use"].setChecked(True)
    rows["as_log_idw"]["n"].setValue(8)
    rows["as_log_idw"]["classing"].setCurrentIndex(0)                          # class values of real data: refused
    assert "quantile" in dlg.validate()
    rows["as_log_idw"]["classing"].setCurrentIndex(1)
    dlg.zero.setCurrentIndex(1)
    dlg.o_layers.setChecked(True)
    assert dlg.validate() is None
    shot(dlg, "58_ebf_dialog")
    params = dlg.params()
    assert [e["classing"] for e in params["evidence"]] == ["values", "values", "quantile", "quantile"]
    assert params["zero_deposit"] == "haldane" and params["evidence"][2]["n_classes"] == 8

    method = REGISTRY["ebf"]
    runs = []
    win.run_task(method.run, lambda res: runs.append(win.method_finished(method, res)), "EBF", params)
    wait_tasks(win, app, timeout=300)
    assert not errors, errors
    run_id, run_dir = runs[0]
    files = set(os.listdir(run_dir))
    assert {"bel.tif", "dis.tif", "unc.tif", "pls.tif", "bel_sb_log_idw.tif", "summary.csv", "layers.csv",
            "pairwise.csv", "classes_geology_host.csv", "validation.csv", "training_points.shp"} <= files
    rd = [d for d in win._dialogs if isinstance(d, EBFResultsDialog)][-1]
    assert rd.result.evaluation.curves["training"].auc > 0.85 and "test" in rd.result.evaluation.curves
    assert rd.tabs.count() == 7 and len(rd.canvases) == 5                    # layers + 4 class charts
    rd.tabs.setCurrentIndex(2)
    if SHOTS:
        rd.show()
        app.processEvents()
        for c in rd.canvases:
            c.draw()
    shot(rd, "58_ebf_results")
    rd.close()
    win.store.dirty = False
    win.close()


@need_data
def test_fuzzy_and_boolean_workflows(app, monkeypatch, tmp_path):
    """Knowledge-driven overlays: dialogs (defaults from the data), runs, run files, results dialogs, ROC."""
    from PyQt5.QtWidgets import QMessageBox
    from core.methods import PLANNED, REGISTRY
    from gui.dialogs.evaluation_dialogs import EvaluationDialog
    from gui.main_window import MainWindow
    from logic.knowledge_driven.boolean_logic.dialogs import BooleanLogicDialog, BooleanLogicResultsDialog
    from logic.knowledge_driven.fuzzy_logic.dialogs import FuzzyLogicDialog, FuzzyLogicResultsDialog

    errors = []
    monkeypatch.setattr(QMessageBox, "critical", lambda *a, **k: errors.append(a[2] if len(a) > 2 else a))
    assert not any(p[2] in ("Boolean logic", "Fuzzy logic overlay") for p in PLANNED)
    win = MainWindow()
    win.resize(1500, 920)
    win.show()
    win.open_project(demo_copy(tmp_path))
    assert win.method_actions["fuzzy_logic"].isEnabled() and win.method_actions["boolean_logic"].isEnabled()

    fd = FuzzyLogicDialog(win.store, win)
    assert fd.validate() is not None
    setup = {"geology_host": ("classes", "2=0.9; 1=0.2", "trap"), "dist_faults": ("z_shape", "500, 5000", "pathway"),
             "sb_log_idw": ("logistic", "0.6, 6", "footprint"), "as_log_idw": ("logistic", None, "footprint")}
    for r in fd._rows:
        if r["layer"].name in setup:
            func, par, grp = setup[r["layer"].name]
            r["use"].setChecked(True)
            r["func"].setCurrentIndex(["linear", "s_shape", "z_shape", "logistic", "gaussian", "large", "small",
                                       "classes"].index(func))
            if par:
                r["params"].setText(par)
            r["group"].setText(grp)
    assert "," in [r for r in fd._rows if r["layer"].name == "as_log_idw"][0]["params"].text()   # data default
    fd.group_op.setCurrentIndex(1)                                          # OR within the footprint group
    assert fd.validate() is None
    shot(fd, "33_fuzzy_dialog")
    params = fd.params()
    assert params["final_op"][0] == "gamma" and set(params["group_ops"]) == {"trap", "pathway", "footprint"}
    method = REGISTRY["fuzzy_logic"]
    runs = []
    win.run_task(method.run, lambda res: runs.append(win.method_finished(method, res)), "Fuzzy", params)
    wait_tasks(win, app, timeout=300)
    assert not errors, errors
    run_id, run_dir = runs[0]
    assert {"favourability.tif", "group_trap.tif", "inference_network.csv", "memberships.csv",
            "validation.csv"} <= set(os.listdir(run_dir))
    rd = [d for d in win._dialogs if isinstance(d, FuzzyLogicResultsDialog)][-1]
    assert rd.result.evaluation.curves["training"].auc > 0.7 and rd.tabs.count() == 3
    rd.tabs.setCurrentIndex(1)
    shot(rd, "34_fuzzy_memberships")
    rd.show_evaluation()
    app.processEvents()
    assert rd.findChildren(EvaluationDialog)

    bd = BooleanLogicDialog(win.store, win)
    for r in bd._rows:
        if r["layer"].name == "geology_host":
            r["use"].setChecked(True)
            r["cond"].setCurrentIndex(4)
            r["values"].setText("2")
        if r["layer"].name == "sb_log_idw":
            r["use"].setChecked(True)
            r["cond"].setCurrentIndex(0)
            r["values"].setText("0.6")
    assert bd.validate() is None
    shot(bd, "35_boolean_dialog")
    method = REGISTRY["boolean_logic"]
    runs = []
    win.run_task(method.run, lambda res: runs.append(win.method_finished(method, res)), "Boolean", bd.params())
    wait_tasks(win, app, timeout=300)
    assert not errors, errors
    run_id, run_dir = runs[0]
    assert {"favourable.tif", "conditions_met.tif", "conditions.csv", "network.csv"} <= set(os.listdir(run_dir))
    br = [d for d in win._dialogs if isinstance(d, BooleanLogicResultsDialog)][-1]
    comb = br.result.extra["boolean"].table.set_index("layer").loc["FAVOURABLE (combined)"]
    assert comb.ratio > 2
    shot(br, "36_boolean_results")
    win.store.dirty = False
    win.close()


@need_data
def test_logistic_geometric_workflow(app, monkeypatch, tmp_path):
    """Logistic fuzzification + geometric average: dialog defaults, run, run files, results, P–A weights."""
    from PyQt5.QtWidgets import QMessageBox
    from core.methods import PLANNED, REGISTRY
    from gui.main_window import MainWindow
    from logic.knowledge_driven.logistic_geometric.dialogs import (LogisticGeometricDialog,
                                                                   LogisticGeometricResultsDialog)

    errors = []
    monkeypatch.setattr(QMessageBox, "critical", lambda *a, **k: errors.append(a[2] if len(a) > 2 else a))
    assert not any(p[2] == "Logistic fuzzification + geometric average" for p in PLANNED)
    win = MainWindow()
    win.resize(1500, 920)
    win.show()
    win.open_project(demo_copy(tmp_path))
    assert win.method_actions["logistic_geometric"].isEnabled()
    dlg = LogisticGeometricDialog(win.store, win)
    setup = {"geology_host": ("classes", "2=0.9; 1=0.1"), "sb_log_idw": ("positive", None),
             "as_log_idw": ("positive", None), "dist_faults": ("negative", None)}
    for r in dlg._rows:
        if r["layer"].name in setup:
            d, spec = setup[r["layer"].name]
            r["use"].setChecked(True)
            r["direction"].setCurrentIndex(["positive", "negative", "classes"].index(d))
            if spec:
                r["spec"].setText(spec)
            assert "," in r["spec"].text() or "=" in r["spec"].text()
    assert not dlg._rows[0]["weight"].isEnabled()
    dlg.weighting.setCurrentIndex(1)                                        # P-A weights
    assert dlg.validate() is None
    shot(dlg, "37_lgfa_dialog")
    method = REGISTRY["logistic_geometric"]
    runs = []
    win.run_task(method.run, lambda res: runs.append(win.method_finished(method, res)), "LGFA", dlg.params())
    wait_tasks(win, app, timeout=300)
    assert not errors, errors
    run_id, run_dir = runs[0]
    assert {"favourability.tif", "evidence.csv", "validation.csv"} <= set(os.listdir(run_dir))
    rd = [d for d in win._dialogs if isinstance(d, LogisticGeometricResultsDialog)][-1]
    t = rd.result.extra["lg"].table.set_index("layer")
    assert t.loc["sb_log_idw", "weight"] > 0 and rd.result.evaluation.curves["training"].auc > 0.7
    rd.tabs.setCurrentIndex(1)
    shot(rd, "38_lgfa_results")
    win.store.dirty = False
    win.close()


@need_data
def test_index_overlay_workflow(app, monkeypatch, tmp_path):
    """Index overlay: scoring rules, AHP matrix editing with live CR, run, run files, results dialog."""
    from PyQt5.QtWidgets import QMessageBox
    from core.methods import PLANNED, REGISTRY
    from gui.main_window import MainWindow
    from logic.knowledge_driven.index_overlay.dialogs import IndexOverlayDialog, IndexOverlayResultsDialog

    errors = []
    monkeypatch.setattr(QMessageBox, "critical", lambda *a, **k: errors.append(a[2] if len(a) > 2 else a))
    assert not any(p[2] in ("Index overlay", "AHP", "Fuzzy AHP", "Dempster-Shafer (expert)") for p in PLANNED)
    win = MainWindow()
    win.resize(1500, 920)
    win.show()
    win.open_project(demo_copy(tmp_path))
    assert win.method_actions["index_overlay"].isEnabled()
    dlg = IndexOverlayDialog(win.store, win)
    assert dlg.validate() is not None
    setup = {"geology_host": ("classes", "2=1; 1=0"), "dist_faults": ("negative", "0, 6000"),
             "as_log_idw": ("positive", None), "sb_log_idw": ("positive", None)}
    for r in dlg._rows:
        if r["layer"].name in setup:
            rule, spec = setup[r["layer"].name]
            r["use"].setChecked(True)
            r["rule"].setCurrentIndex(r["rule"].findData(rule))
            if spec:
                r["spec"].setText(spec)
            assert "," in r["spec"].text() or "=" in r["spec"].text()
    dlg.weighting.setCurrentIndex(dlg.weighting.findData("ahp"))
    assert not dlg.matrix.isHidden() and dlg.matrix.rowCount() == 4
    names = [dlg.matrix.verticalHeaderItem(i).text() for i in range(4)]
    want = {("geology_host", "dist_faults"): "3", ("geology_host", "as_log_idw"): "2",
            ("dist_faults", "as_log_idw"): "1/2", ("dist_faults", "sb_log_idw"): "1/3",
            ("as_log_idw", "sb_log_idw"): "1/2"}
    for (a, b), v in want.items():
        i, j = names.index(a), names.index(b)
        dlg.matrix.item(min(i, j), max(i, j)).setText(v if i < j else f"1/{v}" if "/" not in v else v.split("/")[1])
    app.processEvents()
    assert "CR" in dlg.cr_label.text() and "✓" in dlg.cr_label.text()
    i, j = names.index("dist_faults"), names.index("geology_host")
    assert dlg.matrix.item(max(i, j), min(i, j)).text() in ("1/3", "3")       # reciprocal filled in
    assert dlg.validate() is None
    shot(dlg, "49_index_overlay_dialog")
    params = dlg.params()
    assert params["weighting"] == "ahp" and np.array(params["pairwise"]).shape == (4, 4)
    method = REGISTRY["index_overlay"]
    runs = []
    win.run_task(method.run, lambda res: runs.append(win.method_finished(method, res)), "IO", params)
    wait_tasks(win, app, timeout=300)
    assert not errors, errors
    run_id, run_dir = runs[0]
    assert {"index.tif", "evidence.csv", "ahp.csv", "validation.csv"} <= set(os.listdir(run_dir))
    rd = [d for d in win._dialogs if isinstance(d, IndexOverlayResultsDialog)][-1]
    assert rd.result.evaluation.curves["training"].auc > 0.8 and rd.tabs.count() == 2
    w = rd.result.tables["evidence"].set_index("layer").weight
    assert w["geology_host"] == pytest.approx(w.max())
    shot(rd, "50_index_overlay_results")
    win.store.dirty = False
    win.close()

@need_data
def test_topsis_workflow(app, monkeypatch, tmp_path):
    """TOPSIS / VIKOR: evidence scores, entropy and AHP weights, method switch, run files, results dialog."""
    from PyQt5.QtWidgets import QMessageBox
    from core.methods import PLANNED, REGISTRY
    from gui.main_window import MainWindow
    from logic.knowledge_driven.index_overlay.dialogs import IndexOverlayDialog
    from logic.knowledge_driven.topsis.dialogs import TopsisDialog, TopsisResultsDialog

    errors = []
    monkeypatch.setattr(QMessageBox, "critical", lambda *a, **k: errors.append(a[2] if len(a) > 2 else a))
    assert not any(p[2] == "TOPSIS" for p in PLANNED)
    win = MainWindow()
    win.resize(1500, 920)
    win.show()
    win.open_project(demo_copy(tmp_path))
    assert win.method_actions["topsis"].isEnabled()
    dlg = TopsisDialog(win.store, win)
    assert dlg.validate() is not None and dlg.weighting.findData("entropy") >= 0
    assert IndexOverlayDialog(win.store, win).weighting.findData("entropy") < 0      # TOPSIS-only option
    setup = {"geology_host": ("classes", "2=1; 1=0"), "dist_faults": ("negative", None),
             "as_log_idw": ("positive", None), "sb_log_idw": ("positive", None)}
    for r in dlg._rows:
        if r["layer"].name in setup:
            rule, spec = setup[r["layer"].name]
            r["use"].setChecked(True)
            r["rule"].setCurrentIndex(r["rule"].findData(rule))
            if spec:
                r["spec"].setText(spec)
    # VIKOR: v enabled, normalisation disabled, prefix follows the method
    dlg.method.setCurrentIndex(dlg.method.findData("vikor"))
    assert dlg.v.isEnabled() and not dlg.normalisation.isEnabled() and dlg.prefix.text() == "vikor"
    dlg.method.setCurrentIndex(dlg.method.findData("topsis"))
    assert dlg.normalisation.isEnabled() and not dlg.v.isEnabled() and dlg.prefix.text() == "topsis"
    dlg.weighting.setCurrentIndex(dlg.weighting.findData("ahp"))
    assert not dlg.matrix.isHidden() and dlg.matrix.rowCount() == 4
    names = [dlg.matrix.verticalHeaderItem(i).text() for i in range(4)]
    want = {("geology_host", "dist_faults"): 3, ("geology_host", "as_log_idw"): 2, ("geology_host", "sb_log_idw"): 1,
            ("dist_faults", "as_log_idw"): 1 / 2, ("dist_faults", "sb_log_idw"): 1 / 3, ("as_log_idw", "sb_log_idw"): 1 / 2}
    for (a, b), v in want.items():
        i, j = names.index(a), names.index(b)
        v = v if i < j else 1 / v
        dlg.matrix.item(min(i, j), max(i, j)).setText(f"{v:g}" if v >= 1 else f"1/{1 / v:g}")
    app.processEvents()
    assert "✓" in dlg.cr_label.text()
    dlg.o_parts.setChecked(True)
    assert dlg.validate() is None
    shot(dlg, "51_topsis_dialog")
    params = dlg.params()
    assert params["method"] == "topsis" and params["normalisation"] == "vector" and params["out_parts"]
    method = REGISTRY["topsis"]
    runs = []
    win.run_task(method.run, lambda res: runs.append(win.method_finished(method, res)), "TOPSIS", params)
    wait_tasks(win, app, timeout=300)
    assert not errors, errors
    run_id, run_dir = runs[0]
    assert {"topsis.tif", "d_plus.tif", "d_minus.tif", "evidence.csv", "ahp.csv", "validation.csv"} <= set(
        os.listdir(run_dir))
    rd = [d for d in win._dialogs if isinstance(d, TopsisResultsDialog)][-1]
    assert rd.result.evaluation.curves["training"].auc > 0.8 and rd.tabs.count() == 2
    shot(rd, "52_topsis_results")
    # VIKOR with entropy weights (no AHP tab)
    dlg2 = TopsisDialog(win.store, win)
    assert dlg2.method.currentData() == "topsis" and dlg2.weighting.currentData() == "ahp"     # remembered
    dlg2.method.setCurrentIndex(dlg2.method.findData("vikor"))
    dlg2.weighting.setCurrentIndex(dlg2.weighting.findData("entropy"))
    assert dlg2.matrix.isHidden() and dlg2.validate() is None
    p2 = dlg2.params()
    res = method.run(p2, lambda *a: None)
    assert {o.name for o in res.outputs} >= {"vikor", "S", "R", "Q"} and "ahp" not in res.tables
    assert res.tables["evidence"].weight.sum() == pytest.approx(1.0)
    win.store.dirty = False
    win.close()


# Carlin knowledge table of the hybrid workflows: layer -> (process, direction or class scores, expert weight)
HYBRID_SETUP = {"geology_host": ("trap", "2=0.9; 1=0.1", 1.0), "dist_faults": ("pathway", -1, 1.0),
                "as_log_idw": ("deposition", 1, 1.0), "sb_log_idw": ("deposition", 1, 1.0)}


def _hybrid_workflow(app, monkeypatch, tmp_path, key, dialog_cls, options, files, shot_name, checks=None,
                     setup=HYBRID_SETUP, timeout=900):
    """Hybrid method through the GUI: knowledge table, options, worker run, run files, generic results dialog."""
    from PyQt5.QtWidgets import QMessageBox
    from core.methods import REGISTRY
    from gui.dialogs.evaluation_dialogs import EvaluationDialog
    from gui.main_window import MainWindow
    from logic.hybrid.common.dialogs import HybridResultsDialog

    errors = []
    monkeypatch.setattr(QMessageBox, "critical", lambda *a, **k: errors.append(a[2] if len(a) > 2 else a))
    win = MainWindow()
    win.resize(1500, 920)
    win.show()
    win.open_project(demo_copy(tmp_path))
    assert win.method_actions[key].isEnabled()
    dlg = dialog_cls(win.store, win)
    assert dlg.validate() is not None                           # nothing ticked yet
    kt = dlg.knowledge
    for r in kt.rows:
        if r["layer"].name in setup:
            proc, d, w = setup[r["layer"].name]
            r["use"].setChecked(True)
            r["process"].setCurrentIndex(r["process"].findData(proc))
            if isinstance(d, str):
                assert r["scores"] is not None
                r["scores"].setText(d)
            else:
                r["direction"].setCurrentIndex(0 if d > 0 else 1)
            r["weight"].setValue(w)
    assert set(kt.process_weights()) == {v[0] for v in setup.values()}
    for name, val in options.items():
        w = dlg.opts[name]
        if hasattr(w, "setChecked"):
            w.setChecked(val)
        elif hasattr(w, "findData"):
            w.setCurrentIndex(w.findData(val))
        else:
            w.setValue(val)
    assert dlg.validate() is None
    shot(dlg, shot_name + "_dialog")
    params = dlg.params()
    assert {e["name"]: e["process"] for e in params["evidence"]} == {k: v[0] for k, v in setup.items()}
    method = REGISTRY[key]
    runs = []
    win.run_task(method.run, lambda res: runs.append(win.method_finished(method, res)), key, params)
    wait_tasks(win, app, timeout=timeout)
    assert not errors, errors
    run_id, run_dir = runs[0]
    assert set(files) <= set(os.listdir(run_dir)), sorted(os.listdir(run_dir))
    rd = [d for d in win._dialogs if isinstance(d, HybridResultsDialog)][-1]
    ev = rd.result.evaluation
    assert ev.curves["training"].auc > 0.8 and "test" in ev.curves
    assert rd.tabs.count() >= 2
    shot(rd, shot_name + "_results")
    if checks:
        checks(rd)
    rd.show_evaluation()
    app.processEvents()
    opened = rd.findChildren(EvaluationDialog)
    assert opened and opened[-1].isVisible()
    opened[-1].close()
    win.store.dirty = False
    win.close()
    return rd


@need_data
def test_ms_kprb_workflow(app, monkeypatch, tmp_path):
    from logic.hybrid.ms_kprb.dialogs import KPRBDialog

    def checks(rd):
        lw = rd.result.tables["layer_weights"].set_index("layer")
        assert lw.expert_weight["as_log_idw"] == pytest.approx(0.5)
        assert len(rd.canvases) >= 2                      # weight charts
        assert rd.result.tables["tree_selection"].times_selected.sum() == 3

    _hybrid_workflow(app, monkeypatch, tmp_path, "ms_kprb", KPRBDialog, {"draws": 3, "out_processes": True},
                     {"prob.tif", "prob_sd.tif", "knowledge_prob.tif", "delta.tif", "expert.tif",
                      "limiting_process.tif", "process_trap.tif", "layer_weights.csv", "tree_selection.csv",
                      "validation.csv"}, "40_ms_kprb", checks)


@need_data
def test_kp_rf_workflow(app, monkeypatch, tmp_path):
    from logic.hybrid.kp_rf.dialogs import KPRFDialog

    def checks(rd):
        res = rd.result.extra["kprf"]
        assert res.lam == pytest.approx(20 / (20 + res.stats["deposits"]))
        assert any(n.startswith("F[") for n in rd.result.tables["importance"].layer)

    _hybrid_workflow(app, monkeypatch, tmp_path, "kp_rf", KPRFDialog, {"n_bags": 2, "n_trees": 100},
                     {"prob.tif", "prob_rf.tif", "prob_rf_sd.tif", "prob_k.tif", "expert.tif", "importance.csv",
                      "validation.csv"}, "41_kp_rf", checks)


@need_data
def test_kw_svm_workflow(app, monkeypatch, tmp_path):
    from logic.hybrid.kw_svm.dialogs import KWSVMDialog

    def checks(rd):
        kw = rd.result.tables["kernel_weights"]
        assert kw.kernel_weight.mean() == pytest.approx(1.0)

    _hybrid_workflow(app, monkeypatch, tmp_path, "kw_svm", KWSVMDialog, {"n_bags": 2},
                     {"prob.tif", "prob_sd.tif", "expert.tif", "kernel_weights.csv", "validation.csv"},
                     "42_kw_svm", checks)


@need_data
def test_bs_wofe_workflow(app, monkeypatch, tmp_path):
    from logic.hybrid.bs_wofe.dialogs import BSWofEDialog

    def checks(rd):
        w = rd.result.tables["weights"]
        assert set(w.layer) == set(HYBRID_SETUP) and (w.shrinkage.between(0, 1)).all()
        assert 0 < rd.result.extra["bswofe"].nu <= 1

    _hybrid_workflow(app, monkeypatch, tmp_path, "bs_wofe", BSWofEDialog, {"n_classes": 5},
                     {"prob.tif", "logit_sd.tif", "weights.csv", "layers.csv", "validation.csv"}, "43_bs_wofe",
                     checks)


@need_data
def test_pt_hurdle_workflow(app, monkeypatch, tmp_path):
    from logic.hybrid.pt_hurdle.dialogs import PTHurdleDialog
    setup = {k: v for k, v in HYBRID_SETUP.items() if k != "geology_host"}

    def checks(rd):
        res = rd.result.extra["pthurdle"]
        assert res.stats["deposits_in_tract"] >= res.stats["deposits"] - 2 and 10 < res.stats["tract_pct"] < 40
        assert res.gate.max() == 1.0

    rd = _hybrid_workflow(app, monkeypatch, tmp_path, "pt_hurdle", PTHurdleDialog,
                          {"n_bags": 2, "engine": "bs_wofe"},
                          {"prob.tif", "gate.tif", "conditional.tif", "tract.tif", "weights.csv", "validation.csv"},
                          "44_pt_hurdle", checks, setup=setup)
    assert rd.result.extra["pthurdle"].stats["engine"] == "bs_wofe"


@need_data
def test_fuzzy_wofe_workflow(app, monkeypatch, tmp_path):
    from core.methods import PLANNED
    from logic.hybrid.fuzzy_wofe.dialogs import FuzzyWofEDialog
    assert not any(p[2] == "Fuzzy weights of evidence" for p in PLANNED)

    def checks(rd):
        res = rd.result.extra["fwofe"]
        lay = rd.result.tables["layers"]
        assert set(lay.layer) == set(HYBRID_SETUP) and lay.beta.notna().all()
        assert res.prob.sum() == pytest.approx(res.stats["deposits"], rel=0.02)     # modified: Σ P = N
        assert rd.tabs.count() == 4

    _hybrid_workflow(app, monkeypatch, tmp_path, "fuzzy_wofe", FuzzyWofEDialog, {"n_classes": 8},
                     {"post.tif", "post_classic.tif", "logit_sd.tif", "expert.tif", "weights.csv", "layers.csv",
                      "ci_test.csv", "validation.csv"}, "47_fuzzy_wofe", checks)

@need_data
def test_neuro_fuzzy_workflow(app, monkeypatch, tmp_path):
    from core.methods import PLANNED
    from logic.hybrid.neuro_fuzzy.dialogs import NeuroFuzzyDialog
    assert not any("ANFIS" in p[2] for p in PLANNED)

    def checks(rd):
        res = rd.result.extra["neurofuzzy"]
        assert res.input_mode == "layers" and len(rd.result.tables["rules"]) == 16      # 2^4 rules, 4 layers
        mf = rd.result.tables["membership"]
        assert set(mf.input) == set(HYBRID_SETUP) and len(mf) == 8
        assert set(rd.result.tables["importance"].input) == set(HYBRID_SETUP)
        assert rd.tabs.count() >= 5

    rd = _hybrid_workflow(app, monkeypatch, tmp_path, "neuro_fuzzy", NeuroFuzzyDialog, {"n_bags": 2, "epochs": 40},
                          {"prob.tif", "prob_sd.tif", "expert.tif", "rules.csv", "membership.csv", "validation.csv"},
                          "56_neuro_fuzzy", checks)
    assert rd.result.extra["neurofuzzy"].stats["bags"] == 2


@need_data
def test_dc_fuzzy_workflow(app, monkeypatch, tmp_path):
    from core.methods import PLANNED
    from logic.hybrid.dc_fuzzy.dialogs import DCFuzzyDialog
    assert not any("Data-calibrated" in p[2] for p in PLANNED)

    def checks(rd):
        lay = rd.result.tables["layers"]
        assert set(lay.layer) == set(HYBRID_SETUP) and lay.agrees_with_expert.all()
        mem = rd.result.tables["memberships"]
        assert set(mem.layer) == set(HYBRID_SETUP) and mem.mu_data.between(0.01, 0.99).all()
        assert rd.tabs.count() >= 3

    rd = _hybrid_workflow(app, monkeypatch, tmp_path, "dc_fuzzy", DCFuzzyDialog,
                          {"calibration": "lr", "out_memberships": True},
                          {"fuzzy.tif", "expert.tif", "mu_sb_log_idw.tif", "layers.csv", "memberships.csv",
                           "validation.csv"}, "57_dc_fuzzy", checks)
    assert not rd.result.extra["dcfuzzy"].messages


@need_data
def test_kt_rk_workflow(app, monkeypatch, tmp_path):
    """KT-RK on the Curnamona demo: La of the surface samples estimated from a knowledge trend + forest + kriging."""
    from PyQt5.QtWidgets import QMessageBox
    from core.methods import REGISTRY
    from gui.main_window import MainWindow
    from logic.hybrid.common.dialogs import HybridResultsDialog
    from logic.hybrid.kt_rk.dialogs import KTRKDialog

    errors = []
    monkeypatch.setattr(QMessageBox, "critical", lambda *a, **k: errors.append(a[2] if len(a) > 2 else a))
    win = MainWindow()
    win.resize(1500, 920)
    win.show()
    win.open_project(demo_copy(tmp_path, index=1))
    assert win.method_actions["kt_rk"].isEnabled()
    dlg = KTRKDialog(win.store, win)
    assert dlg.test_pct.value() == 0 and not KTRKDialog.NEED_DEPOSITS
    setup = {"rad_th": ("source", 1), "geochem_nb_log_idw": ("deposition", 1)}
    for r in dlg.knowledge.rows:
        if r["layer"].name in setup:
            r["use"].setChecked(True)
            r["process"].setCurrentIndex(r["process"].findData(setup[r["layer"].name][0]))
    dlg.samples.set_layer(win.store.by_name("geochem_samples"))
    dlg.field.setCurrentIndex(dlg.field.findData("La"))
    assert dlg.field.currentData() == "La"
    dlg.opts["folds"].setValue(5)
    assert dlg.validate() is None
    shot(dlg, "45_kt_rk_dialog")
    params = dlg.params()
    assert params["field"] == "La" and len(params["samples"]) == 615
    method = REGISTRY["kt_rk"]
    runs = []
    win.run_task(method.run, lambda res: runs.append(win.method_finished(method, res)), "KT-RK", params)
    wait_tasks(win, app, timeout=900)
    assert not errors, errors
    run_id, run_dir = runs[0]
    assert {"prediction.tif", "log_prediction.tif", "log_sd.tif", "trend.tif", "cv.csv", "observations.csv"} <=         set(os.listdir(run_dir))
    rd = [d for d in win._dialogs if isinstance(d, HybridResultsDialog)][-1]
    res = rd.result.extra["ktrk"]
    assert res.stats["observations"] > 100 and res.cv.selected.sum() == 1 and rd.tabs.count() == 3
    assert rd.result.evaluation is not None                      # deposits validate the La map
    shot(rd, "46_kt_rk_results")
    win.store.dirty = False
    win.close()


@need_data
def test_curnamona_project_and_wofe(app, tmp_path):
    from gui.main_window import DEMO_PROJECTS, MainWindow
    from core.methods import REGISTRY

    if not os.path.isfile(list(DEMO_PROJECTS.values())[1]):
        pytest.skip("Curnamona demo not built")
    win = MainWindow()
    errors = win.store.load_project(demo_copy(tmp_path, 1))
    assert not errors, errors
    assert win.store.crs.to_epsg() == 28354
    s = win.store
    ev = []
    for n, t in (("dist_faults_class", "ascending"), ("geochem_la_q8", "descending"), ("geology_units", "categorical")):
        l = s.by_name(n)
        ev.append(dict(name=n, data=l.data, grid=l.grid, labels=l.labels(), type=t, criterion="max_contrast",
                       min_stud=0.5))
    m = s.by_name("study_mask")
    params = dict(evidence=ev, deposits=s.by_name("deposits").gdf, mask=(m.data, m.grid), test_fraction=0.0,
                  seed=0, unit_cell_km2=1.0, missing="zero", zero_deposit="zero", prefix="ree")
    res = REGISTRY["wofe"].run(params, lambda p, msg="": None)
    assert res.extra["wofe"].n_deposits == 7
    assert 0.5 < res.evaluation.curves["training"].auc <= 1.0
    win.store.dirty = False                   # discard (no modal "save changes?" prompt in tests)
    win.close()


@need_data
def test_identify_dialog_and_coordinates(app, tmp_path):
    from gui.main_window import DEMO_PROJECTS, MainWindow

    win = MainWindow()
    win.resize(1500, 920)
    win.show()
    win.open_project(demo_copy(tmp_path))
    win.canvas.draw()
    app.processEvents()

    # coordinate bar below the map
    dep = win.store.by_name("deposits").gdf.geometry.iloc[0]
    win.canvas.coords_changed.emit(dep.x, dep.y)
    txt = win.map_panel.xy_label.text()
    assert txt.startswith("X ") and "Y " in txt and "°" not in txt
    win.map_panel._geographic = True                   # a geographic project CRS shows decimal degrees as X / Y
    win.canvas.coords_changed.emit(-116.123456789, 40.5)
    assert win.map_panel.xy_label.text() == "X -116.123457   Y 40.500000"
    win.map_panel._crs_changed()
    win.canvas.coords_cleared.emit()
    assert "—" in win.map_panel.xy_label.text()

    # identify tool opens the dialog; clicking a deposit lists layers + attributes
    win.a_ident.trigger()
    app.processEvents()
    dlg = win.identify_dlg
    assert dlg is not None and dlg.isVisible()
    dlg.scope.setCurrentIndex(0)                       # all layers
    win.canvas.identify_requested.emit(dep.x, dep.y)
    app.processEvents()
    names = [dlg.tree.topLevelItem(i).text(0) for i in range(dlg.tree.topLevelItemCount())]
    assert "deposits" in names and "geology" in names
    dep_item = dlg.tree.topLevelItem(names.index("deposits"))
    dlg.tree.setCurrentItem(dep_item.child(0))
    assert dlg.tree.currentItem() is not None and dlg.attrs.rowCount() > 3
    fields = [dlg.attrs.item(i, 0).text() for i in range(dlg.attrs.rowCount())]
    assert "SITE" in fields
    win.canvas.coords_changed.emit(dep.x + 1234.5, dep.y - 987.6)
    shot(win, "08_identify_window")
    shot(dlg, "09_identify_dialog")
    # selecting the geology layer shows its class attributes
    geo = next(dlg.tree.topLevelItem(i) for i in range(dlg.tree.topLevelItemCount())
               if dlg.tree.topLevelItem(i).text(0) == "geology")
    dlg.tree.setCurrentItem(geo)
    fields = [dlg.attrs.item(i, 0).text() for i in range(dlg.attrs.rowCount())]
    assert "class · label" in fields and "Row, column" in fields

    # switching tool hides the dialog; closing it returns to pan
    win.a_pan.trigger()
    app.processEvents()
    assert not dlg.isVisible()
    win.a_ident.trigger()
    app.processEvents()
    assert dlg.isVisible()
    dlg.close()
    app.processEvents()
    assert win.canvas.mode == "pan" and win.a_pan.isChecked()
    win.store.dirty = False                   # discard (no modal "save changes?" prompt in tests)
    win.close()


@need_data
def test_layer_draw_order_follows_toc(app, tmp_path):
    """The map draws layers in the TOC order: a vector layer below a raster in the TOC is hidden by it, above it is
    drawn on top; labels stay above every layer."""
    from matplotlib.collections import LineCollection, PathCollection
    from gui.main_window import MainWindow

    win = MainWindow()
    win.resize(1400, 900)
    win.show()
    win.store.load_project(demo_copy(tmp_path))
    st = win.store
    geo, dep, flt, area = (st.by_name(n) for n in ("geology", "deposits", "faults", "study_area"))
    for l in st.layers:
        l.visible = l in (geo, dep, flt, area)
    dep.labels = dict(enabled=True, field=next(c for c in dep.gdf.columns if c != "geometry"), size=8)
    assert geo.data is not None and geo.loaded                             # read now (lazy loading otherwise)

    def zorders():
        win.canvas.redraw()
        win.canvas.draw()
        app.processEvents()
        ax = win.canvas.ax
        img = [a.get_zorder() for a in ax.images]
        pts = [a.get_zorder() for a in ax.collections if isinstance(a, PathCollection) and len(a.get_offsets()) > 1]
        lines = [a.get_zorder() for a in ax.collections if isinstance(a, LineCollection)]
        polys = [a.get_zorder() for a in ax.collections if isinstance(a, PathCollection) and len(a.get_offsets()) <= 1]
        labels = [t.get_zorder() for t in win.canvas._label_artists]
        return img, pts, lines, polys, labels

    others = [l for l in st.layers if l not in (geo, dep, flt, area)]
    # TOC (top first): geology raster above the three vector layers
    st.reorder([geo.id, dep.id, flt.id, area.id] + [l.id for l in others])
    img, pts, lines, polys, labels = zorders()
    assert len(img) == 1 and pts and lines and polys
    assert max(pts + lines + polys) < img[0], (img, pts, lines, polys)     # vectors hidden by the raster
    assert min(pts) > max(lines) > max(polys)                              # deposits > faults > study area
    assert labels and min(labels) > img[0]                                 # labels on top of every layer
    shot(win, "48_order_raster_on_top")
    # vectors above the raster
    st.reorder([dep.id, flt.id, geo.id, area.id] + [l.id for l in others])
    img, pts, lines, polys, labels = zorders()
    assert min(pts + lines) > img[0] > max(polys)
    # the TOC move buttons give the same result
    st.move(area, -3)                                                      # study area to the top
    img, pts, lines, polys, labels = zorders()
    assert min(polys) > max(pts + lines + img)
    win.store.dirty = False
    win.close()

@need_data
def test_middle_button_pan(app, tmp_path):
    """Middle-button drag pans in every tool mode with a snapshot; the legend never moves."""
    from PyQt5.QtCore import QEvent, QPoint, QPointF, QRect, Qt
    from PyQt5.QtGui import QMouseEvent
    from PyQt5.QtWidgets import QApplication
    from gui.main_window import DEMO_PROJECTS, MainWindow

    win = MainWindow()
    win.resize(1400, 900)
    win.show()
    win.store.load_project(demo_copy(tmp_path))
    geo = win.store.by_name("geology")
    geo.visible = True
    geo.load()                                      # rasters are read lazily: the snapshots need real pixels
    win.store.set_active(geo)                       # categorical legend in the lower-right corner
    c = win.canvas
    c.redraw()
    c.zoom_full()
    c.draw()
    app.processEvents()
    identified = []
    c.identify_requested.connect(lambda x, y: identified.append((x, y)))

    def send(kind, pos, button, buttons):
        QApplication.sendEvent(c, QMouseEvent(kind, QPointF(pos), button, buttons, Qt.NoModifier))
        app.processEvents()

    def pixels(img, rect):
        import numpy as np
        from PyQt5.QtGui import QImage
        im = img.copy(rect).convertToFormat(QImage.Format_RGBA8888)
        ptr = im.bits()
        ptr.setsize(im.byteCount())
        a = np.frombuffer(ptr, np.uint8).reshape(im.height(), im.bytesPerLine())[:, : im.width() * 4]
        return a.reshape(im.height(), im.width(), 4).astype(int)

    def same(img_a, img_b, rect):
        """True if a region is visually unchanged (allows anti-aliasing noise on a few edge pixels)."""
        d = abs(pixels(img_a, rect) - pixels(img_b, rect)).sum(axis=2)
        return (d > 0).mean() < 0.001 and d.max() <= 12

    def legend_rect():
        r = c._dpr()
        bb = c.ax.get_legend().get_window_extent()
        return QRect(int(bb.x0 / r) + 2, int((c.figure.bbox.height - bb.y1) / r) + 2,
                     int(bb.width / r) - 4, int(bb.height / r) - 4)

    for mode in ("pan", "zoom_in", "identify"):
        c.set_mode(mode)
        c.zoom_full()                   # same start view each time: the test region must cover the map, not blank
        c.draw()
        app.processEvents()
        xl0, yl0 = c.ax.get_xlim(), c.ax.get_ylim()
        upp = (xl0[1] - xl0[0]) / c.ax.bbox.width
        lr = legend_rect()
        send(QEvent.MouseButtonPress, QPoint(400, 300), Qt.MiddleButton, Qt.MiddleButton)
        assert c._pan_pixmap is not None and c._pan_legend is not None
        before = c.grab().toImage()
        for i in range(1, 6):
            send(QEvent.MouseMove, QPoint(400 + 20 * i, 300 + 10 * i), Qt.NoButton, Qt.MiddleButton)
        assert c._pan_offset == QPoint(100, 50)               # snapshot follows the cursor
        assert c.ax.get_xlim() == xl0                         # no re-render while dragging
        after = c.grab().toImage()
        assert same(before, after, lr), "legend moved during panning"
        assert not same(before, after, QRect(300, 200, 200, 200)), "map did not move"
        send(QEvent.MouseButtonRelease, QPoint(500, 350), Qt.MiddleButton, Qt.NoButton)
        assert c._pan_pixmap is None and c._pan is None and c.mode == mode
        xl1, yl1 = c.ax.get_xlim(), c.ax.get_ylim()
        assert xl1[0] - xl0[0] == pytest.approx(-100 * upp, rel=1e-6)   # map moved right with the mouse
        assert yl1[0] - yl0[0] == pytest.approx(50 * upp, rel=1e-6)     # ...and down
        assert c.ax.get_legend() is not None and c.ax.get_legend().get_visible()
    assert not identified                                     # middle click never identifies

    # continuous raster legend (colour bar + title in the figure) also stays put
    post = next((l for l in win.store.rasters() if not l.categorical), None)
    post.visible = True
    post.load()
    win.store.set_active(post)
    c.redraw()
    c.draw()
    app.processEvents()
    cax = c._legend_artists[0]
    r = c._dpr()
    bb = cax.get_window_extent()
    # inset 2 px: the 1-px anti-aliased outline blends with the map pixels underneath
    cr = QRect(int(bb.x0 / r) + 2, int((c.figure.bbox.height - bb.y1) / r) + 2, int(bb.width / r) - 4,
               int(bb.height / r) - 4)
    send(QEvent.MouseButtonPress, QPoint(400, 300), Qt.MiddleButton, Qt.MiddleButton)
    before = c.grab().toImage()
    send(QEvent.MouseMove, QPoint(330, 260), Qt.NoButton, Qt.MiddleButton)
    after = c.grab().toImage()
    assert same(before, after, cr), "colour bar moved during panning"
    send(QEvent.MouseButtonRelease, QPoint(330, 260), Qt.MiddleButton, Qt.NoButton)

    # identify marker is part of the map: it moves with the snapshot
    assert c._marker_xy is None
    xl0 = c.ax.get_xlim()
    send(QEvent.MouseButtonPress, QPoint(300, 300), Qt.MiddleButton, Qt.MiddleButton)
    send(QEvent.MouseButtonRelease, QPoint(300, 300), Qt.MiddleButton, Qt.NoButton)
    assert c.ax.get_xlim() == xl0                             # click without moving: no change
    win.store.dirty = False                   # discard (no modal "save changes?" prompt in tests)
    win.close()


def test_new_project_folder_prompts(app, tmp_path, monkeypatch):
    """New project: non-empty folder -> 'Choose another folder' / 'Continue'; Save has no dialog; Save as copies."""
    from PyQt5.QtWidgets import QFileDialog, QMessageBox
    from gui.main_window import MainWindow

    full = tmp_path / "full"
    full.mkdir()
    (full / "something.txt").write_text("x")
    empty = tmp_path / "empty"
    empty.mkdir()
    picks = [str(full), str(empty)]                         # 1st pick not empty, 2nd pick empty
    monkeypatch.setattr(QFileDialog, "getExistingDirectory", lambda *a, **k: picks.pop(0))
    clicked = []

    def click(label):
        def exec_(box):
            btn = next(b for b in box.buttons() if b.text() == label)
            clicked.append(box.text())
            btn.click()
            return 0
        return exec_

    monkeypatch.setattr(QMessageBox, "exec_", click("Choose another folder"))
    from gui.dialogs.project_dialog import ProjectPropertiesDialog
    opened = []                                             # New project opens Project properties (CRS tab): close it
    monkeypatch.setattr(ProjectPropertiesDialog, "exec_", lambda dlg: opened.append(dlg.tabs.currentIndex()) or 0)
    win = MainWindow()
    win.new_project()
    assert len(opened) == 1
    assert clicked == ["The selected folder is not empty."]
    assert win.store.project_path == str(empty / "project.mpm.json")
    assert (empty / "project.mpm.json").is_file() and (empty / "results").is_dir()
    assert win.a_add_r.isEnabled()

    # Save: no file dialog (getExistingDirectory would fail: picks is empty)
    assert win.save_project()

    # Save as: non-empty target -> Continue -> copied
    target = tmp_path / "copy"
    target.mkdir()
    (target / "old.txt").write_text("x")
    picks.append(str(target))
    monkeypatch.setattr(QMessageBox, "exec_", click("Continue"))
    assert win.save_project_as()
    assert win.store.project_path == str(target / "project.mpm.json") and (target / "results").is_dir()
    win.store.dirty = False
    win.close()


@need_data
def test_target_delineation_dialog(app, tmp_path):
    """P-A / C-A threshold dialog: automatic threshold, drag the line, Confirm -> binary layer 1 red / 0 transparent."""
    from matplotlib.backend_bases import MouseEvent
    from gui.dialogs.target_dialog import TargetDialog
    from gui.main_window import MainWindow

    win = MainWindow()
    win.resize(1400, 900)
    win.open_project(demo_copy(tmp_path))
    dlg = TargetDialog(win.store, win)
    dlg.raster.set_layer(win.store.by_name("as_log_idw"))
    dlg.method.setCurrentIndex(0)                                  # P-A
    dlg.deposits.set_layer(win.store.by_name("deposits"))
    app.processEvents()
    assert dlg._pa is not None and dlg._pa.intersection
    auto = dlg._threshold
    assert auto == pytest.approx(dlg._pa.intersection["threshold"])
    st = dlg._last_stats
    assert st["deposits_in_target_pct"] == pytest.approx(100 - st["target_area_pct"], abs=3)   # P = 100 - A
    shot(dlg, "10_target_pa")

    # drag the dashed line with the mouse to a higher value
    dlg.chart.draw()
    ax = dlg._axes[0]
    def mouse(kind, xdata):
        x, y = ax.transData.transform((xdata, 50))
        MouseEvent(kind, dlg.chart, x, y, button=1)._process()
    new_x = auto + 0.25 * (dlg._values.max() - auto)
    mouse("button_press_event", auto)
    mouse("motion_notify_event", new_x)
    mouse("button_release_event", new_x)
    assert dlg._threshold == pytest.approx(new_x, rel=1e-3) and dlg.thr.value() == pytest.approx(new_x, rel=1e-3)
    assert dlg._last_stats["target_area_pct"] < st["target_area_pct"]

    # C-A fractal
    dlg.method.setCurrentIndex(1)
    app.processEvents()
    assert dlg._ca is not None and len(dlg._ca.breakpoints) == 2
    assert dlg._threshold == pytest.approx(dlg._ca.breakpoints[-1])
    shot(dlg, "11_target_ca")

    # Confirm -> binary layer written into the project run folder
    out = dlg.output()
    lyr = win.add_target_layer(out)
    assert lyr.categorical and set(np.unique(lyr.data.compressed())) <= {0, 1}
    assert lyr.source.endswith("target.tif") and os.path.isfile(lyr.source)
    folder = os.path.dirname(lyr.source)
    assert {"target.tif", "target.classes.csv", "plot.png", "target_report.csv"} <= set(os.listdir(folder))
    from gui.map_canvas import category_colors
    cols = category_colors(lyr, lyr.classes())
    assert cols == ["#00000000", "#e53935"]                          # 0 transparent, 1 red
    win.store.set_active(lyr)
    win.canvas.draw()
    shot(win, "12_target_map")
    win.store.dirty = False
    win.close()


@need_data
def test_small_display_1366x768(app, tmp_path, monkeypatch):
    """Everything fits a 1366 x 768 display (taskbar 40 px); identify dialog 80 px from the right screen edge."""
    from PyQt5.QtCore import QRect
    import gui.widgets as gw
    import gui.main_window as gmw
    from gui.main_window import MainWindow
    avail = QRect(0, 0, 1366, 728)
    monkeypatch.setattr(gw, "available_geometry", lambda widget=None: avail)
    monkeypatch.setattr(gmw, "available_geometry", lambda widget=None: avail)

    win = MainWindow()
    assert win.width() <= 1366 and win.height() <= 728
    win.show()
    win.open_project(demo_copy(tmp_path))
    app.processEvents()

    from gui.dialogs.data_dialogs import (AlignDialog, ClipDialog, ConvertDialog, DistanceDialog,
                                          PolygonToRasterDialog, ReclassDialog)
    from gui.dialogs.evaluation_dialogs import EvaluationDialog, ValidationDialog
    from gui.dialogs.layer_dialogs import LayerPropertiesDialog
    from gui.dialogs.target_dialog import TargetDialog
    from logic.data_driven.wofe.dialogs import WofEDialog
    from logic.data_driven.logistic_regression.dialogs import LogisticRegressionDialog
    from logic.data_driven.random_forest.dialogs import RandomForestDialog
    from logic.data_driven.svm.dialogs import SVMDialog
    from logic.data_driven.xgboost_mpm.dialogs import XGBoostDialog
    from logic.data_driven.rvm.dialogs import RVMDialog
    from logic.data_driven.mlp.dialogs import MLPDialog
    from logic.data_driven.elm.dialogs import ELMDialog
    from logic.data_driven.cnn.dialogs import CNNDialog
    from logic.data_driven.bayesian_network.dialogs import BayesianNetworkDialog
    from logic.knowledge_driven.fuzzy_logic.dialogs import FuzzyLogicDialog
    from logic.knowledge_driven.boolean_logic.dialogs import BooleanLogicDialog
    from logic.knowledge_driven.logistic_geometric.dialogs import LogisticGeometricDialog
    from logic.hybrid.ms_kprb.dialogs import KPRBDialog
    from logic.hybrid.kp_rf.dialogs import KPRFDialog
    from logic.hybrid.kw_svm.dialogs import KWSVMDialog
    from logic.hybrid.bs_wofe.dialogs import BSWofEDialog
    from logic.hybrid.pt_hurdle.dialogs import PTHurdleDialog
    from logic.hybrid.kt_rk.dialogs import KTRKDialog
    from logic.hybrid.fuzzy_wofe.dialogs import FuzzyWofEDialog
    from logic.knowledge_driven.index_overlay.dialogs import IndexOverlayDialog
    from logic.knowledge_driven.topsis.dialogs import TopsisDialog
    from logic.hybrid.neuro_fuzzy.dialogs import NeuroFuzzyDialog
    from logic.hybrid.dc_fuzzy.dialogs import DCFuzzyDialog
    from logic.data_driven.ebf.dialogs import EBFDialog
    from evaluation.metrics import evaluate
    lyr = win.store.by_name("as_log_idw")
    ev = evaluate(lyr.data, [10, 20], [10, 20], score_name="x")
    dialogs = [WofEDialog(win.store, win), LogisticRegressionDialog(win.store, win), RandomForestDialog(win.store, win),
               SVMDialog(win.store, win), XGBoostDialog(win.store, win),
               RVMDialog(win.store, win), MLPDialog(win.store, win),
               ELMDialog(win.store, win), CNNDialog(win.store, win),
               BayesianNetworkDialog(win.store, win), FuzzyLogicDialog(win.store, win),
               BooleanLogicDialog(win.store, win), LogisticGeometricDialog(win.store, win),
               KPRBDialog(win.store, win), KPRFDialog(win.store, win), KWSVMDialog(win.store, win),
               BSWofEDialog(win.store, win), PTHurdleDialog(win.store, win), KTRKDialog(win.store, win),
               FuzzyWofEDialog(win.store, win), IndexOverlayDialog(win.store, win), TopsisDialog(win.store, win),
               NeuroFuzzyDialog(win.store, win), DCFuzzyDialog(win.store, win), EBFDialog(win.store, win),
               TargetDialog(win.store, win),
               ReclassDialog(win.store, win),
               ConvertDialog(win.store, win), AlignDialog(win.store, win), ClipDialog(win.store, win),
               DistanceDialog(win.store, win), PolygonToRasterDialog(win.store, win),
               ValidationDialog(win.store, win), EvaluationDialog(ev, "t", win),
               LayerPropertiesDialog(lyr, win.store, win), LayerPropertiesDialog(win.store.vectors()[0], win.store, win)]
    for d in dialogs:
        d.show()
        app.processEvents()
        fg = d.frameGeometry()
        assert fg.height() <= 728 and fg.width() <= 1366, (type(d).__name__, fg)
        assert avail.contains(fg), (type(d).__name__, fg)
        buttons = getattr(d, "buttons", None)
        if buttons is not None:                         # OK / Cancel stay visible inside the dialog
            assert buttons.isVisible() and buttons.geometry().bottom() <= d.height()
        if type(d).__name__ in ("WofEDialog", "TargetDialog"):
            shot(d, f"13_small_{type(d).__name__}")
        d.close()

    win.a_ident.trigger()
    app.processEvents()
    fg = win.identify_dlg.frameGeometry()
    assert fg.right() + 1 == 1366 - 80                   # right border 80 px from the display edge
    assert avail.contains(fg)
    win.store.dirty = False
    win.close()


@need_data
def test_polygon_to_raster(app, tmp_path):
    """Data ▸ Polygon to raster: map units -> categorical raster on a reference grid or a study-area polygon grid."""
    from gui.dialogs.data_dialogs import PolygonToRasterDialog
    from gui.main_window import DEMO_PROJECTS, MainWindow
    from utils.raster_tools import read_class_table, read_raster

    if not os.path.isfile(list(DEMO_PROJECTS.values())[1]):
        pytest.skip("Curnamona demo not built")
    win = MainWindow()
    win.resize(1200, 800)
    win.show()
    win.open_project(demo_copy(tmp_path, 1))
    s = win.store
    assert win.a_poly2ras in win.menuBar().actions()[[a.text() for a in win.menuBar().actions()].index("&Data")].menu().actions()
    meso = s.by_name("mesoproterozoic")
    d = PolygonToRasterDialog(s, win)
    d.src.set_layer(meso)
    assert d.field.currentData() == "STRATNAME"                     # first text attribute
    d.field.setCurrentIndex(d.field.findData("MAPUNIT"))
    assert not d.codes.model().item(1).isEnabled()                   # text labels: sequential codes only
    assert d.grid_from.currentData() == "raster" and d.ref.layer().name == "study_mask"
    assert d.study.currentText() == "study_area" and d.path.text().endswith("mesoproterozoic_MAPUNIT.tif")
    assert d.name.text() == "mesoproterozoic_MAPUNIT" and "project grid" not in d.summary.text()
    shot(d, "55_polygon_to_raster")
    assert d.validate() is None
    lyr = d.run()
    assert lyr.categorical and lyr.grid.same_as(s.project_grid())
    t = lyr.class_table
    assert t.label.iloc[-1] == "Other" and "STRATNAME" in t.columns and t["count"].sum() == lyr.data.count()
    data, grid, _ = read_raster(lyr.source)                          # GeoTIFF + class table written
    assert grid.same_as(lyr.grid) and np.ma.count(data) == lyr.data.count()
    assert list(read_class_table(lyr.source).label) == list(t.label)
    study = s.by_name("study_area")
    from utils.raster_tools import rasterize_vector
    inside = rasterize_vector(study.gdf, grid, dtype="uint8") == 1
    assert (~np.ma.getmaskarray(lyr.data) == inside).all()           # every study cell classed, nothing outside
    s.add(lyr)

    d = PolygonToRasterDialog(s, win)                                # study-area polygon + coarser cell size
    d.src.set_layer(meso)
    d.grid_from.setCurrentIndex(d.grid_from.findData("polygon"))
    assert not d.study.model().item(1).isEnabled() and d.study.currentText() == "study_area"
    assert d.cell.value() == pytest.approx(s.cell_size) and d.cell.isEnabled()
    d.cell.setValue(s.cell_size * 2)
    d.extent.setCurrentIndex(d.extent.findData("study"))
    d.assign.setCurrentIndex(d.assign.findData("max_area"))
    d.gap.setCurrentIndex(d.gap.findData("nodata"))
    assert not d.gap_label.isEnabled() and "project grid" in d.summary.text()
    g = d.output_grid()
    pg = s.project_grid()
    assert g.res[0] == pytest.approx(2 * s.cell_size)
    assert ((g.bounds[0] - pg.bounds[0]) / g.res[0]) % 1 == pytest.approx(0, abs=1e-9)   # aligned to the project grid
    d.path.setText(str(tmp_path / "meso_coarse.tif"))
    assert d.validate() is None
    lyr2 = d.run()
    assert "Other" not in set(lyr2.class_table.label) and lyr2.data.count() < inside.sum()
    win.store.dirty = False
    win.close()


@need_data
def test_measure_tool(app, tmp_path):
    """Distance / area measuring: live values while the mouse moves, overlay cleared when the tool is left."""
    from PyQt5.QtCore import QSettings
    from gui.dialogs.measure_dialog import SETTINGS_KEY, MeasureDialog
    from gui.main_window import MainWindow
    from version import APP_NAME

    settings = QSettings("MPM", APP_NAME)
    saved = settings.value(SETTINGS_KEY)
    settings.setValue(SETTINGS_KEY, 0)
    try:
        win = MainWindow()
        win.resize(1400, 900)
        win.show()
        win.open_project(demo_copy(tmp_path))
        win.canvas.draw()
        app.processEvents()
        tb = win.findChild(type(win.addToolBar("x")).__mro__[0], "tb_main")
        acts = tb.actions()
        assert acts[acts.index(win.a_pan) + 1] is win.a_measure          # icon right of the pan tool

        win.a_measure.trigger()
        app.processEvents()
        dlg, c = win.measure_dlg, win.canvas
        assert dlg.isVisible() and dlg.tabs.currentIndex() == 0           # Distance is the default tab
        x0, y0 = 0.0, 4700000.0                                           # LCC central meridian (-117 deg)
        c.measure_clicked.emit(x0, y0, False)
        c.coords_changed.emit(x0, y0 + 3000)                              # mouse moves north 3 km
        app.processEvents()
        assert dlg.d_segment.text() == "3.000 km" and dlg.d_azimuth.text().startswith("0.00°")
        assert list(c._ov_artists[1].get_xdata()) == [x0, x0]             # dashed rubber band to the cursor
        c.coords_changed.emit(x0 + 4000, y0 + 3000)                       # moving updates the last segment
        assert dlg.d_segment.text() == "5.000 km"
        c.measure_clicked.emit(x0 + 4000, y0 + 3000, False)
        c.coords_changed.emit(x0 + 4000, y0)                              # second segment, heading south
        assert dlg.d_total.text() == "8.000 km" and dlg.d_segment.text() == "3.000 km"
        assert dlg.d_azimuth.text().startswith("179.") or dlg.d_azimuth.text().startswith("180.")
        c.set_extent(x0 - 4000, y0 - 2000, x0 + 8000, y0 + 5000)          # zoom in for the screenshots
        c.draw()
        app.processEvents()
        c.coords_changed.emit(x0 + 6500, y0 + 1000)                       # live rubber band after the redraw
        shot(win, "14_measure_distance")
        shot(dlg, "14b_measure_dialog")
        c.measure_clicked.emit(float("nan"), float("nan"), True)          # right-click finishes
        assert dlg.finished and dlg.d_total.text() == "5.000 km"

        dlg.tabs.setCurrentIndex(1)                                       # Area tab, remembered
        assert int(QSettings("MPM", APP_NAME).value(SETTINGS_KEY)) == 1 and c._ov is None
        for dx, dy in ((0, 0), (10000, 0), (10000, 10000), (0, 10000)):
            c.measure_clicked.emit(x0 + dx, y0 + dy, False)
        c.measure_clicked.emit(x0, y0 + 10000, True)                      # double-click finishes
        c.set_extent(x0 - 3000, y0 - 3000, x0 + 13000, y0 + 13000)
        c.draw()
        app.processEvents()
        assert dlg.a_area.text() == "100.0000 km²" and dlg.a_perim.text() == "40.000 km"
        xs = list(c._ov_artists[0].get_xdata())
        assert len(xs) == 5 and xs[0] == xs[-1]                           # closed outline, no fill (a line)
        shot(win, "15_measure_area")
        shot(dlg, "15b_measure_dialog_area")
        assert MeasureDialog(win.store, c, win).tabs.currentIndex() == 1   # next time it opens on Area

        win.a_pan.trigger()                                               # other tool -> dialog hidden, cleared
        app.processEvents()
        assert not dlg.isVisible() and c._ov is None and not c._ov_artists
        win.a_measure.trigger()
        c.measure_clicked.emit(x0, y0, False)
        assert c._ov_artists
        win.a_full.trigger()                                              # any other command also ends it
        app.processEvents()
        assert win.canvas.mode == "pan" and c._ov is None
        win.store.dirty = False
        win.close()
    finally:
        settings.setValue(SETTINGS_KEY, saved if saved is not None else 0)


@need_data
def test_clip_raster_by_mask_dialog(app, monkeypatch, tmp_path):
    import geopandas as gpd
    from shapely.geometry import box
    from core.layers import RasterLayer, VectorLayer
    from gui.dialogs.data_dialogs import ClipDialog
    from gui.main_window import MainWindow

    win = MainWindow()
    win.open_project(demo_copy(tmp_path))
    src = win.store.by_name("geology")                     # categorical (class table), integer cells
    left, bottom, right, top = src.grid.bounds
    w, h = right - left, top - bottom
    poly = box(left + 0.2 * w, bottom + 0.3 * h, left + 0.6 * w, bottom + 0.7 * h)
    gdf = gpd.GeoDataFrame({"unit": ["A", "B"]}, geometry=[poly, box(left, bottom, left + 0.1 * w, bottom + 0.1 * h)],
                           crs=src.grid.crs)
    win.store.add(VectorLayer("aoi", gdf))
    win.store.set_active(src)

    dlg = ClipDialog(win.store, win)
    dlg.src.set_layer(src)
    dlg.mask_layer.set_layer(win.store.by_name("aoi"))
    stem = os.path.splitext(os.path.basename(src.source))[0]
    assert dlg.path.text() == os.path.join(os.path.dirname(win.store.project_path), "results", f"{stem}_clip.tif")
    assert dlg.name.text() == f"{stem}_clip" and dlg.nodata.text() == "-9999"
    assert not dlg.crop.isChecked()                        # default: stays on the project grid
    dlg.crop.setChecked(True)
    dlg.field.setCurrentIndex(dlg.field.findData("unit"))
    dlg.value.setCurrentText("A")                          # only the first polygon
    assert "1 of 2" in dlg.summary.text()
    assert dlg.validate() is None
    shot(dlg, "14_clip_dialog")
    lyr = dlg.run()
    assert isinstance(lyr, RasterLayer) and os.path.isfile(lyr.source) and lyr.categorical
    assert lyr.grid.width < src.grid.width * 0.45 and lyr.grid.height < src.grid.height * 0.45
    back = RasterLayer.from_file(lyr.source)
    assert back.nodata == -9999 and back.grid.same_as(lyr.grid)
    assert np.ma.count(back.data) == np.ma.count(lyr.data) > 0

    dlg.nodata.setText("abc")
    assert dlg.validate()
    win.store.dirty = False
    win.close()


@need_data
def test_polygon_fill_colour_from_layer_properties(app, monkeypatch, tmp_path):
    from PyQt5.QtGui import QColor
    from PyQt5.QtWidgets import QColorDialog
    from gui.dialogs.layer_dialogs import LayerPropertiesDialog
    from gui.main_window import MainWindow

    win = MainWindow()
    win.resize(1200, 800)
    win.show()
    win.open_project(demo_copy(tmp_path))
    lyr = win.store.vectors("polygon")[0]
    lyr.style["facecolor"] = "none"
    lyr.visible = True
    parents = []

    def fake_get_color(start, parent, title):
        parents.append(parent)
        return QColor("#ff0000")
    monkeypatch.setattr(QColorDialog, "getColor", staticmethod(fake_get_color))

    dlg = LayerPropertiesDialog(lyr, win.store, win)
    assert dlg.nofill.isChecked()
    dlg.fill.click()                                    # choose red
    assert parents[0] is dlg                            # colour dialog belongs to the window, not to the button
    assert "QPushButton#colorButton" in dlg.fill.styleSheet()   # the button colour does not cascade to children
    assert not dlg.nofill.isChecked()                   # picking a colour switches "no fill" off
    dlg.apply()
    assert lyr.style["facecolor"] == "#ff0000"
    win.canvas.redraw()
    faces = [a for a in win.canvas._layer_artists if hasattr(a, "get_facecolor") and len(a.get_facecolor())]
    red = [a for a in faces if any(tuple(np.round(f[:3], 3)) == (1.0, 0.0, 0.0) for f in a.get_facecolor())]
    assert red, "fill colour not drawn"
    rasters = [a for a in win.canvas._layer_artists if type(a).__name__ == "AxesImage"]
    assert all(red[0].get_zorder() > r.get_zorder() for r in rasters)     # the fill is not hidden under rasters
    from gui import theme
    from matplotlib import colors as mcolors
    assert mcolors.to_hex(win.canvas.ax.get_facecolor()) == theme.MAP_BG          # map stays white with layers
    assert mcolors.to_hex(win.canvas.fig.get_facecolor()) == theme.MAP_BG

    dlg2 = LayerPropertiesDialog(lyr, win.store, win)
    dlg2.nofill.setChecked(True)
    dlg2.apply()
    assert lyr.style["facecolor"] == "none"
    win.canvas.redraw()                                 # "no fill": transparent, not black
    polys = [x for x in win.canvas._layer_artists                 # polygons (point layers are scatters: sizes)
             if type(x).__name__ == "PathCollection" and len(x.get_sizes()) == 0]
    assert polys and all(f[3] == 0 for x in polys for f in x.get_facecolor())
    win.store.dirty = False
    win.close()



@need_data
def test_layer_properties_symbology_and_labels(app, tmp_path):
    import json
    from core import symbology as sym
    from gui.dialogs.layer_dialogs import LayerPropertiesDialog
    from gui.main_window import MainWindow

    win = MainWindow()
    win.resize(1200, 800)
    win.show()
    win.open_project(demo_copy(tmp_path))
    s = win.store
    for l in s.layers:
        l.visible = False

    pts = s.vectors("point")[0]                                   # vector: General / Symbology / Labels
    pts.visible = True
    d = LayerPropertiesDialog(pts, s, win)
    assert [d.tabs.tabText(i) for i in range(d.tabs.count())] == ["General", "Statistics", "Symbology", "Labels"]
    assert d._modes == ["single", "unique", "categorized"]
    d.renderers.setCurrentRow(1)                                  # unique: every feature its own symbol
    assert d.u_model.rowCount() == len(pts.gdf)
    d.u_over["0"] = dict(color="#ff0000", edgecolor="#000000", marker="s", markersize=60)
    text = [k for k in pts.gdf.columns if k != "geometry" and pts.gdf[k].dtype == object][0]
    d.lbl_on.setChecked(True)
    d.lbl_field.setCurrentIndex(d.lbl_field.findData(text))
    d.lbl_size.setValue(9)
    d.apply()
    st = sym.feature_styles(pts)
    assert pts.renderer == "unique" and st[0]["marker"] == "s" and st[0]["color"] == "#ff0000"
    assert len({x["color"] for x in st}) > 1
    win.canvas.redraw()
    win.canvas.draw()                                             # labels are placed for the current view
    labels = win.canvas._label_artists
    assert 0 < len(labels) <= len(pts.gdf)                        # overlapping labels are left out
    assert labels[0].get_fontsize() == 9 and not labels[0].get_path_effects()     # no halo
    import geopandas as gpd                                       # thousands of features: a bounded number of labels
    from shapely.geometry import Point
    from core.layers import VectorLayer
    from gui.map_canvas import MAX_LABELS_DRAWN
    rng = np.random.default_rng(1)
    b = pts.bounds()
    many = VectorLayer("many", gpd.GeoDataFrame(
        {"name": [f"sample {i}" for i in range(4000)]},
        geometry=[Point(x, y) for x, y in zip(rng.uniform(b[0], b[2], 4000), rng.uniform(b[1], b[3], 4000))],
        crs=pts.gdf.crs))
    many.labels.update(enabled=True, field="name")
    s.add(many)
    win.canvas.draw()
    assert 0 < len(win.canvas._label_artists) <= MAX_LABELS_DRAWN
    n_view = len(win.canvas._label_artists)
    win.canvas.draw()                                             # same view: labels are not placed again
    assert len(win.canvas._label_artists) == n_view
    s.remove(many)

    d = LayerPropertiesDialog(pts, s, win)                        # categorized by an attribute
    d.renderers.setCurrentRow(2)
    d.cat.field.setCurrentIndex([d.cat.field.itemData(i)[0] for i in range(d.cat.field.count())].index(text))
    d.cat.classify()
    assert d.cat.classes and sum(c["count"] for c in d.cat.classes) == len(pts.gdf)
    d.apply()
    assert pts.renderer == "categorized" and pts.categorized["field"] == text

    ras = [r for r in s.rasters() if not r.categorical][0]        # raster: four renderers, no Labels tab
    ras.visible = True
    d = LayerPropertiesDialog(ras, s, win)
    assert d.tabs.count() == 3 and d._modes == ["single", "unique", "stretch", "categorized"]
    assert d.renderers.currentRow() == 2                          # continuous raster: stretched
    d.renderers.setCurrentRow(3)
    d.cat.method.setCurrentIndex(d.cat.method.findData("quantile"))
    d.cat.n.setValue(4)
    d.cat.classify()
    assert len(d.cat.classes) == 4 and all("lo" in c for c in d.cat.classes)
    d.apply()
    win.canvas.redraw()
    idx = sym.raster_category_index(ras, ras.data)
    assert idx.max() == 3 and np.ma.count(idx) == np.ma.count(ras.data)    # every valid cell in a class
    d = LayerPropertiesDialog(ras, s, win)
    d.renderers.setCurrentRow(0)
    d.r_color.set_color("#00ff00")
    d.apply()
    win.canvas.redraw()
    assert ras.renderer == "single"

    win.save_project()                                            # saved with the project and restored
    proj = json.load(open(s.project_path, encoding="utf-8"))
    by = {e["name"]: e for e in proj["layers"]}
    assert by[pts.name]["symbology"]["renderer"] == "categorized"
    assert by[pts.name]["symbology"]["labels"]["enabled"] is True
    win2 = MainWindow()
    win2.open_project(s.project_path)
    p2, r2 = win2.store.by_name(pts.name), win2.store.by_name(ras.name)
    assert p2.renderer == "categorized" and p2.labels["field"] == text and p2.unique["styles"]["0"]["marker"] == "s"
    assert r2.renderer == "single" and r2.single_color == "#00ff00" and len(r2.categorized["classes"]) == 4
    for w in (win, win2):
        w.store.dirty = False
        w.close()



@need_data
def test_layer_properties_statistics(app, tmp_path):
    """Statistics tab: histogram by number of intervals / interval width, cumulative curve, statistics box."""
    from PyQt5.QtWidgets import QApplication
    from gui.dialogs.layer_dialogs import LayerPropertiesDialog
    from gui.main_window import MainWindow
    from utils.statistics import describe

    win = MainWindow()
    win.resize(1200, 800)
    win.show()
    win.open_project(demo_copy(tmp_path))
    s = win.store
    ras = s.by_name("as_log_idw")                                  # raster: cell values
    d = LayerPropertiesDialog(ras, s, win, tab="statistics")
    d.show()
    QApplication.processEvents()
    assert d.tabs.tabText(1) == "Statistics" and d.tabs.currentWidget() is d.statistics
    p = d.statistics
    assert p._loaded and not p.field.isVisible()
    v = np.ma.compressed(ras.data).astype(float)
    assert p.stats["count"] == v.size and p.stats["nulls"] == ras.data.size - v.size
    assert p.stats["mean"] == pytest.approx(v.mean()) and p.stats["max"] == pytest.approx(v.max())
    counts, edges = p.hist
    assert counts.sum() == v.size and len(counts) == p.count.value()
    p.count.setValue(25)
    assert len(p.hist[0]) == 25
    p.mode.setCurrentIndex(p.mode.findData("width"))                # interval values
    assert p.width.isVisible() and not p.count.isVisible()
    p.width.setValue(0.1)
    e = p.hist[1]
    assert np.allclose(np.diff(e), 0.1) and e[0] <= v.min() and e[-1] >= v.max() and p.hist[0].sum() == v.size
    ax = p.chart.fig.axes
    assert len(ax) == 2 and ax[1].get_ylabel() == "Cumulative probability"     # ECDF on a second axis
    assert ax[1].lines[0].get_ydata()[-1] == pytest.approx(1.0)
    rows = dict(p.stat_rows())
    assert {"Minimum", "Maximum", "Mean", "Median", "Skewness", "Kurtosis", "NoData"} <= set(rows)
    assert p.stats_artist in ax[1].artists
    p.show_stats.setChecked(False)
    assert p.stats_artist is None and not p.chart.fig.axes[1].artists
    p.show_stats.setChecked(True)
    p.cumulative.setChecked(False)
    assert len(p.chart.fig.axes) == 1
    p.cumulative.setChecked(True)
    p.copy_statistics()
    assert "Kurtosis\t" in QApplication.clipboard().text()
    from PIL import Image
    assert p.save.isEnabled() and p.start_dir.endswith("results")
    assert p.default_png_name() == "as_log_idw_histogram.png"
    out = p.save_histogram(str(tmp_path / "hist"))                 # ".png" added; 300 dpi on white
    im = Image.open(out)
    assert out.endswith("hist.png") and im.size == (2100, 1260)
    assert all(abs(x - 300) < 0.01 for x in im.info["dpi"])
    assert im.convert("RGB").getpixel((5, 5)) == (255, 255, 255)
    assert len(p.chart.fig.axes) == 2 and p.stats_artist in p.chart.fig.axes[1].artists   # screen chart unchanged
    shot(d, "53_layer_statistics_raster")
    d.close()

    dep = s.by_name("deposits")                                     # vector: numeric attributes, first by default
    d = LayerPropertiesDialog(dep, s, win)
    d.show()
    d.tabs.setCurrentIndex(1)
    QApplication.processEvents()
    p = d.statistics
    names = [p.field.itemText(i) for i in range(p.field.count())]
    num = [c for c in dep.gdf.columns if c != "geometry" and np.issubdtype(dep.gdf[c].dtype, np.number)]
    assert names == num and p.field.currentIndex() == 0
    for i, name in enumerate(names):
        p.field.setCurrentIndex(i)
        ref = describe(dep.gdf[name].to_numpy(float))
        assert p.stats["count"] == ref["count"] and p.hist[0].sum() == ref["count"]
        assert p.chart.fig.axes[0].get_xlabel() == name
    p.field.setCurrentIndex(0)
    assert p.default_png_name() == f"deposits_{names[0]}_histogram.png"
    assert os.path.isfile(p.save_histogram(str(tmp_path / p.default_png_name())))
    shot(d, "54_layer_statistics_vector")
    d.close()
    win.store.dirty = False
    win.close()


@need_data
def test_export_legend_png(app, tmp_path):
    from PIL import Image
    from gui.main_window import MainWindow

    win = MainWindow()
    win.open_project(demo_copy(tmp_path))
    s = win.store
    cat = [r for r in s.rasters() if r.categorical][0]
    cont = [r for r in s.rasters() if not r.categorical][0]
    vec = s.vectors()[0]
    sizes = {}
    for lyr in (cat, cont, vec):
        p = win.export_legend(lyr, str(tmp_path / f"{lyr.name}_legend.png"))
        im = Image.open(p)
        assert all(abs(d - 300) < 0.01 for d in im.info["dpi"])        # 300 dpi written into the file
        sizes[lyr.name] = im.size
    n = cat.classes().size                                              # every class, ~10 pt lines at 300 dpi
    assert sizes[cat.name][1] > n * 10 / 72 * 300
    w0 = sizes[vec.name][0]
    vec.name = "a much longer layer name for the legend title"          # width follows the text
    w = Image.open(win.export_legend(vec, str(tmp_path / "long.png"))).size[0]
    assert w > w0 + 300
    s.dirty = False
    win.close()



@need_data
def test_lazy_rasters_read_in_background(app, tmp_path):
    from gui.main_window import MainWindow
    win = MainWindow()
    win.show()
    win.open_project(demo_copy(tmp_path))
    s = win.store
    hidden = [r for r in s.rasters() if not r.visible and r is not s.active]   # (selected: statistics are read)
    assert hidden and not any(r.loaded for r in hidden)            # hidden rasters: only the header was read
    assert all(r.grid.width > 0 and r.crs is not None for r in hidden)
    assert s.wait_loaded(60)                                        # visible ones: read by worker threads
    app.processEvents()
    app.processEvents()
    visible = [r for r in s.rasters() if r.visible]
    assert all(r.loaded for r in visible)
    win.canvas._redraw_timer.stop()
    win.canvas.redraw()
    assert sum(type(a).__name__ == "AxesImage" for a in win.canvas._layer_artists) == len(visible)
    r = hidden[0]
    assert r.data.shape == (r.grid.height, r.grid.width) and r.loaded    # on demand when a method needs it
    assert r.classes() is r.classes()                               # values / counts computed once
    s.dirty = False
    win.close()


@need_data
def test_recent_projects_menu(app, monkeypatch, tmp_path):
    import shutil
    from gui import app_settings
    from gui.main_window import MainWindow

    assert os.environ["MPM_SETTINGS_FILE"].endswith(".ini")       # never the user's registry
    app_settings.clear_recent_projects()
    win = MainWindow()
    from gui import theme
    from matplotlib import colors as mcolors
    assert mcolors.to_hex(win.canvas.fig.get_facecolor()) == theme.BG                    # start-up: theme colour
    assert mcolors.to_hex(win.canvas.ax.get_facecolor()) == theme.BG
    assert not win.canvas.ax.texts                                                       # no start-up hint text
    win.apply_theme("light")                                                             # empty map follows the theme
    assert mcolors.to_hex(win.canvas.fig.get_facecolor()) == theme.LIGHT["BG"]
    win.apply_theme("dark")
    acts = [a for a in win.recent_menu.actions() if not a.isSeparator()]
    assert not acts[0].isEnabled() and not acts[-1].isEnabled()       # "(no recent projects)", "Clear" disabled

    p1 = demo_copy(tmp_path, name="first")
    win.open_project(p1)
    assert mcolors.to_hex(win.canvas.fig.get_facecolor()) == theme.MAP_BG == "#ffffff"  # white with layers
    folders = [os.path.join(str(tmp_path), f"p{i}") for i in range(6)]
    for f in folders:                                                  # 6 more: only the newest 5 are kept
        app_settings.add_recent_project(f)
    app_settings.add_recent_project(os.path.dirname(p1).upper())       # same folder, other case: moved to the top
    rec = app_settings.recent_projects()
    assert len(rec) == 5 and os.path.normcase(rec[0]) == os.path.normcase(os.path.dirname(p1))
    assert folders[5] in rec and folders[0] not in rec

    win._fill_recent_menu()
    acts = [a for a in win.recent_menu.actions() if not a.isSeparator()]
    assert len(acts) == 6 and acts[0].text().lower().startswith("&1  first") and acts[0].isEnabled()
    assert not acts[1].isEnabled()                                     # p5 does not exist -> greyed out

    p2 = demo_copy(tmp_path, name="second")                            # opening from the menu
    app_settings.add_recent_project(os.path.dirname(p2))
    win._fill_recent_menu()
    win.recent_menu.actions()[0].trigger()
    assert os.path.normcase(win.store.project_path) == os.path.normcase(p2)
    assert os.path.normcase(app_settings.recent_projects()[0]) == os.path.normcase(os.path.dirname(p2))

    win.clear_recent()
    assert app_settings.recent_projects() == []
    win.store.dirty = False
    win.close()


@need_data
def test_project_menu_and_properties_dialog(app, monkeypatch, tmp_path):
    from rasterio.crs import CRS
    from core import project
    from gui.dialogs.project_dialog import ProjectPropertiesDialog
    from gui.main_window import MainWindow

    win = MainWindow()
    titles = [a.text() for a in win.menuBar().actions()]
    assert titles[0] == "&Project" and "&File" not in titles
    data_menu = next(a.menu() for a in win.menuBar().actions() if a.text() == "&Data")
    assert data_menu.actions()[0] is win.a_add_r and data_menu.actions()[1] is win.a_add_v
    proj_menu = win.menuBar().actions()[0].menu()
    assert win.a_proj_props in proj_menu.actions() and win.a_add_r not in proj_menu.actions()
    assert not win.a_proj_props.isEnabled()                         # no project yet

    # new, empty project: CRS / cell size editable
    project.create_project(win.store, str(tmp_path / "p"))
    win._update_project_state()
    assert win.a_proj_props.isEnabled()
    dlg = ProjectPropertiesDialog(win.store, win, "crs")
    assert not dlg.locked and dlg.picker.kind.isEnabled() and dlg.picker.kind.currentData() == "none"
    p = dlg.picker
    p.kind.setCurrentIndex(p.kind.findData("utm"))
    p.datum.setCurrentIndex(p.datum.findData(4326))
    assert p.zone.count() == 120
    p.zone.setCurrentIndex(p.zone.findData("11N"))
    dlg.cell.setText("100")
    for (_k, e), v in zip(dlg.ext_edits, ("560000", "4510000", "570000", "4520000")):
        e.setText(v)
    assert "100 columns × 100 rows" in dlg.grid_info.text()
    assert "32611" in dlg.crs_name.text() and dlg.validate() is None
    shot(dlg, "15_project_properties_crs")
    assert dlg.apply()
    assert win.store.crs.to_epsg() == 32611 and win.store.cell_size == 100
    assert win.store.extent == (560000, 4510000, 570000, 4520000)
    assert (win.store.project_grid().width, win.store.project_grid().height) == (100, 100)
    assert "100 m" in win.map_panel.cell_label.text() and win.map_panel.crs_label.text() == "EPSG:32611"
    dlg.cell.setText("-5")
    assert dlg.validate()
    dlg.cell.setText("100")
    dlg.ext_edits[2][1].setText("1")                                # X max < X min
    assert dlg.validate()

    # custom definition, invalid text
    d2 = ProjectPropertiesDialog(win.store, win)
    assert d2.picker.kind.currentData() == "utm" and d2.picker.zone.currentData() == "11N"
    d2.picker.kind.setCurrentIndex(d2.picker.kind.findData("custom"))
    d2.picker.custom.setText("not a crs")
    assert d2.validate()
    d2.picker.custom.setText("2056")
    assert d2.validate() is None and d2.picker.crs() == CRS.from_epsg(2056)

    # with layers: read-only
    win.store.dirty = False
    win.open_project(demo_copy(tmp_path))
    d3 = ProjectPropertiesDialog(win.store, win, "extent")
    assert d3.locked and not d3.picker.kind.isEnabled() and not d3.cell.isEnabled()
    assert not d3.ext_edits[0][1].isEnabled() and np.allclose(d3._extent_values(), win.store.extent, atol=0.01)
    assert d3.picker.crs() == win.store.crs and float(d3.cell.text()) == win.store.cell_size
    assert "columns" in d3.grid_info.text()
    shot(d3, "16_project_properties_extent")
    assert not d3.apply()
    assert d3.nodata.isEnabled() and d3.nodata.text() == "-9999"     # NoData: default -9999, editable with layers
    d3.nodata.setText("abc")
    assert d3.validate()
    d3.nodata.setText("-32768")
    assert d3.validate() is None and d3.apply() and win.store.nodata == -32768
    from core.methods import OutputRaster                            # used by the method outputs
    r = win.store.rasters()[0]
    out = win.add_output_raster(OutputRaster("nd", np.ma.masked_less(r.data.astype("float32"), 1e9), r.grid))
    assert out.nodata == -32768
    win.save_project()
    import json
    assert json.load(open(win.store.project_path, encoding="utf-8"))["nodata"] == -32768
    win.store.dirty = False
    win.open_project(win.store.project_path)
    assert win.store.nodata == -32768
    win.store.dirty = False
    win.close()


def test_input_data_loaded_as_stored(app, monkeypatch, tmp_path):
    import geopandas as gpd
    from PyQt5.QtWidgets import QMessageBox
    from rasterio.crs import CRS
    from rasterio.transform import from_origin
    from shapely.geometry import box
    from core import project
    from core.layers import RasterLayer
    from gui.main_window import MainWindow
    from utils.raster_tools import Grid

    def tif(name, crs, x0, y0, res, n=40, cls=False):
        g = Grid(from_origin(x0, y0, res, res), CRS.from_user_input(crs) if crs else None, n, n)
        data = (np.ma.arange(n * n).reshape(n, n) % 5 + 1).astype("uint8") if cls else \
            np.ma.arange(n * n, dtype="float32").reshape(n, n)
        return RasterLayer(name, data, g).save(str(tmp_path / f"{name}.tif"))

    first = tif("first", "EPSG:32611", 560000, 4520000, 100)              # defines the project grid
    fine = tif("fine", "EPSG:32611", 561000, 4519000, 30)                 # other cell size and extent
    shifted = tif("shifted", "EPSG:32611", 560050, 4520000, 100)          # same cell size, other alignment
    geo = tif("geo", "EPSG:4326", -116.29, 40.83, 0.001)                  # other CRS
    nocrs = tif("nocrs", None, 560000, 4520000, 100)
    far = tif("far", "EPSG:32611", 100000, 1000000, 100)                  # outside the project extent
    vec = str(tmp_path / "aoi.gpkg")
    gpd.GeoDataFrame(geometry=[box(-116.3, 40.8, -116.2, 40.84)], crs="EPSG:4326").to_file(vec)

    asked, answers = [], []

    def fake_question(parent, title, text, buttons=None, default=None):
        asked.append(text)
        return answers.pop(0)
    monkeypatch.setattr(QMessageBox, "question", staticmethod(fake_question))
    monkeypatch.setattr(QMessageBox, "warning", staticmethod(lambda *a, **k: QMessageBox.Ok))

    win = MainWindow()
    project.create_project(win.store, str(tmp_path / "p"))
    s = win.store

    a = win.add_data_file("raster", first)                                # first raster: settings taken from it
    assert not asked and s.crs.to_epsg() == 32611 and s.cell_size == 100
    assert s.extent == (560000, 4516000, 564000, 4520000) and a.grid.same_as(s.project_grid()) and a.source == first
    grid = s.project_grid()
    win.canvas.draw()                                                     # the axes keep filling the whole canvas
    assert win.canvas.ax.get_aspect() == "auto"                           # (imshow's 1:1 box clipped the map)
    assert np.allclose(win.canvas.ax.get_position().bounds, (0, 0, 1, 1))

    # cell size / extent / alignment are not checked: rasters in the project CRS are loaded as stored, no question
    lyr = win.add_data_file("raster", fine)
    assert not asked and lyr.source == fine and lyr.grid.res == (30, 30) and not lyr.grid.same_as(grid)
    lyr = win.add_data_file("raster", shifted)
    assert not asked and lyr.source == shifted and lyr.grid.res == (100, 100) and not lyr.grid.same_as(grid)
    n = len(s.layers)
    lyr = win.add_data_file("raster", far)                                # outside the project extent: loaded too
    assert not asked and lyr is not None and len(s.layers) == n + 1

    # coordinate system is not checked either: other CRS -> loaded exactly as stored, no question;
    # no / unrecognised CRS -> loaded as stored and taken to be the project CRS
    lyr = win.add_data_file("raster", geo)
    assert not asked and lyr.source == geo and lyr.grid.crs.to_epsg() == 4326 and lyr.grid.res == (0.001, 0.001)
    lyr = win.add_data_file("raster", nocrs)
    assert not asked and lyr.source == nocrs and lyr.grid.crs == s.crs and lyr.grid.res == (100, 100)

    v = win.add_data_file("vector", vec)                                  # vectors: reprojected without a question
    assert v.crs == s.crs and v.source == vec

    batch = {}                                                            # several files: still no question
    win.add_data_file("raster", geo, batch, remaining=1)
    win.add_data_file("raster", geo, batch, remaining=0)
    assert not asked
    assert s.crs.to_epsg() == 32611 and s.cell_size == 100                 # project settings never change
    s.dirty = False
    win.close()
