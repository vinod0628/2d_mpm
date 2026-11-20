"""TEMPLATE for a new MPM method package — copy this folder, do not import it directly.

1. Copy ``python/logic/method_template/`` to ``python/logic/<category>/<method_key>/`` where <category> is
   ``data_driven``, ``knowledge_driven`` or ``hybrid``.
2. model.py   : write the algorithm (numpy / scipy / scikit-learn only, no Qt) + a test in python/tests.
3. plugin.py  : set key / name / category / group / description / icon; build the outputs in run().
4. dialogs.py : parameter dialog with params() -> dict (and an optional result dialog).
5. Remove the method's entry from ``core.methods.PLANNED`` (the menu then shows the real one).

Nothing else changes: the plug-in is found automatically (menu entry, toolbar-free) and bundled by the
PyInstaller spec, and its outputs can be validated with the shared ROC/AUC dialog (evaluation.metrics).
This example is a runnable "weighted mean of normalised evidence" so the wiring can be tested as-is.
"""
from .model import weighted_mean_score  # noqa: F401
