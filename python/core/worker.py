"""Run long computations off the GUI thread."""
from __future__ import annotations

import traceback

from PyQt5.QtCore import QThread, pyqtSignal


class Task(QThread):
    """``Task(fn, *args, **kw)``; ``fn`` receives ``progress=callable(int, str)``."""

    progressed = pyqtSignal(int, str)
    succeeded = pyqtSignal(object)
    failed = pyqtSignal(str, str)          # message, traceback

    def __init__(self, fn, *args, parent=None, **kwargs):
        super().__init__(parent)
        self.fn, self.args, self.kwargs = fn, args, kwargs

    def run(self):
        try:
            result = self.fn(*self.args, progress=self._progress, **self.kwargs)
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(str(exc), traceback.format_exc())
        else:
            self.succeeded.emit(result)

    def _progress(self, pct: int, msg: str = ""):
        self.progressed.emit(int(pct), msg)
