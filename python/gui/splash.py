"""Start-up splash screen.

Shows ``gui/resources/splash.png`` (at most 600 x 400 px; a larger replacement image is scaled down, keeping
its aspect ratio) while the application initialises, and closes as soon as the main window is ready — but
never before :data:`MIN_MS` (2 s) have passed, so a fast start does not just flash it.

The built-in image (made by ``tools/make_splash.py``, PNG text key ``mpm-splash = builtin``) carries only the
language-neutral artwork; the tagline and version are drawn here in the interface language. Start-up
messages are drawn in the bottom strip of any image.
"""
from __future__ import annotations

import os
import time
from typing import Callable, Optional

from PyQt5.QtCore import QRect, Qt, QTimer
from PyQt5.QtGui import QColor, QFont, QPainter, QPixmap
from PyQt5.QtWidgets import QApplication, QSplashScreen, QWidget

from i18n import tr

MAX_W, MAX_H = 600, 400
MIN_MS = 2000
STRIP = 44                    # height of the status strip at the bottom of the image (px)


def splash_path() -> str:
    import sys
    base = getattr(sys, "_MEIPASS", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return os.path.join(base, "gui", "resources", "splash.png")


def load_splash_pixmap(path: str) -> tuple:
    """(pixmap scaled to fit 600 x 400, is_builtin) — a null pixmap if the file is missing or unreadable."""
    from PyQt5.QtGui import QImage
    img = QImage(path)
    if img.isNull():
        return QPixmap(), False
    builtin = img.text("mpm-splash") == "builtin"
    if img.width() > MAX_W or img.height() > MAX_H:
        img = img.scaled(MAX_W, MAX_H, Qt.KeepAspectRatio, Qt.SmoothTransformation)
    return QPixmap.fromImage(img), builtin


def _font(px: int, weight: int = QFont.Normal) -> QFont:
    f = QFont("Segoe UI")
    f.setPixelSize(px)
    f.setWeight(weight)
    return f


class StartupSplash(QSplashScreen):
    """Splash screen with a minimum display time; clicks do not close it."""

    def __init__(self, pixmap: QPixmap, builtin: bool = False, version: str = "", min_ms: int = MIN_MS):
        super().__init__(pixmap, Qt.WindowStaysOnTopHint)
        self._builtin, self._version, self._min_ms = builtin, version, min_ms
        self._msg = ""
        self._t0 = time.monotonic()
        self.finished = False

    @classmethod
    def create(cls, version: str = "", path: Optional[str] = None, min_ms: int = MIN_MS) -> Optional["StartupSplash"]:
        """Show the splash; None (no splash) if the image cannot be loaded."""
        pix, builtin = load_splash_pixmap(path or splash_path())
        if pix.isNull():
            return None
        s = cls(pix, builtin, version, min_ms)
        s.show()
        s.raise_()
        QApplication.processEvents()
        s._t0 = time.monotonic()                     # the 2 s count from when it is on screen
        return s

    def elapsed_ms(self) -> float:
        return (time.monotonic() - self._t0) * 1000.0

    def remaining_ms(self) -> int:
        return max(0, int(round(self._min_ms - self.elapsed_ms())))

    def message(self, text: str):
        """Show a start-up step ("Loading methods…") and repaint at once (the event loop is not running yet)."""
        self._msg = text
        self.repaint()
        QApplication.processEvents()

    def finish_when_ready(self, window: Optional[QWidget] = None, then: Optional[Callable[[], None]] = None):
        """Close now if the minimum time has passed, else when it has; then call ``then`` (e.g. open a project)."""
        def done():
            if self.finished:
                return
            self.finished = True
            if window is not None:
                self.finish(window)
            else:
                self.close()
            if then is not None:
                then()
        wait = self.remaining_ms()
        if wait <= 0:
            done()
        else:
            QTimer.singleShot(wait, done)

    def mousePressEvent(self, event):              # QSplashScreen hides on click — keep it for the minimum time
        event.accept()

    def drawContents(self, painter: QPainter):
        painter.setRenderHint(QPainter.TextAntialiasing, True)
        w, h = self.pixmap().width(), self.pixmap().height()
        if self._builtin:
            painter.setPen(QColor("#c9d1dd"))
            painter.setFont(_font(15))
            painter.drawText(QRect(294, 208, w - 304, 24), Qt.AlignLeft | Qt.AlignVCenter,
                             tr("2D mineral prospectivity mapping"))
            if self._version:
                painter.setPen(QColor("#8a95a8"))
                painter.setFont(_font(12))
                painter.drawText(QRect(294, 234, w - 304, 20), Qt.AlignLeft | Qt.AlignVCenter,
                                 tr("Version {v}", v=self._version))
        else:
            painter.fillRect(QRect(0, h - STRIP, w, STRIP), QColor(0, 0, 0, 140))   # readable on any image
        if self._msg:
            painter.setPen(QColor("#aeb8c7"))
            painter.setFont(_font(12))
            painter.drawText(QRect(18, h - STRIP, w - 36, STRIP), Qt.AlignLeft | Qt.AlignVCenter, self._msg)
