"""Start-up splash screen (gui/splash.py): image size limit, minimum display time, start-up messages."""
import os
import time

import pytest

SHOTS = os.environ.get("MPM_SCREENSHOT_DIR")


@pytest.fixture(scope="module")
def app():
    from PyQt5.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def wait(ms):
    from PyQt5.QtWidgets import QApplication
    end = time.monotonic() + ms / 1000
    while time.monotonic() < end:
        QApplication.processEvents()
        time.sleep(0.01)


def test_image_fits_600x400(app, tmp_path):
    from PyQt5.QtGui import QImage
    from gui.splash import load_splash_pixmap, splash_path
    pix, builtin = load_splash_pixmap(splash_path())
    assert (pix.width(), pix.height()) == (600, 400) and builtin                 # the built-in image
    big = QImage(1200, 600, QImage.Format_RGB32)
    big.fill(0x336699)
    big.save(str(tmp_path / "big.png"))
    pix, builtin = load_splash_pixmap(str(tmp_path / "big.png"))
    assert (pix.width(), pix.height()) == (600, 300) and not builtin             # scaled down, aspect kept
    small = QImage(300, 200, QImage.Format_RGB32)
    small.save(str(tmp_path / "small.png"))
    assert load_splash_pixmap(str(tmp_path / "small.png"))[0].width() == 300     # never enlarged
    assert load_splash_pixmap(str(tmp_path / "missing.png"))[0].isNull()


def test_minimum_display_time_and_messages(app):
    from PyQt5.QtCore import QEvent, QPointF, Qt
    from PyQt5.QtGui import QMouseEvent
    from PyQt5.QtWidgets import QApplication, QWidget
    from gui.splash import MIN_MS, StartupSplash
    assert MIN_MS == 2000
    s = StartupSplash.create("9.9.9", min_ms=400)
    assert s is not None and s.isVisible()
    s.message("Loading methods…")
    if SHOTS:
        s.grab().save(os.path.join(SHOTS, "00_splash.png"))
    QApplication.sendEvent(s, QMouseEvent(QEvent.MouseButtonPress, QPointF(10, 10), Qt.LeftButton,
                                          Qt.LeftButton, Qt.NoModifier))
    assert s.isVisible()                                                         # a click does not close it
    win, opened = QWidget(), []
    win.show()
    s.finish_when_ready(win, then=lambda: opened.append(True))                   # fast start: ready at once
    wait(150)
    assert s.isVisible() and not s.finished and not opened                       # ... but kept for the minimum
    wait(400)
    assert s.finished and not s.isVisible() and opened == [True]
    s2 = StartupSplash.create(min_ms=100)                                        # slow start: closes at once
    wait(150)
    s2.finish_when_ready(None)
    assert s2.finished and not s2.isVisible()
    assert StartupSplash.create(path="no_such_file.png") is None                 # no image -> no splash
    win.close()
