"""Settings ▸ General settings: a page list on the left (Language, Theme), one page with its own Apply button each."""
from __future__ import annotations

from PyQt5.QtCore import QRectF, QSize, Qt, QUrl, pyqtSignal
from PyQt5.QtGui import QColor, QDesktopServices, QIcon, QPainter, QPainterPath, QPixmap
from PyQt5.QtWidgets import (QButtonGroup, QDialog, QFrame, QHBoxLayout, QLabel, QListWidget, QListWidgetItem,
                             QPushButton, QRadioButton, QStackedWidget, QToolButton, QVBoxLayout, QWidget)

import i18n
from gui import app_settings, theme
from gui.widgets import fit_to_screen
from i18n import tr


def _section(title: str) -> QWidget:
    """Section heading followed by a horizontal rule (like "Show ────────")."""
    w = QWidget()
    lay = QHBoxLayout(w)
    lay.setContentsMargins(0, 4, 0, 2)
    lay.setSpacing(8)
    lab = QLabel(title)
    lab.setObjectName("sectionTitle")
    line = QFrame()
    line.setObjectName("sectionRule")
    line.setFrameShape(QFrame.HLine)
    line.setFixedHeight(1)
    lay.addWidget(lab)
    lay.addWidget(line, 1)
    return w


def _hint(text: str) -> QLabel:
    lab = QLabel(text)
    lab.setObjectName("hint")
    lab.setWordWrap(True)
    return lab


class _Page(QWidget):
    """Settings page: content on top, [hint ... Apply] at the bottom. Apply is enabled only after a change."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.body = QVBoxLayout()
        self.body.setSpacing(8)
        self.status = _hint("")
        self.apply_btn = QPushButton(tr("Apply"))
        self.apply_btn.setProperty("primary", True)
        self.apply_btn.setEnabled(False)
        self.apply_btn.clicked.connect(self.apply)
        bottom = QHBoxLayout()
        bottom.addWidget(self.status, 1)
        bottom.addWidget(self.apply_btn)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(16, 10, 16, 12)
        lay.addLayout(self.body)
        lay.addStretch(1)
        lay.addLayout(bottom)

    def changed(self):
        self.apply_btn.setEnabled(self.value() != self.saved())
        self.status.setText("")

    def value(self):
        raise NotImplementedError

    def saved(self):
        raise NotImplementedError

    def apply(self):
        raise NotImplementedError


class LanguagePage(_Page):
    applied = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.body.addWidget(_section(tr("Interface language")))
        self.group = QButtonGroup(self)
        current = app_settings.language()
        for code, meta in i18n.available().items():
            text = meta["name"] if meta["name"] == meta["english"] else f"{meta['name']}  ({meta['english']})"
            rb = QRadioButton(text)
            rb.setProperty("code", code)
            rb.setChecked(code == current)
            self.group.addButton(rb)
            self.body.addWidget(rb)
        if self.group.checkedButton() is None and self.group.buttons():
            self.group.buttons()[0].setChecked(True)
        self.body.addWidget(_hint(tr("The new language is used after MPM Studio restarts.")))
        self.body.addSpacing(6)
        self.body.addWidget(_section(tr("Translations")))
        self.body.addWidget(_hint(tr("Each language is one JSON file in the translations folder. Copy an existing "
                                     "file to add a language; missing texts are shown in English.")))
        row = QHBoxLayout()
        folder = QPushButton(theme.icon("mdi6.folder-open-outline"), tr("Open translations folder"))
        folder.clicked.connect(lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(i18n.locale_dir())))
        row.addWidget(folder)
        row.addStretch(1)
        self.body.addLayout(row)
        self.group.buttonToggled.connect(lambda *_: self.changed())

    def value(self):
        b = self.group.checkedButton()
        return b.property("code") if b else "en"

    def saved(self):
        return app_settings.language()

    def apply(self):
        app_settings.set_language(self.value())
        self.apply_btn.setEnabled(False)
        self.status.setText(tr("Saved. Restart MPM Studio to switch the language.")
                            if self.value() != i18n.language() else tr("Saved."))
        self.applied.emit(self.value())


def theme_preview(pal: dict, w: int = 150, h: int = 92) -> QPixmap:
    """Miniature window drawn with a theme palette: toolbar, layer panel, white map, footer."""
    pm = QPixmap(w, h)
    pm.fill(Qt.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.Antialiasing)
    clip = QPainterPath()
    clip.addRoundedRect(QRectF(0.5, 0.5, w - 1, h - 1), 6, 6)
    p.setClipPath(clip)
    p.fillRect(0, 0, w, h, QColor(pal["PANEL"]))
    p.fillRect(0, 0, w, 14, QColor(pal["PANEL"]))
    for i in range(5):                                             # toolbar buttons
        p.fillRect(6 + i * 12, 4, 8, 6, QColor(pal["ACCENT"] if i == 2 else pal["MUTED"]))
    p.fillRect(0, 14, w, 1, QColor(pal["BORDER"]))
    p.fillRect(4, 18, 40, h - 26, QColor(pal["FIELD"]))           # layer panel
    for i in range(5):
        p.fillRect(8, 23 + i * 9, 6, 5, QColor(pal["ACCENT"] if i == 0 else pal["INDICATOR"]))
        p.fillRect(17, 24 + i * 9, 22 - (i % 3) * 4, 3, QColor(pal["TEXT"]))
    p.fillRect(48, 18, w - 52, h - 30, QColor(pal["MAP_BG"]))      # map
    p.setPen(Qt.NoPen)
    p.setBrush(QColor(pal["SERIES"][0]))
    p.drawEllipse(QRectF(70, 30, 34, 24))
    p.setBrush(QColor(pal["SERIES"][1]))
    p.drawEllipse(QRectF(98, 44, 28, 18))
    p.fillRect(48, h - 12, w - 52, 8, QColor(pal["PANEL"]))        # footer
    p.fillRect(52, h - 9, 30, 2, QColor(pal["MUTED"]))
    p.setClipping(False)
    p.setPen(QColor(pal["BORDER"]))
    p.setBrush(Qt.NoBrush)
    p.drawRoundedRect(QRectF(0.5, 0.5, w - 1, h - 1), 6, 6)
    p.end()
    return pm


class ThemePage(_Page):
    applied = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.body.addWidget(_section(tr("Appearance")))
        self.group = QButtonGroup(self)
        row = QHBoxLayout()
        row.setSpacing(14)
        current = theme.MODE
        for mode, label in (("dark", tr("Dark")), ("light", tr("Light"))):
            b = QToolButton()
            b.setObjectName("themeCard")
            b.setCheckable(True)
            b.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
            b.setIcon(_pixmap_icon(theme_preview(theme.MODES[mode])))
            b.setIconSize(QSize(150, 92))
            b.setText(label)
            b.setProperty("mode", mode)
            b.setChecked(mode == current)
            self.group.addButton(b)
            row.addWidget(b)
        row.addStretch(1)
        self.body.addLayout(row)
        self.body.addWidget(_hint(tr("The theme changes immediately. The map background is white as soon as the "
                                     "map has a layer.")))
        self.group.buttonToggled.connect(lambda *_: self.changed())

    def value(self):
        b = self.group.checkedButton()
        return b.property("mode") if b else theme.MODE

    def saved(self):
        return theme.MODE

    def apply(self):
        mode = self.value()
        app_settings.set_theme_mode(mode)
        self.applied.emit(mode)                         # the main window re-styles the application
        self.apply_btn.setEnabled(False)
        self.status.setText(tr("Theme applied."))


def _pixmap_icon(pm: QPixmap) -> QIcon:
    ic = QIcon()
    ic.addPixmap(pm, QIcon.Normal)
    ic.addPixmap(pm, QIcon.Active)
    ic.addPixmap(pm, QIcon.Selected)
    return ic


class GeneralSettingsDialog(QDialog):
    """WPS-style options dialog. ``language_applied(code)`` / ``theme_applied(mode)`` are emitted on Apply."""
    language_applied = pyqtSignal(str)
    theme_applied = pyqtSignal(str)
    PAGE_KEY = "settings/last_page"

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(tr("General settings"))
        self.nav = QListWidget()
        self.nav.setObjectName("settingsNav")
        self.nav.setFixedWidth(170)
        self.stack = QStackedWidget()
        self.stack.setObjectName("settingsStack")
        self.pages = {}
        for key, label, icon, page in (("language", tr("Language"), "mdi6.translate", LanguagePage()),
                                       ("theme", tr("Theme"), "mdi6.palette-outline", ThemePage())):
            it = QListWidgetItem(theme.icon(icon), label)
            it.setData(Qt.UserRole, key)
            self.nav.addItem(it)
            self.stack.addWidget(page)
            self.pages[key] = page
        self.pages["language"].applied.connect(self.language_applied)
        self.pages["theme"].applied.connect(self.theme_applied)
        self.nav.currentRowChanged.connect(self._page_changed)

        close = QPushButton(tr("Close"))
        close.clicked.connect(self.accept)
        bottom = QHBoxLayout()
        bottom.addStretch(1)
        bottom.addWidget(close)
        main = QHBoxLayout()
        main.setSpacing(0)
        main.addWidget(self.nav)
        frame = QFrame()
        frame.setObjectName("settingsPageFrame")
        fl = QVBoxLayout(frame)
        fl.setContentsMargins(0, 0, 0, 0)
        fl.addWidget(self.stack)
        main.addWidget(frame, 1)
        lay = QVBoxLayout(self)
        lay.addLayout(main, 1)
        lay.addLayout(bottom)
        keys = list(self.pages)
        last = str(app_settings.settings().value(self.PAGE_KEY, keys[0]))
        self.nav.setCurrentRow(keys.index(last) if last in keys else 0)
        fit_to_screen(self, 760, 480)

    def _page_changed(self, row):
        self.stack.setCurrentIndex(row)
        if row >= 0:
            app_settings.settings().setValue(self.PAGE_KEY, self.nav.item(row).data(Qt.UserRole))

    def show_page(self, key: str):
        keys = list(self.pages)
        if key in keys:
            self.nav.setCurrentRow(keys.index(key))

