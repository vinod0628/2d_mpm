"""Light / dark theme: Qt palette + stylesheet, matplotlib colours and icons.

The colour names (``theme.TEXT``, ``theme.PANEL``, ...) are module attributes that :func:`set_mode` replaces, so
code that reads them at drawing time follows a theme switch. Icons from :func:`icon` are re-coloured on the fly.
"""
from __future__ import annotations

import os

import matplotlib as mpl
from PyQt5.QtCore import QSize, Qt
from PyQt5.QtGui import QColor, QIcon, QIconEngine, QPalette
from PyQt5.QtWidgets import QApplication

DARK = dict(
    BG="#1e1f22",           # window background
    MAP_BG="#ffffff",       # map canvas background while layers are loaded (no layers: BG)
    PANEL="#2b2d30",        # docks, toolbars, dialogs
    FIELD="#232427",        # inputs, tables
    ALT="#26282b",          # alternate table rows
    BORDER="#3c3f45",
    HOVER="#36383d",
    BTN_HOVER="#404349",
    SCROLL="#4a4d53",
    INDICATOR="#6b6f78",    # check box / radio button frame
    TEXT="#dfe1e5",
    MUTED="#8c9099",
    ACCENT="#3d8bfd",
    ACCENT_DARK="#2f6fd0",
    ACCENT_HOVER="#5a9dfd",
    GOOD="#5fb865",
    WARN="#e5a84b",
    BAD="#e06c75",
    GRID="#34363b",
    CHECKED_BG="#2f6fd0",   # checked tool button (active map tool)
    # categorical palette for charts (colour-blind friendly)
    SERIES=["#3d8bfd", "#f0a35e", "#5fb865", "#e06c75", "#b18cf2", "#4fc1c9", "#e5c07b"],
)
LIGHT = dict(
    BG="#eef0f3",
    MAP_BG="#ffffff",
    PANEL="#f7f8fa",
    FIELD="#ffffff",
    ALT="#f3f5f8",
    BORDER="#d3d7de",
    HOVER="#e7eaef",
    BTN_HOVER="#dde1e7",
    SCROLL="#c3c8d0",
    INDICATOR="#8a909a",
    TEXT="#1f2328",
    MUTED="#656c76",
    ACCENT="#2f6fd0",
    ACCENT_DARK="#2a64bd",
    ACCENT_HOVER="#4a85dc",
    GOOD="#2e8540",
    WARN="#b4690e",
    BAD="#c93545",
    GRID="#e3e6ea",
    CHECKED_BG="#d4e3fa",
    SERIES=["#2f6fd0", "#e07b24", "#2e8540", "#c93545", "#8256d0", "#1f9aa5", "#b88a00"],
)
MODES = {"dark": DARK, "light": LIGHT}
MODE = "dark"
globals().update(DARK)
_ROLES = [k for k, v in DARK.items() if isinstance(v, str)]


def set_mode(mode: str) -> str:
    """Switch the colour names to the "dark" or "light" palette (does not touch widgets; see :func:`apply`)."""
    global MODE
    MODE = mode if mode in MODES else "dark"
    globals().update(MODES[MODE])
    return MODE


def palette() -> dict:
    return MODES[MODE]


def _qss() -> str:
    return f"""
* {{ font-family: "Segoe UI", "Microsoft YaHei UI", "Inter", sans-serif; font-size: 9pt; }}
QMainWindow, QDialog {{ background: {PANEL}; }}
QWidget {{ color: {TEXT}; }}
QToolTip {{ background: {FIELD}; color: {TEXT}; border: 1px solid {BORDER}; padding: 4px; }}
QMenuBar {{ background: {PANEL}; border-bottom: 1px solid {BORDER}; }}
QMenuBar::item {{ padding: 4px 10px; background: transparent; }}
QMenuBar::item:selected {{ background: {HOVER}; border-radius: 4px; }}
QMenu {{ background: {PANEL}; border: 1px solid {BORDER}; padding: 4px; }}
QMenu::item {{ padding: 5px 24px 5px 26px; border-radius: 4px; }}
QMenu::item:selected {{ background: {ACCENT}; color: white; }}
QMenu::item:disabled {{ color: {MUTED}; }}
QMenu::separator {{ height: 1px; background: {BORDER}; margin: 4px 6px; }}
QToolBar {{ background: {PANEL}; border: none; border-bottom: 1px solid {BORDER}; spacing: 2px; padding: 3px; }}
QToolBar::separator {{ width: 1px; background: {BORDER}; margin: 4px 6px; }}
QToolButton {{ background: transparent; border: 1px solid transparent; border-radius: 5px; padding: 4px; }}
QToolButton:hover {{ background: {HOVER}; border-color: {BORDER}; }}
QToolButton:checked {{ background: {CHECKED_BG}; border-color: {ACCENT}; }}
QDockWidget {{ titlebar-close-icon: none; }}
QDockWidget::title {{ background: {PANEL}; padding: 6px 8px; border-bottom: 1px solid {BORDER};
                      font-weight: 600; color: {MUTED}; text-transform: uppercase; }}
QStatusBar {{ background: {PANEL}; border-top: 1px solid {BORDER}; color: {MUTED}; }}
QStatusBar QLabel {{ color: {MUTED}; padding: 0 8px; }}
QTreeWidget, QTreeView, QListWidget, QTableView, QTableWidget, QPlainTextEdit, QTextEdit, QTextBrowser {{
    background: {FIELD}; border: 1px solid {BORDER}; border-radius: 6px;
    selection-background-color: {ACCENT_DARK}; selection-color: white; alternate-background-color: {ALT}; }}
QTreeView::item, QListWidget::item {{ padding: 3px 2px; border-radius: 4px; }}
QTreeView::item:hover, QListWidget::item:hover {{ background: {HOVER}; }}
QTreeView::item:selected, QListWidget::item:selected {{ background: {ACCENT_DARK}; color: white; }}
QHeaderView::section {{ background: {PANEL}; color: {MUTED}; border: none; border-right: 1px solid {BORDER};
                         border-bottom: 1px solid {BORDER}; padding: 4px 6px; font-weight: 600; }}
QTableCornerButton::section {{ background: {PANEL}; border: none; }}
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QPlainTextEdit#input {{
    background: {FIELD}; border: 1px solid {BORDER}; border-radius: 5px; padding: 4px 6px; min-height: 18px; }}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {{ border-color: {ACCENT}; }}
QLineEdit:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled, QComboBox:disabled {{ color: {MUTED}; }}
QComboBox::drop-down {{ border: none; width: 18px; }}
QComboBox QAbstractItemView {{ background: {PANEL}; border: 1px solid {BORDER}; selection-background-color: {ACCENT}; }}
QPushButton {{ background: {HOVER}; border: 1px solid {BORDER}; border-radius: 5px; padding: 5px 14px; }}
QPushButton:hover {{ background: {BTN_HOVER}; }}
QPushButton:pressed {{ background: {BORDER}; }}
QPushButton:disabled {{ color: {MUTED}; background: {PANEL}; }}
QPushButton[primary="true"] {{ background: {ACCENT}; border-color: {ACCENT}; color: white; font-weight: 600; }}
QPushButton[primary="true"]:hover {{ background: {ACCENT_HOVER}; }}
QPushButton:default {{ border-color: {ACCENT}; }}
QGroupBox {{ border: 1px solid {BORDER}; border-radius: 6px; margin-top: 12px; padding: 10px 8px 8px 8px; }}
QGroupBox::title {{ subcontrol-origin: margin; left: 10px; padding: 0 4px; color: {MUTED}; font-weight: 600; }}
QTabWidget::pane {{ border: 1px solid {BORDER}; border-radius: 6px; top: -1px; background: {PANEL}; }}
QTabBar::tab {{ background: transparent; color: {MUTED}; padding: 6px 14px; border: none;
                border-bottom: 2px solid transparent; }}
QTabBar::tab:selected {{ color: {TEXT}; border-bottom: 2px solid {ACCENT}; }}
QTabBar::tab:hover {{ color: {TEXT}; }}
QCheckBox, QRadioButton {{ spacing: 6px; }}
QCheckBox::indicator, QRadioButton::indicator, QTreeView::indicator, QTableView::indicator {{
    width: 14px; height: 14px; border: 1px solid {INDICATOR}; background: {FIELD}; border-radius: 3px; }}
QRadioButton::indicator {{ border-radius: 8px; }}
QCheckBox::indicator:hover, QRadioButton::indicator:hover, QTreeView::indicator:hover {{ border-color: {ACCENT}; }}
QCheckBox::indicator:checked, QTreeView::indicator:checked, QTableView::indicator:checked {{
    background: {ACCENT}; border-color: {ACCENT}; image: url({{CHECK}}); }}
QRadioButton::indicator:checked {{ background: {ACCENT}; border-color: {ACCENT}; image: url({{RADIO}}); }}
QCheckBox::indicator:disabled, QTreeView::indicator:disabled {{ border-color: {BORDER}; background: {PANEL}; }}
QSlider::groove:horizontal {{ height: 4px; background: {BORDER}; border-radius: 2px; }}
QSlider::handle:horizontal {{ background: {ACCENT}; width: 14px; margin: -6px 0; border-radius: 7px; }}
QProgressBar {{ background: {FIELD}; border: 1px solid {BORDER}; border-radius: 4px; text-align: center;
                height: 12px; color: {TEXT}; }}
QProgressBar::chunk {{ background: {ACCENT}; border-radius: 3px; }}
QScrollBar:vertical {{ background: transparent; width: 10px; margin: 0; }}
QScrollBar::handle:vertical {{ background: {SCROLL}; border-radius: 4px; min-height: 24px; }}
QScrollBar:horizontal {{ background: transparent; height: 10px; margin: 0; }}
QScrollBar::handle:horizontal {{ background: {SCROLL}; border-radius: 4px; min-width: 24px; }}
QScrollBar::add-line, QScrollBar::sub-line {{ width: 0; height: 0; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: none; }}
QSplitter::handle {{ background: {BORDER}; }}
QLabel#hint {{ color: {MUTED}; }}
QLabel#title {{ font-size: 12pt; font-weight: 600; }}
QLabel#metric {{ font-size: 18pt; font-weight: 600; color: {TEXT}; }}
QLabel#metricCaption {{ color: {MUTED}; }}
QFrame#mapFooter {{ background: {PANEL}; border-top: 1px solid {BORDER}; }}
QLabel#coords {{ font-family: "Cascadia Mono", "Consolas", monospace; color: {TEXT}; }}
QLabel#coordsMuted {{ font-family: "Cascadia Mono", "Consolas", monospace; color: {MUTED}; }}
QFrame#card {{ background: {FIELD}; border: 1px solid {BORDER}; border-radius: 8px; }}
QScrollArea#dialogScroll, QWidget#dialogBody {{ background: transparent; border: none; }}
QPushButton[primary="true"]:disabled {{ background: {HOVER}; border-color: {BORDER}; color: {MUTED}; font-weight: 600; }}
QListWidget#settingsNav {{ background: {FIELD}; border: 1px solid {BORDER}; border-radius: 6px; padding: 4px;
                           outline: 0; }}
QListWidget#settingsNav::item {{ padding: 7px 8px; margin: 1px 0; }}
QFrame#settingsPageFrame {{ background: {PANEL}; border: 1px solid {BORDER}; border-radius: 6px; margin-left: 8px; }}
QLabel#sectionTitle {{ font-weight: 600; color: {TEXT}; }}
QFrame#sectionRule {{ background: {BORDER}; border: none; }}
QToolButton#themeCard {{ border: 1px solid {BORDER}; border-radius: 8px; padding: 8px 8px 6px 8px; }}
QToolButton#themeCard:hover {{ background: {HOVER}; }}
QToolButton#themeCard:checked {{ background: {HOVER}; border: 2px solid {ACCENT}; }}
"""


def apply(app: QApplication, mode: str = None):
    """Apply the palette, stylesheet and matplotlib defaults of ``mode`` (default: the current mode)."""
    if mode:
        set_mode(mode)
    app.setStyle("Fusion")
    pal = QPalette()
    for role, col in [(QPalette.Window, PANEL), (QPalette.WindowText, TEXT), (QPalette.Base, FIELD),
                      (QPalette.AlternateBase, ALT), (QPalette.ToolTipBase, FIELD),
                      (QPalette.ToolTipText, TEXT), (QPalette.Text, TEXT), (QPalette.Button, HOVER),
                      (QPalette.ButtonText, TEXT), (QPalette.Highlight, ACCENT_DARK),
                      (QPalette.HighlightedText, "#ffffff"), (QPalette.Link, ACCENT),
                      (QPalette.PlaceholderText, MUTED), (QPalette.Light, FIELD), (QPalette.Midlight, HOVER),
                      (QPalette.Mid, BORDER), (QPalette.Dark, BORDER), (QPalette.Shadow, BORDER)]:
        pal.setColor(role, QColor(col))
    for role in (QPalette.Text, QPalette.WindowText, QPalette.ButtonText):
        pal.setColor(QPalette.Disabled, role, QColor(MUTED))
    app.setPalette(pal)
    res = os.path.join(os.path.dirname(os.path.abspath(__file__)), "resources").replace("\\", "/")
    app.setStyleSheet(_qss().replace("{CHECK}", f"{res}/check.svg").replace("{RADIO}", f"{res}/radio.svg"))
    apply_matplotlib()


def apply_matplotlib():
    mpl.rcParams.update({
        "figure.facecolor": PANEL, "axes.facecolor": FIELD, "savefig.facecolor": PANEL,
        "axes.edgecolor": BORDER, "axes.labelcolor": TEXT, "axes.titlecolor": TEXT,
        "xtick.color": MUTED, "ytick.color": MUTED, "text.color": TEXT, "grid.color": GRID,
        "grid.linewidth": 0.6, "axes.grid": True, "axes.prop_cycle": mpl.cycler(color=SERIES),
        "legend.facecolor": PANEL, "legend.edgecolor": BORDER, "legend.framealpha": 0.95,
        "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "semibold",
        "axes.spines.top": False, "axes.spines.right": False,
        "font.family": _font_families(),
    })


_FONTS = None


def _font_families() -> list:
    """DejaVu Sans + an installed CJK font: matplotlib takes glyphs missing from the first family (Chinese UI
    texts) from the next one. Only installed families are listed (a missing one would print warnings)."""
    global _FONTS
    if _FONTS is None:
        from matplotlib import font_manager
        installed = {f.name for f in font_manager.fontManager.ttflist}
        _FONTS = ["DejaVu Sans"] + [f for f in ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Arial Unicode MS")
                                    if f in installed][:1]
    return _FONTS


# ---------------------------------------------------------------------------------------------- icons
_QTA_CACHE = {}


def _qta_icon(name: str, color: str, disabled: str) -> QIcon:
    key = (name, color, disabled)
    if key not in _QTA_CACHE:
        try:
            import qtawesome as qta
            _QTA_CACHE[key] = qta.icon(name, color=color, color_disabled=disabled,
                                        color_selected="#ffffff")   # on the accent-coloured selection
        except Exception:  # noqa: BLE001
            _QTA_CACHE[key] = QIcon()
    return _QTA_CACHE[key]


class _ThemedIconEngine(QIconEngine):
    """Draws a qtawesome icon in the colour its role has in the *current* theme (re-coloured after a switch)."""

    def __init__(self, name: str, role: str):
        super().__init__()
        self._name, self._role = name, role

    def _icon(self) -> QIcon:
        pal = palette()
        return _qta_icon(self._name, pal.get(self._role, self._role), pal["MUTED"])

    def paint(self, painter, rect, mode, state):
        self._icon().paint(painter, rect, Qt.AlignCenter, mode, state)

    def pixmap(self, size, mode, state):
        return self._icon().pixmap(size, mode, state)

    def actualSize(self, size, mode, state):
        return QSize(size)

    def clone(self):
        return _ThemedIconEngine(self._name, self._role)


_ICON_CACHE = {}


def icon(name: str, color: str = None) -> QIcon:
    """qtawesome icon (e.g. 'mdi6.folder-open'); empty icon if unavailable.

    ``color``: None (= TEXT), a role name ("MUTED", "ACCENT", ...) or a colour. A colour equal to the current
    value of a role (e.g. ``theme.MUTED``) is treated as that role, so the icon follows theme switches.
    """
    if _qta_icon(name, TEXT, MUTED).isNull():              # qtawesome missing or unknown icon name
        return QIcon()
    if color is None:
        role = "TEXT"
    elif color in _ROLES:
        role = color
    else:
        pal = palette()
        role = next((k for k in ("TEXT", "MUTED", "ACCENT", "GOOD", "WARN", "BAD") if pal[k].lower() == color.lower()),
                    color)
    key = (name, role)
    if key not in _ICON_CACHE:
        _ICON_CACHE[key] = QIcon(_ThemedIconEngine(name, role))
    return _ICON_CACHE[key]
