"""User preferences (stored with QSettings, i.e. in the Windows registry under HKCU\\Software\\MPM).

Set the environment variable ``MPM_SETTINGS_FILE`` to an .ini path to keep the preferences in that file instead
(the tests do this, so they never change the real recent-projects list or language / theme).
"""
from __future__ import annotations

import os
from typing import List

from PyQt5.QtCore import QSettings

from version import APP_NAME

ORG = "MPM"
K_LANGUAGE = "ui/language"
K_THEME = "ui/theme"
K_RECENT = "projects/recent"
MAX_RECENT = 5


def settings() -> QSettings:
    ini = os.environ.get("MPM_SETTINGS_FILE")
    if ini:
        return QSettings(ini, QSettings.IniFormat)
    return QSettings(ORG, APP_NAME)


def language() -> str:
    return str(settings().value(K_LANGUAGE, "en") or "en")


def set_language(code: str):
    settings().setValue(K_LANGUAGE, code)


def theme_mode() -> str:
    m = str(settings().value(K_THEME, "dark") or "dark")
    return m if m in ("dark", "light") else "dark"


def set_theme_mode(mode: str):
    settings().setValue(K_THEME, mode)


# ------------------------------------------------------------------------------------------- recent projects
def _same(a: str, b: str) -> bool:
    return os.path.normcase(os.path.abspath(a)) == os.path.normcase(os.path.abspath(b))


def recent_projects() -> List[str]:
    """Project folders, most recent first (at most MAX_RECENT)."""
    v = settings().value(K_RECENT, [])
    if isinstance(v, str):                       # QSettings returns a plain string for a one-item list
        v = [v]
    return [str(p) for p in (v or []) if p][:MAX_RECENT]


def add_recent_project(folder: str):
    folder = os.path.abspath(folder)
    items = [folder] + [p for p in recent_projects() if not _same(p, folder)]
    settings().setValue(K_RECENT, items[:MAX_RECENT])


def remove_recent_project(folder: str):
    settings().setValue(K_RECENT, [p for p in recent_projects() if not _same(p, folder)])


def clear_recent_projects():
    settings().setValue(K_RECENT, [])
