"""Minimal multi-language support for MPM Studio.

The English text in the source code is the translation key::

    from i18n import tr, N_
    QLabel(tr("Cell size"))
    log.info(tr("Opened project {path} ({n} layers)", path=p, n=len(layers)))
    CATEGORIES = [N_("Data-driven"), ...]      # marked for translation, translated later with tr(name)

Every other language is one JSON file in this folder (``<code>.json``, e.g. ``zh_CN.json``)::

    {"_meta": {"name": "简体中文", "english": "Chinese (Simplified)", "qt": "zh_CN"},
     "Cell size": "像元大小",
     "Opened project {path} ({n} layers)": "已打开项目 {path}（{n} 个图层）"}

Untranslated or missing entries fall back to English, so a catalog can be completed step by step.
To add a language: copy ``zh_CN.json`` to ``<code>.json``, edit ``_meta`` and the values, then run
``python tools/i18n_update.py`` (adds new source strings to every catalog, reports unused ones).
The language is chosen in Settings ▸ General settings and takes effect at the next start.
"""
from __future__ import annotations

import json
import os
import sys
from typing import Dict

DEFAULT = "en"
_HERE = os.path.dirname(os.path.abspath(__file__))
_catalog: Dict[str, str] = {}
_current = DEFAULT


def locale_dir() -> str:
    base = getattr(sys, "_MEIPASS", None)               # PyInstaller build: bundled next to the other data
    return os.path.join(base, "i18n") if base else _HERE


def _read(code: str) -> dict:
    with open(os.path.join(locale_dir(), f"{code}.json"), encoding="utf-8") as fh:
        return json.load(fh)


def available() -> Dict[str, dict]:
    """code -> meta ({"name": native name, "english": English name, "qt": Qt locale}); English first."""
    langs = {DEFAULT: {"name": "English", "english": "English", "qt": "en"}}
    try:
        files = sorted(f for f in os.listdir(locale_dir()) if f.endswith(".json"))
    except OSError:
        files = []
    for f in files:
        code = f[:-5]
        try:
            meta = _read(code).get("_meta", {})
        except (OSError, ValueError):
            continue
        langs[code] = {"name": meta.get("name", code), "english": meta.get("english", code),
                       "qt": meta.get("qt", code)}
    return langs


def set_language(code: str) -> str:
    """Load a catalog (unknown / broken -> English). Call before the GUI modules are imported."""
    global _catalog, _current
    _catalog, _current = {}, DEFAULT
    if code and code != DEFAULT:
        try:
            data = _read(code)
            _catalog = {k: v for k, v in data.items() if not k.startswith("_") and isinstance(v, str) and v}
            _current = code
        except (OSError, ValueError):
            pass
    return _current


def language() -> str:
    return _current


def qt_locale() -> str:
    return available().get(_current, {}).get("qt", _current)


def tr(text: str, **kw) -> str:
    """Translated ``text``; ``{name}`` placeholders are filled from the keyword arguments."""
    s = _catalog.get(text, text)
    if kw:
        try:
            return s.format(**kw)
        except (KeyError, IndexError, ValueError):     # broken translation: fall back to English
            return text.format(**kw)
    return s


def N_(text: str) -> str:
    """Mark a string for translation without translating it (module constants); translate with tr() on use."""
    return text


def install_qt_translators(app) -> list:
    """Translate Qt's own texts (standard buttons, file dialogs) when Qt ships a catalog for the language."""
    if _current == DEFAULT:
        return []
    from PyQt5.QtCore import QLibraryInfo, QTranslator
    path = QLibraryInfo.location(QLibraryInfo.TranslationsPath)
    loc = qt_locale()
    keep = []
    for base in ("qtbase", "qt"):
        t = QTranslator(app)
        if t.load(f"{base}_{loc}", path) or t.load(f"{base}_{loc.split('_')[0]}", path):
            app.installTranslator(t)
            keep.append(t)
            break                                       # qtbase_xx or the older all-in-one qt_xx
    return keep
