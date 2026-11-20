"""Translation catalogs, tr() fallback and light/dark theme switching."""
import json
import os
import re
import string
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
PY = os.path.dirname(HERE)
sys.path.insert(0, PY)
sys.path.insert(0, os.path.join(PY, "tools"))

import i18n  # noqa: E402
import i18n_update  # noqa: E402

CATALOGS = sorted(f[:-5] for f in os.listdir(os.path.join(PY, "i18n")) if f.endswith(".json"))


def _fields(s):
    return sorted(f[1] for f in string.Formatter().parse(s) if f[1] is not None)


@pytest.fixture(autouse=True)
def english_afterwards():
    yield
    i18n.set_language("en")


def test_tr_fallback_and_format():
    i18n.set_language("en")
    assert i18n.tr("Layers") == "Layers"
    assert i18n.tr("Opened project {path} ({n} layers)", path="p", n=3) == "Opened project p (3 layers)"
    assert i18n.set_language("does_not_exist") == "en"
    assert "en" in i18n.available() and "zh_CN" in i18n.available()


@pytest.mark.parametrize("code", CATALOGS)
def test_catalog_complete_and_consistent(code):
    keys = i18n_update.extract()
    with open(os.path.join(PY, "i18n", f"{code}.json"), encoding="utf-8") as fh:
        cat = json.load(fh)
    assert {"name", "english", "qt"} <= set(cat["_meta"])
    missing = [k for k in keys if not cat.get(k)]
    assert not missing, f"{len(missing)} untranslated, e.g. {missing[:5]}"
    for k, v in cat.items():
        if k.startswith("_") or not v:
            continue
        assert _fields(k) == _fields(v), (k, v)                 # same {placeholders}
        if re.search(r"&\w", k):
            assert re.search(r"&\w", v), (k, v)                  # menu mnemonic kept


def test_chinese_catalog_in_use():
    i18n.set_language("zh_CN")
    assert i18n.language() == "zh_CN"
    assert i18n.tr("Layers") == "图层"
    assert "3" in i18n.tr("Opened project {path} ({n} layers)", path="p", n=3)


def test_theme_switch_recolours_icons():
    from PyQt5.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    from gui import theme

    def colours(ic):
        im = ic.pixmap(24, 24).toImage()
        return {im.pixelColor(x, y).name() for x in range(24) for y in range(24) if im.pixelColor(x, y).alpha() == 255}

    try:
        theme.apply(app, "dark")
        ic = theme.icon("mdi6.ruler")
        assert colours(ic) == {theme.DARK["TEXT"]}
        theme.apply(app, "light")
        assert theme.TEXT == theme.LIGHT["TEXT"] and theme.PANEL == theme.LIGHT["PANEL"]
        assert colours(ic) == {theme.LIGHT["TEXT"]}              # same QIcon, new colour
        assert theme.LIGHT["PANEL"] in app.styleSheet()
    finally:
        theme.apply(app, "dark")
