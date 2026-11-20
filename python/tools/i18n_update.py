"""Keep the translation catalogs (python/i18n/*.json) in step with the source code.

    .venv\\Scripts\\python.exe python\\tools\\i18n_update.py            # add new texts, report the state
    .venv\\Scripts\\python.exe python\\tools\\i18n_update.py --prune    # also delete texts no longer in the code
    .venv\\Scripts\\python.exe python\\tools\\i18n_update.py --new xx   # start catalog xx.json (all texts empty)

Every string literal passed to ``tr("...")`` or ``N_("...")`` is a translation key. New keys are added to each
catalog with an empty value (= shown in English until translated); existing translations are never changed.
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import sys

PY = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
I18N = os.path.join(PY, "i18n")
SKIP_DIRS = {"tests", "distribute", "packaging", "whl", "__pycache__", "tools", ".venv"}


def source_files():
    for root, dirs, files in os.walk(PY):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS and not d.startswith(".")]
        for f in sorted(files):
            if f.endswith(".py"):
                yield os.path.join(root, f)


def extract():
    """Ordered {key: "file:line"} of all tr()/N_() literals."""
    keys = {}
    for path in source_files():
        with open(path, encoding="utf-8") as fh:
            tree = ast.parse(fh.read(), path)
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in ("tr", "N_")
                    and node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str)):
                keys.setdefault(node.args[0].value, f"{os.path.relpath(path, PY)}:{node.lineno}")
    return dict(sorted(keys.items(), key=lambda kv: (kv[1].split(":")[0], int(kv[1].split(":")[1]))))


def update(code: str, keys: dict, prune: bool, create: bool = False) -> dict:
    path = os.path.join(I18N, f"{code}.json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            cat = json.load(fh)
    elif create:
        cat = {"_meta": {"name": code, "english": code, "qt": code}}
    else:
        raise FileNotFoundError(path)
    meta = {k: v for k, v in cat.items() if k.startswith("_")}
    old = {k: v for k, v in cat.items() if not k.startswith("_")}
    new = dict(meta)
    for k in keys:
        new[k] = old.get(k, "")
    obsolete = [k for k in old if k not in keys]
    if not prune:
        for k in obsolete:
            new[k] = old[k]
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(new, fh, ensure_ascii=False, indent=1)
        fh.write("\n")
    return dict(total=len(keys), translated=sum(1 for k in keys if old.get(k)),
                added=sum(1 for k in keys if k not in old), obsolete=len(obsolete))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prune", action="store_true", help="remove keys that are no longer used in the code")
    ap.add_argument("--new", metavar="CODE", help="create catalog CODE.json (e.g. ko, ru, es)")
    ap.add_argument("--missing", metavar="CODE", help="list the untranslated keys of CODE")
    a = ap.parse_args(argv)
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")     # keys contain ▸, …, ±
    except AttributeError:
        pass
    keys = extract()
    if a.new:
        update(a.new, keys, prune=False, create=True)
        print(f"created i18n/{a.new}.json - edit its _meta (name, english, qt) and translate the values")
    codes = sorted(f[:-5] for f in os.listdir(I18N) if f.endswith(".json"))
    print(f"{len(keys)} texts in the source code")
    for code in codes:
        st = update(code, keys, a.prune)
        print(f"  {code:8s} translated {st['translated']}/{st['total']}   new {st['added']}   "
              f"{'removed' if a.prune else 'unused'} {st['obsolete']}")
    if a.missing:
        with open(os.path.join(I18N, f"{a.missing}.json"), encoding="utf-8") as fh:
            cat = json.load(fh)
        for k, where in keys.items():
            if not cat.get(k):
                print(f"{where}: {k!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
