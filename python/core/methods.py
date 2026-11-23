"""Plug-in registry for MPM methods.

Methods live in ``python/logic/<category>/<method>/`` (category = data_driven | knowledge_driven | hybrid).
Each method package has a ``plugin.py`` whose :class:`MPMMethod` subclass is decorated with :func:`register`.
:func:`load_builtin_methods` imports every ``plugin.py`` it finds, and the main window builds the
*Knowledge-driven* / *Data-driven* / *Hybrid* menus from the registry, so adding a method needs no GUI code.
Planned methods are listed (disabled) via :data:`PLANNED`. Copy ``logic/method_template`` to start.

Life cycle::

    dlg = method.create_dialog(parent, store)      # parameter dialog (QDialog)
    if dlg.exec_(): params = dlg.params()          # + params["nodata"] = project NoData value (store.nodata)
    result = method.run(params, progress)          # worker thread, no GUI calls
    method.show_result(main_window, result)        # GUI thread: add layers, open result dialogs
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Type

import numpy as np

from i18n import N_
from utils.raster_tools import Grid

ProgressFn = Callable[[int, str], None]


@dataclass
class OutputRaster:
    name: str                       # short file name inside the run folder, e.g. "post" -> results/<run_id>/post.tif
    data: np.ma.MaskedArray
    grid: Grid
    categorical: bool = False
    cmap: str = "magma"
    description: str = ""
    visible: bool = True


@dataclass
class OutputVector:
    name: str                       # short file name inside the run folder -> results/<run_id>/<name>.gpkg
    gdf: object                     # geopandas.GeoDataFrame
    description: str = ""
    style: Dict[str, object] = field(default_factory=dict)      # VectorLayer style (color, marker, markersize …)
    visible: bool = True


@dataclass
class MethodResult:
    method: str
    outputs: List[OutputRaster] = field(default_factory=list)   # written to results/<run_id>/<name>.tif
    vectors: List[OutputVector] = field(default_factory=list)   # written to results/<run_id>/<name>.gpkg
    label: str = ""                                               # run_id prefix (default: method key)
    tables: Dict[str, object] = field(default_factory=dict)       # name -> pandas.DataFrame
    evaluation: Optional[object] = None                           # evaluation.metrics.Evaluation
    messages: List[str] = field(default_factory=list)
    extra: Dict[str, object] = field(default_factory=dict)


class MPMMethod:
    key: str = ""
    name: str = ""
    category: str = "Data-driven"          # one of CATEGORIES
    group: str = ""                        # sub-menu, e.g. "Statistical", "Machine learning"
    description: str = ""
    icon: str = "mdi6.function-variant"

    def create_dialog(self, parent, store):
        raise NotImplementedError

    def run(self, params: dict, progress: ProgressFn) -> MethodResult:
        raise NotImplementedError

    def show_result(self, window, result: MethodResult):
        """Default: add output rasters (written with the project NoData value) and vectors to the map."""
        for out in result.outputs:
            window.add_output_raster(out)
        for vec in result.vectors:
            window.add_output_vector(vec)


# menu title -> package that holds the methods of that category (menu order = this order).
# Titles are English keys (compared with MPMMethod.category); the GUI shows tr(title).
CATEGORIES = {
    N_("Knowledge-driven"): "logic.knowledge_driven",
    N_("Data-driven"): "logic.data_driven",
    N_("Hybrid"): "logic.hybrid",
}

REGISTRY: Dict[str, MPMMethod] = {}


def register(cls: Type[MPMMethod]) -> Type[MPMMethod]:
    inst = cls()
    if not inst.key:
        raise ValueError(f"{cls.__name__} has no key")
    if inst.category not in CATEGORIES:
        raise ValueError(f"{cls.__name__}: category must be one of {list(CATEGORIES)}")
    REGISTRY[inst.key] = inst
    return cls


def methods(category: Optional[str] = None) -> List[MPMMethod]:
    return [m for m in REGISTRY.values() if category is None or m.category == category]


# (category, group, name) — shown greyed out in the menus until implemented (GUI shows tr(group), tr(name))
PLANNED = [
    ("Data-driven", N_("Machine learning"), N_("One-class / PU learning")),
]


def load_builtin_methods() -> List[str]:
    """Import ``plugin.py`` of every method package under logic/<category>/ (registers the methods).

    Packages whose name starts with "_" are skipped. Returns the imported module names.
    """
    import importlib
    import pkgutil

    loaded = []
    for pkg_name in CATEGORIES.values():
        try:
            pkg = importlib.import_module(pkg_name)
        except ModuleNotFoundError:
            continue
        for info in sorted(pkgutil.iter_modules(pkg.__path__), key=lambda m: m.name):
            if not info.ispkg or info.name.startswith("_"):
                continue
            mod = f"{pkg_name}.{info.name}.plugin"
            try:
                importlib.import_module(mod)
            except ModuleNotFoundError as exc:
                if exc.name != mod:                 # a real missing dependency inside the plug-in
                    raise
                continue                            # package without plugin.py (work in progress)
            loaded.append(mod)
    return loaded
