"""Folder-based MPM projects.

A project is a folder::

    <project>/
        project.mpm.json        layer list (relative paths), styles, map CRS
        data/                   layers created inside the project (saved in-memory layers, tool outputs)
        results/<run_id>/       one folder per method run, run_id = <method>_<MMDD>_<HHMMSS>

Layer paths inside the project folder are stored relative, so a project folder can be copied or moved
(*Save project as* copies it). Short, ASCII names keep the files usable in ArcMap / ArcGIS.
"""
from __future__ import annotations

import datetime as _dt
import glob
import json
import os
import re
import shutil
from typing import List, Optional, Tuple

from core.layers import LayerStore, RasterLayer, VectorLayer
from i18n import tr

PROJECT_FILE = "project.mpm.json"
DATA_DIR = "data"
RESULTS_DIR = "results"
_IGNORE = {"desktop.ini", "thumbs.db", ".ds_store"}


# ------------------------------------------------------------------------------------------- helpers
def is_empty_folder(folder: str) -> bool:
    if not os.path.isdir(folder):
        return True
    return not [f for f in os.listdir(folder) if f.lower() not in _IGNORE]


class ProjectFileError(Exception):
    """The folder does not contain exactly one project file."""


def project_files(folder: str) -> List[str]:
    return sorted(glob.glob(os.path.join(folder, "*.mpm.json")))


def resolve_project_file(folder_or_file: str) -> str:
    """The project file to open: a given *.mpm.json, or the ONE *.mpm.json in the root of a project folder.

    Raises ProjectFileError when the folder has no project file or more than one (ambiguous).
    """
    if os.path.isfile(folder_or_file):
        return folder_or_file
    if not os.path.isdir(folder_or_file):
        raise ProjectFileError(tr("Folder not found:\n{folder}", folder=folder_or_file))
    files = project_files(folder_or_file)
    if not files:
        raise ProjectFileError(tr("This folder is not an MPM project — there is no project file (*.mpm.json) in\n"
                                  "{folder}", folder=folder_or_file))
    if len(files) > 1:
        raise ProjectFileError(tr("This folder contains more than one project file, so it is not clear which one to "
                                  "open:\n\n{files}\n\nKeep only one *.mpm.json in the project folder.",
                                  files="\n".join(os.path.basename(f) for f in files)))
    return files[0]


def _remove_other_project_files(folder: str, keep: str):
    """A project folder holds exactly one project file: remove others (the user confirmed replacing them)."""
    for f in project_files(folder):
        if os.path.normcase(os.path.abspath(f)) != os.path.normcase(os.path.abspath(keep)):
            os.remove(f)


def find_project_file(folder_or_file: str) -> Optional[str]:
    """``project.mpm.json`` (or the single other ``*.mpm.json``) of a folder; a json path is returned as is."""
    if os.path.isfile(folder_or_file):
        return folder_or_file
    if not os.path.isdir(folder_or_file):
        return None
    p = os.path.join(folder_or_file, PROJECT_FILE)
    if os.path.isfile(p):
        return p
    cands = sorted(glob.glob(os.path.join(folder_or_file, "*.mpm.json")))
    return cands[0] if cands else None


def safe_name(name: str, max_len: int = 40) -> str:
    """ASCII letters, digits and '_' only (GIS-friendly file / layer names)."""
    s = re.sub(r"[^A-Za-z0-9_]+", "_", name).strip("_") or "layer"
    return s[:max_len]


def timestamp(now: Optional[_dt.datetime] = None) -> str:
    return (now or _dt.datetime.now()).strftime("%m%d_%H%M%S")


def unique_path(path: str) -> str:
    if not os.path.exists(path):
        return path
    root, ext = os.path.splitext(path)
    if root.endswith(".mpm"):
        root, ext = root[:-4], ".mpm" + ext
    i = 2
    while os.path.exists(f"{root}_{i}{ext}"):
        i += 1
    return f"{root}_{i}{ext}"


def _inside(path: Optional[str], folder: str) -> bool:
    if not path:
        return False
    try:
        return os.path.commonpath([os.path.abspath(path), os.path.abspath(folder)]) == os.path.abspath(folder)
    except ValueError:                      # different drives
        return False


# ------------------------------------------------------------------------------------------- project API
def project_dir(store: LayerStore) -> Optional[str]:
    return os.path.dirname(os.path.abspath(store.project_path)) if store.project_path else None


def create_project(store: LayerStore, folder: str) -> str:
    """Start an empty project in ``folder`` (created if needed) and save it immediately."""
    os.makedirs(folder, exist_ok=True)
    for d in (DATA_DIR, RESULTS_DIR):
        os.makedirs(os.path.join(folder, d), exist_ok=True)
    store.clear()
    path = os.path.join(folder, PROJECT_FILE)
    _remove_other_project_files(folder, path)
    store.save_project(path, name=os.path.basename(os.path.abspath(folder)))
    store.dirty = False
    return path


def open_project(store: LayerStore, folder_or_file: str) -> List[str]:
    path = resolve_project_file(folder_or_file)        # exactly one *.mpm.json, else ProjectFileError
    errors = store.load_project(path)
    folder = project_dir(store)
    for d in (DATA_DIR, RESULTS_DIR):
        os.makedirs(os.path.join(folder, d), exist_ok=True)
    store.dirty = False
    return errors


def unsaved_layers(store: LayerStore) -> List:
    return [l for l in store.layers if not l.source]


def save_project(store: LayerStore) -> List[str]:
    """Write in-memory layers to <project>/data/ and update the project file. Returns the files written."""
    if not store.project_path:
        raise RuntimeError(tr("No project is open."))
    folder = project_dir(store)
    data = os.path.join(folder, DATA_DIR)
    os.makedirs(data, exist_ok=True)
    written = []
    for layer in unsaved_layers(store):
        ext = ".tif" if isinstance(layer, RasterLayer) else ".gpkg"
        path = unique_path(os.path.join(data, safe_name(layer.name) + ext))
        layer.save(path)
        written.append(path)
    store.save_project(store.project_path, name=os.path.basename(folder))
    store.dirty = False
    return written


def save_project_as(store: LayerStore, new_folder: str) -> List[str]:
    """Copy the project folder to ``new_folder``, continue in the copy and save it there."""
    if not store.project_path:
        raise RuntimeError(tr("No project is open."))
    old = project_dir(store)
    new = os.path.abspath(new_folder)
    if os.path.normcase(new) == os.path.normcase(old):
        return save_project(store)
    if _inside(new, old):
        raise ValueError(tr("The new project folder must not be inside the current project folder."))
    shutil.copytree(old, new, dirs_exist_ok=True)
    for layer in store.layers:                       # re-point layers stored inside the old folder
        if _inside(layer.source, old):
            layer.source = os.path.join(new, os.path.relpath(layer.source, old))
    store.project_path = os.path.join(new, os.path.basename(store.project_path))
    _remove_other_project_files(new, store.project_path)
    return save_project(store)


def new_run(store: LayerStore, label: str) -> Tuple[str, str]:
    """Create results/<label>_<MMDD_HHMMSS>/ and return (run_id, folder)."""
    if not store.project_path:
        raise RuntimeError(tr("No project is open."))
    run_id = f"{safe_name(label, 20)}_{timestamp()}"
    folder = unique_path(os.path.join(project_dir(store), RESULTS_DIR, run_id))
    os.makedirs(folder)
    return os.path.basename(folder), folder


def copy_project(src: str, dst_folder: str) -> Tuple[str, List[str]]:
    """Copy a project (folder or project file) into ``dst_folder`` as a new project.

    Layers stored inside the source folder keep relative paths (they are copied); layers stored elsewhere are
    referenced by absolute path; missing ones are dropped. Returns (new project file, dropped layer names).
    """
    src_json = find_project_file(src)
    if src_json is None:
        raise FileNotFoundError(tr("No project file found in {folder}", folder=src))
    src_dir = os.path.dirname(os.path.abspath(src_json))
    dst = os.path.abspath(dst_folder)
    if _inside(dst, src_dir):
        raise ValueError(tr("The new project folder must not be inside the source project folder."))
    shutil.copytree(src_dir, dst, dirs_exist_ok=True)
    with open(src_json, encoding="utf-8") as fh:
        proj = json.load(fh)
    kept, dropped = [], []
    for lyr in proj.get("layers", []):
        p = lyr["path"]
        full = p if os.path.isabs(p) else os.path.normpath(os.path.join(src_dir, p))
        if _inside(full, src_dir) and os.path.exists(full):
            kept.append(lyr)                                   # copied: the relative path stays valid
        elif os.path.exists(full):
            lyr["path"] = full                                 # outside the project: absolute reference
            kept.append(lyr)
        else:
            dropped.append(lyr.get("name", p))
    proj["layers"] = kept
    proj["name"] = os.path.basename(dst)
    new_json = os.path.join(dst, PROJECT_FILE)
    with open(new_json, "w", encoding="utf-8") as fh:
        json.dump(proj, fh, indent=2)
    _remove_other_project_files(dst, new_json)      # incl. the copied source file if it had another name
    for d in (DATA_DIR, RESULTS_DIR):
        os.makedirs(os.path.join(dst, d), exist_ok=True)
    return new_json, dropped
