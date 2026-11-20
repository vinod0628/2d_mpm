"""Build the MPM Studio Windows release.

    .venv\\Scripts\\python.exe python\\packaging\\build_release.py [--skip-build] [--no-zip] [--keep-work]
    (or build_release.bat in the repository root)

All outputs go to python/distribute/ (build/, dist/, release/):
1. PyInstaller one-folder build  -> python/distribute/dist/MPM Studio/ (MPM Studio.exe + _internal/), work dir python/distribute/build/
2. add demo data (processed layers + cleaned project files), docs, README.txt
3. self-test the built exe headless (loads both demo projects, runs WofE, opens dialogs)
4. zip                           -> python/distribute/release/MPM_Studio_<version>_win64.zip
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

SRC = Path(__file__).resolve().parents[1]          # python/
ROOT = SRC.parent                                  # repository root (demo_data/, docs/)
sys.path.insert(0, str(SRC))
from version import APP_NAME, VERSION  # noqa: E402

OUT = SRC / "distribute"                           # all build outputs
DIST = OUT / "dist"
WORK = OUT / "build"
RELEASE = OUT / "release"
APP_DIR = DIST / APP_NAME
DEMOS = ["nevada_carlin_au", "curnamona_ree"]


def step(msg):
    print(f"\n=== {msg}", flush=True)


def stop_running_app():
    """A running copy of dist\\MPM Studio.exe locks the folder: stop it (only copies started from dist\\)."""
    if os.name != "nt":
        return
    ps = ("Get-Process -Name 'MPM Studio' -ErrorAction SilentlyContinue | "
          f"Where-Object {{ $_.Path -like '{DIST}\\*' }} | Stop-Process -Force")
    subprocess.run(["powershell", "-NoProfile", "-Command", ps], check=False)


def build():
    step(f"PyInstaller build of {APP_NAME} {VERSION}")
    stop_running_app()
    for d in (APP_DIR, WORK):
        if d.exists():
            shutil.rmtree(d)
    t0 = time.time()
    cmd = [sys.executable, "-m", "PyInstaller", str(SRC / "packaging" / "mpm_studio.spec"), "--noconfirm",
           "--clean", "--distpath", str(DIST), "--workpath", str(WORK), "--log-level", "WARN"]
    env = dict(os.environ, USE_PYGEOS="0")
    subprocess.run(cmd, check=True, cwd=SRC, env=env)
    print(f"built in {time.time() - t0:.0f}s -> {APP_DIR}")


def _clean_project(path: Path):
    """Keep only layers stored inside the demo folder (drops links to files elsewhere on the build PC)."""
    proj = json.loads(path.read_text(encoding="utf-8"))
    base = path.parent.resolve()
    kept, dropped = [], []
    for lyr in proj.get("layers", []):
        p = Path(lyr["path"])
        full = (base / p).resolve() if not p.is_absolute() else p
        inside = base in full.parents and full.exists()
        (kept if inside else dropped).append(lyr)
    proj["layers"] = kept
    path.write_text(json.dumps(proj, indent=2), encoding="utf-8")
    return [l["name"] for l in dropped]


def add_payload():
    step("Adding demo data, docs and README")
    demo_dst = APP_DIR / "demo_data"
    for name in DEMOS:
        src = ROOT / "demo_data" / name / "processed"
        dst = demo_dst / name / "processed"
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
        for pj in dst.glob("*.mpm.json"):
            dropped = _clean_project(pj)
            if dropped:
                print(f"  {pj.name}: removed layers stored outside the demo folder: {', '.join(dropped)}")
    shutil.copy2(ROOT / "demo_data" / "README.md", demo_dst / "README.md")
    docs = APP_DIR / "docs"
    docs.mkdir(exist_ok=True)
    # repository docs/ (shared) and docs/python/ (Python tool) -> flat docs\ folder in the release
    for src in (ROOT / "docs" / "methods_survey.md", ROOT / "docs" / "python" / "mpm_studio.md"):
        shutil.copy2(src, docs / src.name)
    (APP_DIR / "README.txt").write_text(f"""{APP_NAME} {VERSION} - 2D mineral prospectivity mapping
================================================================

Start:   double-click "{APP_NAME}.exe", then Project > New project... (choose a project folder)
Demo:    Project > Open demo project > Carlin-type Au (Nevada) or REE (Curnamona)
         (the demo is copied into a project folder of your choice; results go to <project>\\results\\)
Guide:   docs\\mpm_studio.md      Methods: docs\\methods_survey.md

Implemented: Weights of Evidence (data-driven) with ROC/AUC, success-rate and
prediction-area validation; data tools (convert, reclassify, align, distance).

Requirements: Windows 10/11 64-bit. No Python installation needed.
Keep the folder structure: "{APP_NAME}.exe" needs the "_internal" folder beside it.
Command line: "{APP_NAME}.exe" [project.mpm.json] | --version | --selftest report.txt

Demo data: USGS Open-File Report 01-221 (public domain); DEEP-SEAM v1.0 / Geological
Survey of South Australia (Zenodo 10.5281/zenodo.17098677) - see demo_data\\README.md
""", encoding="utf-8")


def selftest():
    step("Self-test of the built executable (headless)")
    exe = APP_DIR / f"{APP_NAME}.exe"
    report = WORK / "selftest_report.txt"
    report.parent.mkdir(parents=True, exist_ok=True)
    if report.exists():
        report.unlink()
    env = dict(os.environ, QT_QPA_PLATFORM="offscreen",
               QT_QPA_FONTDIR=os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts"))
    for k in ("PYTHONPATH", "PYTHONHOME"):
        env.pop(k, None)
    t0 = time.time()
    proc = subprocess.Popen([str(exe), "--selftest", str(report)], env=env, cwd=APP_DIR)
    try:
        code = proc.wait(timeout=600)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
        code = "timeout (killed after 600 s)"
    text = report.read_text(encoding="utf-8") if report.exists() else "(no report written)"
    print(text)
    print(f"exit code {code}, {time.time() - t0:.0f}s")
    log = Path(os.environ.get("LOCALAPPDATA", "")) / "MPM Studio" / "mpm_studio.log"
    if code != 0 or "SELFTEST OK" not in text:
        if log.exists():
            print("--- tail of", log)
            print("".join(log.read_text(encoding="utf-8", errors="replace").splitlines(True)[-30:]))
        raise SystemExit("SELF-TEST FAILED")


def make_zip():
    step("Creating release zip")
    RELEASE.mkdir(exist_ok=True)
    base = RELEASE / f"{APP_NAME.replace(' ', '_')}_{VERSION}_win64"
    zip_path = shutil.make_archive(str(base), "zip", root_dir=DIST, base_dir=APP_NAME)
    size = Path(zip_path).stat().st_size / 1e6
    print(f"{zip_path}  ({size:.0f} MB)")
    return zip_path


def folder_size(p: Path) -> float:
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / 1e6


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--skip-build", action="store_true", help="reuse dist/ (only payload, test, zip)")
    ap.add_argument("--no-zip", action="store_true")
    ap.add_argument("--keep-work", action="store_true", help="keep the build/ work folder")
    a = ap.parse_args()
    if not a.skip_build:
        build()
    add_payload()
    selftest()
    print(f"\napplication folder: {APP_DIR}  ({folder_size(APP_DIR):.0f} MB)")
    if not a.no_zip:
        make_zip()
    if not a.keep_work and WORK.exists():
        shutil.rmtree(WORK)
    print("\nRELEASE OK")


if __name__ == "__main__":
    main()
