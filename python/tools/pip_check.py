"""`pip check` that tolerates the one known, harmless metadata conflict.

eis_toolkit 1.1.6 declares `gdal` (the GDAL Python bindings, `osgeo`) as a requirement, but no MPM code needs
it: eis_toolkit only uses `osgeo` in `utilities/gdal.py` and as an optional fast path in `distance_to_anomaly`
(which falls back to numba). GDAL itself is provided by the rasterio / fiona wheels. Exit code 0 = ok.
"""
import subprocess
import sys

KNOWN = ("eis-toolkit 1.1.6 requires gdal",)

out = subprocess.run([sys.executable, "-m", "pip", "check"], capture_output=True, text=True)
lines = [l for l in (out.stdout + out.stderr).splitlines() if l.strip()]
problems = [l for l in lines if not l.startswith(KNOWN) and "No broken requirements" not in l]
for l in lines:
    print(("  (ignored, see python/tools/pip_check.py) " if l.startswith(KNOWN) else "  ") + l)
if problems:
    print("pip check: dependency problems found")
    sys.exit(1)
print("pip check: ok")
