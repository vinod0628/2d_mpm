# PyInstaller runtime hook: runs before main.py inside the frozen application.
import os

# geopandas 0.11 must use Shapely (pygeos is not bundled)
os.environ.setdefault("USE_PYGEOS", "0")
