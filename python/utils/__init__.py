"""Data-preparation utilities for 2D MPM (raster/vector conversion, reclassification, gridding)."""
import os as _os

# pysal pulls in pygeos, whose GEOS build differs from Shapely's; geopandas 0.11 would prefer it.
_os.environ.setdefault("USE_PYGEOS", "0")
