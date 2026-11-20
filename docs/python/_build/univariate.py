"""Single-layer AUC of every demo evidence layer (exploratory data analysis for the method documents).

AUC = probability that a deposit cell has a higher value than a background cell (ties ½); values < 0.5 mean the
deposits sit on LOW values (e.g. distances). Writes results/univariate_<dataset>.json.
"""
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")
os.environ.setdefault("USE_PYGEOS", "0")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "python"))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import geopandas as gpd  # noqa: E402
from evaluation.metrics import delong_auc  # noqa: E402
from utils.raster_tools import points_to_cells, read_raster  # noqa: E402

for key, folder in (("carlin", "nevada_carlin_au"), ("curnamona", "curnamona_ree")):
    base = os.path.join(ROOT, "demo_data", folder, "processed")
    layers = pd.read_csv(os.path.join(base, "layers.csv"))
    m, grid, _ = read_raster(os.path.join(base, "rasters", "study_mask.tif"))
    study = ~np.ma.getmaskarray(m)
    dep = gpd.read_file(os.path.join(base, "vectors", "deposits.gpkg"))
    r, c = points_to_cells(dep, grid, valid_mask=study)
    rows = []
    for _, L in layers[layers.file.str.startswith("rasters/") & (layers.name != "study_mask")].iterrows():
        a, g, _ = read_raster(os.path.join(base, L.file))
        if a.shape != study.shape:
            continue
        ok = study & ~np.ma.getmaskarray(a)
        pos_mask = np.zeros(study.shape, bool)
        pos_mask[r, c] = True
        pos = np.ma.getdata(a)[pos_mask & ok].astype(float)
        neg = np.ma.getdata(a)[ok & ~pos_mask].astype(float)
        if pos.size < 2:
            continue
        auc, se, ci = delong_auc(pos, neg)
        rows.append(dict(layer=L["name"], group=L.group, kind=L.kind, auc=auc, se=se, ci_low=ci[0], ci_high=ci[1],
                         n_dep=int(pos.size), dep_median=float(np.median(pos)), bg_median=float(np.median(neg)),
                         description=L.description))
    df = pd.DataFrame(rows)
    df["strength"] = (df.auc - 0.5).abs()
    df = df.sort_values("strength", ascending=False)
    with open(os.path.join(HERE, "results", f"univariate_{key}.json"), "w", encoding="utf-8") as fh:
        json.dump(df.to_dict("records"), fh, indent=1)
    print(f"=== {key}: {len(r)} deposit cells")
    print(df[["layer", "group", "auc", "ci_low", "ci_high", "dep_median", "bg_median"]].head(20).to_string(index=False))
