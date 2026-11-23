"""Demo experiments of KT-RK (knowledge-trend regression-kriging) for docs/python/tools/kt_rk.docx.

KT-RK estimates a continuous target, so it has its own protocol (the classification tools use experiments.py):
candidate settings are compared by the cross-validated RMSE of ln(y + c) that the tool reports for its selected
(α, s); the selected setting is checked on a 20 % hold-out of the sample cells (model on 80 %) and against the known
deposits (ROC of the predicted map). Results -> results/kt_rk_<dataset>.json, figures/kt_rk_<dataset>_*.png.

    .venv\\Scripts\\python.exe docs\\python\\_build\\experiments_ktrk.py [--dataset carlin] [--quick]

Runs one model at a time (the Carlin kriging of 909 545 cells takes ≈ 9 min and ≈ 0.6 GB).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import experiments as E  # noqa: E402  (sets up paths, matplotlib, the method registry)

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import geopandas as gpd  # noqa: E402
from scipy.stats import pearsonr, spearmanr  # noqa: E402

from core.methods import REGISTRY  # noqa: E402
from logic.hybrid.kt_rk.plugin import sample_cells  # noqa: E402

TIE = 0.005          # CV RMSE differences below this are treated as ties (earliest candidate wins)

# knowledge tables: (layer, process, direction or class scores)
KNOW = {
    "carlin": {
        "host + faults + As": [("geology_host", "trap", "2=0.9; 1=0.1"), ("dist_faults", "pathway", -1),
                               ("as_log_idw", "deposition", 1)],
        "host + faults": [("geology_host", "trap", "2=0.9; 1=0.1"), ("dist_faults", "pathway", -1)],
    },
    "curnamona": {
        "Th + U + granite + silica": [("rad_th", "source", 1), ("rad_u", "source", 1), ("dist_granite", "source", -1),
                                      ("aster_silica_index", "footprint", -1)],
        "… + Nb": [("rad_th", "source", 1), ("rad_u", "source", 1), ("dist_granite", "source", -1),
                   ("aster_silica_index", "footprint", -1), ("geochem_nb_log_idw", "deposition", 1)],
    },
}
TARGET = {
    "carlin": dict(layer="nure_geochem", field="NAA_SB", element="Sb", unit="ppm",
                   description="NURE stream-sediment samples, Sb by neutron activation (ppm)", zero_is_missing=True),
    "curnamona": dict(layer="geochem_samples", field="La", element="La", unit="ppm",
                      description="Surface rock / soil samples, La (ppm)", zero_is_missing=False),
}
CANDIDATES = {
    "carlin": [
        dict(label="host + faults + As, strike 13° (fault grain), 15 × 5 km (default)", know="host + faults + As",
             strike=13, along=15000, across=5000),
        dict(label="host + faults + As, strike 325° (Carlin trend), 15 × 5 km", know="host + faults + As",
             strike=325, along=15000, across=5000),
        dict(label="host + faults + As, isotropic 10 km", know="host + faults + As", strike=0, along=10000,
             across=10000),
        dict(label="host + faults + As, strike 13°, 7.5 × 2.5 km", know="host + faults + As", strike=13,
             along=7500, across=2500),
        dict(label="host + faults, strike 13°, 15 × 5 km", know="host + faults", strike=13, along=15000,
             across=5000),
    ],
    "curnamona": [
        dict(label="Th + U + granite + silica, strike 70° (fault grain), 4 × 1.5 km (default)",
             know="Th + U + granite + silica", strike=70, along=4000, across=1500),
        dict(label="Th + U + granite + silica, isotropic 3 km", know="Th + U + granite + silica", strike=0,
             along=3000, across=3000),
        dict(label="Th + U + granite + silica, strike 70°, 8 × 3 km", know="Th + U + granite + silica", strike=70,
             along=8000, across=3000),
        dict(label="Th + U + granite + silica, strike 160° (across the grain), 4 × 1.5 km",
             know="Th + U + granite + silica", strike=160, along=4000, across=1500),
        dict(label="… + Nb, strike 70°, 4 × 1.5 km", know="… + Nb", strike=70, along=4000, across=1500),
    ],
}


def samples(d: E.Demo, key):
    t = TARGET[key]
    g = gpd.read_file(os.path.join(d.folder, "vectors", t["layer"] + ".gpkg"))
    v = pd.to_numeric(g[t["field"]], errors="coerce").astype(float)
    n0 = int((v == 0).sum())
    if t["zero_is_missing"]:
        v = v.where(v > 0)                      # NURE: 0 = below detection / not analysed -> NoData
    g = g.assign(**{t["field"]: v})
    return g, n0


def params(d, key, cfg, smp, deposits=None):
    ev = []
    for n, proc, dirn in KNOW[key][cfg["know"]]:
        data, grid, _ = d.raster(n)
        cat = isinstance(dirn, str)
        ev.append(dict(name=n, data=data, grid=grid, kind="categorical" if cat else "continuous", process=proc,
                       direction=1 if cat else dirn, scores=dirn if cat else None, weight=1.0))
    p = E.base_params(d, 0.0, 42, deposits)
    p.update(evidence=ev, process_weights={}, samples=smp, field=TARGET[key]["field"], strike=cfg["strike"],
             range_along=cfg["along"], range_across=cfg["across"], offset=0.0, folds=10, max_gp_fit=1500,
             seed=42, prefix="ktrk")
    return p


def run(d, key, cfg, smp, log):
    t0 = time.time()
    res = REGISTRY["kt_rk"].run(params(d, key, cfg, smp), lambda *a: None)
    dt = time.time() - t0
    st = res.extra["ktrk"].stats
    log(f"      CV RMSE {st['cv_rmse']:.4f}  α {st['alpha']:g}  s {st['shrink']:g}  {dt:.0f} s")
    return res, dt


def cell_split(smp, field, grid, study, cells, seed=42, test=0.2):
    """80 / 20 split of the sampled cells (all samples of a cell stay together)."""
    pos, vals, _ = sample_cells(smp, field, grid, study, cells)
    rng = np.random.default_rng(seed)
    te = np.zeros(pos.size, bool)
    te[rng.choice(pos.size, int(round(test * pos.size)), replace=False)] = True
    flat_te = set(cells[pos[te]].tolist())
    g = smp.to_crs(grid.crs) if (smp.crs is not None and grid.crs is not None and smp.crs != grid.crs) else smp
    rows, cols = grid.xy_to_rowcol(np.array([p.centroid.x for p in g.geometry]),
                                   np.array([p.centroid.y for p in g.geometry]))
    flat = rows * grid.width + cols
    in_te = np.array([f in flat_te for f in flat])
    return smp[~in_te], pos[te], vals[te]


def fig_map(res, d, smp, path, title):
    out = next(o for o in res.outputs if o.name == "log_prediction")
    g = out.grid
    fig, ax = plt.subplots(figsize=(6.4, 6.0), dpi=200)
    im = ax.imshow(np.ma.asarray(out.data), extent=g.extent, cmap="inferno", interpolation="nearest")
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label("ln(estimate + c)", fontsize=8)
    s = smp.to_crs(g.crs) if (smp.crs is not None and g.crs is not None) else smp
    ax.scatter([p.x for p in s.geometry], [p.y for p in s.geometry], s=1.2, c="#9ad0ff", alpha=0.6, lw=0,
               label=f"samples ({len(s)})")
    dep = d.deposits.to_crs(g.crs) if (d.deposits.crs is not None and g.crs is not None) else d.deposits
    ax.scatter([p.x for p in dep.geometry], [p.y for p in dep.geometry], s=30, marker="^", c="#43a047",
               edgecolors="white", linewidths=0.6, label=f"deposits ({len(dep)})")
    ax.set_title(title, fontsize=9)
    ax.tick_params(labelsize=6)
    ax.ticklabel_format(style="plain", useOffset=False)
    ax.legend(loc="lower left", fontsize=7, framealpha=0.85)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_panels(res, hold, path):
    r = res.extra["ktrk"]
    fig, axs = plt.subplots(2, 2, figsize=(7.2, 6.4), dpi=200)
    ax = axs[0, 0]
    cv = r.cv
    al, sh = sorted(cv.alpha.unique()), sorted(cv.shrink.unique())
    M = np.array([[float(cv[(cv.alpha == a) & (cv.shrink == s)].cv_rmse.iloc[0]) for s in sh] for a in al])
    im = ax.imshow(M, cmap="viridis_r", origin="lower")
    for i in range(len(al)):
        for j in range(len(sh)):
            sel = bool(cv[(cv.alpha == al[i]) & (cv.shrink == sh[j])].selected.iloc[0])
            ax.text(j, i, f"{M[i, j]:.3f}" + ("*" if sel else ""), ha="center", va="center", fontsize=7,
                    color="white" if M[i, j] > M.mean() else "black", fontweight="bold" if sel else None)
    ax.set_xticks(range(len(sh)))
    ax.set_xticklabels([f"{s:g}" for s in sh])
    ax.set_yticks(range(len(al)))
    ax.set_yticklabels([f"{a:g}" for a in al])
    ax.set(xlabel="forest weight s", ylabel="knowledge-trend weight α", title="(a) CV RMSE of ln(y + c)")
    ax = axs[0, 1]
    ax.scatter(hold["obs"], hold["pred"], s=6, c="#1f77b4", alpha=0.6, lw=0)
    lo, hi = min(np.min(hold["obs"]), np.min(hold["pred"])), max(np.max(hold["obs"]), np.max(hold["pred"]))
    ax.plot([lo, hi], [lo, hi], c="grey", ls="--", lw=0.8)
    ax.set(xlabel="observed ln(y + c)", ylabel="estimated ln(y + c)",
           title=f"(b) 20 % hold-out: r = {hold['pearson']:.2f}")
    for ax, name, cmap, ttl in ((axs[1, 0], "log_sd", "viridis", "(c) kriging s.d. (log scale)"),
                                (axs[1, 1], None, "RdBu_r", "(d) local residual = estimate − trend")):
        if name:
            o = next(o for o in res.outputs if o.name == name)
            arr = np.ma.asarray(o.data)
        else:
            lp = next(o for o in res.outputs if o.name == "log_prediction")
            tr_ = next(o for o in res.outputs if o.name == "trend")
            arr = np.ma.asarray(lp.data) - np.ma.asarray(tr_.data)
            o = lp
        v = np.ma.compressed(arr)
        lim = np.percentile(np.abs(v), 98) if name is None else None
        im = ax.imshow(arr, extent=o.grid.extent, cmap=cmap, interpolation="nearest",
                       vmin=-lim if lim else None, vmax=lim if lim else None)
        fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02).ax.tick_params(labelsize=6)
        ax.set_title(ttl)
        ax.set_xticks([])
        ax.set_yticks([])
    for a in axs.ravel():
        a.tick_params(labelsize=7)
        a.title.set_fontsize(9)
        a.xaxis.label.set_fontsize(8)
        a.yaxis.label.set_fontsize(8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def experiment(key, quick=False, log=print):
    d = E.Demo(key)
    smp, n_zero = samples(d, key)
    t = TARGET[key]
    cands = CANDIDATES[key][:2] if quick else CANDIDATES[key]
    t0 = time.time()
    search, runs = [], {}
    cache_dir = os.path.join(E.OUT, "_cache")
    os.makedirs(cache_dir, exist_ok=True)
    for i, cfg in enumerate(cands):
        log(f"  [kt_rk/{key}] candidate: {cfg['label']}")
        cache = os.path.join(cache_dir, f"kt_rk_{key}_{i}.json")
        if os.path.exists(cache):                      # finished in an earlier, interrupted run
            with open(cache, encoding="utf-8") as fh:
                row = json.load(fh)
            if row.get("label") == cfg["label"]:
                log(f"      (cached) CV RMSE {row['cv_rmse']:.4f}")
                search.append(row)
                continue
        res, dt = run(d, key, cfg, smp, log)
        r = res.extra["ktrk"]
        c = res.evaluation.curves["training"] if res.evaluation is not None else None
        search.append(dict(label=cfg["label"], cfg=cfg, cv_rmse=r.stats["cv_rmse"],
                           cv_rmse_sd=float(r.cv[r.cv.selected].cv_rmse_sd.iloc[0]), alpha=r.alpha, shrink=r.shrink,
                           cv_rmse_kriging_only=r.stats["cv_rmse_mean_only"], corr_k_z=r.stats["corr_k_z"],
                           sill=r.stats["sill"], length_scale=r.stats["length_scale"], nugget=r.stats["nugget"],
                           deposit_auc=c.auc if c is not None else None, seconds=dt))
        runs[cfg["label"]] = res
        with open(cache, "w", encoding="utf-8") as fh:
            json.dump(E.clean(search[-1]), fh, ensure_ascii=False)
    top = min(s["cv_rmse"] for s in search)
    best = next(s for s in search if s["cv_rmse"] <= top + TIE)
    cfg = best["cfg"]
    log(f"  [kt_rk/{key}] selected: {cfg['label']}")
    if cfg["label"] in runs:
        ref = runs[cfg["label"]]
    else:                                              # selected candidate came from the cache: run it again
        log(f"  [kt_rk/{key}] reference run of the selected setting")
        ref, _ = run(d, key, cfg, smp, log)
    for k in list(runs):
        if k != cfg["label"]:
            del runs[k]
    # 20 % hold-out of the sampled cells
    log(f"  [kt_rk/{key}] 20 % hold-out")
    m, mg = d.mask()
    study = ~np.ma.getmaskarray(m)
    cells = np.flatnonzero(study)
    train_smp, te_pos, te_vals = cell_split(smp, t["field"], mg, study, cells)
    hres, hdt = run(d, key, cfg, train_smp, log)
    hr = hres.extra["ktrk"]
    c_ = hr.stats["c"]
    obs = np.log(te_vals + c_)
    lp = np.ma.asarray(next(o for o in hres.outputs if o.name == "log_prediction").data)
    trd = np.ma.asarray(next(o for o in hres.outputs if o.name == "trend").data)
    flat = cells[te_pos]
    pred = np.ma.getdata(lp).ravel()[flat]
    trend = np.ma.getdata(trd).ravel()[flat]
    mean_tr = float(np.mean(np.log(hr.observations.value.to_numpy() + c_)))
    rmse = lambda a: float(np.sqrt(np.mean((a - obs) ** 2)))  # noqa: E731
    hold = dict(train_cells=int(hr.stats["observations"]), test_cells=int(te_pos.size), rmse_log=rmse(pred),
                rmse_trend=rmse(trend), rmse_mean=rmse(np.full(obs.size, mean_tr)),
                r2=float(1 - np.sum((pred - obs) ** 2) / np.sum((obs - obs.mean()) ** 2)),
                pearson=float(pearsonr(obs, pred)[0]), spearman=float(spearmanr(obs, pred)[0]),
                spearman_value=float(spearmanr(te_vals, np.exp(pred) - c_)[0]), seconds=hdt,
                obs=obs.tolist(), pred=pred.tolist())
    log(f"      hold-out RMSE {hold['rmse_log']:.4f} (trend {hold['rmse_trend']:.4f}, mean {hold['rmse_mean']:.4f}), "
        f"r {hold['pearson']:.3f}")
    tag = f"kt_rk_{key}"
    fig_map(ref, d, smp, os.path.join(E.FIG, f"{tag}_map.png"),
            f"KT-RK estimate of {t['element']} — {E.DATASETS[key]['title']}")
    fig_panels(ref, hold, os.path.join(E.FIG, f"{tag}_panels.png"))
    r = ref.extra["ktrk"]
    vals = r.observations.value.to_numpy()
    raw = pd.to_numeric(smp[t["field"]], errors="coerce").to_numpy(float)
    xs, ys = mg.cell_centers()
    area_km2 = study.sum() * mg.cell_area / 1e6
    out = dict(method="kt_rk", dataset=key, dataset_info=d.describe(),
               target=dict(t, samples=int(len(smp)), samples_valid=int(np.isfinite(raw).sum()), zeros=n_zero,
                           cells=int(r.stats["observations"]), min=float(np.min(vals)), median=float(np.median(vals)),
                           p95=float(np.percentile(vals, 95)), max=float(np.max(vals)),
                           spacing_km=float(np.sqrt(area_km2 / r.stats["observations"]))),
               knowledge={k: [dict(layer=a, process=b, direction=(c if isinstance(c, str) else ("+" if c > 0 else "−")))
                              for a, b, c in v] for k, v in KNOW[key].items()},
               evidence=d.layer_info([(n, "categorical" if isinstance(x, str) else "continuous")
                                      for n, _, x in KNOW[key][cfg["know"]]]),
               protocol="regression", search=search, selected=cfg,
               reference=dict(seconds=best["seconds"], summary=r.summary.to_dict("records"),
                              cv=r.cv.to_dict("records"), stats=r.stats, messages=ref.messages,
                              curves={k: E.curve_stats(c) for k, c in ref.evaluation.curves.items()}
                              if ref.evaluation is not None else {}),
               holdout=hold, figures=dict(map=f"{tag}_map.png", panels=f"{tag}_panels.png"),
               runtime_s=time.time() - t0)
    with open(os.path.join(E.OUT, f"{tag}.json"), "w", encoding="utf-8") as fh:
        json.dump(E.clean(out), fh, ensure_ascii=False, indent=1)
    log(f"  [kt_rk/{key}] done in {time.time() - t0:.0f} s")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", action="append", choices=list(E.DATASETS))
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args(argv)
    os.makedirs(E.OUT, exist_ok=True)
    os.makedirs(E.FIG, exist_ok=True)
    for key in a.dataset or ["curnamona", "carlin"]:
        print(f"=== kt_rk / {key}", flush=True)
        experiment(key, a.quick, log=lambda s: print(s, flush=True))
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
