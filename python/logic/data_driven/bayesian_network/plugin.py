"""Bayesian network plug-in: registers the method (dialog -> run in worker -> result display)."""
from __future__ import annotations

import numpy as np

from core.methods import MethodResult, MPMMethod, OutputRaster, register
from evaluation.metrics import evaluate, negative_cells
from i18n import N_, tr
from logic.common.features import Feature
from logic.common.training import deposit_point_outputs, split_deposits, thin_deposits
from utils.raster_tools import align_to_grid, points_to_cells

from .model import BayesianNetworkModel


@register
class BayesianNetworkMethod(MPMMethod):
    key = "bayesian_network"
    name = N_("Bayesian Network")
    category = N_("Data-driven")
    group = N_("Statistical")
    description = N_("Bayesian network classifiers (Porwal et al. 2006): naive Bayes or tree-augmented naive Bayes "
                     "(TAN) that learns the dependencies between evidence layers; posterior probability with "
                     "Dirichlet uncertainty, conditional probability tables and the network structure.")
    icon = "mdi6.graph-outline"

    def create_dialog(self, parent, store):
        from .dialogs import BayesianNetworkDialog
        return BayesianNetworkDialog(store, parent)

    def run(self, params: dict, progress) -> MethodResult:
        evs = params["evidence"]
        grid = evs[0]["grid"]
        progress(3, tr("preparing evidence"))
        feats = []
        for e in evs:
            data = e["data"]
            if not e["grid"].same_as(grid):
                if not params.get("auto_align", True):
                    raise ValueError(tr("{layer} is not on the grid of {ref}", layer=e["name"], ref=evs[0]["name"]))
                data = align_to_grid(data, e["grid"], grid, "nearest" if e["kind"] == "categorical" else "bilinear")
            feats.append(Feature(e["name"], data, e["kind"], labels=e.get("labels") or None))
        mask = None
        if params.get("mask") is not None:
            m, mg = params["mask"]
            if not mg.same_as(grid):
                m = align_to_grid(m, mg, grid, "nearest")
            mask = ~np.ma.getmaskarray(m)
        progress(8, tr("locating deposits"))
        valid_any = mask if mask is not None else np.logical_or.reduce(
            [~np.ma.getmaskarray(np.ma.asarray(f.array)) for f in feats])
        rows, cols = points_to_cells(params["deposits"], grid, valid_mask=valid_any)
        all_rows, all_cols = rows, cols
        rows, cols, thin_msg = thin_deposits(rows, cols, params.get("thin_distance", 0.0), grid.res)
        if len(rows) < 2:
            raise ValueError(tr("Fewer than 2 deposits fall inside the study area / evidence grid."))
        tr_r, tr_c, te_r, te_c = split_deposits(rows, cols, params.get("test_fraction", 0.0), params.get("seed", 0),
                                                params.get("split_method", "spatial"))
        cell_km2 = abs(grid.res[0] * grid.res[1]) / 1e6
        model = BayesianNetworkModel(
            structure=params.get("structure", "tan"), n_classes=params.get("n_classes", 5),
            alpha=params.get("alpha", 1.0), design=params.get("design", "all"), missing=params.get("missing", "class"),
            draws=params.get("draws", 50), unit_cell_km2=params.get("unit_cell_km2", 1.0), cell_area_km2=cell_km2,
            neg_ratio=params.get("neg_ratio", 5.0), min_distance=params.get("min_distance", 0.0),
            min_spacing=params.get("min_spacing", -1.0), res=grid.res, seed=params.get("seed", 0))
        res = model.fit_predict(feats, tr_r, tr_c, mask, known_deposits=(all_rows, all_cols),
                                progress=lambda f, msg: progress(10 + int(80 * f), tr(msg)))
        progress(92, tr("validating"))
        ev = evaluate(res.posterior, tr_r, tr_c, te_r, te_c, valid_mask=res.study_mask, score_name="posterior",
                      negatives=negative_cells(params.get("negatives"), grid, res.study_mask))
        names = ", ".join(f.name for f in feats)
        what = tr("TAN") if params.get("structure", "tan") == "tan" else tr("naive Bayes")
        outs = [OutputRaster("post", res.posterior.astype("float32"), grid, cmap="inferno",
                             description=tr("Bayesian network ({what}) posterior probability per raster cell ({names})",
                                            what=what, names=names))]
        if res.unit_probability is not None and params.get("out_unit", True):
            outs.append(OutputRaster("post_unit", res.unit_probability.astype("float32"), grid, cmap="inferno",
                                     description=tr("Posterior probability per unit cell ({a:g} km²)",
                                                    a=params.get("unit_cell_km2", 1.0)), visible=False))
        if params.get("out_sd", True):
            outs.append(OutputRaster("post_sd", res.posterior_sd.astype("float32"), grid, cmap="viridis",
                                     description=tr("Std. deviation of the posterior (Dirichlet draws of the tables)"),
                                     visible=False))
        vectors = deposit_point_outputs(params["deposits"], grid, tr_r, tr_c, te_r, te_c)
        progress(100, tr("done"))
        tables = {"summary": res.summary, "network_edges": res.edges, "probability_tables": res.cpt_table,
                  "evidence_layers": res.layers, "structure_comparison": res.comparison}
        return MethodResult(self.key, outs, vectors=vectors, label=params.get("prefix", "bn"), tables=tables,
                            evaluation=ev, messages=thin_msg + res.messages,
                            extra=dict(bn=res, deposits_name=params.get("deposits_name"),
                                       train_cells=(tr_r, tr_c), test_cells=(te_r, te_c)))

    def show_result(self, window, result: MethodResult):
        for out in reversed(result.outputs):
            window.add_output_raster(out)
        for vec in result.vectors:
            window.add_output_vector(vec)
        from .dialogs import BayesianNetworkResultsDialog
        dlg = BayesianNetworkResultsDialog(result, window, window)
        dlg.show()
        window.keep_dialog(dlg)
