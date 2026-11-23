"""Evidential belief functions plug-in: registers the method (dialog -> run in worker -> result display)."""
from __future__ import annotations

import numpy as np

from core.methods import MethodResult, MPMMethod, OutputRaster, register
from evaluation.metrics import evaluate, negative_cells
from i18n import N_, tr
from logic.common.training import deposit_point_outputs, split_deposits, thin_deposits
from utils.raster_tools import align_to_grid, points_to_cells

from .model import EBFEvidence, EBFModel

SCORE_NAMES = {"bel": N_("EBF belief Bel ({names})"), "pls": N_("EBF plausibility Pls ({names})"),
               "mid": N_("EBF mid-point (Bel + Pls) / 2 ({names})")}
OTHER_MAPS = {"bel": (N_("EBF belief Bel — lower bound of the support for deposits"), "inferno"),
              "dis": (N_("EBF disbelief Dis — support against deposits"), "viridis"),
              "unc": (N_("EBF uncertainty Unc = 1 − Bel − Dis (ignorance)"), "cividis"),
              "pls": (N_("EBF plausibility Pls = Bel + Unc — upper bound"), "magma")}


def _safe(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in name)


@register
class EBFMethod(MPMMethod):
    key = "ebf"
    name = N_("Evidential Belief Functions")
    category = N_("Data-driven")
    group = N_("Statistical")
    description = N_("Dempster–Shafer evidential belief functions: belief, disbelief, uncertainty and plausibility "
                     "of every evidence class from the deposits, combined with Dempster's rule.")
    icon = "mdi6.set-center"

    def create_dialog(self, parent, store):
        from .dialogs import EBFDialog
        return EBFDialog(store, parent)

    def run(self, params: dict, progress) -> MethodResult:
        evs = params["evidence"]
        grid = evs[0]["grid"]
        progress(5, tr("preparing evidence"))
        evidence = []
        for e in evs:
            data = e["data"]
            if not e["grid"].same_as(grid):
                if not params.get("auto_align", True):
                    raise ValueError(tr("{layer} is not on the grid of {ref}", layer=e["name"], ref=evs[0]["name"]))
                data = align_to_grid(data, e["grid"], grid, "nearest" if e.get("classing", "values") == "values"
                                     else "bilinear")
            evidence.append(EBFEvidence(e["name"], data, e.get("classing", "values"), int(e.get("n_classes", 10)),
                                        labels=e.get("labels") or None))
        mask = None
        if params.get("mask") is not None:
            m, mg = params["mask"]
            if not mg.same_as(grid):
                m = align_to_grid(m, mg, grid, "nearest")
            mask = ~np.ma.getmaskarray(m)
        progress(15, tr("locating deposits"))
        valid_any = mask if mask is not None else np.logical_or.reduce(
            [~np.ma.getmaskarray(np.ma.asarray(e.array)) for e in evidence])
        rows, cols = points_to_cells(params["deposits"], grid, valid_mask=valid_any)
        rows, cols, thin_msg = thin_deposits(rows, cols, params.get("thin_distance", 0.0), grid.res)
        if len(rows) < 2:
            raise ValueError(tr("Fewer than 2 deposits fall inside the study area / evidence grid."))
        tr_r, tr_c, te_r, te_c = split_deposits(rows, cols, params.get("test_fraction", 0.0), params.get("seed", 0),
                                                params.get("split_method", "spatial"))
        progress(30, tr("computing belief functions"))
        score = params.get("score", "bel")
        model = EBFModel(missing=params.get("missing", "vacuous"), zero_deposit=params.get("zero_deposit", "zero"),
                         score=score)
        res = model.fit_predict(evidence, tr_r, tr_c, mask)
        progress(80, tr("validating"))
        neg = negative_cells(params.get("negatives"), grid, res.study_mask)
        ev = evaluate(res.score, tr_r, tr_c, te_r, te_c, valid_mask=res.study_mask, score_name=score, negatives=neg)
        names = ", ".join(e.name for e in evidence)
        maps = dict(bel=res.bel, dis=res.dis, unc=res.unc, pls=res.pls)
        main = score if score != "mid" else "mid"
        outs = [OutputRaster(main, res.score.astype("float32"), grid, cmap="inferno",
                             description=tr(SCORE_NAMES[score], names=names))]
        if params.get("out_maps", True):
            for k, (desc, cmap) in OTHER_MAPS.items():
                if k != main:
                    outs.append(OutputRaster(k, maps[k].astype("float32"), grid, cmap=cmap, description=tr(desc),
                                             visible=False))
        if params.get("out_layers", False):
            for name, lm in res.layer_maps.items():
                outs.append(OutputRaster(f"bel_{_safe(name)}", lm["bel"].astype("float32"), grid, cmap="inferno",
                                         description=tr("EBF belief of {layer}", layer=name), visible=False))
        progress(100, tr("done"))
        vectors = deposit_point_outputs(params["deposits"], grid, tr_r, tr_c, te_r, te_c)
        tables = {"summary": res.summary, "layers": res.layer_table}
        tabs = [("summary", N_("Summary"), N_("<b>Model</b> — evidence, deposits, options and the mean belief "
                                             "functions of the integrated map.")),
                ("layers", N_("Layers"), N_("<b>Layers</b> — classes, largest class belief and belief contrast, the "
                                           "most favourable class, mean uncertainty, the AUC of the layer's belief "
                                           "alone and the mean conflict 1 − β when the layer is combined."))]
        for lw in res.layers:
            key = f"classes_{_safe(lw.name)}"[:31]
            tables[key] = lw.table
            tabs.append((key, lw.name, N_("<b>Classes</b> — cells, training deposits, W_CD (= exp W+) and W_CD̄ "
                                          "(= exp W−), belief Bel, disbelief Dis, uncertainty Unc and plausibility "
                                          "Pls; favourable = W_CD > 1 (more deposits than expected by area).")))
        tables["pairwise"] = res.pairwise
        tabs.append(("pairwise", N_("Independence"), N_("<b>Pairwise χ² tests</b> on deposit cells (favourable / "
                                                         "unfavourable classes); p < 0.05 → the two layers are not "
                                                         "independent, which Dempster's rule assumes.")))
        st = res.stats
        cards = [(f"{res.n_deposits}", N_("training deposits (cells)")), (f"{len(res.layers)}", N_("evidence layers")),
                 (f"{st['mean_unc']:.3f}", N_("mean uncertainty Unc")),
                 (f"{st['deposit_mean_bel']:.3f}", N_("mean Bel at training deposits"))]
        return MethodResult(self.key, outs, vectors=vectors, label=params.get("prefix", "ebf"), tables=tables,
                            evaluation=ev, messages=thin_msg + res.messages +
                            [f"{l.name}: {m}" for l in res.layers for m in l.messages],
                            extra=dict(ebf=res, deposits_name=params.get("deposits_name"), cards=cards, tabs=tabs,
                                       train_cells=(tr_r, tr_c), test_cells=(te_r, te_c)))

    def show_result(self, window, result: MethodResult):
        for out in reversed(result.outputs):
            window.add_output_raster(out)
        for vec in result.vectors:
            window.add_output_vector(vec)
        from .dialogs import EBFResultsDialog
        dlg = EBFResultsDialog(result, window, window)
        dlg.show()
        window.keep_dialog(dlg)
