"""M4 PT-Hurdle plug-in: permissive-tract hurdle model (dialog -> run in worker -> result display)."""
from __future__ import annotations

import numpy as np

from core.methods import MethodResult, MPMMethod, OutputRaster, register
from i18n import N_, tr
from logic.hybrid.common.pipeline import finish, knowledge_outputs, prepare
from utils.raster_tools import align_to_grid

from .model import permissive_mask, pt_hurdle

TITLE = N_("Permissive-Tract Hurdle Model (PT-Hurdle)")


@register
class PTHurdleMethod(MPMMethod):
    key = "pt_hurdle"
    name = TITLE
    category = N_("Hybrid")
    group = N_("Knowledge-constrained learning")
    description = N_("Necessary criterion as a hurdle: a gate that is 1 inside the permissive area (host unit, "
                     "basin) and falls off over the map's positional error, times a conditional model (KP-RF, "
                     "BS-WofE or RF) trained only inside the permissive tract.")
    icon = "mdi6.gate"

    def create_dialog(self, parent, store):
        from .dialogs import PTHurdleDialog
        return PTHurdleDialog(store, parent)

    def run(self, params: dict, progress) -> MethodResult:
        ctx = prepare(params, progress)
        gp = params["gate"]
        data = gp["data"]
        if not gp["grid"].same_as(ctx.grid):
            data = align_to_grid(data, gp["grid"], ctx.grid, "nearest")
        host = permissive_mask(data, gp["rule"], gp["values"])
        if not host.any():
            raise ValueError(tr("No cell satisfies the permissive criterion."))
        res = pt_hurdle(ctx.net, ctx.train_pos, ctx.neg_cand, host, d0=params.get("d0", 2.0),
                        s_gate=params.get("s_gate", 1.0), g0=params.get("g0", 0.2),
                        engine=params.get("engine", "kp_rf"), smooth=params.get("smooth", 0.0),
                        n_bags=int(params.get("n_bags", 10)), ratio=params.get("neg_ratio", 3.0),
                        kappa=params.get("kappa_pool", 20.0), seed=params.get("seed", 0),
                        progress=lambda f, msg: progress(8 + int(84 * f), tr(msg)))
        g, net = ctx.grid, ctx.net
        outs = [OutputRaster("prob", net.grid(res.prob).astype("float32"), g, cmap="inferno",
                             description=tr("PT-Hurdle probability G · P^c (criterion: {layer})", layer=gp["name"])),
                OutputRaster("gate", net.grid(res.gate).astype("float32"), g, cmap="viridis", visible=False,
                             description=tr("Permissive gate G (1 inside the permissive area)")),
                OutputRaster("conditional", net.grid(res.cond).astype("float32"), g, cmap="inferno", visible=False,
                             description=tr("Conditional model P^c (trained inside the tract)")),
                OutputRaster("tract", net.grid(res.tract.astype("uint8"), "uint8"), g, categorical=True,
                             cmap="tab10", visible=False, description=tr("Permissive tract (G > g0) = 1"))]
        outs += knowledge_outputs(ctx, res.F, res.expert, with_processes=params.get("out_processes", False))
        tables = dict(summary=res.summary)
        inner = res.inner
        for k in ("importance", "bags", "weights", "layers"):
            df = getattr(inner, k, None)
            if df is not None:
                tables[k] = df
        tables["conditional_summary"] = inner.summary
        st = res.stats
        extra = dict(pthurdle=res, cards=[(f"{st['deposits_in_tract']} / {st['deposits']}",
                                           N_("deposits inside the tract / all")),
                                          (f"{st['tract_pct']:.1f} %", N_("permissive tract (area)"))],
                     tabs=[("summary", N_("Summary"), N_("<b>Hurdle</b> — gate, permissive tract and conditional "
                                                         "model.")),
                           ("conditional_summary", N_("Conditional model"), N_("<b>Conditional model</b> trained "
                                                                               "inside the tract.")),
                           ("importance", N_("Importance"), N_("<b>Importance</b> inside the tract (forest).")),
                           ("weights", N_("Weights"), N_("<b>Weight table</b> inside the tract (BS-WofE).")),
                           ("bags", N_("Forests"), N_("<b>Forests</b> of the conditional model."))])
        return finish(self.key, ctx, params, outs, tables, res.messages, extra, progress, "pthurdle")

    def show_result(self, window, result: MethodResult):
        from logic.hybrid.common.dialogs import show_hybrid_result
        show_hybrid_result(window, result, tr(TITLE))
