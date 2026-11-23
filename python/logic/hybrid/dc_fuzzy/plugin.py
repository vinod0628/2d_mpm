"""Data-calibrated fuzzy memberships (WofE / LR) plug-in: dialog -> run in worker -> result display."""
from __future__ import annotations

import numpy as np

from core.methods import MethodResult, MPMMethod, OutputRaster, register
from i18n import N_, tr
from logic.hybrid.common.pipeline import _safe, finish, knowledge_outputs, prepare

from .model import dc_fuzzy

TITLE = N_("Data-Calibrated Fuzzy Memberships (WofE / LR)")


@register
class DCFuzzyMethod(MPMMethod):
    key = "dc_fuzzy"
    name = TITLE
    category = N_("Hybrid")
    group = N_("Fuzzy evidence")
    description = N_("Data-driven fuzzy model: the membership of every evidence value is calibrated on the deposits "
                     "— weights of evidence per class (optionally monotone in the expert direction) or a logistic "
                     "regression per layer — and the memberships are combined by the expert inference network "
                     "(OR within a mineral-system process, AND between processes) or the fuzzy γ operator.")
    icon = "mdi6.tune-variant"

    def create_dialog(self, parent, store):
        from .dialogs import DCFuzzyDialog
        return DCFuzzyDialog(store, parent)

    def run(self, params: dict, progress) -> MethodResult:
        ctx = prepare(params, progress)
        progress(30, tr("calibrating memberships"))
        res = dc_fuzzy(ctx.net, ctx.train_pos, calibration=params.get("calibration", "wofe"),
                       scaling=params.get("scaling", "weight"), monotone=params.get("monotone", True),
                       n_classes=int(params.get("n_classes", 10)), kappa=params.get("kappa", 0.0),
                       combination=params.get("combination", "network"), gamma=params.get("gamma", 0.9))
        g, net = ctx.grid, ctx.net
        cal = tr("weights of evidence") if params.get("calibration", "wofe") == "wofe" else tr("logistic regression")
        outs = [OutputRaster("fuzzy", net.grid(res.prob).astype("float32"), g, cmap="inferno",
                             description=tr("Favourability from data-calibrated memberships ({cal})", cal=cal))]
        outs += knowledge_outputs(ctx, res.F, res.expert, with_processes=params.get("out_processes", False))
        if params.get("out_memberships", False):
            for j, l in enumerate(ctx.table.layers):
                outs.append(OutputRaster(f"mu_{_safe(l.name)}", net.grid(np.nan_to_num(res.mu[:, j], nan=0.5))
                                         .astype("float32"), g, cmap="viridis", visible=False,
                                         description=tr("Data-calibrated membership of {layer}", layer=l.name)))
        tables = dict(summary=res.summary, layers=res.layers, memberships=res.memberships)
        st = res.stats
        extra = dict(dcfuzzy=res, cards=[(f"{st['deposits']}", N_("training deposits")),
                                         (f"{st['mean_layer_auc']:.3f}", N_("mean single-layer AUC")),
                                         (f"{st['against_expert']}", N_("layers against the expert direction"))],
                     tabs=[("summary", N_("Summary"), N_("<b>Model</b> — calibration, membership scaling, pooling "
                                                         "with the expert and combination.")),
                           ("layers", N_("Layers"), N_(
                               "<b>Layers</b> — calibration, weight contrast (range of the weights used), logistic "
                               "slope ± s.e., whether the deposits agree with the expert direction, and the AUC of the "
                               "calibrated membership alone.")),
                           ("memberships", N_("Memberships"), N_(
                               "<b>Classes</b> — cells, training deposits, class weight W ± s.d., the weight used "
                               "(monotone fit / logistic regression), data, expert and final membership."))])
        return finish(self.key, ctx, params, outs, tables, res.messages, extra, progress, "dcfuzzy")

    def show_result(self, window, result: MethodResult):
        from logic.hybrid.common.dialogs import show_hybrid_result
        show_hybrid_result(window, result, tr(TITLE))
