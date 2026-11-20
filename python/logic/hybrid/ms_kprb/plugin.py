"""MS-KPRB plug-in: mineral-system knowledge prior + residual boosting (dialog -> run in worker -> result display)."""
from __future__ import annotations

from core.methods import MethodResult, MPMMethod, OutputRaster, register
from i18n import N_, tr
from logic.hybrid.common.pipeline import finish, knowledge_outputs, prepare

from .model import KPRBConfig, fit_kprb

TITLE = N_("Knowledge-Prior Residual Boosting (MS-KPRB)")


@register
class KPRBMethod(MPMMethod):
    key = "ms_kprb"
    name = TITLE
    category = N_("Hybrid")
    group = N_("Knowledge prior + data correction")
    description = N_("Mineral-system knowledge network (OR within processes, AND across processes) calibrated with "
                     "expert priors, plus a logistic boosting of the residuals whose size is chosen by spatial "
                     "cross-validation (0 trees = expert model kept); ensemble over random non-deposit samples.")
    icon = "mdi6.family-tree"

    def create_dialog(self, parent, store):
        from .dialogs import KPRBDialog
        return KPRBDialog(store, parent)

    def run(self, params: dict, progress) -> MethodResult:
        ctx = prepare(params, progress)
        cfg = KPRBConfig(kappa=params.get("kappa", 10.0), tau=params.get("tau", 0.75),
                         learning_rate=params.get("learning_rate", 0.05), depth=int(params.get("depth", 2)),
                         min_leaf=int(params.get("min_leaf", 5)), neg_ratio=params.get("neg_ratio", 3.0),
                         draws=int(params.get("draws", 20)), folds=int(params.get("folds", 4)),
                         guided=params.get("guided", 0.0), prior_correction=params.get("prior_correction", False),
                         seed=params.get("seed", 0))
        res = fit_kprb(ctx.net, ctx.train_pos, ctx.neg_cand, cfg,
                       progress=lambda f, msg: progress(8 + int(84 * f), tr(msg)))
        g, net = ctx.grid, ctx.net
        outs = [OutputRaster("prob", net.grid(res.prob).astype("float32"), g, cmap="inferno",
                             description=tr("MS-KPRB probability (knowledge prior + residual correction; mean of "
                                            "{n} draws)", n=cfg.draws)),
                OutputRaster("prob_sd", net.grid(res.prob_sd).astype("float32"), g, cmap="viridis", visible=False,
                             description=tr("Std. deviation of the MS-KPRB probability over the draws")),
                OutputRaster("knowledge_prob", net.grid(res.knowledge_prob).astype("float32"), g, cmap="inferno",
                             visible=False, description=tr("Calibrated knowledge model π_K (no residual trees)")),
                OutputRaster("delta", net.grid(res.delta).astype("float32"), g, cmap="RdBu_r", visible=False,
                             description=tr("Disagreement Δ = data correction (log-odds): > 0 mineralisation not "
                                            "explained by the expert model, < 0 over-valued criteria"))]
        outs += knowledge_outputs(ctx, res.F_fit, res.expert, with_processes=params.get("out_processes", False))
        tables = dict(summary=res.summary, layer_weights=res.layers, process_weights=res.processes,
                      draws=res.draws, tree_selection=res.t_selection, importance=res.importance)
        st = res.stats
        extra = dict(kprb=res, cards=[(f"{st['deposits']} / {st['non_deposits']}",
                                       N_("training deposits / non-deposits")),
                                      (f"{st['median_trees']:g}", N_("residual trees T (median)")),
                                      (f"{100 * st['share_T0']:.0f} %", N_("draws with T = 0"))],
                     tabs=[("summary", N_("Summary"), N_("<b>Model</b> — knowledge network, priors, residual "
                                                         "boosting and the spatial choice of the number of trees.")),
                           ("layer_weights", N_("Layer weights"), N_(
                               "<b>Layers</b> — expert versus fitted weight inside the process (mean ± s.d. over the "
                               "draws), fitted slope s and inflection c of the membership σ(s(u − c)) on the robust "
                               "scale u = d(x − median)/IQR.")),
                           ("process_weights", N_("Process weights"), N_(
                               "<b>Processes</b> — expert versus fitted weight v and the share of the study area "
                               "where the process is the limiting (weakest) one.")),
                           ("tree_selection", N_("Residual trees"), N_(
                               "<b>Number of residual trees</b> — held-out spatial-block deviance of every T (mean "
                               "over draws) and how often it was selected; T = 0 keeps the calibrated expert model.")),
                           ("importance", N_("Importance"), N_(
                               "<b>Importance</b> — drop of the held-out AUC when a layer is shuffled (one draw).")),
                           ("draws", N_("Draws"), N_("<b>Random non-deposit samples</b> — trees, calibration α, β "
                                                     "and sample AUCs of every draw."))])
        return finish(self.key, ctx, params, outs, tables, res.messages, extra, progress, "kprb")

    def show_result(self, window, result: MethodResult):
        from logic.hybrid.common.dialogs import show_hybrid_result
        show_hybrid_result(window, result, tr(TITLE))
