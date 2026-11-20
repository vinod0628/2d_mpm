"""M1 KP-RF plug-in: knowledge-pooled random forest (dialog -> run in worker -> result display)."""
from __future__ import annotations

from core.methods import MethodResult, MPMMethod, OutputRaster, register
from i18n import N_, tr
from logic.hybrid.common.pipeline import finish, knowledge_outputs, prepare

from .model import kp_rf

TITLE = N_("Knowledge-Pooled Random Forest (KP-RF)")


@register
class KPRFMethod(MPMMethod):
    key = "kp_rf"
    name = TITLE
    category = N_("Hybrid")
    group = N_("Knowledge-constrained learning")
    description = N_("Random forest on the evidence and the expert process scores, trained with knowledge-guided "
                     "non-deposits and pooled with the calibrated expert map: logit P = (1 − λ) logit P_RF + λ η_K, "
                     "λ = κ/(κ + N) — the expert dominates when deposits are few.")
    icon = "mdi6.forest"

    def create_dialog(self, parent, store):
        from .dialogs import KPRFDialog
        return KPRFDialog(store, parent)

    def run(self, params: dict, progress) -> MethodResult:
        ctx = prepare(params, progress)
        lam = None if params.get("lam_auto", True) else params.get("lam", 0.5)
        res = kp_rf(ctx.net, ctx.train_pos, ctx.neg_cand, n_bags=int(params.get("n_bags", 10)),
                    ratio=params.get("neg_ratio", 3.0), a=params.get("guided", 1.0), lam=lam,
                    kappa=params.get("kappa_pool", 20.0), n_trees=int(params.get("n_trees", 200)),
                    min_leaf=int(params.get("min_leaf", 2)), seed=params.get("seed", 0),
                    progress=lambda f, msg: progress(8 + int(84 * f), tr(msg)))
        g, net = ctx.grid, ctx.net
        outs = [OutputRaster("prob", net.grid(res.prob).astype("float32"), g, cmap="inferno",
                             description=tr("KP-RF pooled probability (λ = {lam:.2f})", lam=res.lam)),
                OutputRaster("prob_rf", net.grid(res.prob_rf).astype("float32"), g, cmap="inferno", visible=False,
                             description=tr("Random-forest part P_RF (mean of the forests)")),
                OutputRaster("prob_rf_sd", net.grid(res.prob_rf_sd).astype("float32"), g, cmap="viridis",
                             visible=False, description=tr("Std. deviation of P_RF over the forests")),
                OutputRaster("prob_k", net.grid(res.prob_k).astype("float32"), g, cmap="inferno", visible=False,
                             description=tr("Calibrated expert map σ(η_K)"))]
        outs += knowledge_outputs(ctx, res.F, res.expert, with_processes=params.get("out_processes", False))
        tables = dict(summary=res.summary, importance=res.importance, bags=res.bags)
        st = res.stats
        extra = dict(kprf=res, cards=[(f"{st['deposits']} / {st['non_deposits']}",
                                       N_("training deposits / non-deposits")),
                                      (f"{st['lam']:.2f}", N_("expert pooling weight λ")),
                                      (f"{st['oob_auc']:.3f}", N_("out-of-bag AUC (samples)"))],
                     tabs=[("summary", N_("Summary"), N_("<b>Model</b> — forests, knowledge-guided sampling, "
                                                         "calibration of the expert map and pooling weight λ.")),
                           ("importance", N_("Importance"), N_(
                               "<b>Importance</b> — mean decrease of impurity over the forests; F[...] are the expert "
                               "process scores offered to the forest as features.")),
                           ("bags", N_("Forests"), N_("<b>Forests</b> — out-of-bag AUC of every forest and the mean "
                                                      "expert favourability of its non-deposits."))])
        return finish(self.key, ctx, params, outs, tables, res.messages, extra, progress, "kprf")

    def show_result(self, window, result: MethodResult):
        from logic.hybrid.common.dialogs import show_hybrid_result
        show_hybrid_result(window, result, tr(TITLE))
