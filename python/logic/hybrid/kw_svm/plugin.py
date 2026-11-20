"""M2 KW-SVM plug-in: knowledge-weighted fuzzy SVM (dialog -> run in worker -> result display)."""
from __future__ import annotations

from core.methods import MethodResult, MPMMethod, OutputRaster, register
from i18n import N_, tr
from logic.hybrid.common.pipeline import finish, knowledge_outputs, prepare

from .model import kw_svm

TITLE = N_("Knowledge-Weighted SVM (KW-SVM)")


@register
class KWSVMMethod(MPMMethod):
    key = "kw_svm"
    name = TITLE
    category = N_("Hybrid")
    group = N_("Knowledge-constrained learning")
    description = N_("Fuzzy support vector machine on the expert memberships: an RBF kernel weighted by the expert "
                     "importance of every layer, and non-deposit samples weighted by (1 − K)^a so cells the expert "
                     "calls favourable count less as negatives; Platt calibration, bagging.")
    icon = "mdi6.vector-polyline"

    def create_dialog(self, parent, store):
        from .dialogs import KWSVMDialog
        return KWSVMDialog(store, parent)

    def run(self, params: dict, progress) -> MethodResult:
        ctx = prepare(params, progress)
        res = kw_svm(ctx.net, ctx.train_pos, ctx.neg_cand, n_bags=int(params.get("n_bags", 10)),
                     ratio=params.get("neg_ratio", 3.0), a=params.get("guided", 1.0), C=params.get("C", 1.0),
                     gamma=params.get("gamma", 0.0) or None, seed=params.get("seed", 0),
                     progress=lambda f, msg: progress(8 + int(84 * f), tr(msg)))
        g, net = ctx.grid, ctx.net
        outs = [OutputRaster("prob", net.grid(res.prob).astype("float32"), g, cmap="inferno",
                             description=tr("KW-SVM calibrated score (mean of {n} SVMs)", n=res.stats["bags"])),
                OutputRaster("prob_sd", net.grid(res.prob_sd).astype("float32"), g, cmap="viridis", visible=False,
                             description=tr("Std. deviation of the KW-SVM score over the bags"))]
        outs += knowledge_outputs(ctx, res.F, res.expert, with_processes=params.get("out_processes", False))
        tables = dict(summary=res.summary, kernel_weights=res.kernel_weights, bags=res.bags)
        st = res.stats
        extra = dict(kwsvm=res, cards=[(f"{st['deposits']} / {st['non_deposits']}",
                                        N_("training deposits / non-deposits")),
                                       (f"{st['support_vectors']:.0f}", N_("support vectors (mean)")),
                                       (f"{st['gamma']:.3g}", N_("kernel γ"))],
                     tabs=[("summary", N_("Summary"), N_("<b>Model</b> — kernel, soft margin, fuzzy weights of the "
                                                         "non-deposits and bagging.")),
                           ("kernel_weights", N_("Kernel weights"), N_(
                               "<b>Kernel weights</b> — ω_j = w_j·v_k (mean 1): how strongly a membership difference "
                               "in layer j separates two cells in the kernel.")),
                           ("bags", N_("Bags"), N_("<b>Bags</b> — support vectors, Platt calibration (fitted on the "
                                                   "training decision values) and sample AUC of every SVM."))])
        return finish(self.key, ctx, params, outs, tables, res.messages, extra, progress, "kwsvm")

    def show_result(self, window, result: MethodResult):
        from logic.hybrid.common.dialogs import show_hybrid_result
        show_hybrid_result(window, result, tr(TITLE))
