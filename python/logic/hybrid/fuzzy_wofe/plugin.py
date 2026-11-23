"""Fuzzy weights of evidence plug-in (classic and modified; dialog -> run in worker -> result display)."""
from __future__ import annotations

from core.methods import MethodResult, MPMMethod, OutputRaster, register
from i18n import N_, tr
from logic.hybrid.common.pipeline import finish, knowledge_outputs, prepare

from .model import fuzzy_wofe

TITLE = N_("Fuzzy Weights of Evidence")


@register
class FuzzyWofEMethod(MPMMethod):
    key = "fuzzy_wofe"
    name = TITLE
    category = N_("Hybrid")
    group = N_("Fuzzy evidence")
    description = N_("Fuzzy weights of evidence (Cheng & Agterberg 1999; hybrid form of Porwal et al. 2006): expert or "
                     "data-driven fuzzy memberships of the evidence classes combined with conditional probabilities "
                     "counted from the deposits; classic posterior or the modified model (Agterberg 2011; Zhang et "
                     "al. 2014) whose coefficients are estimated by logistic regression.")
    icon = "mdi6.blur"

    def create_dialog(self, parent, store):
        from .dialogs import FuzzyWofEDialog
        return FuzzyWofEDialog(store, parent)

    def run(self, params: dict, progress) -> MethodResult:
        ctx = prepare(params, progress)
        progress(30, tr("fuzzy weights"))
        res = fuzzy_wofe(ctx.net, ctx.train_pos, membership=params.get("membership", "expert"),
                         pattern=params.get("pattern", "max_contrast"), model=params.get("model", "modified"),
                         n_classes=int(params.get("n_classes", 10)), nonnegative=params.get("nonnegative", True))
        g, net = ctx.grid, ctx.net
        mod = res.model == "modified"
        outs = [OutputRaster("post", net.grid(res.prob).astype("float32"), g, cmap="inferno",
                             description=tr("Posterior probability per cell — {model}",
                                            model=tr("modified fuzzy WofE") if mod else tr("classic fuzzy WofE"))),
                OutputRaster("post_classic" if mod else "post_modified",
                             net.grid(res.prob_classic if mod else res.prob_modified).astype("float32"), g,
                             cmap="inferno", visible=False,
                             description=tr("Posterior probability per cell — {model}",
                                            model=tr("classic fuzzy WofE") if mod else tr("modified fuzzy WofE"))),
                OutputRaster("logit_sd", net.grid(res.logit_sd).astype("float32"), g, cmap="viridis", visible=False,
                             description=tr("S.d. of the posterior log-odds from the fuzzy weights (delta method)"))]
        outs += knowledge_outputs(ctx, res.F, res.expert, with_processes=params.get("out_processes", False))
        tables = dict(summary=res.summary, layers=res.layers, weights=res.weights, ci_test=res.ci_test)
        st = res.stats
        extra = dict(fwofe=res, cards=[(f"{st['deposits']}", N_("training deposits")),
                                       (f"{st['ci_ratio']:.2f}", N_("CI ratio N / S (classic)")),
                                       (f"{st['mean_beta']:.2f}", N_("mean modified coefficient β"))],
                     tabs=[("summary", N_("Summary"), N_("<b>Model</b> — memberships, favourable patterns and the "
                                                         "reported posterior.")),
                           ("layers", N_("Layers"), N_(
                               "<b>Layers</b> — crisp favourable pattern B (classes, area, deposits), W⁺, W⁻, contrast, "
                               "studentized contrast and the modified coefficient β ± s.e. (1 = fuzzy weights used as "
                               "they are; > 1 = the memberships understate the evidence; smaller than alone = the layer "
                               "repeats information of another).")),
                           ("weights", N_("Fuzzy weights"), N_(
                               "<b>Classes</b> — membership μ, cells, deposits, whether the class belongs to B, class "
                               "contrast and the fuzzy weight W(μ) ± s.d.")),
                           ("ci_test", N_("Independence test"), N_(
                               "<b>Conditional independence</b> — observed versus predicted deposits of the classic "
                               "model and the Agterberg–Cheng test; the modified model reproduces N."))])
        return finish(self.key, ctx, params, outs, tables, res.messages, extra, progress, "fwofe")

    def show_result(self, window, result: MethodResult):
        from logic.hybrid.common.dialogs import show_hybrid_result
        show_hybrid_result(window, result, tr(TITLE))
