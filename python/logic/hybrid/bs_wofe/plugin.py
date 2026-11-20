"""M3 BS-WofE plug-in: Bayesian-shrinkage fuzzy weights of evidence (dialog -> run in worker -> result display)."""
from __future__ import annotations

from core.methods import MethodResult, MPMMethod, OutputRaster, register
from i18n import N_, tr
from logic.hybrid.common.pipeline import finish, knowledge_outputs, prepare

from .model import bs_wofe

TITLE = N_("Bayesian-Shrinkage Fuzzy WofE (BS-WofE)")


@register
class BSWofEMethod(MPMMethod):
    key = "bs_wofe"
    name = TITLE
    category = N_("Hybrid")
    group = N_("Fuzzy evidence")
    description = N_("Weights of evidence on classes of the expert memberships; every data weight is shrunk toward "
                     "an expert prior weight by its precision (classes with few deposits keep the expert value) and "
                     "the sum of the weights is deflated for dependent layers. Transparent weight table.")
    icon = "mdi6.scale-balance"

    def create_dialog(self, parent, store):
        from .dialogs import BSWofEDialog
        return BSWofEDialog(store, parent)

    def run(self, params: dict, progress) -> MethodResult:
        ctx = prepare(params, progress)
        progress(30, tr("weights of evidence"))
        res = bs_wofe(ctx.net, ctx.train_pos, int(params.get("n_classes", 5)), params.get("kappa_w", 4.0),
                      params.get("gamma", 2.0))
        g, net = ctx.grid, ctx.net
        outs = [OutputRaster("prob", net.grid(res.prob).astype("float32"), g, cmap="inferno",
                             description=tr("BS-WofE posterior probability (ν = {nu:.2f})", nu=res.nu)),
                OutputRaster("logit_sd", net.grid(res.eta_sd).astype("float32"), g, cmap="viridis", visible=False,
                             description=tr("Posterior s.d. of the log-odds (uncertainty of the weights)"))]
        outs += knowledge_outputs(ctx, res.F, res.expert, with_processes=params.get("out_processes", False))
        tables = dict(summary=res.summary, layers=res.layers, weights=res.weights)
        st = res.stats
        extra = dict(bswofe=res, cards=[(f"{st['deposits']}", N_("training deposits")),
                                        (f"{st['nu']:.2f}", N_("dependence deflation ν")),
                                        (f"{st['rho']:.2f}", N_("mean |correlation| ρ̄"))],
                     tabs=[("summary", N_("Summary"), N_("<b>Model</b> — classes, prior precision, dependence "
                                                         "deflation and intercept.")),
                           ("weights", N_("Weights"), N_(
                               "<b>Weight table</b> — per class: cells, deposits, data weight W^d ± s, expert prior "
                               "weight W⁰, posterior W ± s.d. and the shrinkage κ/(1/s² + κ) toward the prior (1 = "
                               "expert only).")),
                           ("layers", N_("Layers"), N_(
                               "<b>Layers</b> — contrast of the posterior weights and the correlation between data "
                               "and prior weights; a negative correlation flags a direction to review."))])
        return finish(self.key, ctx, params, outs, tables, res.messages, extra, progress, "bswofe")

    def show_result(self, window, result: MethodResult):
        from logic.hybrid.common.dialogs import show_hybrid_result
        show_hybrid_result(window, result, tr(TITLE))
