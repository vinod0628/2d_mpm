"""Neuro-fuzzy (ANFIS-type) plug-in: dialog -> run in worker -> result display."""
from __future__ import annotations

from core.methods import MethodResult, MPMMethod, OutputRaster, register
from i18n import N_, tr
from logic.hybrid.common.pipeline import finish, knowledge_outputs, prepare

from .model import neuro_fuzzy

TITLE = N_("Neuro-Fuzzy Inference (ANFIS-type)")


@register
class NeuroFuzzyMethod(MPMMethod):
    key = "neuro_fuzzy"
    name = TITLE
    category = N_("Hybrid")
    group = N_("Neuro-fuzzy")
    description = N_("Takagi–Sugeno fuzzy inference system whose membership functions start from the expert "
                     "knowledge (low / high on the expert memberships or process scores) and are tuned on deposits "
                     "with the ANFIS hybrid learning rule (least squares + gradient descent); rule outputs are "
                     "shrunk towards the expert favourability, checking-set early stopping, bagging.")
    icon = "mdi6.brain"

    def create_dialog(self, parent, store):
        from .dialogs import NeuroFuzzyDialog
        return NeuroFuzzyDialog(store, parent)

    def run(self, params: dict, progress) -> MethodResult:
        ctx = prepare(params, progress)
        res = neuro_fuzzy(ctx.net, ctx.train_pos, ctx.neg_cand, input_mode=params.get("inputs", "auto"),
                          n_mf=int(params.get("n_mf", 2)), consequent=params.get("consequent", "linear"),
                          kappa=params.get("kappa", 1.0), epochs=int(params.get("epochs", 100)),
                          lr=params.get("lr", 0.02), check_fraction=params.get("check_pct", 25.0) / 100.0,
                          n_bags=int(params.get("n_bags", 10)), ratio=params.get("neg_ratio", 3.0),
                          seed=params.get("seed", 0), progress=lambda f, msg: progress(8 + int(84 * f), tr(msg)))
        g, net = ctx.grid, ctx.net
        outs = [OutputRaster("prob", net.grid(res.prob).astype("float32"), g, cmap="inferno",
                             description=tr("Neuro-fuzzy calibrated score (mean of {n} models, {r} rules)",
                                            n=res.stats["bags"], r=res.stats["rules"])),
                OutputRaster("prob_sd", net.grid(res.prob_sd).astype("float32"), g, cmap="viridis", visible=False,
                             description=tr("Std. deviation of the neuro-fuzzy score over the models"))]
        outs += knowledge_outputs(ctx, res.F, res.expert, with_processes=params.get("out_processes", False))
        tables = dict(summary=res.summary, rules=res.rules, membership=res.mf, importance=res.importance,
                      bags=res.bags)
        st = res.stats
        extra = dict(neurofuzzy=res, cards=[(f"{st['deposits']} / {st['non_deposits']}",
                                             N_("training deposits / non-deposits")),
                                            (f"{st['inputs']} / {st['rules']}", N_("inputs / rules")),
                                            (f"{st['best_epoch']:.0f}", N_("best epoch (mean)"))],
                     tabs=[("summary", N_("Summary"), N_("<b>Model</b> — inputs, membership functions, rule base, "
                                                         "knowledge prior and hybrid learning.")),
                           ("rules", N_("Rules"), N_(
                               "<b>Rules</b> — IF … THEN f_r, sorted by the learnt output at the rule prototype. "
                               "expert_K = calibrated expert favourability the rule started from; firing = share of "
                               "the study area / of the training deposits the rule covers (normalised firing "
                               "strength); lift = deposit share / area share.")),
                           ("membership", N_("Membership functions"), N_(
                               "<b>Membership functions</b> — Gaussian A(x) = exp(−(x − c)² / 2σ²) of every input: "
                               "expert start and learnt centre c and width σ (mean over the models).")),
                           ("importance", N_("Importance"), N_(
                               "<b>Importance</b> — drop of the sample AUC when one input is permuted (mean ± s.d. "
                               "over the models).")),
                           ("bags", N_("Models"), N_("<b>Models</b> — training / checking samples, epoch with the "
                                                     "smallest checking error, errors, Platt calibration and "
                                                     "sample AUC of every bagged model."))])
        return finish(self.key, ctx, params, outs, tables, res.messages, extra, progress, "neurofuzzy")

    def show_result(self, window, result: MethodResult):
        from logic.hybrid.common.dialogs import show_hybrid_result
        show_hybrid_result(window, result, tr(TITLE))
