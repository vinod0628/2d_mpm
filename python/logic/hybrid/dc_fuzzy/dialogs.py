"""Data-calibrated fuzzy memberships parameter dialog (results: the generic hybrid results dialog)."""
from __future__ import annotations

from i18n import N_
from logic.hybrid.common.dialogs import HybridDialog


class DCFuzzyDialog(HybridDialog):
    KEY = "dc_fuzzy"
    TITLE = N_("Data-Calibrated Fuzzy Memberships (WofE / LR)")
    SUBTITLE = N_("The expert decides the structure — which layers, which mineral-system process, which direction "
                  "and how the processes combine; the deposits decide the memberships: weights of evidence of every "
                  "class (kept monotone in the expert direction) or a logistic regression per layer, turned into "
                  "fuzzy memberships and combined OR-like within a process and AND-like between processes.")
    PREFIX = "dcfuzzy"
    USES_NEGATIVES = False

    def build_options(self):
        self.opt_combo("calibration", "Membership calibration", [
            ("wofe", N_("weights of evidence per class")),
            ("lr", N_("logistic regression per layer"))], "wofe",
            "WofE: one weight per class (quantile classes of continuous layers). LR: a smooth logistic function of "
            "the robustly scaled values; categorical layers always use class weights.")
        self.opt_combo("scaling", "Membership scaling", [
            ("weight", N_("μ = σ(W) — keeps the strength of each layer")),
            ("stretch", N_("stretched to 0.01 … 0.99 per layer"))], "weight",
            "σ(W): a weight of 0 gives 0.5 (no evidence), weak layers stay near 0.5. Stretched: every layer spans "
            "the full range, as expert memberships do.")
        self.opt_check("monotone", "Keep the expert direction (monotone memberships)", True,
                       "WofE: weighted isotonic fit of the class weights along the expert direction. LR: a slope "
                       "against the expert direction is set to 0 (layer neutral) and reported.")
        self.opt_spin("n_classes", "Classes per continuous layer", 2, 50, 10, integer=True,
                      tip="Quantile classes of continuous layers (WofE calibration; table only for LR).")
        self.opt_spin("kappa", "Expert worth κ (deposits)", 0, 1000, 0.0, 1, 5, special="0 (data only)",
                      tip="Pool with the expert memberships: μ = λ μ_data + (1 − λ) μ_expert, λ = N / (N + κ).")
        self.opt_combo("combination", "Combination", [
            ("network", N_("expert inference network (OR within, AND between processes)")),
            ("gamma", N_("fuzzy γ operator over all layers"))], "network")
        self.opt_spin("gamma", "γ (fuzzy γ operator)", 0, 1, 0.9, 2, 0.05)
        self.opt_check("out_memberships", "Output membership maps", False)
        self.opt_check("out_processes", "Output process maps", False)
        self.opts["combination"].currentIndexChanged.connect(self._sync)
        self._sync()

    def _sync(self, *_):
        self.opts["gamma"].setEnabled(self.opts["combination"].currentData() == "gamma")
