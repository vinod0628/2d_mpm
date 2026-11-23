"""Fuzzy weights of evidence parameter dialog (results: the generic hybrid results dialog)."""
from __future__ import annotations

from i18n import N_
from logic.hybrid.common.dialogs import HybridDialog


class FuzzyWofEDialog(HybridDialog):
    KEY = "fuzzy_wofe"
    TITLE = N_("Fuzzy Weights of Evidence")
    SUBTITLE = N_("Each evidence class gets a fuzzy weight between W⁻ and W⁺: its membership μ in the "
                  "favourable set (from the knowledge table, or from the data) mixes the conditional probabilities of "
                  "the crisp favourable pattern and its complement counted from the deposits. The modified model "
                  "estimates the layer coefficients by logistic regression, so correlated layers are not double "
                  "counted and the predicted number of deposits equals the observed one.")
    PREFIX = "fwofe"
    USES_NEGATIVES = False

    def build_options(self):
        self.opt_combo("model", "Posterior model", [("modified", N_("modified fuzzy WofE (β by logistic regression)")),
                                                    ("classic", N_("classic fuzzy WofE (Σ W)"))], "modified",
                       "Classic: logit P = prior + Σ W (conditional independence assumed). Modified: logit P = β0 + "
                       "Σ β W, coefficients estimated on all cells (Agterberg 2011; Zhang et al. 2014).")
        self.opt_combo("membership", "Class memberships", [("expert", N_("expert (knowledge table)")),
                                                           ("data", N_("data (logistic of the class contrast)"))],
                       "expert", "Expert: class scores and the expert direction (hybrid fuzzy WofE, Porwal et al. "
                                 "2006). Data: memberships from the contrast of every class (data-driven).")
        self.opt_combo("pattern", "Favourable pattern B", [("max_contrast", N_("maximum studentized contrast")),
                                                           ("half", N_("classes with μ ≥ 0.5"))], "max_contrast",
                       "Crisp pattern whose conditional probabilities define W⁺ (μ = 1) and W⁻ (μ = 0).")
        self.opt_check("nonnegative", "Modified model: β ≥ 0 (keep the expert directions)", True,
                       "A layer whose weights would be used against its expert direction once the other layers are "
                       "accounted for gets β = 0 (no extra information) instead of a negative coefficient.")
        self.opt_spin("n_classes", "Classes per continuous layer", 2, 50, 10, integer=True,
                      tip="Quantile classes of a continuous layer; use fewer with few deposits.")
        self.opt_check("out_processes", "Output process maps", False)
