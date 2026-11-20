"""BS-WofE parameter dialog (results: the generic hybrid results dialog)."""
from __future__ import annotations

from i18n import N_
from logic.hybrid.common.dialogs import HybridDialog


class BSWofEDialog(HybridDialog):
    KEY = "bs_wofe"
    TITLE = N_("Bayesian-Shrinkage Fuzzy WofE (BS-WofE)")
    SUBTITLE = N_("Each layer is cut into classes of its expert membership. The data weight of a class is shrunk "
                  "toward the expert prior weight in proportion to its precision: classes with many deposits follow "
                  "the data, classes with few keep the expert value. Correlated layers are deflated. Use fewer "
                  "classes below 10 deposits.")
    PREFIX = "bswofe"

    def build_options(self):
        self.opt_spin("n_classes", "Classes per continuous layer Q", 2, 20, 5, integer=True)
        self.opt_spin("kappa_w", "Prior precision κ", 0, 1000, 4.0, 1, 1,
                      "κ = 4 ↔ prior s.d. 0.5 of a weight; 0 = data weights only (classical WofE on fuzzy classes).")
        self.opt_spin("gamma", "Prior scale γ", 0, 20, 2.0, 1, 0.5,
                      "Expert prior weight W⁰ = γ·ω_j·(mean membership of the class − mean of the layer).")
        self.opt_check("out_processes", "Output process maps", False)
