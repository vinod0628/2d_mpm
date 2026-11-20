"""KW-SVM parameter dialog (results: the generic hybrid results dialog)."""
from __future__ import annotations

from i18n import N_
from logic.hybrid.common.dialogs import HybridDialog


class KWSVMDialog(HybridDialog):
    KEY = "kw_svm"
    TITLE = N_("Knowledge-Weighted SVM (KW-SVM)")
    SUBTITLE = N_("An SVM in the space of the expert memberships: every layer's membership is scaled by its expert "
                  "importance (weight in the process × process weight), and non-deposits in cells the expert calls "
                  "favourable get small weights (they may be undiscovered deposits). A wrong expert direction hurts "
                  "this method directly — check the directions first (e.g. with BS-WofE's weight table).")
    PREFIX = "kwsvm"

    def build_options(self):
        self.opt_spin("C", "C (soft margin)", 0.001, 1e4, 1.0, 3, 0.5)
        self.opt_spin("gamma", "Kernel γ", 0, 100, 0.0, 3, 0.1, special="auto (1/m)",
                      tip="RBF width on the weighted memberships; auto = 1 / number of layers.")
        self.opt_spin("guided", "Fuzzy weight exponent a", 0, 5, 1.0, 1, 0.5,
                      "Weight of a non-deposit ∝ (1 − K)^a, scaled to sum N; 0 = all weights equal.")
        self.opt_spin("neg_ratio", "Non-deposits per deposit", 0.5, 50, 3.0, 1, 0.5)
        self.opt_spin("n_bags", "SVMs (bags)", 1, 100, 10, integer=True)
        self.opt_check("out_processes", "Output process maps", False)
