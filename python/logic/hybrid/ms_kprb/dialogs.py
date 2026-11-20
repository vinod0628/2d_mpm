"""MS-KPRB parameter dialog (results: the generic hybrid results dialog)."""
from __future__ import annotations

from i18n import N_
from logic.hybrid.common.dialogs import HybridDialog


class KPRBDialog(HybridDialog):
    KEY = "ms_kprb"
    TITLE = N_("Knowledge-Prior Residual Boosting (MS-KPRB)")
    SUBTITLE = N_("logit P = calibrated knowledge log-odds + data correction. The expert network is calibrated with "
                  "the expert weights as prior modes (confidence κ); small residual trees then correct it where the "
                  "deposits demand it. The number of trees is chosen on held-out spatial blocks — 0 trees keeps the "
                  "expert model. Repeated for random non-deposit samples (mean and s.d.).")
    PREFIX = "kprb"

    def build_options(self):
        self.opt_spin("kappa", "Expert confidence κ", 0, 1000, 10.0, 1, 1,
                      "Dirichlet pseudo-count of the expert weights: 3 = low confidence, 10 = default, 30 = high.")
        self.opt_spin("tau", "Membership tolerance τ", 0.05, 10, 0.75, 2, 0.05,
                      "Prior s.d. of ln(slope) and inflection of the memberships (robust scale).")
        self.opt_spin("neg_ratio", "Non-deposits per deposit", 0.5, 50, 3.0, 1, 0.5)
        self.opt_spin("draws", "Random draws B", 1, 200, 20, integer=True,
                      tip="Number of random non-deposit samples; mean and s.d. of the probability are reported.")
        self.opt_spin("learning_rate", "Learning rate ν", 0.005, 1, 0.05, 3, 0.01)
        self.opt_spin("depth", "Tree depth", 1, 6, 2, integer=True,
                      tip="Depth 2 lets a tree express one two-layer interaction.")
        self.opt_spin("min_leaf", "Min. samples per leaf", 1, 100, 5, integer=True)
        self.opt_spin("folds", "Spatial folds", 2, 16, 4, integer=True,
                      tip="Blocks of a 4 × 4 tiling are dealt to this many folds to choose the number of trees.")
        self.opt_combo("guided", "Non-deposit sampling", [(0.0, N_("uniform (outside the buffer)")),
                                                          (1.0, N_("knowledge-guided ∝ (1 − K)"))], 0.0,
                       "Knowledge-guided sampling prefers cells the expert model calls unfavourable.")
        self.opt_check("prior_correction", "Correct for the sampling ratio (absolute scale)", False,
                       "Subtract ln(unlabelled cells / sampled non-deposits) from the log-odds; ranks are unchanged.")
        self.opt_check("out_processes", "Output process maps", False)
