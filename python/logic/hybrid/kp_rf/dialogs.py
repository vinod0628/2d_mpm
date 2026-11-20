"""KP-RF parameter dialog (results: the generic hybrid results dialog)."""
from __future__ import annotations

from i18n import N_
from logic.hybrid.common.dialogs import HybridDialog


class KPRFDialog(HybridDialog):
    KEY = "kp_rf"
    TITLE = N_("Knowledge-Pooled Random Forest (KP-RF)")
    SUBTITLE = N_("A random forest learns from the evidence and the expert process scores; non-deposits are drawn "
                  "where the expert model is unfavourable. The forest is pooled with the calibrated expert map: "
                  "logit P = (1 − λ) logit P_RF + λ η_K with λ = κ/(κ + N) — the expert is worth κ pseudo-deposits, "
                  "so it dominates when deposits are few. Do not tune λ by cross-validation with few deposits.")
    PREFIX = "kprf"

    def build_options(self):
        self.opt_spin("kappa_pool", "Expert worth κ (pseudo-deposits)", 0, 1000, 20.0, 1, 5,
                      "λ = κ/(κ + N): κ = 20 gives λ = 0.71 for 8, 0.57 for 15 and 0.29 for 50 deposits.")
        self.opt_check("lam_auto", "λ = κ/(κ + N)", True, "Untick to fix λ (0 = forest only, 1 = expert only).")
        self.opt_spin("lam", "Fixed λ", 0, 1, 0.5, 2, 0.05)
        self.opt_spin("guided", "Knowledge guidance a of non-deposits", 0, 5, 1.0, 1, 0.5,
                      "Non-deposits are drawn with probability ∝ (1 − K)^a outside the buffer; 0 = uniform.")
        self.opt_spin("neg_ratio", "Non-deposits per deposit", 0.5, 50, 3.0, 1, 0.5)
        self.opt_spin("n_bags", "Forests (bags)", 1, 100, 10, integer=True)
        self.opt_spin("n_trees", "Trees per forest", 10, 5000, 200, integer=True, step=50)
        self.opt_spin("min_leaf", "Min. samples per leaf", 1, 50, 2, integer=True)
        self.opt_check("out_processes", "Output process maps", False)
        self.opts["lam_auto"].toggled.connect(lambda on: self.opts["lam"].setEnabled(not on))
        self.opts["lam"].setEnabled(not self.opts["lam_auto"].isChecked())
