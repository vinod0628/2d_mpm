"""Neuro-fuzzy (ANFIS-type) parameter dialog (results: the generic hybrid results dialog)."""
from __future__ import annotations

from i18n import N_, tr
from logic.hybrid.common.dialogs import HybridDialog

from .model import MAX_RULES

INPUT_LABELS = [("auto", N_("auto (layers if ≤ 64 rules, else processes)")),
                ("processes", N_("one per mineral-system process")), ("layers", N_("one per evidence layer"))]
CONSEQUENT_LABELS = [("linear", N_("first order  f = p·x + q")), ("constant", N_("zero order  f = q"))]
MF_LABELS = [(2, N_("2 (low, high)")), (3, N_("3 (low, medium, high)"))]


class NeuroFuzzyDialog(HybridDialog):
    KEY = "neuro_fuzzy"
    TITLE = N_("Neuro-Fuzzy Inference (ANFIS-type)")
    SUBTITLE = N_("A Takagi–Sugeno fuzzy rule base (IF source is high AND pathway is low THEN …) built on the expert "
                  "knowledge: the membership functions start at low = 0 / high = 1 of the expert memberships (or "
                  "process scores) and every rule starts from the expert favourability. ANFIS hybrid learning then "
                  "tunes them on the deposits — least squares for the rule outputs, gradient descent for the "
                  "membership functions — and keeps the epoch with the smallest error on a checking set.")
    PREFIX = "neurofuzzy"

    def build_options(self):
        self.opt_combo("inputs", "Inputs", INPUT_LABELS, "auto",
                       "Processes: few inputs and rules (robust with few deposits). Layers: the data shapes every "
                       "layer, but the rules grow as M^layers.")
        self.opt_combo("n_mf", "Membership functions per input", MF_LABELS, 2)
        self.opt_combo("consequent", "Rule outputs", CONSEQUENT_LABELS, "linear")
        self.opt_spin("kappa", "Knowledge prior κ", 0, 1000, 1.0, 2, 0.5,
                      "Ridge weight pulling every rule output towards the expert favourability of the rule (and "
                      "the slopes towards 0); 0 = data only, large = expert rule base.")
        self.opt_spin("epochs", "Epochs", 1, 2000, 100, integer=True)
        self.opt_spin("lr", "Learning rate (premises)", 0.0, 1.0, 0.02, 3, 0.005,
                      "Adam step of the membership-function centres and log widths; 0 = keep the expert "
                      "membership functions (consequents only).")
        self.opt_spin("check_pct", "Checking set (%)", 0, 50, 25, 0, 5, special="off",
                      tip="Share of the samples held out to pick the epoch with the smallest checking error.")
        self.opt_spin("neg_ratio", "Non-deposits per deposit", 0.5, 50, 3.0, 1, 0.5)
        self.opt_spin("n_bags", "Models (bags)", 1, 100, 10, integer=True)
        self.opt_check("out_processes", "Output process maps", False)

    def extra_validate(self):
        sel = self.knowledge.selected()
        m = len(sel)
        k = len({self.knowledge.process_of(r) for r in sel})
        mode = self.opts["inputs"].currentData()
        M = self.opts["n_mf"].currentData()
        if mode == "auto":
            mode = "layers" if M ** m <= 64 or k < 2 else "processes"
        n = m if mode == "layers" else k
        if M ** n > MAX_RULES:
            return tr("{M}^{n} = {r} rules — more than {max}. Use one input per process, 2 membership functions or "
                      "fewer layers.", M=M, n=n, r=M ** n, max=MAX_RULES)
        return None
