"""PT-Hurdle parameter dialog (results: the generic hybrid results dialog)."""
from __future__ import annotations

import numpy as np
from PyQt5.QtWidgets import QComboBox, QLineEdit

from gui.widgets import LayerCombo
from i18n import N_, tr
from logic.hybrid.common.dialogs import HybridDialog

RULES = [("in", N_("class is one of")), ("ge", N_("value ≥")), ("le", N_("value ≤"))]
ENGINE_LABELS = [("kp_rf", N_("KP-RF (knowledge-pooled forest)")), ("bs_wofe", N_("BS-WofE (weight table)")),
                 ("rf", N_("random forest only"))]


class PTHurdleDialog(HybridDialog):
    KEY = "pt_hurdle"
    TITLE = N_("Permissive-Tract Hurdle Model (PT-Hurdle)")
    SUBTITLE = N_("For deposits that need a necessary criterion (host unit, basin, permissive formation): a gate G "
                  "(1 inside the permissive area, falling off over the map's positional error) times a conditional "
                  "model trained only inside the permissive tract — so the model learns what distinguishes deposits "
                  "within the host, not host versus non-host. Do not tick the criterion layer in the knowledge table.")
    PREFIX = "pthurdle"

    def build_options(self):
        last = self.last.get("gate", {})
        self.gate_layer = LayerCombo(self.store, "raster")
        host = next((l for l in self.store.rasters() if "host" in l.name.lower()), None)
        name = last.get("layer")
        lyr = self.store.by_name(name) if name else host
        if lyr is not None:
            self.gate_layer.set_layer(lyr)
        self.gate_rule = QComboBox()
        for k, v in RULES:
            self.gate_rule.addItem(tr(v), k)
        self.gate_rule.setCurrentIndex([k for k, _ in RULES].index(last.get("rule", "in")))
        self.gate_values = QLineEdit(last.get("values", "2" if host is not None else ""))
        self.gate_values.setToolTip(tr("Classes separated by ';' (rule 'class is one of') or one threshold."))
        self._place("_gate_layer", "Permissive criterion layer", self.gate_layer)
        self._place("_gate_rule", "Permissive if", self.gate_rule)
        self._place("_gate_values", "Classes / threshold", self.gate_values)
        for k in ("_gate_layer", "_gate_rule", "_gate_values"):
            self.opts.pop(k)
        self.opt_combo("engine", "Conditional model", ENGINE_LABELS, "kp_rf")
        self.opt_spin("d0", "Gate offset d0 (cells)", 0, 100, 2.0, 1, 0.5,
                      "≈ positional error of the map: cells up to d0 outside the permissive area stay favourable.")
        self.opt_spin("s_gate", "Gate width s (cells)", 0.1, 100, 1.0, 1, 0.5)
        self.opt_spin("g0", "Tract threshold g0", 0.01, 0.99, 0.2, 2, 0.05)
        self.opt_spin("smooth", "Smoothing along the tract (cells)", 0, 50, 0.0, 1, 0.5, special="none")
        self.opt_spin("kappa_pool", "KP-RF: expert worth κ", 0, 1000, 20.0, 1, 5)
        self.opt_spin("neg_ratio", "Non-deposits per deposit", 0.5, 50, 3.0, 1, 0.5)
        self.opt_spin("n_bags", "Forests (bags)", 1, 100, 10, integer=True)
        self.opt_check("out_processes", "Output process maps", False)

    def extra_validate(self):
        lyr = self.gate_layer.layer()
        if lyr is None:
            return tr("Choose the layer of the permissive criterion.")
        if any(r["layer"] is lyr for r in self.knowledge.selected()):
            return tr("The permissive criterion is the gate — do not also tick it in the knowledge table.")
        vals = [v for v in self.gate_values.text().replace(",", ";").split(";") if v.strip()]
        try:
            [float(v) for v in vals]
        except ValueError:
            vals = []
        if not vals:
            return tr("Enter the permissive classes or a threshold.")
        return None

    def extra_params(self):
        lyr = self.gate_layer.layer()
        self.last["gate"] = dict(layer=lyr.name, rule=self.gate_rule.currentData(), values=self.gate_values.text())
        return dict(gate=dict(name=lyr.name, data=lyr.data, grid=lyr.grid, rule=self.gate_rule.currentData(),
                              values=self.gate_values.text()))
