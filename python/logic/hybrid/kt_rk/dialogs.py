"""KT-RK parameter dialog (results: the generic hybrid results dialog)."""
from __future__ import annotations

import pandas as pd
from PyQt5.QtWidgets import QComboBox

from gui.widgets import LayerCombo
from i18n import N_, tr
from logic.hybrid.common.dialogs import HybridDialog


class KTRKDialog(HybridDialog):
    KEY = "kt_rk"
    TITLE = N_("Knowledge-Trend Regression-Kriging (KT-RK)")
    SUBTITLE = N_("Estimates a continuous target from point values (composited drill-hole grade, thickness or "
                  "accumulation; or a geochemical value): trend on the expert favourability + random forest on the "
                  "residuals, both shrunk by the one-standard-error rule, plus anisotropic kriging along the expert "
                  "strike. About 25–30 or more observations are needed. Deposits (optional) only validate the map.")
    PREFIX = "ktrk"
    NEED_DEPOSITS = False
    REGRESSION = True
    TEST_DEFAULT = 0

    def build_options(self):
        last = self.last.get("target", {})
        self.samples = LayerCombo(self.store, "point")
        lyr = self.store.by_name(last["layer"]) if last.get("layer") else next(
            (l for l in self.store.vectors("point") if "deposit" not in l.name.lower()), None)
        if lyr is not None:
            self.samples.set_layer(lyr)
        self.field = QComboBox()
        self.field.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.field.setMinimumContentsLength(14)
        self._place("_samples", "Observations (points)", self.samples)
        self._place("_field", "Value field", self.field)
        for k in ("_samples", "_field"):
            self.opts.pop(k)
        self.samples.currentIndexChanged.connect(lambda _=None: self._fields(last.get("field")))
        self._fields(last.get("field"))
        cell = self.cell
        self.opt_spin("strike", "Strike (° from north)", -180, 360, 0.0, 1, 5,
                      "Direction of the longest continuity (e.g. of the host structure or bed).")
        self.opt_spin("range_along", "Range along strike (map units)", 1, 1e7, 20 * cell, 0, 100)
        self.opt_spin("range_across", "Range across strike (map units)", 1, 1e7, 6 * cell, 0, 100)
        self.opt_spin("offset", "Log offset c", 0, 1e6, 0.0, 4, 0.01, special="auto (0.1 × median)")
        self.opt_spin("folds", "Cross-validation folds", 3, 50, 10, integer=True)
        self.opt_spin("max_gp_fit", "Max. points for the covariance fit", 100, 10000, 1500, integer=True, step=100,
                      tip="The covariance parameters are fitted on a random subset (the cost grows with n³); "
                          "kriging uses every observation.")
        self.opt_check("out_processes", "Output process maps", False)

    def _fields(self, prefer=None):
        self.field.clear()
        lyr = self.samples.layer()
        if lyr is None:
            return
        for c in lyr.gdf.columns:
            if c != "geometry" and pd.api.types.is_numeric_dtype(lyr.gdf[c]):
                self.field.addItem(c, c)
        i = self.field.findData(prefer) if prefer else -1
        if i >= 0:
            self.field.setCurrentIndex(i)

    def extra_validate(self):
        if self.samples.layer() is None or self.field.currentData() is None:
            return tr("Choose the observation points and a numeric value field.")
        if len(self.samples.layer().gdf) < 10:
            return tr("At least 10 observations are needed (about 25–30 or more).")
        return None

    def extra_params(self):
        lyr = self.samples.layer()
        self.last["target"] = dict(layer=lyr.name, field=self.field.currentData())
        return dict(samples=lyr.gdf, samples_name=lyr.name, field=self.field.currentData())
