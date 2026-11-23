"""TOPSIS / VIKOR: parameter dialog (index-overlay evidence table and weights + MCDM options) and results dialog."""
from __future__ import annotations

from PyQt5.QtWidgets import QCheckBox, QComboBox, QDoubleSpinBox, QGridLayout, QHBoxLayout, QLabel

from gui.widgets import metric_card
from i18n import N_, tr

from ..index_overlay.dialogs import IndexOverlayDialog, IndexOverlayResultsDialog
from .model import METHOD_LABELS, NORMALISATION_LABELS, WEIGHTING_LABELS

_LAST = {}


class TopsisDialog(IndexOverlayDialog):
    TITLE = N_("TOPSIS / VIKOR (AHP / Fuzzy AHP / entropy weights)")
    SUBTITLE = N_("Every cell is an alternative and every evidence layer a criterion, scored 0 … 1 like the index "
                  "overlay (score per class, linear rescaling in the favourable direction, or a binary condition). "
                  "TOPSIS ranks the cells by their relative closeness to the ideal solution (best score of every "
                  "criterion) and distance from the negative ideal; VIKOR by a compromise between the group utility "
                  "S (weighted sum of the shortfalls) and the individual regret R (largest weighted shortfall). "
                  "Weights: equal, custom, AHP, fuzzy AHP or Shannon entropy. Deposits are used only for validation.")
    WEIGHTINGS = WEIGHTING_LABELS
    PREFIX = "topsis"
    LAST = _LAST

    def _extra_options(self, grid: QGridLayout, row: int):
        last = self.LAST
        self.method = QComboBox()
        for k, v in METHOD_LABELS.items():
            self.method.addItem(tr(v), k)
        self.method.setCurrentIndex(max(0, self.method.findData(last.get("method", "topsis"))))
        self.normalisation = QComboBox()
        for k, v in NORMALISATION_LABELS.items():
            self.normalisation.addItem(tr(v), k)
        self.normalisation.setCurrentIndex(max(0, self.normalisation.findData(last.get("normalisation", "vector"))))
        self.normalisation.setToolTip(tr("TOPSIS: vector normalisation divides every criterion by its root sum of "
                                         "squares (Hwang & Yoon 1981), so criteria with low average scores gain "
                                         "relative weight; 'none' uses the 0 … 1 scores directly."))
        self.v = QDoubleSpinBox()
        self.v.setRange(0.0, 1.0)
        self.v.setDecimals(2)
        self.v.setSingleStep(0.1)
        self.v.setValue(last.get("v", 0.5))
        self.v.setToolTip(tr("VIKOR strategy weight v: 1 = group utility S only (all criteria trade off), "
                             "0 = individual regret R only (the weakest criterion decides), 0.5 = consensus."))
        self.o_parts = QCheckBox(tr("distance / utility maps"))
        self.o_parts.setChecked(last.get("out_parts", False))
        grid.addWidget(QLabel(tr("Method")), row, 0)
        grid.addWidget(self.method, row, 1, 1, 3)
        grid.addWidget(QLabel(tr("Normalisation (TOPSIS)")), row + 1, 0)
        grid.addWidget(self.normalisation, row + 1, 1)
        grid.addWidget(QLabel(tr("strategy weight v (VIKOR)")), row + 1, 2)
        grid.addWidget(self.v, row + 1, 3)
        h = QHBoxLayout()
        h.addWidget(self.o_parts)
        h.addStretch(1)
        grid.addWidget(QLabel(tr("More outputs")), row + 2, 0)
        grid.addLayout(h, row + 2, 1, 1, 3)
        self.method.currentIndexChanged.connect(self._sync_method)
        self._sync_method()

    def _sync_method(self):
        vikor = self.method.currentData() == "vikor"
        self.normalisation.setEnabled(not vikor)
        self.v.setEnabled(vikor)
        if self.prefix.text() in ("topsis", "vikor"):
            self.prefix.setText(self.method.currentData())

    def params(self) -> dict:
        p = super().params()
        method = self.method.currentData()
        self.LAST.update(method=method, normalisation=self.normalisation.currentData(), v=self.v.value(),
                         out_parts=self.o_parts.isChecked())
        p.update(method=method, normalisation=self.normalisation.currentData(), v=self.v.value(),
                 out_parts=self.o_parts.isChecked(), prefix=self.prefix.text().strip() or method)
        return p


class TopsisResultsDialog(IndexOverlayResultsDialog):
    TITLE = N_("TOPSIS / VIKOR (AHP / Fuzzy AHP / entropy weights)")
    WEIGHTINGS = WEIGHTING_LABELS
    RESULT_KEY = "mcdm"
    EXPORT_NAME = "topsis_results.xlsx"

    def _method_cards(self, cards, res):
        ex = self.result.extra
        if ex["method"] == "topsis":
            norm = tr("vector normalisation") if ex["normalisation"] == "vector" else tr("no normalisation")
            cards.addWidget(metric_card("TOPSIS", norm))
        else:
            cards.addWidget(metric_card("VIKOR", f"v = {ex['v']:g}"))
