"""Parameter dialog of the method. ``params()`` must return plain data (arrays, grids, numbers) for run()."""
from __future__ import annotations

from PyQt5.QtWidgets import QDoubleSpinBox, QFormLayout

from gui.widgets import BaseDialog, LayerCombo
from i18n import tr


class TemplateDialog(BaseDialog):
    def __init__(self, store, parent=None):
        # every user-visible text goes through tr(); placeholders are named: tr("Evidence {n}", n=1)
        super().__init__(tr("Template: weighted mean"),
                         tr("Example parameter dialog — replace with the method's settings."), parent, ok_text=tr("Run"))
        self.store = store
        form = QFormLayout()
        self.rows = []
        for i in range(2):
            combo, weight = LayerCombo(store, "raster"), QDoubleSpinBox()
            weight.setRange(0, 100)
            weight.setValue(1.0)
            form.addRow(tr("Evidence {n}", n=i + 1), combo)
            form.addRow("  " + tr("weight"), weight)
            self.rows.append((combo, weight))
        self.deposits = LayerCombo(store, "point", allow_none=True, none_text=tr("— no validation —"))
        form.addRow(tr("Deposits (validation)"), self.deposits)
        self.body.addLayout(form)

    def validate(self):
        if any(c.layer() is None for c, _ in self.rows):
            return tr("Choose the evidence rasters.")
        return None

    def params(self) -> dict:
        return dict(evidence=[dict(name=c.layer().name, data=c.layer().data, grid=c.layer().grid, weight=w.value())
                              for c, w in self.rows],
                    deposits=self.deposits.layer().gdf if self.deposits.layer() is not None else None)
