"""Hybrid MPM methods — combine expert knowledge and training data (e.g. data-calibrated fuzzy memberships).

Same layout as ``data_driven`` (one sub-package per method with model.py / plugin.py / dialogs.py),
discovered automatically by ``core.methods.load_builtin_methods()``.

Planned: neuro_fuzzy (ANFIS-type; Porwal et al. 2004), fuzzy_wofe (Cheng & Agterberg 1999),
wofe_lr_fuzzy_memberships (Porwal et al. 2003). Start from ``python/logic/method_template``.
"""
