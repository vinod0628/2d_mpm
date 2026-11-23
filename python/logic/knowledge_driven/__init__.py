"""Knowledge-driven MPM methods — weights / memberships come from expert judgement (deposits only validate).

Same layout as ``data_driven`` (one sub-package per method with model.py / plugin.py / dialogs.py),
discovered automatically by ``core.methods.load_builtin_methods()``.

Implemented: boolean_logic, fuzzy_logic (inference network), logistic_geometric (Yousefi & Carranza 2015),
index_overlay (AHP / fuzzy-AHP weights), topsis (TOPSIS / VIKOR). Start new ones from ``python/logic/method_template``.
"""
