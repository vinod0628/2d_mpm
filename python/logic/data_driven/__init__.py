"""Data-driven MPM methods — weights/parameters are learned from known deposits (training points).

Layout: one sub-package per method, discovered automatically by ``core.methods.load_builtin_methods()``::

    data_driven/<method>/
        __init__.py   re-exports the public model API
        model.py      algorithm only (numpy / scipy / scikit-learn) — no Qt, unit-testable
        plugin.py     @register-ed MPMMethod: create_dialog() -> run() (worker thread) -> show_result()
        dialogs.py    parameter + result dialogs (imported lazily by plugin.py)

Implemented: wofe (Weights of Evidence), logistic_regression (weighted / sample-based LR), random_forest, svm
(support vector machine), xgboost_mpm (XGBoost gradient boosting), rvm
(relevance vector machine), mlp (neural network), elm
(extreme learning machine), cnn (patch CNN, TensorFlow),
bayesian_network (naive Bayes / TAN).
Planned: evidential_belief, frequency_ratio, pu_learning.
Start from ``python/logic/method_template``. Non-deposit sampling: ``logic.common.training``; feature matrices for
machine-learning methods: ``logic.common.features``.
"""
