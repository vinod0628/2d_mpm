"""MPM logic: all prospectivity methods, grouped by category, plus helpers shared between methods.

    logic/
      data_driven/       methods trained on known deposits (wofe, ...)
      knowledge_driven/  methods driven by expert weights / memberships
      hybrid/            methods combining expert knowledge and training data
      common/            GUI-free helpers used by several methods (training data, ...)
      method_template/   copy-me skeleton for a new method (not loaded)

Every method is a sub-package ``logic/<category>/<method>/`` with model.py (algorithm), plugin.py
(registration) and dialogs.py (UI); ``core.methods.load_builtin_methods()`` discovers them automatically.
"""
