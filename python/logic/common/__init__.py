"""GUI-free helpers shared by several MPM methods (training data, feature stacks, ...).

Rule: code used by more than one method lives here (or in ``evaluation`` / ``utils``), never inside one
method's package, so methods stay independent of each other.
"""
