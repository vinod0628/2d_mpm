"""Fuzzy logic overlay (Bonham-Carter 1994; Yousefi & Carranza 2015). Public model API (GUI-free)."""
from .model import (MEMBERSHIP_LABELS, MEMBERSHIPS, OPERATOR_LABELS, OPERATORS, PARAM_HINTS, FuzzyLayer,  # noqa: F401
                    FuzzyResult, combine, default_params, fuzzy_overlay, membership, parse_params)
