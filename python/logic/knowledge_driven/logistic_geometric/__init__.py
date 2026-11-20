"""Logistic-function fuzzification + geometric average (Yousefi & Carranza 2015). Public model API (GUI-free)."""
from .model import (DIRECTION_LABELS, DIRECTIONS, WEIGHTING_LABELS, WEIGHTINGS, LGLayer, LGResult,  # noqa: F401
                    fuzzify, logistic, logistic_geometric, logistic_parameters, parse_range, parse_scores)
