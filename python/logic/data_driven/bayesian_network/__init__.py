"""Bayesian network classifiers (naive Bayes, TAN) for prospectivity mapping. Public model API (GUI-free)."""
from .model import (DESIGN_LABELS, DESIGNS, STRUCTURE_LABELS, STRUCTURES, BayesianNetworkModel, BNResult,  # noqa: F401
                    discretise, fit_bn, log_odds, mutual_information)
