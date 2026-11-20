"""XGBoost (Chen & Guestrin 2016) for prospectivity mapping. Public model API (GUI-free).

The package is called ``xgboost_mpm`` so that it never shadows the ``xgboost`` library.
"""
from .model import (CLASS_WEIGHT_LABELS, CLASS_WEIGHTS, DEVICE_LABELS, DEVICES, MISSING_LABELS,  # noqa: F401
                    MONOTONE_LABELS, XGBoostModel, XGBResult, best_round, cuda_available)
