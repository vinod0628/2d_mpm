"""Support vector machine (Vapnik 1995) for prospectivity mapping. Public model API (GUI-free)."""
from .model import (CLASS_WEIGHT_LABELS, CLASS_WEIGHTS, KERNEL_LABELS, KERNELS, SCALING_LABELS, SCALINGS,  # noqa: F401
                    SVMModel, SVMResult, platt_fit, platt_prob)
