"""Boolean logic overlay (Bonham-Carter 1994). Public model API (GUI-free)."""
from .model import (BOOL_OP_LABELS, BOOL_OPS, CONDITION_LABELS, CONDITIONS, VALUE_HINTS, BooleanResult,  # noqa: F401
                    BoolLayer, boolean_overlay, condition, parse_values)
