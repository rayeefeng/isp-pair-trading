"""isppairtradingv2 — hedge-ratio-aware lead-lag analysis.

Builds on v1 (`pairtrade`) by adding a rolling hedge ratio (β) fit at the
recommended lead-lag delay and "capture ratio" diagnostics for each event:
how much of the predicted follower move actually materialized.
"""

from . import hedge_ratio, leadlag, plots

__all__ = ["hedge_ratio", "leadlag", "plots"]
