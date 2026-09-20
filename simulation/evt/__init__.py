"""What the experiments ask this package for.

Deliberately narrow: the limit laws in `laws.py`, the per-test critical values and rejection
rules in `tests.py`, and the internals of a gap rule in `gap_rules.py` are all reachable by
importing that module directly. Only the names below are re-exported, so this list stays a
statement about what is used rather than an inventory of everything defined.
"""

from .gap_rules import (
    GapRule,
    build_procedures,
    gap_statistics_batch,
    sorted_cache,
    sqrt_2_log,
    validate_names,
)
from .tests import (
    REJECT,
    STATISTIC_BATCH,
    TESTS,
    asymptotic_critical_values,
    empirical_critical,
)

__all__ = [
    "GapRule",
    "REJECT",
    "STATISTIC_BATCH",
    "TESTS",
    "asymptotic_critical_values",
    "build_procedures",
    "empirical_critical",
    "gap_statistics_batch",
    "sorted_cache",
    "sqrt_2_log",
    "validate_names",
]
