"""Hartmann transfer learning regression benchmarks."""

from benchmarks.domains.regression.hartmann.hartmann_tl_3_20_15_regr import (
    hartmann_tl_3_20_15_regr_benchmark,
)
from benchmarks.domains.regression.hartmann.hartmann_increased_noise_tl_regr import (
    hartmann_increased_noise_tl_regr_benchmark,
)
from benchmarks.domains.regression.hartmann.hartmann_partially_inverted_tl_regr import (
    hartmann_partially_inverted_tl_regr_benchmark,
)
from benchmarks.domains.regression.hartmann.hartmann_fully_inverted_tl_regr import (
    hartmann_fully_inverted_tl_regr_benchmark,
)

__all__ = [
    "hartmann_tl_3_20_15_regr_benchmark",
    "hartmann_increased_noise_tl_regr_benchmark",
    "hartmann_partially_inverted_tl_regr_benchmark", 
    "hartmann_fully_inverted_tl_regr_benchmark",
]