"""Sigmoid transfer learning regression benchmarks."""

from benchmarks.domains.regression.sigmoid.sigmoid_partially_inverted_tl_regr import (
    sigmoid_partially_inverted_tl_regr_benchmark,
)
from benchmarks.domains.regression.sigmoid.sigmoid_partially_inverted_noisy_tl_regr import (
    sigmoid_partially_inverted_noisy_tl_regr_benchmark,
)

__all__ = [
    "sigmoid_partially_inverted_tl_regr_benchmark",
    "sigmoid_partially_inverted_noisy_tl_regr_benchmark",
]