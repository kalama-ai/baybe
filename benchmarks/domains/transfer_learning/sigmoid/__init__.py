"""Sigmoid transfer learning benchmarks."""

from benchmarks.domains.transfer_learning.sigmoid.sigmoid_partially_inverted_tl import (
    sigmoid_partially_inverted_tl_benchmark,
)
from benchmarks.domains.transfer_learning.sigmoid.sigmoid_partially_inverted_noisy_tl import (
    sigmoid_partially_inverted_noisy_tl_benchmark,
)

__all__ = [
    "sigmoid_partially_inverted_tl_benchmark",
    "sigmoid_partially_inverted_noisy_tl_benchmark",
]