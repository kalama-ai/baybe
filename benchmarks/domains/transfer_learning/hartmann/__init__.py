"""Synthetic transfer learning benchmarks based on the Hartmann function."""

from benchmarks.domains.transfer_learning.hartmann.hartmann_tl_3_20_15 import (
    hartmann_tl_3_20_15_benchmark,
)
from benchmarks.domains.transfer_learning.hartmann.hartmann_increased_noise_tl import (
    hartmann_increased_noise_tl_benchmark,
)
from benchmarks.domains.transfer_learning.hartmann.hartmann_partially_inverted_tl import (
    hartmann_partially_inverted_tl_benchmark,
)
from benchmarks.domains.transfer_learning.hartmann.hartmann_fully_inverted_tl import (
    hartmann_fully_inverted_tl_benchmark,
)

__all__ = [
    "hartmann_tl_3_20_15_benchmark",
    "hartmann_increased_noise_tl_benchmark", 
    "hartmann_partially_inverted_tl_benchmark",
    "hartmann_fully_inverted_tl_benchmark",
]
