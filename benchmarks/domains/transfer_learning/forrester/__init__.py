"""Forrester transfer learning benchmarks.

This module provides a comprehensive set of transfer learning benchmarks using
the Forrester function with various source-target transformations:

• Noise-based benchmarks: Test robustness to different noise levels
• Parameter variants: Test cross-parameter transfer (amplified, low-fidelity, inverted)
• Spatial shifts: Test domain shift robustness with input transformations

All benchmarks use the clean original Forrester function as target and
different configurations as source tasks.
"""