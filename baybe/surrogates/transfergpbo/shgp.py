"""SHGP (Sequential Hierarchical GP) surrogate for transfer learning."""

from typing import ClassVar

from attrs import define
from botorch.models.model import Model
from typing_extensions import override

from baybe.surrogates.transfergpbo.base import TransferGPBOSurrogate
from baybe.surrogates.transfergpbo.models import SHGPModel


@define
class SHGPGaussianProcessSurrogate(TransferGPBOSurrogate):
    """Sequential Hierarchical Gaussian Process surrogate with uncertainty propagation.

    This surrogate implements the SHGP (Sequential Hierarchical GP) approach from
    "Transfer Learning with Gaussian Processes for Bayesian Optimization" by
    Tighineanu et al. (2022). SHGP improves over MHGP by propagating uncertainty
    through the stack of GPs.

    The key innovation is that SHGP includes not only the mean from source GPs
    as prior information, but also propagates the uncertainty (covariance) from
    source GPs into the target GP. This leads to better uncertainty quantification
    and improved transfer learning performance.

    For the target prior, SHGP uses:
    p[f_t(x)] = N[m_s(x), K_s(x,x) + K_t^(0)(x,x)]

    Where:
    - m_s(x): posterior mean from source GPs (same as MHGP)
    - K_s(x,x): posterior covariance from source GPs (uncertainty propagation)
    - K_t^(0)(x,x): prior covariance of target GP

    Computational Complexity:
    - Training: O(N_t * N_s^2) where N_s is source points, N_t is target points
    - Inference: O(N_s^2)
    - Compare to MHGP: O(N_t * N_s) training, O(N_s) inference

    Examples:
        >>> from baybe.parameters import NumericalDiscreteParameter, TaskParameter
        >>> from baybe.searchspace import SearchSpace
        >>> from baybe.surrogates.transfergpbo import SHGPGaussianProcessSurrogate
        >>> import pandas as pd
        >>>
        >>> # Create search space with task parameter
        >>> searchspace = SearchSpace.from_product([
        ...     NumericalDiscreteParameter("x", [1, 2, 3]),
        ...     TaskParameter("task", ["source", "target"], active_values=["target"])
        ... ])
        >>>
        >>> # Create surrogate
        >>> surrogate = SHGPGaussianProcessSurrogate(input_dim=1)
        >>>
        >>> # Training data with source/target tasks
        >>> data = pd.DataFrame({
        ...     "x": [1, 2, 3, 2],
        ...     "task": ["source", "source", "source", "target"],
        ...     "Target": [0.1, 0.4, 0.9, 0.5]
        ... })
        >>>
        >>> # Fit surrogate
        >>> from baybe.objectives import SingleTargetObjective
        >>> from baybe.targets import NumericalTarget
        >>> objective = SingleTargetObjective(NumericalTarget("Target", "MIN"))
        >>> surrogate.fit(searchspace, objective, data)
        >>>
        >>> # Make predictions
        >>> test_data = pd.DataFrame({"x": [1.5], "task": ["target"]})
        >>> posterior = surrogate.posterior(test_data)
    """

    # Class variables
    supports_transfer_learning: ClassVar[bool] = True
    """Class variable encoding transfer learning support."""

    supports_multi_output: ClassVar[bool] = False
    """Class variable encoding multi-output compatibility."""

    # Enable numerical stability by default for SHGP
    numerical_stability: bool = True
    """Whether to enable numerical stability enhancements."""

    @override
    def _create_model(self) -> Model:
        """Create the SHGP model instance.

        Returns:
            The created SHGPModel instance configured for transfer learning
            with uncertainty propagation.
        """
        return SHGPModel(
            input_dim=self.input_dim,
            numerical_stability=self.numerical_stability,
        )

    def __str__(self) -> str:
        """Return string representation."""
        base_str = super().__str__()
        return base_str.replace(
            "MHGPGaussianProcessSurrogate", "SHGPGaussianProcessSurrogate"
        ).replace("MHGP", "SHGP")
