# Source GP defintion requires
# X_source shape (in Model)
# parameter_bounds from search space (SP in Wrapper)
# pasing context is sufficient

"""Torch Models for "Transfer Learning with GPs for BO" by Tighineanu et al. (2022)."""

import torch
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.model import Model
from botorch.posteriors import GPyTorchPosterior
from gpytorch.distributions import MultivariateNormal
from gpytorch.mlls import ExactMarginalLogLikelihood
from torch import Tensor
from typing_extensions import override

from baybe.surrogates.source_prior.source_prior import GPBuilder
from baybe.surrogates.transfergpbo.utils import (
    is_pd,
    nearest_pd,
)


class MHGPModel(Model):
    """Multi-task Hierarchical Gaussian Process model for transfer learning.

    This model implements the MHGP (Multi-task Hierarchical GP) approach from
    "Transfer Learning with Gaussian Processes for Bayesian Optimization" by
    Tighineanu et al. (2022). The model sequentially trains a stack of Gaussian
    processes where each GP models the residuals from the previous GPs in the stack.

    The key idea is that each GP uses the posterior mean of the previous GP as its
    prior mean, creating a hierarchical structure that enables transfer learning
    from source tasks to a target task.

    Args:
        input_dim: Dimensionality of the input space (excluding task feature).
        numerical_stability: Flag whether to enable numerical stable predictions.

    Note:
        This is the basic implementation without numerical stability enhancements.
        For production use with small datasets or ill-conditioned problems,
        consider using :class:`MHGPModelStable` instead.

    Examples:
        >>> import torch
        >>>
        >>> # Create model and fit with MultiTaskGP-like interface
        >>> model = MHGPModel(input_dim=2)
        >>>
        >>> # X_multi includes task indices, Y contains all outputs
        >>> X_multi = torch.tensor([[0.1, 0.2, 0], [0.3, 0.4, 0], [0.5, 0.6, 1]])
        >>> Y = torch.tensor([[0.5], [0.7], [0.9]])
        >>>
        >>> model.meta_fit(X_multi, Y, task_feature=-1, target_task=1)
        >>> model.fit(X_multi, Y, task_feature=-1, target_task=1)
        >>>
        >>> # Make predictions
        >>> X_test = torch.tensor([[0.1, 0.4, 0], [0.5, 0.8, 1], [0.7, 0.8, 1]])
        >>> posterior = model.posterior(X_test)
        >>> mean = posterior.mean
        >>> variance = posterior.variance
    """

    def __init__(self, input_dim: int, numerical_stability: bool) -> None:
        super().__init__()
        self.input_dim = input_dim
        """Input dimension excluding TaskParameter"""
        self.numerical_stability = numerical_stability
        """Whether to use numerically stable implementation."""
        self.task_feature: int | None = None
        """The index of the task descriptors in the data"""
        self.target_task: int | None = None
        """The descriptor of the target task"""
        self.source_gps: list[SingleTaskGP] = []
        """List of fitted source Gaussian Process models."""
        self.target_gp: SingleTaskGP | None = None
        """The target Gaussian Process model."""
        self._fitted: bool = False
        """Whether the model has been fully fitted (including target task)."""

        # NEW: GP builder for consistent GP construction
        self.gp_builder: "GPBuilder" | None = None
        """GP builder for creating GPs with BayBE's configuration."""

    def set_gp_builder(self, gp_builder: "GPBuilder") -> None:
        """Set the GP builder for consistent GP construction.

        Args:
            gp_builder: BayBEGPBuilder instance configured with BayBE's kernel/noise settings.
        """
        self.gp_builder = gp_builder

    @property
    def num_outputs(self) -> int:
        """Number of outputs of the model."""
        return 1

    def _extract_task_data(
        self, X: Tensor, Y: Tensor, task_feature: int, target_task: int
    ) -> tuple[list[tuple[Tensor, Tensor]], tuple[Tensor, Tensor]]:
        """Extract source and target data from multi-task format.

        Args:
            X: Input data including task indices, shape (n_total, input_dim + 1).
            Y: Output data, shape (n_total, 1).
            task_feature: Index of the task feature dimension.
            target_task: Task ID for the target task.

        Returns:
            Tuple of (source_data_list, target_data_tuple).
        """
        # Extract task indices
        task_indices = X[:, task_feature].long()

        # Extract input features (without task index)
        if task_feature == -1:
            X_features = X[:, :-1]
        else:
            X_features = torch.cat(
                [X[:, :task_feature], X[:, task_feature + 1 :]], dim=1
            )

        # Get unique task IDs and sort them
        unique_tasks = torch.unique(task_indices)
        source_tasks = unique_tasks[unique_tasks != target_task].sort().values

        # Extract source data
        source_data = []
        for task_id in source_tasks:
            mask = task_indices == task_id
            X_task = X_features[mask]
            Y_task = Y[mask]
            source_data.append((X_task, Y_task))

        # Extract target data
        target_mask = task_indices == target_task
        X_target = X_features[target_mask]
        Y_target = Y[target_mask]

        return source_data, (X_target, Y_target)

    def meta_fit(
        self, X: Tensor, Y: Tensor, task_feature: int = 0, target_task: int = 0
    ) -> None:
        """Fit source GPs sequentially on residuals.

        This method implements the core MHGP training procedure by fitting each
        source GP to the residuals left by the previous GPs in the stack.

        Args:
            X: Input data including task indices, shape (n_total, input_dim + 1).
            Y: Output data, shape (n_total, 1).
            task_feature: Index of the task feature dimension (default: -1).
            target_task: Task ID for the target task (default: 2).
        """
        # Store task configuration
        self.task_feature = task_feature
        self.target_task = target_task
        # Extract source and target data
        source_data, _ = self._extract_task_data(X, Y, task_feature, target_task)

        if len(source_data) == 0:
            print(
                "No source data was provided to train the model."
                "MHGP will fall back to standard GP."
            )

        for i, (X_source, Y_source) in enumerate(source_data):
            # Compute residuals from previous GPs
            if i == 0:
                residuals = Y_source.clone()
            else:
                residuals = Y_source.clone()
                for j in range(i):
                    with torch.no_grad():
                        pred_mean = self.source_gps[j].posterior(X_source).mean
                        residuals = residuals - pred_mean.detach()

            # Ensure residuals are properly detached and cloned
            residuals = residuals.detach().clone()
            X_source_clean = X_source.detach().clone()

            # Create GP using BayBE configuration if builder available
            if self.gp_builder is not None:
                gp = self.gp_builder.create_gp(X_source_clean, residuals)
            else:
                print(f"No GPBuilder available for source GP {i}, using fallback.")
                # Fallback to original logic for backward compatibility
                gp = SingleTaskGP(X_source_clean, residuals)
                mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
                fit_gpytorch_mll(mll)

            self.source_gps.append(gp)

    def fit(
        self, X: Tensor, Y: Tensor, task_feature: int = -1, target_task: int = 2
    ) -> None:
        """Fit target GP on residuals from all source GPs.

        This method completes the MHGP training by fitting the final target GP
        to the residuals left after removing predictions from all source GPs.

        Args:
            X: Input data including task indices, shape (n_total, input_dim + 1).
            Y: Output data, shape (n_total, 1).
            task_feature: Index of the task feature dimension (default: -1).
            target_task: Task ID for the target task (default: 2).
        """
        # Extract target data
        _, (X_target, Y_target) = self._extract_task_data(
            X, Y, task_feature, target_task
        )

        if X_target.shape[0] == 0:
            print(
                "No target data provided for fitting the model."
                "The posterior will fall back to the last model in the stack."
            )
        else:
            # Fit target GP on residuals from all source GPs
            if len(self.source_gps) == 0:
                residuals = Y_target.clone()
            else:
                residuals = Y_target.clone()
                for gp in self.source_gps:
                    with torch.no_grad():
                        pred_mean = gp.posterior(X_target).mean
                        residuals = residuals - pred_mean.detach()

            # Ensure clean tensors
            residuals = residuals.detach().clone()
            X_target_clean = X_target.detach().clone()

            # Create target GP using BayBE configuration if builder available
            if self.gp_builder is not None:
                self.target_gp = self.gp_builder.create_gp(X_target_clean, residuals)
            else:
                print("No GPBuilder available, using fallback.")
                # Fallback to original logic
                self.target_gp = SingleTaskGP(X_target_clean, residuals)
                mll = ExactMarginalLogLikelihood(
                    self.target_gp.likelihood, self.target_gp
                )
                fit_gpytorch_mll(mll)

        self._fitted = True

    @override
    def posterior(self, X: Tensor, **kwargs) -> GPyTorchPosterior:
        """Compute posterior distribution with multi-task interface.

        This method now accepts input data with task indices (like MultiTaskGP) and
        returns predictions from the corresponding model in the stack for each input.

        Args:
            X: Input locations including task indices, with shape
                (batch_shape, n_points, input_dim + 1). The last column should
                contain task indices.
            **kwargs: Additional keyword arguments (currently unused).

        Returns:
            A GPyTorchPosterior object representing the posterior distribution
            at the input locations from the corresponding models in the stack.

        Raises:
            RuntimeError: If the model has not been fitted yet.
            ValueError: If task indices are invalid or out of range.
        """
        if not self._fitted:
            raise RuntimeError("Model must be fitted first. Call meta_fit() and fit().")
        if self.task_feature is None:
            raise RuntimeError("Task feature not configured. Call meta_fit() first.")

        # Handle non-batched inputs by adding a batch dimension
        if X.dim() == 2:
            X = X.unsqueeze(0)
            unbatch_output = True
        else:
            unbatch_output = False

        # Save original shape for reshaping outputs later
        batch_shape = X.shape[:-2]  # Everything except the last two dimensions
        n_points = X.shape[-2]  # Number of points

        # Extract task feature index
        task_feature = self.task_feature

        # Extract task indices and features
        if task_feature == -1:
            # Task is last column
            task_indices = X[..., -1].long()
            X_features = X[..., :-1]
        else:
            # Task is at specified position
            task_indices = X[..., task_feature].long()
            # Remove task column while preserving batch dimensions
            if task_feature == 0:
                # Task is first column
                X_features = X[..., 1:]
            elif task_feature == X.shape[-1] - 1:
                # Task is last column
                X_features = X[..., :-1]
            else:
                # Task is in the middle
                X_features = torch.cat(
                    [X[..., :task_feature], X[..., task_feature + 1 :]], dim=-1
                )

        # Validate max task ID
        max_task_id = len(self.source_gps)
        if torch.any(task_indices < 0) or torch.any(task_indices > max_task_id):
            raise ValueError(
                f"Task indices must be in range [0, {max_task_id}]. "
                f"Got invalid task indices."
            )

        # Process each batch element separately
        batch_means = []
        batch_covs = []

        with torch.set_grad_enabled(X.requires_grad):
            # Iterate over batch elements
            for batch_idx in range(batch_shape.numel()):
                # Get flat batch index
                batch_indices = []
                remaining = batch_idx
                for dim_size in reversed(batch_shape):
                    batch_indices.insert(0, remaining % dim_size)
                    remaining = remaining // dim_size

                # Extract data for this batch
                if batch_shape.numel() == 1:
                    # Single batch dimension
                    task_indices_batch = task_indices[batch_idx]
                    X_features_batch = X_features[batch_idx]
                else:
                    # Multiple batch dimensions
                    task_indices_batch = task_indices[tuple(batch_indices)]
                    X_features_batch = X_features[tuple(batch_indices)]

                # Initialize output tensors for this batch
                batch_mean = torch.zeros(n_points, 1, dtype=X.dtype, device=X.device)
                batch_cov = torch.zeros(
                    n_points, n_points, dtype=X.dtype, device=X.device
                )

                # Process each unique task in this batch
                unique_tasks_batch = torch.unique(task_indices_batch)
                for task_id in unique_tasks_batch:
                    task_mask = task_indices_batch == task_id
                    task_indices_list = torch.where(task_mask)[0]
                    X_task = X_features_batch[task_mask]

                    # Prediction from source GP
                    task_mean, task_cov = self._predict_from_stack(
                        X_task, task_id.item()
                    )

                    # Store results
                    batch_mean[task_mask] = task_mean

                    # Assign covariance block using meshgrid indexing
                    idx_i, idx_j = torch.meshgrid(
                        task_indices_list, task_indices_list, indexing="ij"
                    )
                    batch_cov[idx_i, idx_j] = task_cov

                batch_means.append(batch_mean)
                batch_covs.append(batch_cov)

        # Stack results along batch dimension
        if batch_shape.numel() == 1:
            # Single batch dimension
            stacked_means = torch.stack(batch_means, dim=0)
            stacked_covs = torch.stack(batch_covs, dim=0)
        else:
            # Multiple batch dimensions - reshape to original batch shape
            stacked_means = torch.stack(batch_means, dim=0).reshape(
                *batch_shape, n_points, 1
            )
            stacked_covs = torch.stack(batch_covs, dim=0).reshape(
                *batch_shape, n_points, n_points
            )

        # Remove extra batch dimension if input wasn't batched
        if unbatch_output:
            stacked_means = stacked_means.squeeze(0)
            stacked_covs = stacked_covs.squeeze(0)

        # Create the MultivariateNormal distribution
        mvn = MultivariateNormal(
            stacked_means.squeeze(-1),  # Remove last dimension: (..., n, 1) -> (..., n)
            stacked_covs,
        )

        return GPyTorchPosterior(mvn)

    def _predict_from_stack(
        self, X: Tensor, up_to_task_id: int
    ) -> tuple[Tensor, Tensor]:
        """Predict using the stack up to the specified task ID.

        Args:
            X: Input features (without task indices).
            up_to_task_id: Task ID to predict up to (inclusive).

        Returns:
            Tuple of (mean, variance) predictions.
        """
        # Sum predictions from source GPs up to the specified task
        total_mean = torch.zeros(X.shape[0], 1, dtype=X.dtype, device=X.device)

        # Check if we need to include target GP
        if up_to_task_id == len(self.source_gps):
            for gp in self.source_gps:
                gp_posterior = gp.posterior(X)
                total_mean += gp_posterior.mean

            if self.target_gp is None:
                print(
                    "No target data provided."
                    "Falling back to the last model in the stack for predictions."
                )
                covar_matrix = (
                    self.source_gps[up_to_task_id - 1].posterior(X).covariance_matrix
                )

            else:
                # Add target GP prediction
                target_posterior = self.target_gp.posterior(X)
                total_mean = total_mean + target_posterior.mean

                # Apply numerical stability fixes (TransferGPBO style)
                covar_matrix = target_posterior.covariance_matrix

        else:
            for task_id in range(up_to_task_id):
                gp_posterior = self.source_gps[task_id].posterior(X)
                total_mean += gp_posterior.mean

            # Apply numerical stability fixes (TransferGPBO style)
            covar_matrix = self.source_gps[up_to_task_id].posterior(X).covariance_matrix

        # Check if covariance is positive definite, fix if not
        if not is_pd(covar_matrix) and self.numerical_stability:
            covar_matrix = nearest_pd(covar_matrix)

        return total_mean, covar_matrix


class SHGPModel(MHGPModel):
    """Sequential Hierarchical GP model with uncertainty propagation.

    This model implements the SHGP (Sequential Hierarchical GP) approach from
    "Transfer Learning with Gaussian Processes for Bayesian Optimization" by
    Tighineanu et al. (2022). SHGP improves over MHGP by propagating uncertainty
    through the stack of GPs.

    The key idea is to include, in addition to the mean, also the uncertainty
    of the source posterior into the prior of the target GP. This is achieved
    by adding the posterior covariance of the previous GP as an additional
    kernel term when training each GP in the stack.

    For the target prior, SHGP uses:
    p[f_t(x)] = N[m_s(x), K_s(x,x) + K_t^(0)(x,x)]

    Where:
    - m_s(x): posterior mean from source GP
    - K_s(x,x): posterior covariance from source GP (uncertainty propagation)
    - K_t^(0)(x,x): prior covariance of target GP

    Args:
        input_dim: Dimensionality of the input space (excluding task feature).
        numerical_stability: Flag to enable numerical stability enhancements.

    Examples:
        >>> import torch
        >>> model = SHGPModel(input_dim=2, numerical_stability=True)
        >>> # Training data with task indices
        >>> X_multi = torch.tensor([[0.1, 0.2, 0], [0.3, 0.4, 0], [0.5, 0.6, 1]])
        >>> Y = torch.tensor([[0.5], [0.7], [0.9]])
        >>> model.meta_fit(X_multi, Y, task_feature=-1, target_task=1)
        >>> model.fit(X_multi, Y, task_feature=-1, target_task=1)
        >>> # Make predictions
        >>> X_test = torch.tensor([[0.1, 0.4, 1]])
        >>> posterior = model.posterior(X_test)
    """

    def __init__(self, input_dim: int, numerical_stability: bool = True) -> None:
        super().__init__(input_dim, numerical_stability)
        self._cholesky_cache: list[torch.Tensor | None] = []
        """Cached Cholesky decompositions for each GP in the stack."""

    def _compute_cholesky(
        self, gp: SingleTaskGP, prior_cov: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Compute Cholesky decomposition for a GP with optional prior covariance.

        Args:
            gp: The GP model to compute Cholesky for.
            prior_cov: Optional prior covariance matrix to add to the kernel.

        Returns:
            Lower triangular Cholesky decomposition.
        """
        if gp.train_inputs is None or gp.train_targets is None:
            raise RuntimeError("GP must be fitted before computing Cholesky")

        X_train = gp.train_inputs[0]

        # Get kernel matrix
        with torch.no_grad():
            covar_matrix = gp.covar_module(X_train).evaluate()

            # Add prior covariance if provided (this is the key SHGP modification)
            if prior_cov is not None:
                covar_matrix = covar_matrix + prior_cov

            # Add noise
            noise_var = gp.likelihood.noise
            covar_matrix = (
                covar_matrix
                + torch.eye(
                    covar_matrix.shape[-1],
                    device=covar_matrix.device,
                    dtype=covar_matrix.dtype,
                )
                * noise_var
            )

            # Compute Cholesky with numerical stability
            try:
                cholesky = torch.linalg.cholesky(covar_matrix)
            except RuntimeError:
                # Add jitter for numerical stability
                jitter = 1e-6
                while jitter < 1e-1:
                    try:
                        covar_matrix_jittered = (
                            covar_matrix
                            + torch.eye(
                                covar_matrix.shape[-1],
                                device=covar_matrix.device,
                                dtype=covar_matrix.dtype,
                            )
                            * jitter
                        )
                        cholesky = torch.linalg.cholesky(covar_matrix_jittered)
                        break
                    except RuntimeError:
                        jitter *= 10
                else:
                    raise RuntimeError(
                        "Failed to compute Cholesky decomposition even with jitter"
                    )

            return cholesky

    def _cache_cholesky_decompositions(self) -> None:
        """Cache Cholesky decompositions for all GPs in the stack."""
        self._cholesky_cache = []

        # First source GP (no prior covariance)
        if self.source_gps:
            chol_0 = self._compute_cholesky(self.source_gps[0])
            self._cholesky_cache.append(chol_0)

            # Subsequent source GPs (with prior covariance from previous)
            for i in range(1, len(self.source_gps)):
                gp_prev = self.source_gps[i - 1]
                gp_curr = self.source_gps[i]

                # Get training inputs for current GP
                X_curr = gp_curr.train_inputs[0]

                # Get posterior covariance from previous GP at current GP's training points
                with torch.no_grad():
                    prior_cov = gp_prev.posterior(X_curr).covariance_matrix

                chol_i = self._compute_cholesky(gp_curr, prior_cov)
                self._cholesky_cache.append(chol_i)

        # Target GP (with prior covariance from last source GP)
        if self.target_gp is not None:
            if self.source_gps:
                gp_last_source = self.source_gps[-1]
                X_target = self.target_gp.train_inputs[0]

                with torch.no_grad():
                    prior_cov = gp_last_source.posterior(X_target).covariance_matrix

                chol_target = self._compute_cholesky(self.target_gp, prior_cov)
            else:
                chol_target = self._compute_cholesky(self.target_gp)

            self._cholesky_cache.append(chol_target)

    def fit(
        self, X: Tensor, Y: Tensor, task_feature: int = -1, target_task: int = 2
    ) -> None:
        """Fit target GP with uncertainty propagation from source GPs.

        This method extends the base MHGP fit by incorporating uncertainty
        propagation through Cholesky decomposition caching.
        """
        # First call parent fit method
        super().fit(X, Y, task_feature, target_task)

        # Then cache Cholesky decompositions for SHGP uncertainty propagation
        self._cache_cholesky_decompositions()

    def _predict_from_stack(
        self, X: Tensor, up_to_task_id: int
    ) -> tuple[Tensor, Tensor]:
        """Predict using the SHGP stack with uncertainty propagation.

        This method overrides the MHGP prediction to implement proper uncertainty
        propagation through the stack. The key difference is that SHGP adds the
        posterior covariance from previous GPs to the current prediction's covariance.

        Args:
            X: Input features (without task indices).
            up_to_task_id: Task ID to predict up to (inclusive).

        Returns:
            Tuple of (mean, covariance) predictions with uncertainty propagation.
        """
        device = X.device
        dtype = X.dtype
        n_points = X.shape[0]

        # Initialize mean prediction (same as MHGP)
        total_mean = torch.zeros(n_points, 1, dtype=dtype, device=device)

        # Check if we're predicting from target GP
        if up_to_task_id == len(self.source_gps):
            # Add predictions from all source GPs (same as MHGP)
            for gp in self.source_gps:
                gp_posterior = gp.posterior(X)
                total_mean += gp_posterior.mean

            if self.target_gp is None:
                print(
                    "No target data provided."
                    "Falling back to the last model in the stack for predictions."
                )
                if self.source_gps:
                    # SHGP: Use covariance with uncertainty propagation
                    covar_matrix = self._predict_covariance_with_uncertainty(
                        X, len(self.source_gps) - 1
                    )
                else:
                    covar_matrix = torch.eye(n_points, dtype=dtype, device=device)
            else:
                # Add target GP prediction
                target_posterior = self.target_gp.posterior(X)
                total_mean += target_posterior.mean

                # SHGP: Use target covariance with uncertainty propagation from all source GPs
                covar_matrix = self._predict_covariance_with_uncertainty(
                    X, len(self.source_gps)
                )

        else:
            # Predicting from source GPs only (same mean as MHGP)
            for task_id in range(up_to_task_id + 1):
                gp_posterior = self.source_gps[task_id].posterior(X)
                total_mean += gp_posterior.mean

            # SHGP: Use covariance with uncertainty propagation
            covar_matrix = self._predict_covariance_with_uncertainty(X, up_to_task_id)

        # Apply numerical stability if needed
        if self.numerical_stability and not is_pd(covar_matrix):
            covar_matrix = nearest_pd(covar_matrix)

        return total_mean, covar_matrix

    def _predict_covariance_with_uncertainty(self, X: Tensor, up_to_idx: int) -> Tensor:
        """Predict covariance with uncertainty propagation from the stack.

        This implements the core SHGP covariance computation:
        K_SHGP(x,x') = K_current(x,x') + sum(K_prev_posterior(x,x'))

        Args:
            X: Input points for prediction.
            up_to_idx: Index of the GP to predict up to (inclusive).

        Returns:
            Covariance matrix with uncertainty propagation.
        """
        if up_to_idx < 0:
            # No GPs in stack yet
            n_points = X.shape[0]
            return torch.eye(n_points, dtype=X.dtype, device=X.device)

        # Start with the current GP's covariance
        if up_to_idx < len(self.source_gps):
            # Source GP
            current_gp = self.source_gps[up_to_idx]
        else:
            # Target GP
            current_gp = self.target_gp

        if current_gp is None:
            # Fallback to identity
            n_points = X.shape[0]
            return torch.eye(n_points, dtype=X.dtype, device=X.device)

        # Get current GP's posterior covariance
        with torch.no_grad():
            covar_matrix = current_gp.posterior(X).covariance_matrix

            # SHGP: Add uncertainty from all previous GPs in the stack
            for i in range(up_to_idx):
                if i < len(self.source_gps):
                    prev_gp = self.source_gps[i]
                    prev_covar = prev_gp.posterior(X).covariance_matrix
                    covar_matrix = covar_matrix + prev_covar

        return covar_matrix
