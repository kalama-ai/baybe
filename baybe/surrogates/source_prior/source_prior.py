"""Source prior transfer learning Gaussian process surrogate implementation."""

from typing import TYPE_CHECKING, ClassVar

from attrs import define, field
from botorch.models.model import Model
from botorch.posteriors import Posterior
from torch import Tensor
from typing_extensions import override

from baybe.exceptions import ModelNotTrainedError
from baybe.parameters import TaskParameter
from baybe.parameters.base import Parameter  # Add this import

# Add these imports for the scaler factory
if TYPE_CHECKING:
    from botorch.models.transforms.input import InputTransform

from copy import deepcopy

import gpytorch
import torch
from botorch.models import SingleTaskGP

from baybe.searchspace import SearchSpace
from baybe.surrogates import GaussianProcessSurrogate
from baybe.surrogates.gaussian_process.core import _ModelContext
from baybe.surrogates.gaussian_process.presets.default import (
    _default_noise_factory,
)


class GPyTMean(gpytorch.means.Mean):
    """Mean module that uses a GP as prior mean function.

    This class wraps a pre-trained GP to use as a mean function in another GP.
    """

    def __init__(self, gp, batch_shape=torch.Size(), **kwargs):
        super().__init__()
        # See https://github.com/cornellius-gp/gpytorch/issues/743
        self.gp = deepcopy(gp)
        self.batch_shape = batch_shape
        for param in self.gp.parameters():
            param.requires_grad = False

    def reset_gp(self) -> None:
        """Reset the GP to evaluation mode."""
        self.gp.eval()
        self.gp.likelihood.eval()

    def forward(self, input: Tensor) -> Tensor:
        """Forward pass through the mean function.

        Args:
            input: Input tensor.

        Returns:
            Mean predictions from the wrapped GP.
        """
        self.reset_gp()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            with gpytorch.settings.detach_test_caches(False):
                mean = self.gp(input).mean.detach()
        mean = mean.reshape(torch.broadcast_shapes(self.batch_shape, input.shape[:-1]))
        return mean


class GPyTKernel(gpytorch.kernels.Kernel):
    """Kernel module that uses a pre-trained kernel.

    This class wraps a pre-trained kernel to use in another GP.
    """

    def __init__(self, kernel, **kwargs):
        super().__init__()
        # See https://github.com/cornellius-gp/gpytorch/issues/743
        self.base_kernel = deepcopy(kernel)
        for param in self.base_kernel.parameters():
            param.requires_grad = False

    def reset(self) -> None:
        """Reset the kernel to evaluation mode."""
        self.base_kernel.eval()

    def forward(self, x1: Tensor, x2: Tensor, **params) -> Tensor:
        """Forward pass through the kernel.

        Args:
            x1: First input tensor.
            x2: Second input tensor.
            **params: Additional parameters.

        Returns:
            Kernel matrix.
        """
        self.reset()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            with gpytorch.settings.detach_test_caches(False):
                k = self.base_kernel.forward(x1, x2, **params).detach()
        return k


@define
class SourcePriorGaussianProcessSurrogate(GaussianProcessSurrogate):
    """Source prior transfer learning Gaussian process surrogate.

    This surrogate implements transfer learning by:
    1. Training a source GP on source task data (without task dimension)
    2. Using the source GP as a mean prior for the target GP
    3. Training the target GP on target data with source-informed priors
    """

    # Class variables
    supports_transfer_learning: ClassVar[bool] = True
    """Class variable encoding transfer learning support."""

    supports_multi_output: ClassVar[bool] = False
    """Class variable encoding multi-output compatibility."""

    # Input dim of the problem
    input_dim: int | None = field(default=None)
    """Dimensionality of the input space (excluding task feature)."""

    numerical_stability: bool = field(default=True)
    """Whether to use numerically stable implementation."""

    # Private attributes for storing model state
    _model: Model = field(init=False, default=None, eq=False)
    """The actual SourcePriorModel instance."""

    _target_task_id: int = field(init=False, default=None, eq=False)
    """Numeric ID of the target task."""

    _task_column_idx: int = field(init=False, default=None, eq=False)
    """Column index of task feature in BayBE's computational representation."""

    _source_gp: SingleTaskGP | None = field(init=False, default=None, eq=False)
    """Fitted source Gaussian Process model."""

    _target_gp: SingleTaskGP | None = field(init=False, default=None, eq=False)
    """Fitted target Gaussian Process model with source prior."""

    def _identify_target_task(self) -> int:
        """Identify target task from TaskParameter active_values.

        Extracts the target task from the TaskParameter's active_values
        and converts the task name to its corresponding numeric ID.

        Returns:
            Numeric ID of the target task.

        Raises:
            ValueError: If no TaskParameter found in search space.
            ValueError: If not exactly one active task is specified.
        """
        # Find the TaskParameter
        task_param = None
        for param in self._searchspace.parameters:
            if isinstance(param, TaskParameter):
                task_param = param
                break

        if task_param is None:
            raise ValueError(
                "No TaskParameter found in search space. "
                "Transfer learning requires a TaskParameter."
            )

        # Get active values (target tasks)
        active_values = task_param.active_values

        if len(active_values) != 1:
            raise ValueError(
                f"Got {len(active_values)}: {active_values} actives values. "
                f"Transfer learning requires exactly one target task."
            )

        target_task_name = active_values[0]

        # Convert task name to numeric ID using TaskParameter's values order
        # TaskParameter uses integer encoding: values[0]→0, values[1]→1, etc.
        task_name_to_id = {name: idx for idx, name in enumerate(task_param.values)}

        if target_task_name not in task_name_to_id:
            raise ValueError(
                f"Target task '{target_task_name}' not found in TaskParameter values: "
                f"{list(task_name_to_id.keys())}"
            )

        target_task_id = task_name_to_id[target_task_name]

        return target_task_id

    def _validate_transfer_learning_context(self) -> None:
        """Validate that we have a proper transfer learning setup.

        Raises:
            ValueError: If no task parameter found in search space.
            ValueError: If input dimensions don't match expected format.
        """
        if self._searchspace.task_idx is None:
            raise ValueError(
                "No task parameter found in search space. "
                "TransferGPBOSurrogate requires a TaskParameter for transfer learning."
            )
        # Set input_dim if not provided at initialization
        if self.input_dim is None:
            self.input_dim = len(self._searchspace.comp_rep_columns) - 1

        # Validate that we have the expected number of feature dimensions
        expected_total_dims = self.input_dim + 1  # features + task
        actual_total_dims = len(self._searchspace.comp_rep_columns)

        if actual_total_dims != expected_total_dims:
            raise ValueError(
                f"Expected {expected_total_dims} total dimensions "
                f"({self.input_dim} features + 1 task), "
                f"but got {actual_total_dims} from search space."
            )

    def _extract_task_data(
        self, X: Tensor, Y: Tensor = None, task_feature: int = -1, target_task: int = 0
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
            Y_task = Y[mask] if Y is not None else None
            source_data.append((X_task, Y_task))

        # Extract target data
        target_mask = task_indices == target_task
        X_target = X_features[target_mask]
        Y_target = Y[target_mask] if Y is not None else None

        return source_data, (X_target, Y_target)

    @staticmethod
    @override
    def _make_parameter_scaler_factory(
        parameter: Parameter,
    ) -> type["InputTransform"] | None:
        """Prevent task parameters from being normalized."""
        from botorch.models.transforms.input import Normalize

        from baybe.parameters import TaskParameter

        if isinstance(parameter, TaskParameter):
            return None  # No scaling for task parameters
        return Normalize  # Normal scaling for continuous parameters

    def _fit_with_and_without_prior(
        self,
        searchspace: SearchSpace,
        train_x: Tensor,
        train_y: Tensor,
        prior: SingleTaskGP = None,
    ) -> SingleTaskGP:
        """Fit a GP model with or without a source prior.

        Args:
            searchspace: The search space for the GP.
            train_x: Training input data.
            train_y: Training target data.
            prior: Optional source GP to use as prior.

        Returns:
            Fitted SingleTaskGP model.
        """
        import botorch
        import gpytorch
        import torch

        context = _ModelContext(searchspace)

        numerical_idxs = context.get_numerical_indices(train_x.shape[-1])

        if prior is None:
            mean_module = gpytorch.means.ConstantMean()
            # For GPs, we let botorch handle the scaling.
            input_transform = botorch.models.transforms.Normalize(
                train_x.shape[-1],
                bounds=context.parameter_bounds,
                indices=numerical_idxs,
            )
            outcome_transform = botorch.models.transforms.Standardize(1)

        else:
            # Use source GP as prior - inherit transforms from source
            prior_model = deepcopy(prior)
            mean_module = GPyTMean(prior_model)
            input_transform = prior_model.input_transform
            outcome_transform = prior_model.outcome_transform

        # extract the batch shape of the training data
        batch_shape = train_x.shape[:-2]
        mean_module.batch_shape = batch_shape

        # define the covariance module for the numeric dimensions
        base_covar_module = self.kernel_factory(
            context.searchspace, train_x, train_y
        ).to_gpytorch(
            ard_num_dims=train_x.shape[-1] - context.n_task_dimensions,
            active_dims=numerical_idxs,
            batch_shape=batch_shape,
        )
        # Note: prior_base_kernel is not used in current implementation

        # create GP covariance
        if not context.is_multitask:
            covar_module = base_covar_module
        else:
            task_covar_module = gpytorch.kernels.IndexKernel(
                num_tasks=context.n_tasks,
                active_dims=context.task_idx,
                rank=context.n_tasks,  # TODO: make controllable
            )
            covar_module = base_covar_module * task_covar_module

        # create GP likelihood
        noise_prior = _default_noise_factory(context.searchspace, train_x, train_y)
        likelihood = gpytorch.likelihoods.GaussianLikelihood(
            noise_prior=noise_prior[0].to_gpytorch(), batch_shape=batch_shape
        )
        likelihood.noise = torch.tensor([noise_prior[1]])

        # construct and fit the Gaussian process
        model = botorch.models.SingleTaskGP(
            train_x,
            train_y,
            input_transform=input_transform,
            outcome_transform=outcome_transform,
            mean_module=mean_module,
            covar_module=covar_module,
            likelihood=likelihood,
        )

        mll = gpytorch.mlls.LeaveOneOutPseudoLikelihood(model.likelihood, model)
        botorch.fit.fit_gpytorch_mll(mll, max_attempts=50)
        return model

    def _reduce_searchspace(self, searchspace: SearchSpace) -> SearchSpace:
        """Remove TaskParameter from a SearchSpace if it exists.

        Args:
            searchspace: The SearchSpace to process.

        Returns:
            A new SearchSpace without TaskParameter, or the original SearchSpace
            if no TaskParameter exists.
        """
        # Get all parameters from the search space
        parameters = list(searchspace.parameters)

        # Filter out TaskParameter instances
        filtered_parameters = [
            param for param in parameters if not isinstance(param, TaskParameter)
        ]

        # If no TaskParameter was found, return the original searchspace
        if len(filtered_parameters) == len(parameters):
            return searchspace

        # If all parameters were TaskParameters, create empty SearchSpace
        if not filtered_parameters:
            return SearchSpace()

        # Create new SearchSpace with filtered parameters and original constraints
        return SearchSpace.from_product(
            parameters=filtered_parameters, constraints=list(searchspace.constraints)
        )

    @override
    def _fit(self, train_x: Tensor, train_y: Tensor) -> None:
        """Fit the transfer learning model.

        This method handles the common training workflow for all transfergpbo models:
        1. Validate transfer learning context
        2. Identify target task from TaskParameter
        3. Reorder tensors to match transfergpbo format
        4. Train the model using meta_fit() and fit()

        Args:
            train_x: Training inputs in BayBE's computational representation.
                    Shape: (n_points, n_features + 1) last column may be task indices.
            train_y: Training targets. Shape: (n_points, 1).

        Raises:
            ValueError: If received empty training data or if context is invalid.
        """
        # FIXME[typing]: It seems there is currently no better way to inform the type
        #   checker that the attribute is available at the time of the function call
        assert self._searchspace is not None

        # Check if we receive empty data
        if train_x.shape[0] == 0 or train_y.shape[0] == 0:
            raise ValueError(
                f"Received empty training data! train_x.shape={train_x.shape},"
                f" train_y.shape={train_y.shape}"
            )

        # 1. Validate transfer learning context
        self._validate_transfer_learning_context()

        # 2. Identify target task from TaskParameter active_values
        self._target_task_id = self._identify_target_task()
        self._task_column_idx = self._searchspace.task_idx

        source_data, (X_target, Y_target) = self._extract_task_data(
            train_x, train_y, self._task_column_idx, self._target_task_id
        )
        source_data = source_data[0]
        X_source, Y_source = source_data

        # Remove the task parameter from searchspace before training the GPs
        reduced_searchspace = self._reduce_searchspace(searchspace=self._searchspace)

        # Fit the source GP
        self._source_gp = self._fit_with_and_without_prior(
            train_x=X_source,
            train_y=Y_source,
            prior=None,
            searchspace=reduced_searchspace,
        )

        # Fit the target GP using the source GP as a prior
        self._target_gp = self._fit_with_and_without_prior(
            train_x=X_target,
            train_y=Y_target,
            prior=self._source_gp,
            searchspace=reduced_searchspace,
        )

    @override
    def _posterior(self, candidates_comp_scaled: Tensor, /) -> Posterior:
        """Compute posterior predictions.

        This method handles prediction for candidates that may contain
        task indices for any of the available tasks (source or target).

        Args:
            candidates_comp_scaled: Candidate points in computational
                representation. Should include task indices in same format as
                training data.

        Returns:
            Posterior distribution for the candidate points.

        Raises:
            ModelNotTrainedError: If model hasn't been trained yet.
        """
        if self._target_gp is None:
            raise ModelNotTrainedError(
                "Model must be fitted before making predictions. Call fit() first."
            )

        source_data, (X_target, Y_target) = self._extract_task_data(
            X=candidates_comp_scaled,
            Y=None,
            task_feature=self._task_column_idx,
            target_task=self._target_task_id,
        )

        if len(source_data) > 0:
            raise NotImplementedError("Can only make predictions on target task.")

        posterior = self._target_gp.posterior(X_target)
        return posterior

    @override
    def to_botorch(self) -> Model:
        """Convert surrogate to BoTorch model.

        Returns:
            The target GP model for BoTorch compatibility.

        Raises:
            ModelNotTrainedError: If model hasn't been trained yet.
        """
        if self._target_gp is None:
            raise ModelNotTrainedError(
                "Model must be fitted before conversion. Call fit() first."
            )
        return self._target_gp
