"""Abstract base class for transfergpbo surrogates."""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, ClassVar

from attrs import define, field
from botorch.models.model import Model
from botorch.posteriors import Posterior
from torch import Tensor
from typing_extensions import override

from baybe.exceptions import ModelNotTrainedError
from baybe.parameters import TaskParameter
from baybe.parameters.base import Parameter  # Add this import
from baybe.surrogates.base import Surrogate
from baybe.surrogates import GaussianProcessSurrogate

# Add these imports for the scaler factory
if TYPE_CHECKING:
    from botorch.models.transforms.input import InputTransform

from baybe.surrogates.source_prior.source_prior import GPBuilder


@define
class TransferGPBOSurrogate(GaussianProcessSurrogate, ABC):
    """Abstract base class for all transfergpbo model wrappers.

    This class handles the common BayBE integration logic for transfer learning
    models from the transfergpbo package. It translates between BayBE's search
    space representation and the tensor format expected by transfergpbo models.

    Key responsibilities:
    - Extract task information from BayBE's TaskParameter
    - Identify target task from active_values
    - Reorder tensors to match transfergpbo's expected format (task_feature=-1)
    - Handle training and prediction workflows
    """

    # Class variables
    supports_transfer_learning: ClassVar[bool] = True
    """Class variable encoding transfer learning support."""

    supports_multi_output: ClassVar[bool] = False
    """Class variable encoding multi-output compatibility."""

    # Input dim of the problem
    input_dim: int | None = field(default=None)
    """Dimensionality of the input space (excluding task feature)."""

    # Private attributes for storing model state
    _model: Model = field(init=False, default=None, eq=False)
    """The actual transfergpbo model instance."""

    _target_task_id: int = field(init=False, default=None, eq=False)
    """Numeric ID of the target task."""

    _task_column_idx: int = field(init=False, default=None, eq=False)
    """Column index of task feature in BayBE's computational representation."""

    @abstractmethod
    def _create_model(self) -> Model:
        """Create the specific transfergpbo model instance.

        This method must be implemented by subclasses to instantiate
        their specific model (e.g., MHGPModel, MHGPModelStable).

        Returns:
            The created transfergpbo model instance.
        """
        pass

    def _identify_target_task(self) -> tuple[int, float]:
        """Identify the TaskParameter and return its column index and target value.

        This function identifies the TaskParameter within the search space, retrieves
        its active value, and returns both the column index of the TaskParameter and
        the computational representation value for the active value. This is useful
        for filtering tensor rows that correspond to the target task.

        Returns:
            A tuple containing:
            - task_idx (int): The column index of the TaskParameter in the computational
                            representation of the search space
            - target_value (float): The computational representation value for the
                        active value of the TaskParameter (used for filtering)

        Example:
            >>> # Filter tensor rows for target task
            >>> task_idx, target_value = _identify_target_task(searchspace)
            >>> target_mask = candidates_comp_scaled[:, task_idx] == target_value
            >>> target_candidates = candidates_comp_scaled[target_mask]
        """
        searchspace = self._searchspace
        # Find the TaskParameter in the search space
        task_params = [
            p for p in searchspace.parameters if isinstance(p, TaskParameter)
        ]

        task_param = task_params[0]

        # Get the active value
        active_value = task_param.active_values[0]

        # Get the index of the TaskParameter in the computational representation
        task_idx = searchspace.task_idx

        # Get the computational representation value for the active value
        # TaskParameter uses INT encoding, so comp_df has a single column with
        # integer values
        comp_df = task_param.comp_df

        # Extract the single computational representation value
        target_value = float(comp_df.loc[active_value].iloc[0])

        return task_idx, target_value

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

    # @staticmethod
    # @override
    # def _make_parameter_scaler_factory(
    #     parameter: Parameter,
    # ) -> type["InputTransform"] | None:
    #     """Prevent task parameters from being normalized."""
    #     from botorch.models.transforms.input import Normalize

    #     from baybe.parameters import TaskParameter

    #     if isinstance(parameter, TaskParameter):
    #         return None  # No scaling for task parameters
    #     return Normalize  # Normal scaling for continuous parameters

    def _reduce_searchspace(self, searchspace):
        """Remove TaskParameter from a SearchSpace if it exists.

        Args:
            searchspace: The SearchSpace to process.

        Returns:
            A new SearchSpace without TaskParameter, or the original SearchSpace
            if no TaskParameter exists.
        """
        from baybe.parameters import TaskParameter
        from baybe.searchspace import SearchSpace

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

        # Create new SearchSpace with filtered parameters and constraints
        return SearchSpace.from_product(
            parameters=filtered_parameters,
            constraints=list(searchspace.constraints),
        )

    @override
    def _fit(self, train_x: Tensor, train_y: Tensor) -> None:
        """Fit the transfer learning model.

        This method handles the common training workflow for all transfergpbo models:
        1. Validate transfer learning context
        2. Identify target task from TaskParameter
        3. Create GPBuilder and reduced searchspace
        4. Train the model using meta_fit() and fit()

        Args:
            train_x: Training inputs in BayBE's computational representation.
                    Shape: (n_points, n_features + 1) last column may be task indices.
            train_y: Training targets. Shape: (n_points, 1).

        Raises:
            ValueError: If received empty training data or if context is invalid.
        """
        # Check if we receive empty data
        if train_x.shape[0] == 0 or train_y.shape[0] == 0:
            raise ValueError(
                f"Received empty training data! train_x.shape={train_x.shape},"
                f" train_y.shape={train_y.shape}"
            )

        # 1. Validate transfer learning context
        self._validate_transfer_learning_context()

        # 2. Identify target task from TaskParameter active_values
        self._task_column_idx, self._target_task_id = self._identify_target_task()

        # 3. Create reduced searchspace (without TaskParameter) and GPBuilder
        reduced_searchspace = self._reduce_searchspace(self._searchspace)
        gp_builder = GPBuilder(
            searchspace=reduced_searchspace, kernel_factory=self.kernel_factory
        )

        # 4. Create model if not exists and set GPBuilder
        if self._model is None:
            self._model = self._create_model()

        # Pass the GPBuilder to the model
        self._model.set_gp_builder(gp_builder)

        # 5. Train the transfergpbo model
        # meta_fit: Train source GPs on residuals
        self._model.meta_fit(
            train_x,
            train_y,
            task_feature=self._task_column_idx,
            target_task=self._target_task_id,
        )

        # fit: Train target GP on remaining residuals
        self._model.fit(
            train_x,
            train_y,
            task_feature=self._task_column_idx,
            target_task=self._target_task_id,
        )

    @override
    def _posterior(self, candidates_comp_scaled: Tensor, /) -> Posterior:
        """Compute posterior predictions.

        This method handles prediction for candidates that may contain
        task indices for any of the available tasks (source or target).

        Args:
            candidates_comp_scaled: Candidate points in computational representation.
                            Should include task indices in same format as training data.

        Returns:
            Posterior distribution for the candidate points.

        Raises:
            ModelNotTrainedError: If model hasn't been trained yet.
        """
        if self._model is None:
            raise ModelNotTrainedError(
                "Model must be fitted before making predictions. Call fit() first."
            )
        posterior = self._model.posterior(candidates_comp_scaled)
        return posterior

    @override
    def to_botorch(self) -> Model:
        """Return the trained transfergpbo model.

        Returns:
            The underlying transfergpbo model instance.

        Raises:
            ModelNotTrainedError: If model hasn't been trained yet.
        """
        if self._model is None:
            raise ModelNotTrainedError(
                "Model must be fitted before accessing the BoTorch model. "
                "Call fit() first."
            )
        return self._model

    def __str__(self) -> str:
        """Return string representation of the surrogate."""
        fields = [
            f"Input Dim: {self.input_dim}",
            f"Supports Transfer Learning: {self.supports_transfer_learning}",
            f"Model Type: {self.__class__.__name__}",
        ]

        if self._model is not None:
            fields.append("Status: Trained")
            if self._target_task_id is not None:
                fields.append(f"Target Task ID: {self._target_task_id}")
        else:
            fields.append("Status: Not Trained")

        return f"{self.__class__.__name__}({', '.join(fields)})"
