"""Wrapper surrogate for pre-trained BoTorch SingleTaskGP models."""

from typing import TYPE_CHECKING, ClassVar

from attrs import define, field
from botorch.models import SingleTaskGP
from botorch.models.model import Model
from botorch.posteriors import Posterior
from torch import Tensor
from typing_extensions import override

from baybe.surrogates.gaussian_process.core import GaussianProcessSurrogate


@define
class PretrainedSingleTaskGPSurrogate(GaussianProcessSurrogate):
    """A BayBE surrogate wrapping a pre-trained BoTorch SingleTaskGP model.
    
    This surrogate allows using pre-trained BoTorch models within BayBE's 
    surrogate framework. It's particularly useful for:
    
    1. **Transfer Learning**: Use models pre-trained on combined source+target data 
       in optimization contexts that require task-parameter-free search spaces
    2. **External Models**: Integrate externally trained BoTorch models into BayBE
    3. **Model Reuse**: Avoid retraining when switching between optimization contexts
    
    The wrapper handles the interface conversion while preserving the pre-trained
    model's knowledge and performance characteristics.
    
    Example:
        >>> from botorch.models import SingleTaskGP
        >>> import torch
        >>> 
        >>> # Pre-train a BoTorch model (with transfer learning, external data, etc.)
        >>> train_x = torch.randn(10, 2)
        >>> train_y = torch.randn(10, 1) 
        >>> pretrained_model = SingleTaskGP(train_x, train_y)
        >>> 
        >>> # Wrap for use in BayBE
        >>> surrogate = PretrainedSingleTaskGPSurrogate.from_botorch_model(pretrained_model)
        >>> 
        >>> # Use in BayBE campaign as usual
        >>> campaign = Campaign(surrogate=surrogate, ...)
    """
    
    # Class variables
    supports_transfer_learning: ClassVar[bool] = True
    """Class variable encoding transfer learning support."""
    
    supports_multi_output: ClassVar[bool] = False  
    """Class variable encoding multi-output compatibility."""
    
    # Private attributes
    _pretrained_model: SingleTaskGP = field(init=False, default=None, eq=False)
    """The pre-trained BoTorch SingleTaskGP model."""
    
    @classmethod
    def from_botorch_model(
        cls,
        pretrained_model: SingleTaskGP,
        **kwargs
    ) -> "PretrainedSingleTaskGPSurrogate":
        """Create a PretrainedSingleTaskGPSurrogate from a fitted BoTorch model.
        
        Args:
            pretrained_model: A fitted BoTorch SingleTaskGP model.
            **kwargs: Additional keyword arguments for surrogate initialization.
                Typically not needed since the model is pre-trained.
                
        Returns:
            PretrainedSingleTaskGPSurrogate ready for use in BayBE.
            
        Raises:
            ValueError: If the model hasn't been fitted or is incompatible.
        """
        # Validate model state
        if not hasattr(pretrained_model, 'train_targets') or pretrained_model.train_targets is None:
            raise ValueError(
                "BoTorch model appears to be unfitted. Ensure the SingleTaskGP "
                "has been trained before wrapping it."
            )
        
        # Create instance - use default kernel factory since model is pre-trained
        instance = cls(**kwargs)
        
        # Store the pre-trained model
        instance._pretrained_model = pretrained_model
        
        # Also set the inherited _model attribute for compatibility
        instance._model = pretrained_model
        
        return instance
    
    @override
    def _fit(self, train_x: Tensor, train_y: Tensor) -> None:
        """Fit method for pre-trained surrogate.
        
        Since this surrogate wraps a pre-trained model, the _fit method validates
        that the pre-trained model is available but doesn't perform additional training.
        
        The train_x and train_y are ignored since the model is already fitted.
        However, they could potentially be used for validation or fine-tuning
        in future extensions.
        
        Args:
            train_x: Training inputs (ignored, model is pre-trained).
            train_y: Training targets (ignored, model is pre-trained).
            
        Raises:
            RuntimeError: If no pre-trained model is available.
        """
        if self._pretrained_model is None:
            raise RuntimeError(
                "No pre-trained BoTorch model available. "
                "Use PretrainedSingleTaskGPSurrogate.from_botorch_model() to create instance."
            )
            
        # Validation: Ensure the model looks properly trained
        if not hasattr(self._pretrained_model, 'train_targets'):
            raise RuntimeError(
                "Pre-trained model appears to be missing training data. "
                "Ensure the BoTorch model was properly fitted."
            )
        
        # Set the inherited _model attribute for compatibility with parent class
        self._model = self._pretrained_model
    
    @override
    def _posterior(self, candidates_comp_scaled: Tensor, /) -> Posterior:
        """Compute posterior predictions using the pre-trained BoTorch model.
        
        This method delegates directly to the wrapped BoTorch model's posterior
        method. The model should already be properly configured for the input
        format and scaling.
        
        Args:
            candidates_comp_scaled: Candidate points in computational representation,
                properly scaled for the model. Shape: (..., n_points, n_features)
                
        Returns:
            Posterior distribution from the pre-trained model.
            
        Raises:
            RuntimeError: If no pre-trained model is available.
        """
        if self._pretrained_model is None:
            raise RuntimeError(
                "No pre-trained model available for posterior computation."
            )
        
        # Delegate to the wrapped BoTorch model
        return self._pretrained_model.posterior(candidates_comp_scaled)
    
    @override
    def to_botorch(self) -> Model:
        """Return the wrapped pre-trained BoTorch model.
        
        Returns:
            The pre-trained SingleTaskGP model.
            
        Raises:
            RuntimeError: If no pre-trained model is available.
        """
        if self._pretrained_model is None:
            raise RuntimeError(
                "No pre-trained model available."
            )
        
        return self._pretrained_model


# Future Extension Ideas:
# 
# class PretrainedModelSurrogate(GaussianProcessSurrogate):
#     """More general wrapper for any BoTorch model (not just SingleTaskGP)."""
#     
#     @classmethod 
#     def from_botorch_model(cls, model: Model, **kwargs):
#         # Handle different model types: MultiTaskGP, FixedNoiseGP, etc.
#         # Detect model capabilities and set class variables accordingly
#         pass
#
# This could support:
# - MultiTaskGP → supports_multi_output = True
# - HeteroskedasticSingleTaskGP → specialized noise handling  
# - ModelListGP → composite surrogate behavior
# - Custom BoTorch models → generic wrapping