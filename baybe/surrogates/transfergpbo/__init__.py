"""Transfer Learning Surrogates from Tighineanu et al. (2022)."""

# from baybe.surrogates.transfergpbo.hgp import HGPGaussianProcessSurrogate
from baybe.surrogates.transfergpbo.mhgp import MHGPGaussianProcessSurrogate
from baybe.surrogates.transfergpbo.shgp import SHGPGaussianProcessSurrogate

__all__ = [
    "MHGPGaussianProcessSurrogate",
    "SHGPGaussianProcessSurrogate",
]
