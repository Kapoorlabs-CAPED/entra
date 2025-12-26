try:
    from ._version import version as __version__
except ImportError:
    __version__ = "unknown"

from .scalar_basis import ScalarBasis
from .tensor_basis import TensorBasis
from .utils import (
    divergence,
    divergence_components,
    gradient_component,
    is_divergence_free,
    tensor_basis_column_divergence,
    verify_divergence_free_symmetric,
    verify_tensor_basis_divergence_free,
)
from .vector_sampler import VectorSampler

__all__ = [
    "VectorSampler",
    "ScalarBasis",
    "TensorBasis",
    "gradient_component",
    "divergence_components",
    "divergence",
    "is_divergence_free",
    "tensor_basis_column_divergence",
    "verify_tensor_basis_divergence_free",
    "verify_divergence_free_symmetric",
]
