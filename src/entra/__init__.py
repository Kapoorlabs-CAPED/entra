try:
    from ._version import version as __version__
except ImportError:
    __version__ = "unknown"

from .covariance_minimizer import CovarianceMinimizer
from .effective_basis import (
    EffectiveCovarianceMinimizer,
    EffectiveTransformation,
)
from .scalar_basis import ScalarBasis
from .tensor_basis import TensorBasis
from .transformation import Transformation
from .utils import (
    divergence,
    divergence_components,
    gradient_component,
    is_divergence_free,
    shannon_entropy_gaussian,
    shannon_entropy_knn,
    shannon_entropy_uniform,
    tensor_basis_column_divergence,
    verify_divergence_free_symmetric,
    verify_tensor_basis_divergence_free,
)
from .vector_sampler import VectorSampler

__all__ = [
    "VectorSampler",
    "ScalarBasis",
    "TensorBasis",
    "Transformation",
    "CovarianceMinimizer",
    "EffectiveTransformation",
    "EffectiveCovarianceMinimizer",
    "gradient_component",
    "divergence_components",
    "divergence",
    "is_divergence_free",
    "tensor_basis_column_divergence",
    "verify_tensor_basis_divergence_free",
    "verify_divergence_free_symmetric",
    "shannon_entropy_gaussian",
    "shannon_entropy_knn",
    "shannon_entropy_uniform",
]
