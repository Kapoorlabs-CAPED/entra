try:
    from ._version import version as __version__
except ImportError:
    __version__ = "unknown"


from .distributions import (
    generate_k_distributions,
    plot_distributions_histogram,
)

__all__ = ["generate_k_distributions", "plot_distributions_histogram"]
