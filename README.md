# entra

[![License BSD-3](https://img.shields.io/pypi/l/entra.svg?color=green)](https://github.com/Kapoorlabs-CAPED/entra/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/entra.svg?color=green)](https://pypi.org/project/entra)
[![Python Version](https://img.shields.io/pypi/pyversions/entra.svg?color=green)](https://python.org)
[![tests](https://github.com/Kapoorlabs-CAPED/entra/workflows/tests/badge.svg)](https://github.com/Kapoorlabs-CAPED/entra/actions)
[![codecov](https://codecov.io/gh/Kapoorlabs-CAPED/entra/branch/main/graph/badge.svg)](https://codecov.io/gh/Kapoorlabs-CAPED/entra)

Divergence-free tensor basis functions for incompressible vector field representation and entropy-conserving transformations.

----------------------------------

## Overview

This package implements a method for constructing **divergence-free vector basis functions** by applying a differential operator to Gaussian radial basis functions (RBFs). The resulting basis functions are ideal for representing incompressible vector fields and entropy-conserving transformations.

## Theoretical Foundation

### Maximum Entropy Principle

A fundamental theorem in information theory states that **among all distributions with a given covariance matrix, the Gaussian distribution has maximum entropy**. This package exploits a corollary of this theorem: by minimizing the covariance determinant of a point distribution using volume-preserving transformations, we can transform any distribution towards a Gaussian.

### Why Divergence-Free?

Divergence-free vector fields are **volume-preserving** (incompressible). When we use divergence-free basis functions to define a transformation:

- The Jacobian determinant equals 1 everywhere
- Total volume is conserved
- **Entropy is conserved** under the transformation

This allows us to iteratively minimize the covariance determinant while preserving the fundamental entropy of the distribution, effectively reshaping any distribution into a Gaussian form.

### The Divergence-Free Operator

The construction of divergence-free vector fields from scalar RBFs uses the differential operator discovered by Lowitzsch [1]:

**Ô = -I∇² + ∇∇ᵀ**

When applied to a scalar function φ(x), this operator produces a D×D matrix where each column is a divergence-free vector field.

## References

[1] S. Lowitzsch, *Approximation and Interpolation Employing Divergence-Free Radial Basis Functions With Applications*, PhD thesis, Department of Mathematics, Texas A&M University, 2002.

## Algorithm Flowchart

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         DIVERGENCE-FREE BASIS PIPELINE                       │
└─────────────────────────────────────────────────────────────────────────────┘

    ┌─────────────────────────────────────────┐
    │  INPUT PARAMETERS                       │
    │  • D: dimension                         │
    │  • δx: grid spacing                     │
    │  • n: points per dimension              │
    │  • σ = 0.7 × δx: RBF width             │
    └────────────────────┬────────────────────┘
                         │
                         ▼
    ╔═════════════════════════════════════════╗
    ║  STEP 1: GRID SAMPLING                  ║
    ║  VectorSampler                          ║
    ╠═════════════════════════════════════════╣
    ║  Sample J = n^D points on regular grid  ║
    ║                                         ║
    ║  x_{i} = x_center + i × δx              ║
    ║                                         ║
    ║  Output: eval_points (J, D)             ║
    ╚════════════════════┬════════════════════╝
                         │
                         ▼
    ╔═════════════════════════════════════════╗
    ║  STEP 2: CHOOSE L CENTERS               ║
    ╠═════════════════════════════════════════╣
    ║  Select L center points c_1, ..., c_L   ║
    ║  for basis functions                    ║
    ║                                         ║
    ║  Output: centers (L, D)                 ║
    ╚════════════════════┬════════════════════╝
                         │
                         ▼
    ╔═════════════════════════════════════════╗
    ║  STEP 3: SCALAR BASIS FUNCTIONS         ║
    ║  ScalarBasis                            ║
    ╠═════════════════════════════════════════╣
    ║  Gaussian RBF for each center:          ║
    ║                                         ║
    ║  φ_l(x) = exp(-||x - c_l||² / 2σ²)     ║
    ║                                         ║
    ║  Output: φ values (J, L)                ║
    ╚════════════════════┬════════════════════╝
                         │
                         ▼
    ╔═════════════════════════════════════════╗
    ║  STEP 4: APPLY TENSOR OPERATOR          ║
    ║  TensorBasis                            ║
    ╠═════════════════════════════════════════╣
    ║  Apply Ô = -I∇² + ∇∇ᵀ to each φ_l      ║
    ║                                         ║
    ║  Φ_l(x) = Ô φ_l(x)                      ║
    ║                                         ║
    ║  Output: Φ values (J, L, D, D)          ║
    ║          ───────────────────            ║
    ║          D×D matrix at each point       ║
    ╚════════════════════┬════════════════════╝
                         │
                         ▼
    ╔═════════════════════════════════════════╗
    ║  STEP 5: EXTRACT COLUMN VECTOR FIELDS   ║
    ╠═════════════════════════════════════════╣
    ║  Each column d of Φ is a vector field:  ║
    ║                                         ║
    ║  V_d = Φ[:, :, d]  shape: (J, D)        ║
    ║                                         ║
    ║  For D=2:                               ║
    ║  • Column 0: V_0 = [Φ_00, Φ_10]ᵀ       ║
    ║  • Column 1: V_1 = [Φ_01, Φ_11]ᵀ       ║
    ╚════════════════════┬════════════════════╝
                         │
                         ▼
    ╔═════════════════════════════════════════╗
    ║  STEP 6: VERIFY DIVERGENCE-FREE         ║
    ╠═════════════════════════════════════════╣
    ║  Compute divergence of each column:     ║
    ║                                         ║
    ║  div(V_d) = ∂V_{d,x}/∂x + ∂V_{d,y}/∂y  ║
    ║                                         ║
    ║  Each column should satisfy:            ║
    ║  div(V_d) ≈ 0                           ║
    ╚════════════════════┬════════════════════╝
                         │
                         ▼
    ┌─────────────────────────────────────────┐
    │  OUTPUT                                 │
    │  • D divergence-free vector fields      │
    │    per center (L × D total)             │
    │  • Each field has J discrete values     │
    └─────────────────────────────────────────┘
```

## Mathematical Details

For the complete mathematical formulation including:
- Grid sampling formulas
- Gaussian RBF definitions
- Tensor operator derivation
- Proof of divergence-free property
- Discrete divergence computation

See **[docs/equations.rst](docs/equations.rst)**

## Quick Start

```python
import numpy as np
from entra import VectorSampler, TensorBasis, verify_tensor_basis_divergence_free

# Step 1: Create evaluation grid (10×10 = 100 points)
sampler = VectorSampler(
    center=[0.0, 0.0],
    delta_x=0.1,
    num_points_per_dim=10,
    distribution="uniform"
)
eval_points = sampler.sample()  # (100, 2)

# Step 2: Define centers
centers = np.array([[0.0, 0.0]])  # L=1 center

# Step 3 & 4: Create tensor basis and evaluate
sigma = 0.7 * 0.1  # 0.7 × delta_x
basis = TensorBasis(centers, sigma=sigma)
Phi = basis.evaluate(eval_points)  # (100, 1, 2, 2)

# Step 5: Extract columns as vector fields
V0 = Phi[:, 0, :, 0]  # Column 0: (100, 2)
V1 = Phi[:, 0, :, 1]  # Column 1: (100, 2)

# Step 6: Verify divergence-free
all_free, rel_divs = verify_tensor_basis_divergence_free(
    Phi, dx=0.1, grid_shape=(10, 10)
)
print(f"All columns divergence-free: {all_free}")
```

## API Reference

### Classes

| Class | Description |
|-------|-------------|
| `VectorSampler` | Sample points on a regular D-dimensional grid |
| `ScalarBasis` | Gaussian RBF scalar basis functions |
| `TensorBasis` | Apply tensor operator to produce D×D matrix basis |

### Functions

| Function | Description |
|----------|-------------|
| `divergence(V, dx, grid_shape)` | Compute divergence of vector field (signed sum) |
| `divergence_components(V, dx, grid_shape)` | Get individual ∂V_d/∂x_d terms (signed) |
| `gradient_component(f, dx, axis, grid_shape)` | Compute ∂f/∂x_axis for scalar field |
| `is_divergence_free(V, dx, grid_shape)` | Check if field is divergence-free |
| `tensor_basis_column_divergence(Phi, dx, grid_shape)` | Divergence of all tensor basis columns |
| `verify_tensor_basis_divergence_free(Phi, dx, grid_shape)` | Verify all columns are divergence-free |

**Note:** All divergence computations preserve signs. Use `divergence_components()` to see how
individual terms (which may be positive or negative) sum to give the total divergence.

## Shape Conventions

| Symbol | Meaning |
|--------|---------|
| D | Dimension of space |
| J | Number of evaluation points (grid points) |
| L | Number of basis function centers |

| Array | Shape | Description |
|-------|-------|-------------|
| `eval_points` | (J, D) | Evaluation grid points |
| `centers` | (L, D) | Basis function centers |
| `ScalarBasis.evaluate()` | (J, L) | Scalar basis values |
| `TensorBasis.evaluate()` | (J, L, D, D) | Tensor basis matrices |
| `divergence()` | (J,) | Divergence at each point |

## Examples

See the `notebooks/` directory for Jupyter notebook examples:
- `2d_vector_field_example.ipynb` - 2D visualization of tensor basis columns

See the `scripts/` directory for standalone demos:
- `basis_pipeline_demo.py` - Full pipeline with shape verification

## Installation

You can install `entra` via [pip]:

    pip install entra

To install latest development version:

    pip install git+https://github.com/Kapoorlabs-CAPED/entra.git

## Contributing

Contributions are very welcome. Tests can be run with [tox], please ensure
the coverage at least stays the same before you submit a pull request.

## License

Distributed under the terms of the [BSD-3] license,
"entra" is free and open source software

## Issues

If you encounter any problems, please [file an issue] along with a detailed description.

[pip]: https://pypi.org/project/pip/
[caped]: https://github.com/Kapoorlabs-CAPED
[Cookiecutter]: https://github.com/audreyr/cookiecutter
[@caped]: https://github.com/Kapoorlabs-CAPED
[MIT]: http://opensource.org/licenses/MIT
[BSD-3]: http://opensource.org/licenses/BSD-3-Clause
[GNU GPL v3.0]: http://www.gnu.org/licenses/gpl-3.0.txt
[GNU LGPL v3.0]: http://www.gnu.org/licenses/lgpl-3.0.txt
[Apache Software License 2.0]: http://www.apache.org/licenses/LICENSE-2.0
[Mozilla Public License 2.0]: https://www.mozilla.org/media/MPL/2.0/index.txt
[cookiecutter-template]: https://github.com/Kapoorlabs-CAPED/cookiecutter-template
[file an issue]: https://github.com/Kapoorlabs-CAPED/entra/issues
[tox]: https://tox.readthedocs.io/en/latest/
[PyPI]: https://pypi.org/
