---
title: 'entra: Entropy-Conserving Transformations Using Divergence-Free Vector Fields'
tags:
  - Python
  - entropy
  - divergence-free
  - volume-preserving
  - Gaussianization
  - radial basis functions
  - information theory
authors:
  - name: Varun Kapoor
    orcid: 0000-0001-5331-7966
    affiliation: "1, 2"
affiliations:
  - name: Kapoorlabs, Paris, France
    index: 1
  - name: Universität Osnabrück, Osnabrück, Germany
    index: 2
date: 27 December 2024
bibliography: paper.bib
---

# Summary

`entra` is a Python package that transforms arbitrary probability distributions towards Gaussian form while rigorously conserving entropy. The method constructs divergence-free vector fields from Gaussian radial basis functions, ensuring volume-preserving transformations that maintain the information content of the data. By iteratively minimizing the covariance determinant using the Levenberg-Marquardt algorithm, `entra` exploits the maximum entropy property of Gaussian distributions to achieve principled Gaussianization.

# Statement of Need

Many statistical methods and machine learning algorithms assume Gaussian-distributed data. Standard normalization techniques—Box-Cox transforms, quantile normalization, or z-scoring—reshape distributions but do not preserve entropy, fundamentally altering the information content. Previous Gaussianization approaches [@chen2001; @laparra2011] focus on achieving marginal Gaussianity through iterative rotations or ICA-based methods [@hyvarinen2000], but do not explicitly preserve differential entropy. In applications where information preservation is critical (thermodynamic simulations, generative modeling, information-theoretic analysis), there is a need for transformations that can normalize distributions while maintaining their entropy.

`entra` fills this gap by providing entropy-conserving transformations grounded in differential geometry and information theory. The theoretical foundation rests on two principles: (1) divergence-free vector fields generate volume-preserving flows that conserve differential entropy, and (2) among all distributions with a given covariance, the Gaussian has maximum entropy. By minimizing covariance while preserving entropy, any distribution converges towards Gaussian form.

# Theoretical Foundation

## Maximum Entropy Principle

For a D-dimensional distribution with covariance matrix $\Sigma$, the differential entropy is bounded by:

$$H(X) \leq \frac{D}{2}\left(1 + \ln(2\pi)\right) + \frac{1}{2}\ln\det(\Sigma) = H_{\text{Gaussian}}(\Sigma)$$

with equality if and only if $X$ is Gaussian [@cover2006]. This means for any distribution with entropy $H_0$, if we can reduce the covariance determinant while keeping entropy fixed at $H_0$, we reduce the Gaussian entropy bound $H_{\text{Gaussian}}(\Sigma)$ until it equals $H_0$—at which point the distribution must be Gaussian.

## Entropy Conservation via Divergence-Free Fields

Under a smooth transformation $Y = T(X)$, the differential entropy [@shannon1948; @cover2006] transforms as:

$$H(Y) = H(X) + \mathbb{E}_X\left[\ln\lvert\det(J_T(X))\rvert\right]$$

where $J_T$ is the Jacobian matrix. For entropy conservation, we require transformations where $\lvert\det(J_T)\rvert = 1$ everywhere—i.e., volume-preserving transformations.

Divergence-free vector fields $v(x)$ with $\nabla \cdot v = 0$ generate volume-preserving flows. This follows from Liouville's equation [@arnold1989]: $\frac{d}{dt}\ln\lvert\det(J)\rvert = \nabla \cdot v$. When divergence vanishes, the Jacobian determinant remains unity, guaranteeing entropy conservation.

## Divergence-Free Basis Construction

Following Lowitzsch [@lowitzsch2002], we construct divergence-free vector fields by applying the differential operator:

$$\hat{O} = -I\nabla^2 + \nabla\nabla^T$$

to Gaussian radial basis functions $\phi_l(x) = \exp(-\|x - c_l\|^2 / 2\sigma^2)$ centered at points $c_l$ [@wendland2004]. This produces matrix-valued basis functions $\Phi_l(x) = \hat{O}\phi_l(x)$ of shape $D \times D$ [@narcowich1994], where each column is a divergence-free vector field.

For Gaussian RBFs, the explicit form is:

$$\Phi_{ij}(x) = \frac{1}{\sigma^4}\left[-\delta_{ij}\left(\|x-c\|^2 - D\sigma^2\right) + (x_i-c_i)(x_j-c_j) - \delta_{ij}\sigma^2\right]\phi(x)$$

# Algorithm

The algorithm proceeds in two stages, progressively refining the transformation.

## Stage 1: Tensor Basis Optimization

Given $J$ sample points and $L$ basis function centers:

1. **Construct tensor basis**: Evaluate $\Phi_l(x)$ at all sample points, yielding a tensor of shape $(J, L, D, D)$

2. **Define parameterized transformation**:
   $$y' = y + \sum_{l=1}^{L} \Phi_l(y) \cdot c_l$$
   where $c_l \in \mathbb{R}^D$ are coefficient vectors, giving $L \times D$ total parameters

3. **Minimize covariance determinant**: Using Levenberg-Marquardt optimization [@levenberg1944; @marquardt1963], find coefficients that minimize $\det(\text{Cov}(y'))$

4. **Collapse to effective basis**: Compute $V_l = \Phi_l \cdot c_l$, reducing the $(J, L, D, D)$ tensor to $(J, L, D)$—now $L$ vector fields instead of $L \times D$ matrix fields

## Stage 2: Iterative Refinement

After Stage 1, the tensor basis $(J, L, D, D)$ with optimized coefficients $(L, D)$ collapses to an **effective basis** $(J, L, D)$ representing $L$ learned vector fields. This basis captures the displacement directions discovered during Stage 1.

Each outer iteration then:

1. **Define scalar-weighted transformation**:
   $$y' = y + \sum_{l=1}^{L} \alpha_l V_l$$
   with only $L$ scalar coefficients $\alpha_l$

2. **Optimize**: Minimize $\det(\text{Cov}(y'))$ over the $L$ scalars using Levenberg-Marquardt (up to 1000 iterations)

3. **Update basis**: $V_l \leftarrow \alpha_l V_l$ (absorb coefficients into basis, preserving learned directions)

4. **Transform points**: $y \leftarrow y'$

5. **Repeat** for multiple outer rounds (typically 5), each refining the point positions using the progressively improved basis

This iterative refinement is crucial: each outer round operates on the transformed points from the previous round, allowing the algorithm to make incremental adjustments that compound across rounds. The effective basis adapts at each round, capturing increasingly fine-grained displacement patterns that further reduce the covariance determinant.

The Levenberg-Marquardt algorithm adaptively adjusts its damping parameter, eliminating the need for learning rate tuning. This makes the optimization robust across different data scales and distributions.

## Computational Complexity

- Tensor basis evaluation: $O(J \times L \times D^2)$
- Covariance computation per iteration: $O(J \times D^2)$
- LM optimization: Typically 10-100 iterations per outer loop

# Implementation

`entra` provides a high-level `DataFrameTransformer` API for pandas DataFrames and lower-level classes for custom pipelines:

```python
from entra import DataFrameTransformer, VectorSampler
import pandas as pd

# Generate 2D uniform distribution (400 points)
sampler = VectorSampler(center=[0.0, 0.0], delta_x=1, num_points_per_dim=20)
points = sampler.sample()
df = pd.DataFrame({'x': points[:, 0], 'y': points[:, 1]})

# Transform towards Gaussian
transformer = DataFrameTransformer(sigma=5.0, max_iterations=100)
df_transformed = transformer.fit_transform(df, columns=['x', 'y'])

# Verify entropy conservation
entropy = transformer.get_entropy_comparison(df, df_transformed)
print(f"Target H(uniform): {entropy['original']['uniform_entropy']:.4f} nats")
print(f"Final H(Gaussian): {entropy['transformed']['gaussian_entropy']:.4f} nats")
```

The package also provides:

- `VectorSampler`: Generate uniform or Gaussian point distributions
- `TensorBasis`: Evaluate divergence-free tensor basis functions
- `EffectiveBasis`: Manage collapsed vector field basis
- `CovarianceMinimizer`: Low-level LM optimization interface
- Utility functions for divergence verification

# Parameter Tuning

The RBF width parameter $\sigma$ controls the spatial scale of the divergence-free basis functions and significantly affects optimization performance. Too small a $\sigma$ produces basis functions that are too localized to create meaningful global displacements; too large a $\sigma$ produces nearly constant fields with negligible gradients.

## Hydra Configuration

`entra` uses Hydra for configuration management, enabling systematic parameter sweeps:

```yaml
# conf/scenario_entra.yaml
defaults:
  - entra_base
  - sweep: fine

sampling:
  dimension: 2
  num_points_per_dim: 20

stage1:
  max_iterations: 1000

stage2:
  n_outer: 5
  max_iterations: 1000
```

Sweep configurations define sigma ranges to test:

```yaml
# conf/sweep/fine.yaml
enabled: true
sigmas: [4.0, 4.5, 5.0, 5.5, 6.0, 6.5]
```

## Sigma Sweep Results

The optimal $\sigma$ minimizes the absolute gap between the final Gaussian entropy and the target uniform entropy. Table 1 shows results for a 2D uniform distribution with 400 points ($H_{\text{uniform}} = 5.889$ nats, initial determinant $= 1.111 \times 10^3$).

**Table 1.** Sigma sweep results for 2D uniform distribution ($20 \times 20$ grid).

| $\sigma$ | Final Det | $H_{\text{Gaussian}}$ | Gap (nats) | Det Reduction |
|----------|-----------|----------------------|------------|---------------|
| 2.0 | $1.09 \times 10^3$ | 6.337 | +0.448 | 1.02$\times$ |
| 3.0 | $9.92 \times 10^2$ | 6.288 | +0.399 | 1.12$\times$ |
| **4.0** | $4.32 \times 10^2$ | **5.872** | **-0.017** | 2.57$\times$ |
| 5.0 | $3.36 \times 10^2$ | 5.746 | -0.143 | 3.31$\times$ |
| 7.0 | $1.10 \times 10^2$ | 5.188 | -0.701 | 10.10$\times$ |
| 10.0 | $3.11 \times 10^2$ | 5.708 | -0.180 | 3.57$\times$ |

The optimal $\sigma = 4.0$ achieves a gap of only $-0.017$ nats, confirming successful entropy-conserving Gaussianization. Smaller $\sigma$ values (2.0, 3.0) produce basis functions too localized to effectively reduce the covariance, resulting in positive gaps (under-compression). Larger $\sigma$ values (7.0) over-compress the distribution, yielding large negative gaps. The non-monotonic behavior at $\sigma = 10.0$ reflects the interplay between basis function overlap and optimization dynamics.

# Validation

Entropy conservation is validated by comparing the analytical entropy of the initial uniform distribution $H_{\text{uniform}} = \ln(V)$ with the Gaussian entropy of the transformed distribution $H_{\text{Gaussian}} = \frac{D}{2}(1 + \ln 2\pi) + \frac{1}{2}\ln\det(\Sigma)$ [@cover2006]. As the transformation converges, these values approach equality, confirming both entropy conservation and successful Gaussianization. For non-uniform initial distributions, k-nearest-neighbor entropy estimators [@kozachenko1987] can be used to verify conservation empirically.

Additionally, the package includes functions to numerically verify that all basis function columns satisfy the divergence-free condition $\nabla \cdot V \approx 0$.

# Availability

`entra` is available via PyPI (`pip install entra`) and includes:

- Interactive web demos (Gradio and Streamlit) on Hugging Face Spaces
- Jupyter notebook examples for 2D and 3D transformations
- Comprehensive API documentation

Source code: https://github.com/Kapoorlabs-CAPED/entra

# Acknowledgements

We acknowledge the foundational work of Lowitzsch on divergence-free radial basis functions.

# References
