---
title: 'entra: Entropy-Conserving Transformations Using Divergence-Free Vector Fields'
tags:
  - Python
  - entropy
  - divergence-free
  - volume-preserving
  - Gaussian
  - radial basis functions
authors:
  - name: Varun Kapoor
    orcid: 0000-0000-0000-0000
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

`entra` is a Python package that transforms arbitrary probability distributions towards Gaussian form while conserving entropy. The package constructs divergence-free vector fields from Gaussian radial basis functions using the Lowitzsch operator, ensuring that all transformations are volume-preserving. By iteratively minimizing the covariance determinant of transformed point clouds, `entra` exploits the maximum entropy property of Gaussian distributions to achieve Gaussianization without altering the fundamental information content of the data.

# Statement of Need

Many statistical and machine learning methods assume Gaussian-distributed data, yet real-world distributions rarely satisfy this assumption. Existing normalization techniques such as Box-Cox transforms or quantile normalization do not preserve entropy, fundamentally altering the information content of the data. There is a need for transformation methods that can reshape distributions towards Gaussian form while maintaining their entropy—a property essential for applications in information theory, statistical mechanics, and generative modeling.

`entra` addresses this gap by providing entropy-conserving transformations based on divergence-free vector fields. The key theoretical insight is that divergence-free velocity fields generate volume-preserving flows, which in turn conserve differential entropy under transformation. Combined with the maximum entropy principle—that the Gaussian distribution maximizes entropy for a given covariance—minimizing the covariance determinant while preserving entropy drives any distribution towards Gaussian form.

The package is designed for researchers in machine learning, statistical physics, and computational biology who need to normalize data distributions without losing information. It provides a principled alternative to heuristic normalization methods and offers theoretical guarantees on entropy conservation.

# Mathematics

The transformation is based on divergence-free basis functions constructed using the Lowitzsch operator [@lowitzsch2002]:

$$\hat{O} = -I\nabla^2 + \nabla\nabla^T$$

Applied to Gaussian radial basis functions $\phi_l(x) = \exp(-\|x - c_l\|^2 / 2\sigma^2)$, this operator produces matrix-valued functions whose columns are divergence-free vector fields. The divergence-free property ensures that the Jacobian determinant of the induced transformation equals unity, guaranteeing volume preservation and thus entropy conservation.

The optimization minimizes the covariance determinant using the Levenberg-Marquardt algorithm, which adaptively adjusts step sizes without requiring a learning rate parameter.

# Example Usage

```python
import pandas as pd
from entra import DataFrameTransformer, VectorSampler

# Generate uniform distribution
sampler = VectorSampler(center=[0.0, 0.0], delta_x=1, num_points_per_dim=20)
points = sampler.sample()
df = pd.DataFrame({'x': points[:, 0], 'y': points[:, 1]})

# Transform towards Gaussian
transformer = DataFrameTransformer(sigma=5.0, max_iterations=100)
df_gaussian = transformer.fit_transform(df, columns=['x', 'y'])

# Verify entropy conservation
entropy = transformer.get_entropy_comparison(df, df_gaussian)
print(f"Original H(uniform): {entropy['original']['uniform_entropy']:.4f}")
print(f"Final H(Gaussian): {entropy['transformed']['gaussian_entropy']:.4f}")
```

# Availability

`entra` is available on PyPI (`pip install entra`) and includes interactive demonstrations via Gradio and Streamlit web applications hosted on Hugging Face Spaces. Documentation, Jupyter notebook examples, and source code are available at https://github.com/Kapoorlabs-CAPED/entra.

# Acknowledgements

We acknowledge the theoretical foundations established by Lowitzsch in the construction of divergence-free radial basis functions.

# References
