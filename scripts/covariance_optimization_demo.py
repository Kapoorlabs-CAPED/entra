"""
Covariance Optimization Demo

Demonstrates:
1. Sample points from a uniform distribution
2. Create 5 centers for divergence-free basis
3. Iteratively minimize covariance determinant
4. Track Shannon entropy at each iteration
"""

import numpy as np

from entra import (
    CovarianceMinimizer,
    TensorBasis,
    Transformation,
    VectorSampler,
    shannon_entropy_gaussian,
)


def main():
    print("=" * 70)
    print("COVARIANCE OPTIMIZATION WITH DIVERGENCE-FREE TRANSFORMATION")
    print("=" * 70)

    D = 2
    num_points_per_dim = 20  # 10 points per dimension
    delta_x = 1  # Grid spacing in both x and y
    sigma = 5

    print("\nParameters:")
    print(f"  D (dimension)           = {D}")
    print(f"  num_points_per_dim      = {num_points_per_dim}")
    print(f"  delta_x                 = {delta_x}")
    print(f"  J (total grid points)   = {num_points_per_dim**D}")
    print(f"  sigma                   = {sigma:.4f}")

    # Step 1: Sample points from uniform distribution
    print("\n" + "-" * 70)
    print("STEP 1: Sample points from uniform distribution")
    print("-" * 70)

    sampler = VectorSampler(
        center=[0.0, 0.0],
        delta_x=delta_x,
        num_points_per_dim=num_points_per_dim,
        distribution="uniform",
    )

    eval_points = sampler.sample()
    print(f"  Points shape: {eval_points.shape}")
    print(f"  Range: [{eval_points.min():.2f}, {eval_points.max():.2f}]")

    # Initial covariance
    initial_cov = np.cov(eval_points, rowvar=False)
    initial_det = np.linalg.det(initial_cov)
    initial_entropy = shannon_entropy_gaussian(initial_cov)

    print("\n  Initial covariance matrix:")
    print(f"    [[{initial_cov[0, 0]:.6f}, {initial_cov[0, 1]:.6f}],")
    print(f"     [{initial_cov[1, 0]:.6f}, {initial_cov[1, 1]:.6f}]]")
    print(f"  Initial determinant: {initial_det:.6e}")
    print(f"  Initial entropy:     {initial_entropy:.6f} nats")

    # Step 2: Create 5 centers
    print("\n" + "-" * 70)
    print("STEP 2: Create 5 centers for basis functions")
    print("-" * 70)

    # Place centers in a pattern
    center_list = []
    for i in range(int(eval_points[:, 0].max())):
        center_list.append([i, 0.0])
        center_list.append([-i, 0.0])
        center_list.append([0.0, i])
        center_list.append([0.0, -i])
    centers = np.asarray(center_list)

    print(f"  Centers shape: {centers.shape}")
    for i, c in enumerate(centers):
        print(f"    Center {i}: ({c[0]:+.2f}, {c[1]:+.2f})")

    # Step 3: Create transformation
    print("\n" + "-" * 70)
    print("STEP 3: Create divergence-free transformation")
    print("-" * 70)

    basis = TensorBasis(centers, sigma=sigma)
    transformation = Transformation(basis)

    print(f"  TensorBasis: L={basis.L}, D={basis.D}, sigma={sigma}")
    print(f"  Transformation: {transformation.num_parameters} parameters")

    # Step 4: Optimize with custom LM showing each iteration
    print("\n" + "-" * 70)
    print("STEP 4: Iterative optimization (Levenberg-Marquardt)")
    print("-" * 70)

    minimizer = CovarianceMinimizer(transformation, eval_points)

    # Custom optimization loop to show entropy at each step
    x = transformation.get_coefficients_flat().copy()
    n_params = len(x)

    lam = 1.0  # Damping parameter
    eps = 1e-7  # For finite difference

    print(
        f"\n  {'Iter':>5}  {'Determinant':>14}  {'Entropy':>12}  {'Lambda':>10}"
    )
    print("  " + "-" * 50)

    # Initial values
    cov = minimizer.compute_covariance(x)
    det_val = np.linalg.det(cov)
    entropy = shannon_entropy_gaussian(cov)
    print(f"  {0:>5}  {det_val:>14.6e}  {entropy:>12.6f}  {lam:>10.2e}")

    max_iterations = 1000
    tolerance = 1e-8

    history = {
        "iteration": [0],
        "determinant": [det_val],
        "entropy": [entropy],
        "coefficients": [x.copy()],
    }

    for iteration in range(1, max_iterations + 1):
        # Compute residuals and Jacobian
        r = minimizer.residuals_for_lm(x)

        J_mat = np.zeros((len(r), n_params))
        for i in range(n_params):
            x_plus = x.copy()
            x_plus[i] += eps
            r_plus = minimizer.residuals_for_lm(x_plus)
            J_mat[:, i] = (r_plus - r) / eps

        # LM update
        JTJ = J_mat.T @ J_mat
        JTr = J_mat.T @ r

        try:
            delta = np.linalg.solve(JTJ + lam * np.eye(n_params), -JTr)
        except np.linalg.LinAlgError:
            delta = -JTr / (np.diag(JTJ) + lam + 1e-10)

        x_new = x + delta
        obj_new = minimizer.objective_logdet(x_new)
        obj_old = minimizer.objective_logdet(x)

        if obj_new < obj_old:
            x = x_new
            lam *= 0.1
            improvement = obj_old - obj_new

            # Compute new covariance and entropy
            cov = minimizer.compute_covariance(x)
            det_val = np.linalg.det(cov)
            entropy = shannon_entropy_gaussian(cov)

            history["iteration"].append(iteration)
            history["determinant"].append(det_val)
            history["entropy"].append(entropy)
            history["coefficients"].append(x.copy())

            print(
                f"  {iteration:>5}  {det_val:>14.6e}  {entropy:>12.6f}  {lam:>10.2e}"
            )

            if improvement < tolerance:
                print(f"\n  Converged after {iteration} iterations")
                break
        else:
            lam *= 10.0

    # Set final coefficients
    transformation.set_coefficients_flat(x)

    # Step 5: Final results
    print("\n" + "-" * 70)
    print("STEP 5: Final results")
    print("-" * 70)

    final_cov = minimizer.compute_covariance()
    final_det = np.linalg.det(final_cov)
    final_entropy = shannon_entropy_gaussian(final_cov)

    print("\n  Final covariance matrix:")
    print(f"    [[{final_cov[0, 0]:.6f}, {final_cov[0, 1]:.6f}],")
    print(f"     [{final_cov[1, 0]:.6f}, {final_cov[1, 1]:.6f}]]")

    print("\n  Comparison:")
    print(f"    {'':20} {'Initial':>14} {'Final':>14} {'Reduction':>12}")
    print(f"    {'-' * 60}")
    print(
        f"    {'Determinant':20} {initial_det:>14.6e} {final_det:>14.6e} "
        f"{initial_det / final_det:>12.2f}x"
    )
    print(
        f"    {'Entropy (nats)':20} {initial_entropy:>14.6f} {final_entropy:>14.6f} "
        f"{initial_entropy - final_entropy:>12.6f}"
    )

    print(f"\n  Optimized coefficients (L={centers.shape}, D={D}):")
    coeffs = transformation.coefficients
    for i in range(centers.shape):
        print(f"    Center {i}: [{coeffs[i, 0]:+.6f}, {coeffs[i, 1]:+.6f}]")

    print("\n" + "=" * 70)
    print("DEMO COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
