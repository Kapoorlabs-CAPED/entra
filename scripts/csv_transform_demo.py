"""
CSV Transform Demo

Demonstrates how to use entra to transform real data from CSV files.

This script:
1. Generates sample data (uniform distribution) and saves to CSV
2. Reads the CSV and transforms towards Gaussian
3. Saves the transformed data to a new CSV
4. Compares entropy before and after
"""

import numpy as np
import pandas as pd

from entra import DataFrameTransformer, transform_csv


def generate_sample_data(
    n_points: int = 400, dimensions: int = 2
) -> pd.DataFrame:
    """Generate uniform sample data for demonstration."""
    # Create uniform grid
    n_per_dim = int(np.ceil(n_points ** (1 / dimensions)))

    if dimensions == 2:
        x = np.linspace(-10, 10, n_per_dim)
        y = np.linspace(-10, 10, n_per_dim)
        xx, yy = np.meshgrid(x, y)
        data = {
            "id": range(len(xx.ravel())),
            "x": xx.ravel(),
            "y": yy.ravel(),
            "label": [
                "A" if i % 2 == 0 else "B" for i in range(len(xx.ravel()))
            ],
        }
    elif dimensions == 3:
        n_per_dim = int(np.ceil(n_points ** (1 / 3)))
        x = np.linspace(-10, 10, n_per_dim)
        y = np.linspace(-10, 10, n_per_dim)
        z = np.linspace(-10, 10, n_per_dim)
        xx, yy, zz = np.meshgrid(x, y, z)
        data = {
            "id": range(len(xx.ravel())),
            "x": xx.ravel(),
            "y": yy.ravel(),
            "z": zz.ravel(),
            "label": [
                "A" if i % 2 == 0 else "B" for i in range(len(xx.ravel()))
            ],
        }
    else:
        raise ValueError("Only 2D and 3D supported in this demo")

    return pd.DataFrame(data)


def demo_dataframe_api():
    """Demonstrate the DataFrame API."""
    print("=" * 70)
    print("DEMO 1: DataFrame API")
    print("=" * 70)

    # Generate sample data
    print("\nGenerating 2D uniform sample data...")
    df = generate_sample_data(n_points=400, dimensions=2)
    print(f"  Shape: {df.shape}")
    print(f"  Columns: {list(df.columns)}")
    print("\nFirst 5 rows:")
    print(df.head())

    # Transform using DataFrame API
    print("\n" + "-" * 70)
    print("Transforming with DataFrameTransformer...")
    print("-" * 70)

    transformer = DataFrameTransformer(
        sigma=5.0,
        center_stride=1,  # Use all points as centers
        max_iterations=100,
        verbose=True,
    )

    df_transformed = transformer.fit_transform(df, columns=["x", "y"])

    # Compare entropy
    print("\n" + "-" * 70)
    print("Entropy Comparison")
    print("-" * 70)
    entropy = transformer.get_entropy_comparison(df, df_transformed)
    print(
        f"  Original k-NN entropy:    {entropy['original']['knn_entropy']:.6f}"
    )
    print(
        f"  Transformed k-NN entropy: {entropy['transformed']['knn_entropy']:.6f}"
    )
    print(
        f"  Transformed Gaussian H:   {entropy['transformed']['gaussian_entropy']:.6f}"
    )
    print(
        f"\n  Original determinant:     {entropy['original']['determinant']:.6e}"
    )
    print(
        f"  Transformed determinant:  {entropy['transformed']['determinant']:.6e}"
    )

    # Show transformed data
    print("\nTransformed data (first 5 rows):")
    print(df_transformed.head())

    # Note: other columns (id, label) are preserved
    print(
        f"\nNote: Non-coordinate columns preserved: {list(df_transformed.columns)}"
    )

    return df, df_transformed


def demo_csv_api():
    """Demonstrate the CSV file API."""
    print("\n" + "=" * 70)
    print("DEMO 2: CSV File API")
    print("=" * 70)

    # Generate and save sample data
    print("\nGenerating 3D uniform sample data...")
    df = generate_sample_data(n_points=1000, dimensions=3)

    input_path = "/tmp/sample_vectors.csv"
    output_path = "/tmp/transformed_vectors.csv"

    print(f"Saving to: {input_path}")
    df.to_csv(input_path, index=False)

    # Transform using one-liner
    print("\n" + "-" * 70)
    print("Transforming with transform_csv()...")
    print("-" * 70)

    transform_csv(
        input_path=input_path,
        output_path=output_path,
        columns=["x", "y", "z"],
        sigma=5.0,
        center_stride=1,
        verbose=True,
    )

    # Read back and show
    print(f"\nReading transformed data from: {output_path}")
    df_result = pd.read_csv(output_path)
    print(f"  Shape: {df_result.shape}")
    print("\nFirst 5 rows:")
    print(df_result.head())


def demo_large_dataset():
    """Demonstrate handling larger datasets with center_stride."""
    print("\n" + "=" * 70)
    print("DEMO 3: Large Dataset (using center_stride)")
    print("=" * 70)

    # Generate larger dataset
    print("\nGenerating larger 2D dataset (10000 points)...")
    n_per_dim = 100
    x = np.linspace(-10, 10, n_per_dim)
    y = np.linspace(-10, 10, n_per_dim)
    xx, yy = np.meshgrid(x, y)

    df = pd.DataFrame(
        {
            "x": xx.ravel(),
            "y": yy.ravel(),
        }
    )
    print(f"  Shape: {df.shape}")

    # Use center_stride to reduce computation
    # Instead of 10000 centers, use every 10th point = 1000 centers
    print("\n" + "-" * 70)
    print(
        "Transforming with center_stride=10 (1000 centers instead of 10000)..."
    )
    print("-" * 70)

    transformer = DataFrameTransformer(
        sigma=5.0,
        center_stride=10,  # Use every 10th point
        max_iterations=50,
        verbose=True,
    )

    df_transformed = transformer.fit_transform(df, columns=["x", "y"])

    entropy = transformer.get_entropy_comparison(df, df_transformed)
    print(
        f"\n  Original k-NN entropy:    {entropy['original']['knn_entropy']:.6f}"
    )
    print(
        f"  Transformed k-NN entropy: {entropy['transformed']['knn_entropy']:.6f}"
    )
    print(
        f"  Determinant reduction:    {entropy['original']['determinant'] / entropy['transformed']['determinant']:.2f}x"
    )


def main():
    demo_dataframe_api()
    demo_csv_api()
    demo_large_dataset()

    print("\n" + "=" * 70)
    print("ALL DEMOS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
