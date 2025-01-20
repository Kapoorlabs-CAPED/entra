import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def generate_k_distributions(n=5, K=4, vectors_per_class=100):
    """
    Generate K classes with different distributions.

    Parameters:
    - n: int, dimensionality of the vectors.
    - K: int, number of classes (each class corresponds to a different distribution).
    - vectors_per_class: int, number of vectors per class.

    Returns:
    - data: np.ndarray, array of shape (K * vectors_per_class, n), generated vectors.
    - labels: np.ndarray, array of shape (K * vectors_per_class,), class labels for each vector.
    """
    distributions = [
        ("Gaussian", lambda size: np.random.normal(0, 1, size)),
        ("Uniform", lambda size: np.random.uniform(-1, 1, size)),
        ("Exponential", lambda size: np.random.exponential(1, size)),
        ("Beta", lambda size: np.random.beta(2, 5, size)),
        ("Poisson", lambda size: np.random.poisson(3, size)),
    ]

    if K > len(distributions):
        raise ValueError(
            f"Only {len(distributions)} distributions are implemented, but K={K} was requested."
        )

    data = []
    labels = []
    distribution_names = []

    for class_id in range(K):
        dist_name, dist_func = distributions[class_id]
        distribution_names.append(dist_name)
        class_vectors = dist_func((vectors_per_class, n))
        data.append(class_vectors)
        labels.extend([class_id] * vectors_per_class)

    data = np.vstack(data)
    labels = np.array(labels)

    return data, labels, distribution_names


def plot_distributions_histogram(
    data, labels, distribution_names, dimensions_to_plot
):
    """
    Plots histograms for the specified dimensions.

    Parameters:
    - data: np.ndarray, dataset of shape (N, n), where N is the number of samples, and n is the dimensionality.
    - labels: np.ndarray, class labels for each sample in the dataset.
    - distribution_names: list, names of the distributions for each class.
    - dimensions_to_plot: list of integers, specifying the dimensions to include in histograms.
    """
    for dim in dimensions_to_plot:
        plt.figure(figsize=(12, 6))
        for class_id, dist_name in enumerate(distribution_names):
            sns.histplot(
                data[labels == class_id, dim],
                kde=True,
                label=f"Class {class_id}: {dist_name}",
                bins=30,
                alpha=0.6,
            )
        plt.title(f"Distribution of Dimension {dim + 1}")
        plt.xlabel(f"Dimension {dim + 1}")
        plt.ylabel("Frequency")
        plt.legend()
        plt.grid(True)
        plt.show()
