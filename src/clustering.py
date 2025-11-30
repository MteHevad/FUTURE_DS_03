"""
clustering.py

Performs KMeans clustering on PCA-transformed data.
Includes:
- Elbow method to determine optimal k
- Final clustering assignment
- Cluster profile generation
"""

import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from pca_analysis import apply_pca


def elbow_method(df, max_k: int = 7):
    """Plot inertia values for k=1..max_k to determine elbow."""
    scaled_components, _, _ = apply_pca(df)
    inertia = []

    for k in range(1, max_k + 1):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(scaled_components)
        inertia.append(km.inertia_)

    plt.plot(range(1, max_k + 1), inertia, "-o")
    plt.xlabel("k")
    plt.ylabel("Inertia")
    plt.title("Elbow Method")
    plt.show()


def apply_kmeans(df: pd.DataFrame, k: int = 3):
    """Apply KMeans clustering with selected k and return updated dataframe."""
    components, _, _ = apply_pca(df)

    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    df["Cluster"] = km.fit_predict(components)
    return df


def cluster_profiles(df: pd.DataFrame):
    """Return mean profile per cluster."""
    rating_cols = [c for c in df.columns if c not in ["Student ID", "Cluster"]]
    return df.groupby("Cluster")[rating_cols].mean()


if __name__ == "__main__":
    df = pd.read_csv("../data/student_feedback_processed.csv")
    elbow_method(df)
    df = apply_kmeans(df, k=3)
    print(cluster_profiles(df))
