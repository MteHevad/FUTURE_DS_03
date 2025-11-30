"""
visualizations.py

Module that creates all essential visualizations:
- Rating item histograms
- Correlation heatmap
- PCA scatter plot
- Cluster visualization
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pca_analysis import apply_pca


def plot_histograms(df: pd.DataFrame):
    """Histogram for each rating item."""
    rating_cols = [c for c in df.columns if c not in ["Student ID", "Overall_Score", "Cluster"]]
    df[rating_cols].hist(bins=10, figsize=(14, 10))
    plt.suptitle("Rating Item Distributions")
    plt.show()


def plot_correlation_heatmap(df: pd.DataFrame):
    """Plot heatmap of correlations across rating items."""
    rating_cols = [c for c in df.columns if c not in ["Student ID", "Overall_Score", "Cluster"]]
    corr = df[rating_cols].corr()

    plt.figure(figsize=(10, 7))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Heatmap")
    plt.show()


def plot_pca(df: pd.DataFrame):
    """Plot PCA scatter (PC1 vs PC2)."""
    components, variance, _ = apply_pca(df)

    plt.scatter(components[:, 0], components[:, 1], alpha=0.6)
    plt.xlabel(f"PC1 ({variance[0]*100:.1f}%)")
    plt.ylabel(f"PC2 ({variance[1]*100:.1f}%)")
    plt.title("PCA Scatter Plot")
    plt.show()


def plot_clusters(df: pd.DataFrame):
    """Cluster visualization in PCA space."""
    components, _, _ = apply_pca(df)

    plt.scatter(components[:, 0], components[:, 1], c=df["Cluster"], cmap="viridis")
    plt.title("Cluster Visualization (PCA Space)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.show()


if __name__ == "__main__":
    df = pd.read_csv("../data/student_feedback_processed.csv")
    plot_histograms(df)
    plot_correlation_heatmap(df)
    plot_pca(df)
    if "Cluster" in df.columns:
        plot_clusters(df)
