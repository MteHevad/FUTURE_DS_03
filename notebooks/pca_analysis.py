"""
pca_analysis.py

Module for performing PCA analysis on the cleaned student feedback dataset.
Generates:
- Scaled rating matrix
- PCA components
- Explained variance ratios
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def prepare_data_for_pca(df: pd.DataFrame):
    """Extract rating columns and standardize them."""
    rating_cols = [c for c in df.columns if c not in ["Student ID", "Overall_Score"]]
    scaler = StandardScaler()
    scaled = scaler.fit_transform(df[rating_cols])
    return scaled, rating_cols


def apply_pca(df: pd.DataFrame, n_components: int = 2):
    """Apply PCA and return components + explained variance."""
    scaled, rating_cols = prepare_data_for_pca(df)

    pca = PCA(n_components=n_components)
    components = pca.fit_transform(scaled)
    variance = pca.explained_variance_ratio_

    return components, variance, rating_cols


if __name__ == "__main__":
    df = pd.read_csv("../data/student_feedback_processed.csv")
    components, variance, cols = apply_pca(df)
    print("Explained variance:", variance)
