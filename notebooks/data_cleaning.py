"""
data_cleaning.py

Module for cleaning and preprocessing the student feedback dataset.
Includes:
- Removing redundant columns
- Converting rating fields to numeric
- Handling missing values
- Creating an overall satisfaction score
"""

import pandas as pd
import numpy as np


def load_data(path: str) -> pd.DataFrame:
    """Loads CSV dataset from the given path."""
    df = pd.read_csv(path)
    return df


def remove_redundant_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Remove unnecessary export-index columns like 'Unnamed: 0'."""
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
    return df


def convert_rating_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Convert all rating columns to numeric type."""
    rating_cols = [c for c in df.columns if c != "Student ID"]
    df[rating_cols] = df[rating_cols].apply(pd.to_numeric, errors="coerce")
    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Handle missing values:
       - Drop rows with >50% ratings missing
       - Impute remaining NaNs with row mean
    """
    rating_cols = [c for c in df.columns if c != "Student ID"]

    row_missing = df[rating_cols].isnull().sum(axis=1)
    threshold = len(rating_cols) // 2

    # Drop rows with excessive missing
    df = df[row_missing <= threshold].copy()

    # Row-mean imputation
    df[rating_cols] = df[rating_cols].apply(lambda row: row.fillna(row.mean()), axis=1)

    return df


def create_overall_score(df: pd.DataFrame) -> pd.DataFrame:
    """Creates a composite score as the mean of all rating items."""
    rating_cols = [c for c in df.columns if c != "Student ID"]
    df["Overall_Score"] = df[rating_cols].mean(axis=1)
    return df


def clean_data(path: str) -> pd.DataFrame:
    """Full cleaning pipeline for external use."""
    df = load_data(path)
    df = remove_redundant_columns(df)
    df = convert_rating_columns(df)
    df = handle_missing_values(df)
    df = create_overall_score(df)
    return df


if __name__ == "__main__":
    cleaned = clean_data("../data/student_feedback_raw.csv")
    cleaned.to_csv("../data/student_feedback_processed.csv", index=False)
    print("Data cleaned successfully and saved to data/student_feedback_processed.csv")
