"""
Exploratory Data Analysis
for Student Academic Performance
Prediction System
"""

import os

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from preprocessing import (
    load_dataset,
    create_target,
    handle_missing_values,
    handle_duplicates,
    detect_outliers
)


OUTPUT_DIR = "eda_output"


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

def create_output_directory():

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )


# ============================================================
# PASS FAIL
# ============================================================

def pass_fail_distribution(df, y):

    pass_count = int(
        (y == 1).sum()
    )

    fail_count = int(
        (y == 0).sum()
    )

    print(
        f"PASS: {pass_count}"
    )

    print(
        f"FAIL: {fail_count}"
    )

    plt.figure(
        figsize=(8, 5)
    )

    plt.bar(
        ["PASS", "FAIL"],
        [pass_count, fail_count]
    )

    plt.title(
        "Student Pass / Fail Distribution"
    )

    plt.xlabel(
        "Performance"
    )

    plt.ylabel(
        "Number of Students"
    )

    plt.tight_layout()

    plt.savefig(
        f"{OUTPUT_DIR}/01_pass_fail_distribution.png",
        dpi=300
    )

    plt.close()


# ============================================================
# AGE
# ============================================================

def age_distribution(df):

    plt.figure(
        figsize=(10, 5)
    )

    plt.hist(
        df["age"],
        bins=15,
        edgecolor="black"
    )

    plt.title(
        "Student Age Distribution"
    )

    plt.xlabel(
        "Age"
    )

    plt.ylabel(
        "Number of Students"
    )

    plt.tight_layout()

    plt.savefig(
        f"{OUTPUT_DIR}/02_age_distribution.png",
        dpi=300
    )

    plt.close()


# ============================================================
# ABSENCES
# ============================================================

def absence_boxplot(df):

    plt.figure(
        figsize=(8, 5)
    )

    plt.boxplot(
        df["absences"]
    )

    plt.title(
        "Student Absence Distribution"
    )

    plt.ylabel(
        "Number of Absences"
    )

    plt.tight_layout()

    plt.savefig(
        f"{OUTPUT_DIR}/03_absence_boxplot.png",
        dpi=300
    )

    plt.close()


# ============================================================
# STUDY TIME
# ============================================================

def studytime_performance(df, y):

    plt.figure(
        figsize=(10, 6)
    )

    plt.scatter(
        df.loc[y == 1, "studytime"],
        df.loc[y == 1].index,
        label="PASS",
        alpha=0.7
    )

    plt.scatter(
        df.loc[y == 0, "studytime"],
        df.loc[y == 0].index,
        label="FAIL",
        alpha=0.7
    )

    plt.xlabel(
        "Study Time"
    )

    plt.ylabel(
        "Student Index"
    )

    plt.title(
        "Study Time vs Academic Performance"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        f"{OUTPUT_DIR}/04_studytime_vs_performance.png",
        dpi=300
    )

    plt.close()


# ============================================================
# AVERAGE GRADE
# ============================================================

def average_grade_by_studytime(df):

    grouped = (
        df.groupby("studytime")["G3"]
        .mean()
    )

    plt.figure(
        figsize=(10, 5)
    )

    plt.plot(
        grouped.index,
        grouped.values,
        marker="o"
    )

    plt.xlabel(
        "Study Time"
    )

    plt.ylabel(
        "Average Final Grade G3"
    )

    plt.title(
        "Average Grade by Study Time"
    )

    plt.grid(
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        f"{OUTPUT_DIR}/05_average_grade_by_studytime.png",
        dpi=300
    )

    plt.close()


# ============================================================
# CORRELATION
# ============================================================

def correlation_heatmap(df):

    numeric_df = df.select_dtypes(
        include=[np.number]
    )

    correlation = numeric_df.corr()

    plt.figure(
        figsize=(14, 10)
    )

    sns.heatmap(
        correlation,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0
    )

    plt.title(
        "Correlation Heatmap"
    )

    plt.tight_layout()

    plt.savefig(
        f"{OUTPUT_DIR}/06_correlation_heatmap.png",
        dpi=300
    )

    plt.close()


# ============================================================
# MAIN EDA
# ============================================================

def main():

    print("\n")
    print("=" * 60)

    print(
        "EXPLORATORY DATA ANALYSIS"
    )

    print("=" * 60)


    create_output_directory()


    # Load
    df = load_dataset()


    # Clean
    df = handle_missing_values(df)

    df = handle_duplicates(df)


    # Target
    y = create_target(df)


    # Overview

    print("\nDataset Shape:")

    print(
        df.shape
    )

    print("\nData Types:")

    print(
        df.dtypes
    )

    print("\nFirst 5 rows:")

    print(
        df.head()
    )

    print("\nDescriptive Statistics:")

    print(
        df.describe()
    )


    # Missing values

    print("\nMissing Values:")

    print(
        df.isnull().sum().sum()
    )


    # Duplicate

    print("\nDuplicates:")

    print(
        df.duplicated().sum()
    )


    # Outliers

    detect_outliers(df)


    # Charts

    print(
        "\nGenerating visualizations..."
    )

    pass_fail_distribution(
        df,
        y
    )

    age_distribution(
        df
    )

    absence_boxplot(
        df
    )

    studytime_performance(
        df,
        y
    )

    average_grade_by_studytime(
        df
    )

    correlation_heatmap(
        df
    )


    print("\n")
    print("=" * 60)

    print(
        "EDA COMPLETED"
    )

    print("=" * 60)

    print(
        "\nCharts saved in:"
    )

    print(
        OUTPUT_DIR
    )


if __name__ == "__main__":

    main()