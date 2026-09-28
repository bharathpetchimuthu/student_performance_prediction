import pandas as pd
import numpy as np


def clean_dataframe(df):

    df = df.copy()

    # Remove empty rows
    df = df.dropna(
        axis=0,
        how="all"
    )

    # Remove empty columns
    df = df.dropna(
        axis=1,
        how="all"
    )

    # Remove duplicate rows
    df = df.drop_duplicates()

    # Clean column names
    df.columns = [
        str(column)
        .replace("\ufeff", "")
        .strip()
        for column in df.columns
    ]

    # Replace common missing values
    df = df.replace(
        [
            "",
            "NA",
            "N/A",
            "na",
            "null",
            "NULL",
            "None"
        ],
        np.nan
    )

    return df