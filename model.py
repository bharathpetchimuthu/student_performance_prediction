# ============================================================
# STUDENT ACADEMIC PERFORMANCE PREDICTION
# model.py
# ============================================================

import os
import re
from io import StringIO

import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder
)

from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


# ============================================================
# FOLDERS
# ============================================================

MODEL_FOLDER = "models"

OUTPUT_FOLDER = "eda_output"

os.makedirs(
    MODEL_FOLDER,
    exist_ok=True
)

os.makedirs(
    OUTPUT_FOLDER,
    exist_ok=True
)

MODEL_PATH = os.path.join(
    MODEL_FOLDER,
    "pass_fail_model.pkl"
)


# ============================================================
# NORMALIZE COLUMN
# ============================================================

def normalize_column_name(name):

    return re.sub(
        r"[^a-z0-9]",
        "",
        str(name).lower()
    )


# ============================================================
# CLEAN DATAFRAME
# ============================================================

def repair_dataframe(df):

    if df is None:

        raise ValueError(
            "Dataset is empty."
        )

    df = df.copy()

    # Remove completely empty rows
    df = df.dropna(
        axis=0,
        how="all"
    )

    # Remove completely empty columns
    df = df.dropna(
        axis=1,
        how="all"
    )

    # Clean column names
    df.columns = [
        str(column).strip()
        for column in df.columns
    ]

    # --------------------------------------------------------
    # Repair semicolon CSV
    # --------------------------------------------------------

    if len(df.columns) == 1:

        first_column = str(
            df.columns[0]
        )

        if ";" in first_column:

            try:

                csv_text = (
                    first_column
                    + "\n"
                    + "\n".join(
                        str(value)
                        for value in df.iloc[:, 0]
                    )
                )

                repaired = pd.read_csv(
                    StringIO(csv_text),
                    sep=";"
                )

                if len(repaired.columns) > 1:

                    df = repaired

            except Exception:

                pass

    # Remove duplicate rows
    df = df.drop_duplicates()

    # Clean names again
    df.columns = [
        str(column).strip()
        for column in df.columns
    ]

    return df


# ============================================================
# FIND COLUMN
# ============================================================

def find_column(
    df,
    possible_names
):

    normalized_columns = {

        normalize_column_name(column):
            column

        for column in df.columns
    }

    for name in possible_names:

        normalized = normalize_column_name(
            name
        )

        if normalized in normalized_columns:

            return normalized_columns[
                normalized
            ]

    return None


# ============================================================
# CONVERT TARGET
#
# For G3:
#
# 10 - 20 = PASS
# 0  - 9  = FAIL
# ============================================================

def convert_target(
    series,
    column_name=""
):

    original = series.copy()

    numeric = pd.to_numeric(
        original,
        errors="coerce"
    )

    valid_numeric = numeric.dropna()

    # --------------------------------------------------------
    # Numeric target
    # --------------------------------------------------------

    if len(valid_numeric) > 0:

        unique_values = sorted(
            valid_numeric.unique()
        )

        # 0 / 1
        if set(unique_values).issubset(
            {0, 1}
        ):

            return numeric.map(

                lambda value:

                np.nan
                if pd.isna(value)
                else int(value)
            )

        normalized_name = normalize_column_name(
            column_name
        )

        # ----------------------------------------------------
        # G3 / marks / score / result
        # ----------------------------------------------------

        if (

            normalized_name == "g3"

            or "mark" in normalized_name

            or "score" in normalized_name

            or "grade" in normalized_name

            or "result" in normalized_name

            or "performance" in normalized_name

            or "final" in normalized_name

        ):

            maximum = float(
                valid_numeric.max()
            )

            # Out of 20
            if maximum <= 20:

                return numeric.map(

                    lambda value:

                    np.nan
                    if pd.isna(value)
                    else int(value >= 10)
                )

            # Out of 100
            if maximum <= 100:

                return numeric.map(

                    lambda value:

                    np.nan
                    if pd.isna(value)
                    else int(value >= 50)
                )

        # ----------------------------------------------------
        # Two numeric classes
        # ----------------------------------------------------

        if len(unique_values) == 2:

            second_value = unique_values[1]

            return numeric.map(

                lambda value:

                np.nan
                if pd.isna(value)
                else int(
                    value == second_value
                )
            )

        # ----------------------------------------------------
        # Generic numeric target
        # ----------------------------------------------------

        threshold = float(
            valid_numeric.median()
        )

        return numeric.map(

            lambda value:

            np.nan
            if pd.isna(value)
            else int(
                value >= threshold
            )
        )

    # --------------------------------------------------------
    # Text target
    # --------------------------------------------------------

    text = (
        original
        .astype(str)
        .str.strip()
        .str.lower()
    )

    pass_words = {
        "pass",
        "passed",
        "p",
        "yes",
        "y",
        "true",
        "good",
        "excellent",
        "high",
        "success",
        "successful"
    }

    fail_words = {
        "fail",
        "failed",
        "f",
        "no",
        "n",
        "false",
        "poor",
        "low",
        "failure",
        "unsuccessful"
    }

    output = []

    for value in text:

        if value in pass_words:

            output.append(1)

        elif value in fail_words:

            output.append(0)

        else:

            output.append(np.nan)

    result = pd.Series(
        output,
        index=series.index
    )

    # If PASS/FAIL was detected
    if result.notna().sum() > 0:

        return result

    # --------------------------------------------------------
    # Generic two text classes
    # --------------------------------------------------------

    valid_text = text[
        text != "nan"
    ]

    unique_text = list(
        pd.unique(valid_text)
    )

    if len(unique_text) == 2:

        second_class = unique_text[1]

        return text.map(

            lambda value:

            np.nan
            if value == "nan"
            else int(
                value == second_class
            )
        )

    return pd.Series(
        np.nan,
        index=series.index
    )


# ============================================================
# FIND TARGET COLUMN
# ============================================================

def find_target_column(df):

    # G3 gets highest priority
    priority = [

        "G3",

        "Result",

        "Final_Result",

        "Final Result",

        "Pass_Fail",

        "Pass Fail",

        "Performance",

        "Status",

        "Outcome",

        "Target",

        "Academic_Performance",

        "Academic Result"
    ]

    # --------------------------------------------------------
    # First search
    # --------------------------------------------------------

    for name in priority:

        column = find_column(
            df,
            [name]
        )

        if column is None:

            continue

        converted = convert_target(
            df[column],
            column
        )

        valid = converted.dropna()

        if (

            len(valid) >= 2

            and valid.nunique() >= 2

        ):

            return column

    # --------------------------------------------------------
    # Keyword search
    # --------------------------------------------------------

    keywords = [

        "result",

        "performance",

        "status",

        "outcome",

        "pass",

        "fail",

        "target",

        "final"
    ]

    for column in df.columns:

        normalized = normalize_column_name(
            column
        )

        if any(
            keyword in normalized
            for keyword in keywords
        ):

            converted = convert_target(
                df[column],
                column
            )

            valid = converted.dropna()

            if (

                len(valid) >= 2

                and valid.nunique() >= 2

            ):

                return column

    # --------------------------------------------------------
    # Last fallback
    # --------------------------------------------------------

    for column in df.columns:

        converted = convert_target(
            df[column],
            column
        )

        valid = converted.dropna()

        if (

            len(valid) >= 4

            and valid.nunique() >= 2

        ):

            return column

    raise ValueError(

        "Could not find a usable target column. "
        "Your dataset must contain G3, Result, "
        "Pass_Fail, Performance, Status or Outcome."
    )


# ============================================================
# PREPROCESSOR
# ============================================================

def create_preprocessor(X):

    numeric_columns = X.select_dtypes(
        include=["number"]
    ).columns.tolist()

    categorical_columns = X.select_dtypes(
        exclude=["number"]
    ).columns.tolist()

    # --------------------------------------------------------
    # Numeric
    # --------------------------------------------------------

    numeric_pipeline = Pipeline(

        steps=[

            (
                "imputer",

                SimpleImputer(
                    strategy="median"
                )
            ),

            (
                "scaler",

                StandardScaler()
            )
        ]
    )

    # --------------------------------------------------------
    # Categorical encoder
    # --------------------------------------------------------

    try:

        encoder = OneHotEncoder(

            handle_unknown="ignore",

            sparse_output=False
        )

    except TypeError:

        encoder = OneHotEncoder(

            handle_unknown="ignore",

            sparse=False
        )

    # --------------------------------------------------------
    # Categorical
    # --------------------------------------------------------

    categorical_pipeline = Pipeline(

        steps=[

            (
                "imputer",

                SimpleImputer(
                    strategy="most_frequent"
                )
            ),

            (
                "encoder",

                encoder
            )
        ]
    )

    # --------------------------------------------------------
    # Column transformer
    # --------------------------------------------------------

    return ColumnTransformer(

        transformers=[

            (
                "numeric",

                numeric_pipeline,

                numeric_columns
            ),

            (
                "categorical",

                categorical_pipeline,

                categorical_columns
            )
        ],

        remainder="drop"
    )


# ============================================================
# PREPARE FEATURES
# ============================================================

def prepare_features(X):

    X = X.copy()

    remove_columns = []

    for column in X.columns:

        normalized = normalize_column_name(
            column
        )

        # ----------------------------------------------------
        # Remove IDs
        # ----------------------------------------------------

        if normalized in {

            "id",

            "studentid",

            "studentnumber",

            "rollno",

            "rollnumber",

            "serialno",

            "sno"

        }:

            remove_columns.append(
                column
            )

        # ----------------------------------------------------
        # G1 and G2 are excluded from automatic dataset
        # classification because G3 is the final target.
        # ----------------------------------------------------

        if normalized in {

            "g1",

            "g2"

        }:

            remove_columns.append(
                column
            )

    # --------------------------------------------------------
    # Remove constant columns
    # --------------------------------------------------------

    for column in X.columns:

        if X[column].nunique(
            dropna=False
        ) <= 1:

            remove_columns.append(
                column
            )

    remove_columns = list(
        dict.fromkeys(
            remove_columns
        )
    )

    if remove_columns:

        X = X.drop(

            columns=remove_columns,

            errors="ignore"
        )

    return X


# ============================================================
# DATASET MODE
# ============================================================

def train_and_predict(
    df,
    model_path=MODEL_PATH
):

    print()
    print("=" * 70)
    print("DATASET MODE STARTED")
    print("=" * 70)

    # --------------------------------------------------------
    # Clean
    # --------------------------------------------------------

    df = repair_dataframe(df)

    if df.empty:

        raise ValueError(
            "Dataset is empty."
        )

    print(
        "Rows:",
        len(df)
    )

    print(
        "Columns:",
        list(df.columns)
    )

    # --------------------------------------------------------
    # Target
    # --------------------------------------------------------

    target_column = find_target_column(
        df
    )

    print(
        "Target:",
        target_column
    )

    # --------------------------------------------------------
    # Convert target
    # --------------------------------------------------------

    target = convert_target(

        df[target_column],

        target_column
    )

    valid_rows = target.notna()

    df = df.loc[
        valid_rows
    ].copy()

    target = target.loc[
        valid_rows
    ].astype(int)

    # --------------------------------------------------------
    # Check PASS/FAIL classes
    # --------------------------------------------------------

    print(
        "PASS:",
        int(
            (target == 1).sum()
        )
    )

    print(
        "FAIL:",
        int(
            (target == 0).sum()
        )
    )

    if target.nunique() < 2:

        raise ValueError(

            "The selected target must contain "
            "at least two classes (PASS and FAIL). "
            "If you are using G3, the dataset must "
            "contain students with both passing and "
            "failing final marks."
        )

    # --------------------------------------------------------
    # Features
    # --------------------------------------------------------

    X = df.drop(
        columns=[target_column]
    )

    X = prepare_features(
        X
    )

    if X.shape[1] == 0:

        raise ValueError(
            "No usable feature columns were found."
        )

    # --------------------------------------------------------
    # Train/Test split
    # --------------------------------------------------------

    class_counts = target.value_counts()

    if (

        len(target) >= 10

        and class_counts.min() >= 2

    ):

        X_train, X_test, y_train, y_test = train_test_split(

            X,

            target,

            test_size=0.20,

            random_state=42,

            stratify=target
        )

    else:

        X_train, X_test, y_train, y_test = train_test_split(

            X,

            target,

            test_size=0.20,

            random_state=42
        )

    # --------------------------------------------------------
    # Preprocessor
    # --------------------------------------------------------

    preprocessor = create_preprocessor(
        X_train
    )

    # --------------------------------------------------------
    # Models
    # --------------------------------------------------------

    models = {

        "Logistic Regression":

            LogisticRegression(
                max_iter=2000
            ),

        "Decision Tree":

            DecisionTreeClassifier(
                random_state=42
            ),

        "Random Forest":

            RandomForestClassifier(
                n_estimators=200,

                random_state=42
            ),

        "KNN":

            KNeighborsClassifier(
                n_neighbors=min(
                    5,
                    max(
                        1,
                        len(X_train)
                    )
                )
            ),

        "Naive Bayes":

            GaussianNB()
    }

    results = []

    trained_models = {}

    # ========================================================
    # TRAIN MODELS
    # ========================================================

    for model_name, classifier in models.items():

        try:

            pipeline = Pipeline(

                steps=[

                    (
                        "preprocessor",

                        preprocessor
                    ),

                    (
                        "classifier",

                        classifier
                    )
                ]
            )

            pipeline.fit(
                X_train,
                y_train
            )

            predicted = pipeline.predict(
                X_test
            )

            accuracy = accuracy_score(
                y_test,
                predicted
            )

            precision = precision_score(
                y_test,
                predicted,
                zero_division=0
            )

            recall = recall_score(
                y_test,
                predicted,
                zero_division=0
            )

            f1 = f1_score(
                y_test,
                predicted,
                zero_division=0
            )

            results.append({

                "Model":
                    model_name,

                "Accuracy":
                    round(
                        accuracy * 100,
                        2
                    ),

                "Precision":
                    round(
                        precision * 100,
                        2
                    ),

                "Recall":
                    round(
                        recall * 100,
                        2
                    ),

                "F1 Score":
                    round(
                        f1 * 100,
                        2
                    )
            })

            trained_models[
                model_name
            ] = pipeline

            print(
                model_name,
                "trained successfully"
            )

        except Exception as error:

            print(
                model_name,
                "failed:",
                error
            )

    if not trained_models:

        raise ValueError(
            "No machine learning model could be trained."
        )

    # --------------------------------------------------------
    # Model comparison
    # --------------------------------------------------------

    comparison_df = pd.DataFrame(
        results
    )

    comparison_df = comparison_df.sort_values(

        by="F1 Score",

        ascending=False
    ).reset_index(
        drop=True
    )

    best_model_name = comparison_df.iloc[0][
        "Model"
    ]

    best_model = trained_models[
        best_model_name
    ]

    # --------------------------------------------------------
    # Predict all students
    # --------------------------------------------------------

    predictions = best_model.predict(
        X
    )

    # --------------------------------------------------------
    # Probabilities
    # --------------------------------------------------------

    try:

        probabilities = best_model.predict_proba(
            X
        )

        classes = list(
            best_model.classes_
        )

        pass_index = classes.index(
            1
        )

        fail_index = classes.index(
            0
        )

        pass_probability = (

            probabilities[
                :,
                pass_index
            ]

            * 100
        )

        fail_probability = (

            probabilities[
                :,
                fail_index
            ]

            * 100
        )

    except Exception:

        pass_probability = np.where(

            predictions == 1,

            100,

            0
        )

        fail_probability = np.where(

            predictions == 0,

            100,

            0
        )

    # --------------------------------------------------------
    # Create prediction table
    # --------------------------------------------------------

    prediction_df = df.copy()

    prediction_df.insert(

        0,

        "Student_Number",

        range(
            1,
            len(prediction_df) + 1
        )
    )

    prediction_df[
        "Prediction"
    ] = np.where(

        predictions == 1,

        "PASS",

        "FAIL"
    )

    prediction_df[
        "Pass_Probability"
    ] = np.round(

        pass_probability,

        2
    )

    prediction_df[
        "Fail_Probability"
    ] = np.round(

        fail_probability,

        2
    )

    prediction_df[
        "Risk_Level"
    ] = [

        "LOW"

        if probability >= 75

        else

        "MEDIUM"

        if probability >= 50

        else

        "HIGH"

        for probability
        in pass_probability
    ]

    recommendation_list = []

    for prediction, probability in zip(

        predictions,

        pass_probability
    ):

        if (

            prediction == 1

            and probability >= 80

        ):

            recommendation_list.append(

                "Excellent performance. "
                "Continue the current study routine."
            )

        elif prediction == 1:

            recommendation_list.append(

                "Good performance. "
                "Continue regular preparation."
            )

        else:

            recommendation_list.append(

                "Needs improvement. "
                "Increase study time and revision."
            )

    prediction_df[
        "Recommendation"
    ] = recommendation_list

    # --------------------------------------------------------
    # Save predictions
    # --------------------------------------------------------

    prediction_path = os.path.join(

        OUTPUT_FOLDER,

        "student_predictions.csv"
    )

    prediction_df.to_csv(

        prediction_path,

        index=False
    )

    # --------------------------------------------------------
    # Save model comparison
    # --------------------------------------------------------

    comparison_path = os.path.join(

        OUTPUT_FOLDER,

        "model_comparison.csv"
    )

    comparison_df.to_csv(

        comparison_path,

        index=False
    )

    # --------------------------------------------------------
    # Save trained model
    # --------------------------------------------------------

    model_data = {

        "model":
            best_model,

        "features":
            list(X.columns),

        "target":
            target_column,

        "model_name":
            best_model_name,

        "comparison":
            comparison_df.to_dict(
                orient="records"
            )
    }

    joblib.dump(

        model_data,

        model_path
    )

    print()
    print(
        "Best model:",
        best_model_name
    )

    print(
        "Prediction file:",
        prediction_path
    )

    print(
        "=" * 70
    )

    # ========================================================
    # IMPORTANT
    #
    # Return DICTIONARY
    # ========================================================

    return {

        "predictions":
            prediction_df,

        "comparison":
            comparison_df,

        "best_model":
            best_model_name,

        "target":
            target_column,

        "total_students":
            len(prediction_df),

        "pass_count":
            int(
                (predictions == 1).sum()
            ),

        "fail_count":
            int(
                (predictions == 0).sum()
            ),

        "pass_percentage":
            round(

                (
                    (predictions == 1).sum()
                    /
                    len(predictions)
                )
                * 100,

                2
            ),

        "fail_percentage":
            round(

                (
                    (predictions == 0).sum()
                    /
                    len(predictions)
                )
                * 100,

                2
            )
    }


# ============================================================
# MANUAL MODE
#
# Only four values:
#
# 1. Study Hours
# 2. Attendance Percentage
# 3. Previous Marks
# 4. Assignment Score
#
# ============================================================

def manual_predict(

    study_hours,

    attendance,

    previous_marks,

    assignment_score

):

    # --------------------------------------------------------
    # Convert to numbers
    # --------------------------------------------------------

    try:

        study_hours = float(
            study_hours
        )

        attendance = float(
            attendance
        )

        previous_marks = float(
            previous_marks
        )

        assignment_score = float(
            assignment_score
        )

    except (
        ValueError,
        TypeError
    ):

        raise ValueError(

            "Please enter valid numeric values."
        )

    # --------------------------------------------------------
    # Validate study hours
    # --------------------------------------------------------

    if not 0 <= study_hours <= 24:

        raise ValueError(

            "Study Hours must be between 0 and 24."
        )

    # --------------------------------------------------------
    # Validate attendance
    # --------------------------------------------------------

    if not 0 <= attendance <= 100:

        raise ValueError(

            "Attendance must be between 0 and 100."
        )

    # --------------------------------------------------------
    # Validate previous marks
    # --------------------------------------------------------

    if not 0 <= previous_marks <= 100:

        raise ValueError(

            "Previous Marks must be between 0 and 100."
        )

    # --------------------------------------------------------
    # Validate assignment
    # --------------------------------------------------------

    if not 0 <= assignment_score <= 100:

        raise ValueError(

            "Assignment Score must be between 0 and 100."
        )

    # ========================================================
    # STUDY SCORE
    #
    # 12 hours/day is treated as 100% study score.
    # ========================================================

    study_score = (

        study_hours
        /
        12
    ) * 100

    study_score = min(

        max(
            study_score,
            0
        ),

        100
    )

    # ========================================================
    # FINAL SCORE
    #
    # Previous Marks    = 40%
    # Attendance        = 20%
    # Assignment       = 20%
    # Study Hours       = 20%
    # ========================================================

    estimated_mark = (

        previous_marks * 0.40

        +

        attendance * 0.20

        +

        assignment_score * 0.20

        +

        study_score * 0.20
    )

    estimated_mark = min(

        max(
            estimated_mark,
            0
        ),

        100
    )

    estimated_mark = round(

        estimated_mark,

        2
    )

    # ========================================================
    # PASS / FAIL
    #
    # 50 or above = PASS
    # ========================================================

    if estimated_mark >= 50:

        prediction = "PASS"

    else:

        prediction = "FAIL"

    # ========================================================
    # PROBABILITY
    # ========================================================

    pass_probability = round(

        estimated_mark,

        2
    )

    fail_probability = round(

        100 - estimated_mark,

        2
    )

    # ========================================================
    # RISK
    # ========================================================

    if estimated_mark >= 75:

        risk_level = "LOW"

    elif estimated_mark >= 50:

        risk_level = "MEDIUM"

    else:

        risk_level = "HIGH"

    # ========================================================
    # RECOMMENDATION
    # ========================================================

    if estimated_mark >= 80:

        recommendation = (

            "Excellent academic performance. "
            "Continue the current study routine."
        )

    elif estimated_mark >= 65:

        recommendation = (

            "Good academic performance. "
            "Maintain regular study and revision."
        )

    elif estimated_mark >= 50:

        recommendation = (

            "Satisfactory performance. "
            "More consistent study can improve the result."
        )

    else:

        recommendation = (

            "Performance needs improvement. "
            "Increase study hours, attendance, "
            "assignment performance and revision."
        )

    # ========================================================
    # RETURN RESULT
    #
    # Include all names used by the HTML.
    # ========================================================

    return {

        "estimated_mark":
            estimated_mark,

        "predicted_score":
            estimated_mark,

        "prediction":
            prediction,

        "result":
            prediction,

        "probability":
            pass_probability,

        "pass_probability":
            pass_probability,

        "fail_probability":
            fail_probability,

        "risk_level":
            risk_level,

        "risk":
            risk_level,

        "recommendation":
            recommendation,

        "study_hours":
            study_hours,

        "attendance":
            attendance,

        "attendance_percentage":
            attendance,

        "previous_marks":
            previous_marks,

        "assignment_score":
            assignment_score
    }


# ============================================================
# END OF FILE
# ============================================================