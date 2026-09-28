# ============================================================
# STUDENT ACADEMIC PERFORMANCE PREDICTION
# app.py
# ============================================================

from flask import Flask, render_template, request, redirect, url_for

import os
import pandas as pd

from model import train_and_predict, manual_predict


# ============================================================
# FLASK APP
# ============================================================

app = Flask(__name__)


# ============================================================
# FOLDERS
# ============================================================

UPLOAD_FOLDER = "dataset"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


# ============================================================
# HOME PAGE
# ============================================================

@app.route("/")
def index():

    return render_template("index.html")


# ============================================================
# DATASET MODE
#
# IMPORTANT:
# The endpoint name is predict_dataset because your
# existing index.html is calling:
#
# url_for('predict_dataset')
# ============================================================

@app.route(
    "/predict",
    methods=["POST"],
    endpoint="predict_dataset"
)
def predict_dataset():

    try:

        print()
        print("=" * 70)
        print("DATASET PREDICTION REQUEST")
        print("=" * 70)

        # ----------------------------------------------------
        # Get uploaded file
        # ----------------------------------------------------

        uploaded_file = request.files.get("dataset")

        # Support alternate field name
        if uploaded_file is None:

            uploaded_file = request.files.get("file")

        if uploaded_file is None:

            return render_template(
                "index.html",
                error="Please select a dataset file."
            )

        if uploaded_file.filename == "":

            return render_template(
                "index.html",
                error="Please select a dataset file."
            )

        filename = uploaded_file.filename

        print("File:", filename)

        # ----------------------------------------------------
        # Save uploaded file
        # ----------------------------------------------------

        filepath = os.path.join(
            app.config["UPLOAD_FOLDER"],
            filename
        )

        uploaded_file.save(filepath)

        print("Saved:", filepath)

        # ----------------------------------------------------
        # Find extension
        # ----------------------------------------------------

        extension = (
            filename
            .lower()
            .split(".")[-1]
        )

        # ----------------------------------------------------
        # Read CSV
        # ----------------------------------------------------

        if extension == "csv":

            try:

                # Automatically detects comma/semicolon/etc.
                df = pd.read_csv(
                    filepath,
                    sep=None,
                    engine="python"
                )

            except Exception:

                # Fallback for semicolon CSV
                df = pd.read_csv(
                    filepath,
                    sep=";"
                )

        # ----------------------------------------------------
        # Excel
        # ----------------------------------------------------

        elif extension == "xlsx":

            df = pd.read_excel(
                filepath
            )

        elif extension == "xls":

            df = pd.read_excel(
                filepath
            )

        else:

            return render_template(
                "index.html",
                error=(
                    "Unsupported file format. "
                    "Please upload CSV, XLSX or XLS."
                )
            )

        # ----------------------------------------------------
        # Check dataset
        # ----------------------------------------------------

        if df is None or df.empty:

            return render_template(
                "index.html",
                error="The uploaded dataset is empty."
            )

        print("Rows:", len(df))

        print(
            "Columns:",
            list(df.columns)
        )

        # ----------------------------------------------------
        # TRAIN + PREDICT
        # ----------------------------------------------------

        result = train_and_predict(df)

        # ----------------------------------------------------
        # Get result dictionary
        # ----------------------------------------------------

        predictions = result["predictions"]

        comparison = result["comparison"]

        best_model = result["best_model"]

        target = result["target"]

        total_students = result["total_students"]

        pass_count = result["pass_count"]

        fail_count = result["fail_count"]

        pass_percentage = result["pass_percentage"]

        fail_percentage = result["fail_percentage"]

        # ----------------------------------------------------
        # Print summary
        # ----------------------------------------------------

        print()
        print("=" * 70)
        print("DATASET MODE COMPLETED SUCCESSFULLY")
        print("=" * 70)

        print(
            "Total:",
            total_students
        )

        print(
            "PASS:",
            pass_count
        )

        print(
            "FAIL:",
            fail_count
        )

        print(
            "Best Model:",
            best_model
        )

        print(
            "Target:",
            target
        )

        # ----------------------------------------------------
        # Show result
        # ----------------------------------------------------

        return render_template(

            "result.html",

            predictions=predictions,

            comparison=comparison,

            best_model=best_model,

            target=target,

            total_students=total_students,

            pass_count=pass_count,

            fail_count=fail_count,

            pass_percentage=pass_percentage,

            fail_percentage=fail_percentage
        )

    except Exception as error:

        print()
        print("=" * 70)
        print("DATASET PREDICTION ERROR")
        print("=" * 70)

        print(error)

        return render_template(

            "index.html",

            error=(
                "Dataset prediction error: "
                + str(error)
            )
        )


# ============================================================
# MANUAL MODE
# ============================================================

@app.route(
    "/manual-predict",
    methods=["POST"]
)
def manual_prediction():

    try:

        print()
        print("=" * 70)
        print("MANUAL PREDICTION REQUEST")
        print("=" * 70)

        # ----------------------------------------------------
        # Student name
        # ----------------------------------------------------

        student_name = request.form.get(
            "student_name",
            ""
        )

        # ----------------------------------------------------
        # Study hours
        # ----------------------------------------------------

        study_hours = request.form.get(
            "study_hours"
        )

        if study_hours is None:

            study_hours = request.form.get(
                "study"
            )

        # ----------------------------------------------------
        # Attendance
        # ----------------------------------------------------

        attendance = request.form.get(
            "attendance"
        )

        if attendance is None:

            attendance = request.form.get(
                "attendance_percentage"
            )

        # ----------------------------------------------------
        # Previous marks
        # ----------------------------------------------------

        previous_marks = request.form.get(
            "previous_marks"
        )

        if previous_marks is None:

            previous_marks = request.form.get(
                "previous_mark"
            )

        # ----------------------------------------------------
        # Assignment score
        # ----------------------------------------------------

        assignment_score = request.form.get(
            "assignment_score"
        )

        if assignment_score is None:

            assignment_score = request.form.get(
                "assignment"
            )

        # ----------------------------------------------------
        # Check fields
        # ----------------------------------------------------

        if study_hours is None:

            raise ValueError(
                "Study Hours field is missing."
            )

        if attendance is None:

            raise ValueError(
                "Attendance field is missing."
            )

        if previous_marks is None:

            raise ValueError(
                "Previous Marks field is missing."
            )

        if assignment_score is None:

            raise ValueError(
                "Assignment Score field is missing."
            )

        # ----------------------------------------------------
        # Predict
        # ----------------------------------------------------

        result = manual_predict(

            study_hours=study_hours,

            attendance=attendance,

            previous_marks=previous_marks,

            assignment_score=assignment_score
        )

        # ----------------------------------------------------
        # Add student name
        # ----------------------------------------------------

        result["student_name"] = student_name

        print(
            "Student:",
            student_name
        )

        print(
            "Estimated Mark:",
            result["estimated_mark"]
        )

        print(
            "Prediction:",
            result["prediction"]
        )

        print("=" * 70)

        # ----------------------------------------------------
        # Show manual result
        # ----------------------------------------------------

        return render_template(

            "manual_result.html",

            **result
        )

    except Exception as error:

        print()
        print("=" * 70)
        print("MANUAL PREDICTION ERROR")
        print("=" * 70)

        print(error)

        return render_template(

            "index.html",

            error=(
                "Manual prediction error: "
                + str(error)
            )
        )


# ============================================================
# RUN APPLICATION
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 70)
    print("STUDENT ACADEMIC PERFORMANCE PREDICTION")
    print("=" * 70)
    print("Dataset Mode : /predict")
    print("Manual Mode  : /manual-predict")
    print("URL          : http://127.0.0.1:5000")
    print("=" * 70)
    print()

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )