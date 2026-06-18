import pickle

import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request

app = Flask(__name__)

MODEL_PATH = "model.pkl"
FEATURE_COLUMNS = [
    "study_hours_per_day",
    "attendance_percentage",
    "assignments_completed",
    "previous_semester_marks",
    "class_participation",
]

with open(MODEL_PATH, "rb") as model_file:
    model = pickle.load(model_file)


def performance_category(score: float) -> str:
    if score >= 85:
        return "Excellent"
    if score >= 75:
        return "Good"
    if score >= 60:
        return "Average"
    return "Needs Improvement"


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)
    try:
        study_hours = float(data.get("study_hours"))
        attendance = float(data.get("attendance"))
        assignments = float(data.get("assignments"))
        previous_marks = float(data.get("previous_marks"))
        participation = float(data.get("participation"))
    except (TypeError, ValueError):
        return jsonify({"success": False, "message": "Invalid input values."}), 400

    features = pd.DataFrame(
        [[study_hours, attendance, assignments, previous_marks, participation]],
        columns=FEATURE_COLUMNS,
    )
    prediction = model.predict(features)
    final_grade = float(np.clip(prediction[0], 0, 100))
    category = performance_category(final_grade)

    return jsonify(
        success=True,
        predicted_grade=round(final_grade, 1),
        performance_category=category,
        message="Prediction generated successfully."
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False, use_reloader=False)
