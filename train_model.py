import pickle

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

DATA_PATH = "data/student_performance.csv"
MODEL_PATH = "model.pkl"


def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def build_model() -> RandomForestRegressor:
    return RandomForestRegressor(n_estimators=100, random_state=42)


def main() -> None:
    df = load_data(DATA_PATH)
    features = [
        "study_hours_per_day",
        "attendance_percentage",
        "assignments_completed",
        "previous_semester_marks",
        "class_participation",
    ]
    X = df[features]
    y = df["final_grade"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = build_model()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Model training complete")
    print(f"Mean Absolute Error: {mae:.2f}")
    print(f"R2 Score: {r2:.2f}")

    with open(MODEL_PATH, "wb") as model_file:
        pickle.dump(model, model_file)
    print(f"Saved trained model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
