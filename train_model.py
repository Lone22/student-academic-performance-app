import pickle

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

DATA_PATH = "data/student_performance.csv"
MODEL_PATH = "model.pkl"
TARGET_COLUMN = "final_grade"
FEATURE_COLUMNS = [
    "study_hours_per_day",
    "attendance_percentage",
    "assignments_completed",
    "previous_semester_marks",
    "class_participation",
]


def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    cleaned_df = df.copy()

    # 1. Data Cleaning
    cleaned_df = cleaned_df.drop_duplicates()

    # 2. Missing Value Handling
    for column in cleaned_df.columns:
        if cleaned_df[column].isnull().any():
            if cleaned_df[column].dtype == "object":
                mode_value = cleaned_df[column].mode(dropna=True)
                cleaned_df[column] = cleaned_df[column].fillna(
                    mode_value.iloc[0] if not mode_value.empty else "Unknown"
                )
            else:
                cleaned_df[column] = cleaned_df[column].fillna(cleaned_df[column].mean())

    return cleaned_df


def visualize_data(df: pd.DataFrame) -> None:
    # 5. Data Visualization
    os.makedirs("plots", exist_ok=True)

    plt.figure(figsize=(10, 6))
    sns.heatmap(
        df[FEATURE_COLUMNS + [TARGET_COLUMN]].corr(),
        annot=True,
        cmap="Blues",
        fmt=".2f",
    )
    plt.title("Feature Correlation Matrix")
    plt.tight_layout()
    plt.savefig("plots/correlation_matrix.png", dpi=150)
    plt.close()

    plt.figure(figsize=(10, 6))
    sns.histplot(df[TARGET_COLUMN], bins=20, kde=True)
    plt.title("Distribution of Final Grades")
    plt.xlabel("Final Grade")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig("plots/final_grade_distribution.png", dpi=150)
    plt.close()


def evaluate_model(model, X_train, X_test, y_train, y_test) -> dict:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    # For regression tasks, accuracy is not the usual metric.
    # We compute a simple rounded accuracy proxy for display purposes.
    accuracy = accuracy_score(
        (y_test >= 50).astype(int),
        (y_pred >= 50).astype(int),
    )

    return {
        "model": model,
        "accuracy": accuracy,
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "r2": r2,
    }


def main() -> None:
    df = load_data(DATA_PATH)
    df = clean_data(df)
    visualize_data(df)

    # 4. Feature Selection and 6. Train-Test Split
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 3. Machine Learning Models
    models = {
        "Linear Regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("regressor", LinearRegression()),
            ]
        ),
        "Random Forest Regressor": RandomForestRegressor(
            n_estimators=300,
            random_state=42,
            min_samples_leaf=2,
        ),
    }

    results = {}
    for name, model in models.items():
        metrics = evaluate_model(model, X_train, X_test, y_train, y_test)
        results[name] = metrics

        print(f"{name}:")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  MAE     : {metrics['mae']:.2f}")
        print(f"  MSE     : {metrics['mse']:.2f}")
        print(f"  RMSE    : {metrics['rmse']:.2f}")
        print(f"  R2      : {metrics['r2']:.4f}")

    # Select best model by highest R2 score
    best_model_name = max(results, key=lambda name: results[name]["r2"])
    best_model = results[best_model_name]["model"]

    print(f"\nBest model selected: {best_model_name}")

    with open(MODEL_PATH, "wb") as model_file:
        pickle.dump(best_model, model_file)

    print(f"Saved trained model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
