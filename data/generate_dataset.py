import numpy as np
import pandas as pd

OUTPUT_PATH = "data/student_performance.csv"


def generate_dataset(num_records: int = 500) -> pd.DataFrame:
    np.random.seed(42)
    study_hours = np.round(np.random.uniform(0.5, 8.0, size=num_records), 1)
    attendance = np.round(np.random.uniform(50.0, 100.0, size=num_records), 1)
    assignments = np.random.randint(0, 11, size=num_records)
    previous_marks = np.round(np.random.uniform(25.0, 100.0, size=num_records), 1)
    participation = np.random.randint(1, 6, size=num_records)

    raw_score = (
        study_hours * 6.0
        + attendance * 0.25
        + assignments * 3.5
        + previous_marks * 0.35
        + participation * 2.0
    )
    noise = np.random.normal(0, 6, size=num_records)
    final_grade = np.clip(np.round(raw_score + noise, 1), 20.0, 100.0)

    performance_level = pd.cut(
        final_grade,
        bins=[0, 59, 74, 84, 100],
        labels=["Low", "Average", "Good", "Excellent"],
    )

    return pd.DataFrame(
        {
            "study_hours_per_day": study_hours,
            "attendance_percentage": attendance,
            "assignments_completed": assignments,
            "previous_semester_marks": previous_marks,
            "class_participation": participation,
            "final_grade": final_grade,
            "performance_level": performance_level,
        }
    )


def main() -> None:
    df = generate_dataset(500)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Generated dataset with {len(df)} records to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
