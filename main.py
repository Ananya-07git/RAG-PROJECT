import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


def create_student_dataset() -> pd.DataFrame:
    data = {
        "Study_Hours": [
            1, 2, 2, 3, 3, 3, 4, 5, 5, 6,
            6, 7, 7, 8, 8, 9, 9, 10, 10, 11,
        ],
        "Attendance": [
            50, 55, 60, 60, 65, 65, 65, 70, 75, 75,
            80, 80, 85, 85, 90, 90, 92, 95, 96, 98,
        ],
        "Exam_Marks": [
            25, 30, 35, 40, 42, 48, 50, 55, 60, 62,
            68, 70, 75, 78, 82, 85, 88, 92, 95, 98,
        ],
    }

    df = pd.DataFrame(data)
    df["Result"] = (df["Exam_Marks"] > 40).astype(int)
    return df


def train_linear_regression(df: pd.DataFrame) -> LinearRegression:
    x = df[["Study_Hours", "Attendance"]]
    y_marks = df["Exam_Marks"]

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y_marks,
        test_size=0.25,
        random_state=42,
    )

    linear_model = LinearRegression()
    linear_model.fit(x_train, y_train)
    marks_pred = linear_model.predict(x_test)

    print("\n--- LINEAR REGESSION ---")
    print("Actual Marks:", y_test.values)
    print("Predicted Marks:", marks_pred.round(2))
    print("MAE:", mean_absolute_error(y_test, marks_pred))
    print("MSE:", mean_squared_error(y_test, marks_pred))
    print("R2 Score:", r2_score(y_test, marks_pred))

    new_student = pd.DataFrame({
        "Study_Hours": [7],
        "Attendance": [80],
    })
    predicted_marks = linear_model.predict(new_student)

    print("\nNew Student:")
    print("Study_Hours:", new_student["Study_Hours"].iloc[0])
    print("Attendance:", new_student["Attendance"].iloc[0])
    print("Predicted Exam Marks:", round(predicted_marks[0], 2))
    return linear_model


def train_logistic_regression(df: pd.DataFrame) -> LogisticRegression:
    x = df[["Study_Hours", "Attendance"]]
    y_result = df["Result"]

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y_result,
        test_size=0.25,
        random_state=42,
    )

    logistic_model = LogisticRegression()
    logistic_model.fit(x_train, y_train)
    result_pred = logistic_model.predict(x_test)

    print("\n--- LOGISTIC REGESSION ---")
    print("Actual Results:", y_test.values)
    print("Predicted Results:", result_pred)
    print("Accuracy:", accuracy_score(y_test, result_pred))

    new_student = pd.DataFrame({
        "Study_Hours": [7],
        "Attendance": [80],
    })
    predicted_result = logistic_model.predict(new_student)

    print("\nPredict the result for a new student")
    print("Study_Hours:", new_student["Study_Hours"].iloc[0])
    print("Attendance:", new_student["Attendance"].iloc[0])
    print("Predicted Result:", predicted_result[0])
    return logistic_model


def main() -> None:
    df = create_student_dataset()
    print("Student Dataset:")
    print(df)

    train_linear_regression(df)
    train_logistic_regression(df)


if __name__ == "__main__":
    main()
