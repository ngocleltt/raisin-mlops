
from pathlib import Path

import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "Raisin_Dataset.xlsx"
MODEL_PATH = BASE_DIR / "models" / "raisin_model.joblib"


def load_data():
    # Загрузка датасета
    df = pd.read_excel(DATA_PATH)

    # Разделение признаков и целевой переменной
    X = df.drop(columns=["Class"])
    y = df["Class"]

    # Формирование обучающей и тестовой выборок
    return train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )


def main():
    X_train, X_test, y_train, y_test = load_data()

    # Создание модели Random Forest
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=8,
        random_state=42
    )

    # Обучение модели
    model.fit(X_train, y_train)

    # Получение предсказаний
    y_pred = model.predict(X_test)

    # Оценка качества модели
    print("Train size:", len(X_train))
    print("Test size:", len(X_test))

    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"F1-score: {f1_score(y_test, y_pred, average='macro'):.4f}")

    print("\nClassification report:")
    print(classification_report(y_test, y_pred))

    print("\nConfusion matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Сохранение обученной модели
    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    print(f"\nModel saved: {MODEL_PATH}")


if __name__ == "__main__":
    main()
