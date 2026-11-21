
import mlflow
import mlflow.sklearn

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate
from train import load_data

# Настройка MLflow Tracking
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Raisin Classification")

X_train, X_test, y_train, y_test = load_data()

# Конфигурации гиперпараметров
configs = [
    {"n_estimators": 50, "max_depth": 4},
    {"n_estimators": 100, "max_depth": 8},
    {"n_estimators": 200, "max_depth": 12}
]

# Проведение экспериментов
for params in configs:
    run_name = f"RF_{params['n_estimators']}_depth_{params['max_depth']}"

    with mlflow.start_run(run_name=run_name):

        model = RandomForestClassifier(
            **params,
            random_state=42
        )

        # Кросс-валидация
        scores = cross_validate(
            model,
            X_train,
            y_train,
            cv=5,
            scoring=["accuracy", "f1_macro"]
        )

        model.fit(X_train, y_train)

        # Логирование параметров
        mlflow.log_params(model.get_params())
        mlflow.log_param("cv_folds", 5)

        # Логирование метрик
        accuracy = scores["test_accuracy"].mean()
        f1 = scores["test_f1_macro"].mean()

        mlflow.log_metrics({
            "cv_accuracy": accuracy,
            "cv_f1_macro": f1
        })

        # Логирование обученной модели
        mlflow.sklearn.log_model(
            model,
            name="raisin_model",
            input_example=X_train.head(3)
        )

        print(f"{run_name}: Accuracy={accuracy:.4f}, F1={f1:.4f}")
