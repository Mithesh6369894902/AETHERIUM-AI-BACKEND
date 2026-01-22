import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, r2_score


def run_benchmark(payload: dict):
    data = pd.DataFrame(payload["data"])
    target = payload["target"]

    X = data.drop(columns=[target])
    y = data[target]

    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
    numerical_cols = X.select_dtypes(include=np.number).columns.tolist()

    task_type = "classification" if y.nunique() <= 10 else "regression"

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
        ]
    )

    if task_type == "classification":
        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "Random Forest": RandomForestClassifier()
        }
        metric_name = "Accuracy"
    else:
        models = {
            "Linear Regression": LinearRegression(),
            "Random Forest Regressor": RandomForestRegressor()
        }
        metric_name = "R2"

    benchmark = []

    for name, model in models.items():
        pipe = Pipeline([
            ("preprocessing", preprocessor),
            ("model", model)
        ])

        scores = cross_val_score(pipe, X, y, cv=5)
        benchmark.append({
            "Model": name,
            "Mean Score": round(scores.mean(), 4),
            "Std Dev": round(scores.std(), 4)
        })

    best_model_name = max(benchmark, key=lambda x: x["Mean Score"])["Model"]
    best_model = models[best_model_name]

    final_pipeline = Pipeline([
        ("preprocessing", preprocessor),
        ("model", best_model)
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    final_pipeline.fit(X_train, y_train)
    preds = final_pipeline.predict(X_test)

    final_score = (
        accuracy_score(y_test, preds)
        if task_type == "classification"
        else r2_score(y_test, preds)
    )

    return {
        "task_type": task_type,
        "best_model": best_model_name,
        "benchmark": benchmark,
        "metric": metric_name,
        "final_score": round(float(final_score), 4)
    }

