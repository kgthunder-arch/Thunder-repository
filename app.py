from __future__ import annotations

import os
import uuid
from dataclasses import dataclass
from typing import Any

import pandas as pd
from flask import Flask, flash, redirect, render_template, request, url_for
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor
from werkzeug.utils import secure_filename


BASE_DIR = os.path.abspath(os.path.dirname(__file__))
UPLOAD_FOLDER = (
    os.path.join("/tmp", "uploads")
    if os.getenv("VERCEL")
    else os.path.join(BASE_DIR, "uploads")
)
SAMPLE_DATA_PATH = os.path.join(BASE_DIR, "sample_data", "agribusiness_sample.csv")
ALLOWED_EXTENSIONS = {"csv", "xlsx", "xls"}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config["SECRET_KEY"] = "agribusiness-predictive-analysis-secret"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


@dataclass
class DatasetSummary:
    path: str
    filename: str
    row_count: int
    column_count: int
    columns: list[str]
    numeric_columns: list[str]
    preview_html: str
    suggestions: dict[str, str]


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def read_dataset(file_path: str) -> pd.DataFrame:
    extension = file_path.rsplit(".", 1)[1].lower()
    if extension == "csv":
        return pd.read_csv(file_path)
    return pd.read_excel(file_path)


def suggest_targets(columns: list[str]) -> dict[str, str]:
    suggestions: dict[str, str] = {}
    keywords = {
        "yield": ["yield", "production"],
        "price": ["price", "revenue"],
        "demand": ["demand", "sales", "consumption"],
    }
    for forecast_type, aliases in keywords.items():
        for column in columns:
            name = column.lower()
            if any(alias in name for alias in aliases):
                suggestions[forecast_type] = column
                break
        suggestions.setdefault(forecast_type, "")
    return suggestions


def summarize_dataset(file_path: str, filename: str) -> DatasetSummary:
    dataframe = read_dataset(file_path)
    preview = dataframe.head(8).to_html(
        classes="data-table preview-table", index=False, border=0, justify="left"
    )
    numeric_columns = dataframe.select_dtypes(include="number").columns.tolist()
    return DatasetSummary(
        path=file_path,
        filename=filename,
        row_count=int(dataframe.shape[0]),
        column_count=int(dataframe.shape[1]),
        columns=dataframe.columns.tolist(),
        numeric_columns=numeric_columns,
        preview_html=preview,
        suggestions=suggest_targets(dataframe.columns.tolist()),
    )


def expand_datetime_features(dataframe: pd.DataFrame) -> pd.DataFrame:
    transformed = dataframe.copy()
    columns_to_drop: list[str] = []

    for column in transformed.columns:
        series = transformed[column]
        parsed = None

        if pd.api.types.is_datetime64_any_dtype(series):
            parsed = pd.to_datetime(series, errors="coerce")
        elif series.dtype == "object" and "date" in column.lower():
            parsed = pd.to_datetime(series, errors="coerce")
            if parsed.notna().mean() < 0.75:
                parsed = None

        if parsed is not None and parsed.notna().any():
            transformed[f"{column}_year"] = parsed.dt.year
            transformed[f"{column}_month"] = parsed.dt.month
            transformed[f"{column}_quarter"] = parsed.dt.quarter
            transformed[f"{column}_day_of_year"] = parsed.dt.dayofyear
            columns_to_drop.append(column)

    if columns_to_drop:
        transformed = transformed.drop(columns=columns_to_drop)

    return transformed


def build_preprocessor(dataframe: pd.DataFrame) -> ColumnTransformer:
    numeric_features = dataframe.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_features = [column for column in dataframe.columns if column not in numeric_features]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ]
    )


def rank_models(metrics_df: pd.DataFrame) -> pd.DataFrame:
    ranked = metrics_df.copy()
    ranked["rmse_rank"] = ranked["RMSE"].rank(method="min")
    ranked["mae_rank"] = ranked["MAE"].rank(method="min")
    ranked["r2_rank"] = ranked["R2"].rank(ascending=False, method="min")
    ranked["score"] = ranked["rmse_rank"] + ranked["mae_rank"] + ranked["r2_rank"]
    return ranked.sort_values(by=["score", "RMSE", "MAE", "R2"], ascending=[True, True, True, False])


def calculate_cv_summary(
    feature_frame: pd.DataFrame, target_series: pd.Series, estimator: Any, cv_folds: int
) -> dict[str, float]:
    pipeline = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(feature_frame)),
            ("model", estimator),
        ]
    )
    scores = cross_validate(
        pipeline,
        feature_frame,
        target_series,
        cv=cv_folds,
        scoring={
            "mae": "neg_mean_absolute_error",
            "rmse": "neg_root_mean_squared_error",
            "r2": "r2",
        },
        n_jobs=None,
    )
    return {
        "cv_mae": float((-scores["test_mae"]).mean()),
        "cv_rmse": float((-scores["test_rmse"]).mean()),
        "cv_r2": float(scores["test_r2"].mean()),
    }


def chart_payload(data: list[dict[str, Any]], layout: dict[str, Any]) -> dict[str, Any]:
    return {
        "data": data,
        "layout": layout,
        "config": {"displayModeBar": False, "responsive": True},
    }


def create_metrics_chart(metrics_df: pd.DataFrame, forecast_label: str) -> dict[str, Any]:
    metric_styles = {
        "MAE": "#5f8d4e",
        "RMSE": "#a4be7b",
        "R2": "#285430",
    }

    data: list[dict[str, Any]] = []
    for metric_name, color in metric_styles.items():
        data.append(
            {
                "type": "bar",
                "x": metrics_df["Model"].tolist(),
                "y": metrics_df[metric_name].tolist(),
                "name": metric_name,
                "marker": {"color": color},
            }
        )

    return chart_payload(
        data=data,
        layout={
            "title": {"text": f"{forecast_label} Model Comparison"},
            "barmode": "group",
            "paper_bgcolor": "rgba(0,0,0,0)",
            "plot_bgcolor": "rgba(0,0,0,0)",
            "margin": {"l": 20, "r": 20, "t": 60, "b": 20},
            "legend": {"title": {"text": ""}},
        },
    )


def create_prediction_chart(
    prediction_df: pd.DataFrame, forecast_label: str, model_name: str
) -> dict[str, Any]:
    return chart_payload(
        data=[
            {
                "type": "scatter",
                "x": prediction_df["Record"].tolist(),
                "y": prediction_df["Actual"].tolist(),
                "mode": "lines+markers",
                "name": "Actual",
                "line": {"color": "#d8f3dc", "width": 3},
            },
            {
                "type": "scatter",
                "x": prediction_df["Record"].tolist(),
                "y": prediction_df["Predicted"].tolist(),
                "mode": "lines+markers",
                "name": "Predicted",
                "line": {"color": "#f4a259", "width": 3},
            },
        ],
        layout={
            "title": {"text": f"{forecast_label} Forecast: Actual vs Predicted ({model_name})"},
            "xaxis": {"title": {"text": "Test Record"}},
            "yaxis": {"title": {"text": forecast_label}},
            "paper_bgcolor": "rgba(0,0,0,0)",
            "plot_bgcolor": "rgba(0,0,0,0)",
            "margin": {"l": 20, "r": 20, "t": 60, "b": 20},
            "legend": {"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
        },
    )


def create_scatter_chart(prediction_df: pd.DataFrame, forecast_label: str, model_name: str) -> dict[str, Any]:
    return chart_payload(
        data=[
            {
                "type": "scatter",
                "x": prediction_df["Actual"].tolist(),
                "y": prediction_df["Predicted"].tolist(),
                "mode": "markers",
                "marker": {"color": "#bc4749", "size": 10, "opacity": 0.82},
                "name": "Prediction",
            }
        ],
        layout={
            "title": {"text": f"{forecast_label} Accuracy Scatter ({model_name})"},
            "xaxis": {"title": {"text": "Actual"}},
            "yaxis": {"title": {"text": "Predicted"}},
            "paper_bgcolor": "rgba(0,0,0,0)",
            "plot_bgcolor": "rgba(0,0,0,0)",
            "margin": {"l": 20, "r": 20, "t": 60, "b": 20},
        },
    )


def recommendation_text(
    forecast_label: str,
    best_row: pd.Series,
    holdout_used: bool,
    cv_enabled: bool,
) -> str:
    evaluation_basis = "holdout test set" if holdout_used else "full dataset"
    sentence = (
        f"{best_row['Model']} is the strongest option for {forecast_label.lower()} forecasting "
        f"because it achieved the lowest error profile on the {evaluation_basis}: "
        f"MAE {best_row['MAE']:.3f}, RMSE {best_row['RMSE']:.3f}, and R2 {best_row['R2']:.3f}."
    )
    if cv_enabled and pd.notna(best_row.get("CV RMSE")):
        sentence += (
            f" Cross-validation stayed consistent with an average RMSE of "
            f"{best_row['CV RMSE']:.3f}, which makes this model a safer deployment choice."
        )
    return sentence


def analyze_target(
    dataframe: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
    forecast_label: str,
    test_size: float,
    random_state: int,
    use_holdout: bool,
    use_cv: bool,
    cv_folds: int,
) -> dict[str, Any]:
    filtered = dataframe[feature_columns + [target_column]].copy()
    filtered[target_column] = pd.to_numeric(filtered[target_column], errors="coerce")
    filtered = filtered.dropna(subset=[target_column])

    feature_frame = expand_datetime_features(filtered[feature_columns])
    target_series = filtered[target_column]

    minimum_rows = max(cv_folds + 2, 10) if use_cv else 10
    if len(filtered) < minimum_rows:
        raise ValueError(f"{forecast_label} needs at least {minimum_rows} valid rows for training and validation.")

    if use_holdout:
        estimated_test_rows = int(round(len(filtered) * test_size))
        if estimated_test_rows < 2:
            raise ValueError(
                f"{forecast_label} needs a larger test split or more rows so the holdout set has at least 2 records."
            )

    models: dict[str, Any] = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(max_depth=8, random_state=random_state),
        "Random Forest": RandomForestRegressor(
            n_estimators=250,
            max_depth=10,
            random_state=random_state,
            n_jobs=None,
        ),
    }

    metrics_rows: list[dict[str, Any]] = []
    predictions_by_model: dict[str, pd.DataFrame] = {}

    if use_holdout:
        x_train, x_test, y_train, y_test = train_test_split(
            feature_frame,
            target_series,
            test_size=test_size,
            random_state=random_state,
        )
    else:
        x_train, x_test, y_train, y_test = feature_frame, feature_frame, target_series, target_series

    for model_name, estimator in models.items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", build_preprocessor(feature_frame)),
                ("model", estimator),
            ]
        )
        pipeline.fit(x_train, y_train)
        predictions = pipeline.predict(x_test)

        row: dict[str, Any] = {
            "Model": model_name,
            "MAE": float(mean_absolute_error(y_test, predictions)),
            "RMSE": float(mean_squared_error(y_test, predictions) ** 0.5),
            "R2": float(r2_score(y_test, predictions)),
            "CV MAE": None,
            "CV RMSE": None,
            "CV R2": None,
        }

        if use_cv:
            cv_metrics = calculate_cv_summary(feature_frame, target_series, estimator, cv_folds)
            row["CV MAE"] = cv_metrics["cv_mae"]
            row["CV RMSE"] = cv_metrics["cv_rmse"]
            row["CV R2"] = cv_metrics["cv_r2"]

        prediction_frame = pd.DataFrame(
            {
                "Record": list(range(1, len(y_test) + 1)),
                "Actual": y_test.to_numpy(),
                "Predicted": predictions,
                "Residual": y_test.to_numpy() - predictions,
            }
        ).round(3)

        metrics_rows.append(row)
        predictions_by_model[model_name] = prediction_frame

    metrics_df = pd.DataFrame(metrics_rows).round(3)
    ranked_df = rank_models(metrics_df)
    best_model = ranked_df.iloc[0]["Model"]
    best_row = ranked_df.iloc[0]
    best_predictions = predictions_by_model[best_model]

    metrics_table = ranked_df[
        ["Model", "MAE", "RMSE", "R2", "CV MAE", "CV RMSE", "CV R2"]
    ].to_html(classes="data-table", index=False, border=0, justify="left")
    predictions_table = best_predictions.to_html(classes="data-table", index=False, border=0, justify="left")

    return {
        "forecast_label": forecast_label,
        "target_column": target_column,
        "best_model": best_model,
        "recommendation": recommendation_text(forecast_label, best_row, use_holdout, use_cv),
        "metrics_table": metrics_table,
        "predictions_table": predictions_table,
        "metrics_chart": create_metrics_chart(metrics_df, forecast_label),
        "prediction_chart": create_prediction_chart(best_predictions, forecast_label, best_model),
        "scatter_chart": create_scatter_chart(best_predictions, forecast_label, best_model),
    }


def default_feature_columns(columns: list[str], selected_targets: list[str]) -> list[str]:
    excluded = set(selected_targets)
    return [column for column in columns if column not in excluded]


@app.get("/")
def agribusiness_index() -> str:
    sample_summary = summarize_dataset(SAMPLE_DATA_PATH, "agribusiness_sample.csv")
    return render_template("index.html", sample_summary=sample_summary)


@app.post("/upload")
def upload() -> str:
    source = request.form.get("source", "sample")

    try:
        if source == "sample":
            summary = summarize_dataset(SAMPLE_DATA_PATH, "agribusiness_sample.csv")
        else:
            uploaded_file = request.files.get("dataset")
            if uploaded_file is None or uploaded_file.filename == "":
                flash("Please choose a dataset file to continue.", "error")
                return redirect(url_for("agribusiness_index"))
            if not allowed_file(uploaded_file.filename):
                flash("Upload a CSV or Excel dataset file.", "error")
                return redirect(url_for("agribusiness_index"))

            safe_name = secure_filename(uploaded_file.filename)
            stored_name = f"{uuid.uuid4().hex}_{safe_name}"
            stored_path = os.path.join(app.config["UPLOAD_FOLDER"], stored_name)
            uploaded_file.save(stored_path)
            summary = summarize_dataset(stored_path, safe_name)
    except Exception as exc:
        flash(f"Unable to read the dataset: {exc}", "error")
        return redirect(url_for("agribusiness_index"))

    return render_template("configure.html", summary=summary)


@app.post("/analyze")
def analyze() -> str:
    dataset_path = request.form.get("dataset_path", "")
    filename = request.form.get("filename", "dataset")

    if not dataset_path or not os.path.exists(dataset_path):
        flash("The dataset session expired. Upload the file again.", "error")
        return redirect(url_for("agribusiness_index"))

    try:
        dataframe = read_dataset(dataset_path)
        selected_targets = {
            "Crop Yield": request.form.get("yield_target", "").strip(),
            "Market Price": request.form.get("price_target", "").strip(),
            "Demand": request.form.get("demand_target", "").strip(),
        }
        feature_columns = request.form.getlist("feature_columns")
        targets_in_use = [value for value in selected_targets.values() if value]

        if not targets_in_use:
            flash("Select at least one target column for yield, price, or demand.", "error")
            return redirect(url_for("agribusiness_index"))

        if not feature_columns:
            feature_columns = default_feature_columns(dataframe.columns.tolist(), targets_in_use)

        feature_columns = [column for column in feature_columns if column not in targets_in_use]
        if not feature_columns:
            flash("Choose at least one feature column after excluding target variables.", "error")
            return redirect(url_for("agribusiness_index"))

        use_holdout = request.form.get("use_holdout") == "on"
        use_cv = request.form.get("use_cv") == "on"
        test_size = max(0.2, float(request.form.get("test_size", "0.2")))
        cv_folds = int(request.form.get("cv_folds", "5"))
        random_state = int(request.form.get("random_state", "42"))

        results: list[dict[str, Any]] = []
        for forecast_label, target_column in selected_targets.items():
            if not target_column:
                continue
            results.append(
                analyze_target(
                    dataframe=dataframe,
                    feature_columns=feature_columns,
                    target_column=target_column,
                    forecast_label=forecast_label,
                    test_size=test_size,
                    random_state=random_state,
                    use_holdout=use_holdout,
                    use_cv=use_cv,
                    cv_folds=cv_folds,
                )
            )
    except Exception as exc:
        flash(f"Analysis could not be completed: {exc}", "error")
        return redirect(url_for("agribusiness_index"))

    return render_template(
        "results.html",
        filename=filename,
        results=results,
        use_holdout=use_holdout,
        use_cv=use_cv,
        feature_columns=feature_columns,
    )


if __name__ == "__main__":
    app.run(debug=True)
