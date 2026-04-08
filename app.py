from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any
from urllib.error import HTTPError
from urllib.parse import parse_qsl, quote, urlencode, urlparse, urlunparse
from urllib.request import Request, urlopen

os.environ.setdefault("JOBLIB_MULTIPROCESSING", "0")

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
IS_VERCEL = bool(os.getenv("VERCEL"))
BLOB_TOKEN = os.getenv("BLOB_READ_WRITE_TOKEN")
BLOB_API_URL = (
    os.getenv("VERCEL_BLOB_API_URL")
    or os.getenv("NEXT_PUBLIC_VERCEL_BLOB_API_URL")
    or "https://vercel.com/api/blob"
)
BLOB_API_VERSION = (
    os.getenv("VERCEL_BLOB_API_VERSION_OVERRIDE")
    or os.getenv("NEXT_PUBLIC_VERCEL_BLOB_API_VERSION_OVERRIDE")
    or "11"
)
UPLOAD_FOLDER = os.path.join("/tmp", "uploads") if IS_VERCEL else os.path.join(BASE_DIR, "uploads")
PERSISTENT_STORAGE_DIR = os.path.join(BASE_DIR, "storage")
PERSISTENT_UPLOADS_DIR = os.path.join(PERSISTENT_STORAGE_DIR, "uploads")
PERSISTENT_ANALYSES_DIR = os.path.join(PERSISTENT_STORAGE_DIR, "analyses")
SAMPLE_DATA_PATH = os.path.join(BASE_DIR, "sample_data", "agribusiness_sample.csv")
ALLOWED_EXTENSIONS = {"csv", "xlsx", "xls"}
RECENT_ANALYSIS_LIMIT = 4

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
if not IS_VERCEL:
    os.makedirs(PERSISTENT_UPLOADS_DIR, exist_ok=True)
    os.makedirs(PERSISTENT_ANALYSES_DIR, exist_ok=True)

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
    source_kind: str
    storage_label: str
    validation_report: "DatasetValidationReport"


@dataclass
class DatasetValidationReport:
    blocking_issues: list[str]
    warnings: list[str]
    duplicate_rows: int
    missing_cell_count: int
    parseable_date_columns: list[str]
    suspicious_date_columns: list[str]
    high_missing_columns: list[str]
    missing_target_types: list[str]
    non_numeric_target_columns: list[str]


@dataclass
class SavedAnalysisSummary:
    analysis_id: str
    filename: str
    created_at_label: str
    targets: list[str]
    best_models: list[str]
    storage_label: str


@dataclass
class BlobPutResult:
    pathname: str


@dataclass
class BlobListItem:
    pathname: str
    uploaded_at: str


@dataclass
class BlobListResult:
    blobs: list[BlobListItem]


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def blob_storage_enabled() -> bool:
    return bool(BLOB_TOKEN)


def active_storage_mode() -> str:
    if blob_storage_enabled():
        return "blob"
    if IS_VERCEL:
        return "session"
    return "local"


def storage_label_for_source(source_kind: str) -> str:
    labels = {
        "sample": "Bundled sample dataset",
        "blob": "Durable Vercel Blob storage",
        "local": "Local saved storage",
        "session": "Temporary Vercel runtime storage",
    }
    return labels.get(source_kind, "Dataset storage")


def current_storage_status() -> dict[str, str]:
    mode = active_storage_mode()
    descriptions = {
        "blob": "Uploads and saved analyses are stored durably in Vercel Blob.",
        "local": "Uploads and saved analyses are stored locally on disk for development.",
        "session": "Uploads use temporary runtime storage. Add BLOB_READ_WRITE_TOKEN to enable durable Vercel persistence.",
    }
    return {
        "mode": mode,
        "label": storage_label_for_source(mode),
        "description": descriptions[mode],
    }


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def humanize_timestamp(value: str) -> str:
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return value
    return parsed.astimezone(timezone.utc).strftime("%b %d, %Y %H:%M UTC")


def blob_store_id(token: str) -> str:
    parts = token.split("_")
    return parts[3] if len(parts) > 3 else ""


def blob_request_headers(extra_headers: dict[str, str] | None = None) -> dict[str, str]:
    headers = {
        "Authorization": f"Bearer {BLOB_TOKEN}",
        "x-api-version": BLOB_API_VERSION,
    }
    if extra_headers:
        headers.update(extra_headers)
    return headers


def blob_api_json_request(
    method: str,
    *,
    query: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
    body: bytes | None = None,
) -> dict[str, Any]:
    url = BLOB_API_URL
    if query:
        filtered = {key: value for key, value in query.items() if value is not None}
        if filtered:
            url = f"{url}?{urlencode(filtered)}"

    request = Request(
        url,
        data=body,
        headers=blob_request_headers(headers),
        method=method,
    )
    try:
        with urlopen(request, timeout=30) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"Blob API request failed ({exc.code}): {detail}") from exc


def blob_download_url(pathname: str) -> str:
    store_id = blob_store_id(BLOB_TOKEN or "")
    if not store_id:
        metadata = blob_api_json_request("GET", query={"url": pathname})
        return metadata["url"]

    safe_pathname = quote(pathname.lstrip("/"), safe="/")
    return f"https://{store_id}.private.blob.vercel-storage.com/{safe_pathname}"


def put_blob(
    path: str,
    body: bytes,
    *,
    access: str,
    add_random_suffix: bool,
    content_type: str | None,
    overwrite: bool = False,
) -> BlobPutResult:
    response = blob_api_json_request(
        "PUT",
        query={"pathname": path},
        headers={
            "x-vercel-blob-access": access,
            "x-add-random-suffix": "1" if add_random_suffix else "0",
            "x-allow-overwrite": "1" if overwrite else "0",
            **({"x-content-type": content_type} if content_type else {}),
        },
        body=body,
    )
    return BlobPutResult(pathname=response["pathname"])


def list_objects(*, limit: int | None = None, prefix: str | None = None) -> BlobListResult:
    response = blob_api_json_request(
        "GET",
        query={
            "limit": limit,
            "prefix": prefix,
        },
    )
    blobs = [
        BlobListItem(
            pathname=item["pathname"],
            uploaded_at=item.get("uploadedAt", ""),
        )
        for item in response.get("blobs", [])
    ]
    return BlobListResult(blobs=blobs)


def read_dataset(file_path: str) -> pd.DataFrame:
    extension = file_path.rsplit(".", 1)[1].lower()
    if extension == "csv":
        return pd.read_csv(file_path)
    return pd.read_excel(file_path)


def read_dataset_content(filename: str, content: bytes) -> pd.DataFrame:
    extension = filename.rsplit(".", 1)[1].lower()
    buffer = BytesIO(content)
    if extension == "csv":
        return pd.read_csv(buffer)
    return pd.read_excel(buffer)


def load_blob_bytes(pathname: str) -> bytes:
    request = Request(
        blob_download_url(pathname),
        headers={"Authorization": f"Bearer {BLOB_TOKEN}"},
        method="GET",
    )
    try:
        with urlopen(request, timeout=30) as response:
            return response.read()
    except HTTPError as exc:
        if exc.code == 404:
            raise FileNotFoundError(f"Blob not found: {pathname}") from exc
        detail = exc.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"Blob download failed ({exc.code}): {detail}") from exc


def load_dataset_from_reference(source_kind: str, reference: str, filename: str) -> pd.DataFrame:
    if source_kind in {"sample", "local", "session"}:
        if not os.path.exists(reference):
            raise FileNotFoundError(reference)
        return read_dataset(reference)
    if source_kind == "blob":
        return read_dataset_content(filename, load_blob_bytes(reference))
    raise ValueError(f"Unsupported dataset source: {source_kind}")


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


def format_percent(value: float) -> str:
    return f"{value * 100:.0f}%"


def date_column_quality(dataframe: pd.DataFrame) -> tuple[list[str], list[str]]:
    parseable: list[str] = []
    suspicious: list[str] = []
    keywords = ("date", "time")

    for column in dataframe.columns:
        series = dataframe[column]
        if pd.api.types.is_datetime64_any_dtype(series):
            parseable.append(column)
            continue

        if not pd.api.types.is_object_dtype(series) and not pd.api.types.is_string_dtype(series):
            continue

        if not any(keyword in column.lower() for keyword in keywords):
            continue

        non_empty = series.astype("string").str.strip().replace({"": pd.NA})
        populated = int(non_empty.notna().sum())
        if populated == 0:
            continue

        parsed = pd.to_datetime(series, errors="coerce")
        parse_ratio = float(parsed.notna().sum() / populated)
        if parse_ratio >= 0.75:
            parseable.append(column)
        else:
            suspicious.append(column)

    return parseable, suspicious


def build_validation_report(dataframe: pd.DataFrame, suggestions: dict[str, str]) -> DatasetValidationReport:
    blocking_issues: list[str] = []
    warnings: list[str] = []

    if dataframe.empty:
        blocking_issues.append("The dataset has no rows to analyze.")
    if dataframe.shape[1] < 2:
        blocking_issues.append("Add at least two columns so the app has both targets and predictors to work with.")

    numeric_columns = dataframe.select_dtypes(include="number").columns.tolist()
    if not numeric_columns:
        blocking_issues.append("No numeric columns were detected. Add numeric yield, price, weather, or demand fields.")

    duplicate_rows = int(dataframe.duplicated().sum())
    if duplicate_rows:
        warnings.append(
            f"{duplicate_rows} duplicate row{'s' if duplicate_rows != 1 else ''} detected. Duplicate records can overstate model confidence."
        )

    missing_cell_count = int(dataframe.isna().sum().sum())
    high_missing_columns: list[str] = []
    for column in dataframe.columns:
        missing_ratio = float(dataframe[column].isna().mean())
        if missing_ratio >= 0.2:
            high_missing_columns.append(f"{column} ({format_percent(missing_ratio)} missing)")
    if high_missing_columns:
        warnings.append(
            "High missing-value columns detected: " + ", ".join(high_missing_columns[:4]) + "."
        )

    parseable_date_columns, suspicious_date_columns = date_column_quality(dataframe)
    if suspicious_date_columns:
        warnings.append(
            "Date-like columns with inconsistent formatting: " + ", ".join(suspicious_date_columns) + "."
        )

    missing_target_types = [name.title() for name, column in suggestions.items() if not column]
    if missing_target_types:
        warnings.append(
            "No obvious " + ", ".join(target.lower() for target in missing_target_types) + " target column was detected automatically."
        )

    non_numeric_target_columns: list[str] = []
    for forecast_type, column in suggestions.items():
        if not column:
            continue
        populated = dataframe[column].dropna()
        if populated.empty:
            non_numeric_target_columns.append(column)
            continue
        numeric_ratio = float(pd.to_numeric(populated, errors="coerce").notna().mean())
        if numeric_ratio < 0.8:
            non_numeric_target_columns.append(column)

    if non_numeric_target_columns:
        warnings.append(
            "Suggested targets that do not look numeric enough for forecasting: "
            + ", ".join(non_numeric_target_columns)
            + "."
        )

    return DatasetValidationReport(
        blocking_issues=blocking_issues,
        warnings=warnings,
        duplicate_rows=duplicate_rows,
        missing_cell_count=missing_cell_count,
        parseable_date_columns=parseable_date_columns,
        suspicious_date_columns=suspicious_date_columns,
        high_missing_columns=high_missing_columns,
        missing_target_types=missing_target_types,
        non_numeric_target_columns=non_numeric_target_columns,
    )


def build_dataset_summary(
    dataframe: pd.DataFrame,
    filename: str,
    reference: str,
    source_kind: str,
) -> DatasetSummary:
    preview = dataframe.head(8).to_html(
        classes="data-table preview-table",
        index=False,
        border=0,
        justify="left",
    )
    numeric_columns = dataframe.select_dtypes(include="number").columns.tolist()
    suggestions = suggest_targets(dataframe.columns.tolist())
    return DatasetSummary(
        path=reference,
        filename=filename,
        row_count=int(dataframe.shape[0]),
        column_count=int(dataframe.shape[1]),
        columns=dataframe.columns.tolist(),
        numeric_columns=numeric_columns,
        preview_html=preview,
        suggestions=suggestions,
        source_kind=source_kind,
        storage_label=storage_label_for_source(source_kind),
        validation_report=build_validation_report(dataframe, suggestions),
    )


def summarize_dataset(file_path: str, filename: str, source_kind: str) -> DatasetSummary:
    return build_dataset_summary(read_dataset(file_path), filename, file_path, source_kind)


def persist_uploaded_dataset(uploaded_file: Any) -> tuple[str, str, pd.DataFrame, str]:
    safe_name = secure_filename(uploaded_file.filename or "dataset.csv")
    if not allowed_file(safe_name):
        raise ValueError("Upload a CSV or Excel dataset file.")

    content = uploaded_file.read()
    dataset_id = uuid.uuid4().hex
    content_type = uploaded_file.mimetype or "application/octet-stream"
    dataframe = read_dataset_content(safe_name, content)
    storage_mode = active_storage_mode()

    if storage_mode == "blob":
        blob = put_blob(
            f"datasets/{dataset_id}-{safe_name}",
            content,
            access="private",
            add_random_suffix=False,
            content_type=content_type,
        )
        return "blob", blob.pathname, dataframe, safe_name

    target_dir = PERSISTENT_UPLOADS_DIR if storage_mode == "local" else UPLOAD_FOLDER
    stored_path = os.path.join(target_dir, f"{dataset_id}_{safe_name}")
    with open(stored_path, "wb") as saved_file:
        saved_file.write(content)
    return storage_mode, stored_path, dataframe, safe_name


def local_analysis_path(analysis_id: str) -> str:
    return os.path.join(PERSISTENT_ANALYSES_DIR, f"{analysis_id}.json")


def session_analysis_path(analysis_id: str) -> str:
    return os.path.join(UPLOAD_FOLDER, f"analysis_{analysis_id}.json")


def save_analysis_snapshot(payload: dict[str, Any]) -> dict[str, Any]:
    storage_mode = active_storage_mode()
    analysis_id = uuid.uuid4().hex[:12]
    saved_at = utc_now().isoformat()
    record = {
        "analysis_id": analysis_id,
        "saved_at": saved_at,
        "storage_mode": storage_mode,
        "storage_label": storage_label_for_source(storage_mode),
        **payload,
    }
    serialized = json.dumps(record, ensure_ascii=True).encode("utf-8")

    if storage_mode == "blob":
        put_blob(
            f"analyses/{analysis_id}.json",
            serialized,
            access="private",
            add_random_suffix=False,
            content_type="application/json",
        )
    else:
        output_path = local_analysis_path(analysis_id) if storage_mode == "local" else session_analysis_path(analysis_id)
        with open(output_path, "wb") as output_file:
            output_file.write(serialized)

    return record


def load_analysis_snapshot(analysis_id: str) -> dict[str, Any] | None:
    storage_mode = active_storage_mode()
    try:
        if storage_mode == "blob":
            raw_bytes = load_blob_bytes(f"analyses/{analysis_id}.json")
        else:
            snapshot_path = local_analysis_path(analysis_id) if storage_mode == "local" else session_analysis_path(analysis_id)
            if not os.path.exists(snapshot_path):
                return None
            with open(snapshot_path, "rb") as snapshot_file:
                raw_bytes = snapshot_file.read()
    except FileNotFoundError:
        return None

    return json.loads(raw_bytes.decode("utf-8"))


def recent_saved_analyses(limit: int = RECENT_ANALYSIS_LIMIT) -> list[SavedAnalysisSummary]:
    storage_mode = active_storage_mode()
    records: list[dict[str, Any]] = []

    if storage_mode == "blob":
        listing = list_objects(prefix="analyses/", limit=50)
        for item in listing.blobs:
            if not item.pathname.endswith(".json"):
                continue
            try:
                record = json.loads(load_blob_bytes(item.pathname).decode("utf-8"))
            except Exception:
                continue
            records.append(record)
    elif storage_mode == "local":
        analysis_dir = Path(PERSISTENT_ANALYSES_DIR)
        for path in analysis_dir.glob("*.json"):
            try:
                records.append(json.loads(path.read_text(encoding="utf-8")))
            except Exception:
                continue
    else:
        return []

    records.sort(key=lambda item: item.get("saved_at", ""), reverse=True)
    summaries: list[SavedAnalysisSummary] = []
    for record in records[:limit]:
        summaries.append(
            SavedAnalysisSummary(
                analysis_id=record["analysis_id"],
                filename=record.get("filename", "dataset"),
                created_at_label=humanize_timestamp(record.get("saved_at", "")),
                targets=[result["forecast_label"] for result in record.get("results", [])],
                best_models=[
                    f"{result['forecast_label']}: {result['best_model']}"
                    for result in record.get("results", [])
                ],
                storage_label=record.get("storage_label", storage_label_for_source(record.get("storage_mode", "local"))),
            )
        )
    return summaries


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


def format_engineered_feature_name(name: str) -> str:
    cleaned = name.replace("numeric__", "").replace("categorical__", "")
    cleaned = cleaned.replace("_", " ")
    for suffix in (" year", " month", " quarter", " day of year"):
        if cleaned.endswith(suffix):
            return cleaned
    return cleaned


def create_importance_chart(
    importance_df: pd.DataFrame,
    forecast_label: str,
    model_name: str,
) -> dict[str, Any]:
    trimmed = importance_df.head(8).iloc[::-1]
    return chart_payload(
        data=[
            {
                "type": "bar",
                "orientation": "h",
                "x": trimmed["Importance"].round(4).tolist(),
                "y": trimmed["Feature"].tolist(),
                "marker": {"color": "#5f8d4e"},
                "name": "Importance",
            }
        ],
        layout={
            "title": {"text": f"{forecast_label} Top Drivers ({model_name})"},
            "xaxis": {"title": {"text": "Relative importance"}},
            "paper_bgcolor": "rgba(0,0,0,0)",
            "plot_bgcolor": "rgba(0,0,0,0)",
            "margin": {"l": 20, "r": 20, "t": 60, "b": 20},
        },
    )


def build_tree_explainability(
    pipeline: Pipeline,
    model_name: str,
    forecast_label: str,
) -> dict[str, Any] | None:
    fitted_model = pipeline.named_steps["model"]
    importances = getattr(fitted_model, "feature_importances_", None)
    if importances is None:
        return None

    feature_names = [
        format_engineered_feature_name(name)
        for name in pipeline.named_steps["preprocessor"].get_feature_names_out()
    ]
    importance_df = (
        pd.DataFrame({"Feature": feature_names, "Importance": importances})
        .sort_values(by="Importance", ascending=False)
        .head(8)
        .round(4)
    )
    if importance_df.empty:
        return None

    top_features = importance_df["Feature"].head(3).tolist()
    if len(top_features) == 1:
        top_feature_summary = top_features[0]
    elif len(top_features) == 2:
        top_feature_summary = f"{top_features[0]} and {top_features[1]}"
    else:
        top_feature_summary = f"{', '.join(top_features[:-1])}, and {top_features[-1]}"
    summary = f"{model_name} leaned most on {top_feature_summary} when scoring this forecast."

    return {
        "model_name": model_name,
        "summary": summary,
        "importance_table": importance_df.to_html(classes="data-table", index=False, border=0, justify="left"),
        "importance_chart": create_importance_chart(importance_df, forecast_label, model_name),
    }


def validate_selected_columns(
    dataframe: pd.DataFrame,
    selected_targets: dict[str, str],
    feature_columns: list[str],
) -> list[str]:
    errors: list[str] = []
    targets_in_use = [value for value in selected_targets.values() if value]

    if not targets_in_use:
        errors.append("Select at least one target column for yield, price, or demand.")
        return errors

    duplicate_targets = [column for column in set(targets_in_use) if targets_in_use.count(column) > 1]
    if duplicate_targets:
        errors.append("Each forecast target must use a different dataset column.")

    missing_columns = [column for column in targets_in_use + feature_columns if column not in dataframe.columns]
    if missing_columns:
        errors.append("Some selected columns no longer exist in the dataset. Reload the dataset and try again.")

    for forecast_label, target_column in selected_targets.items():
        if not target_column:
            continue
        populated = dataframe[target_column].dropna()
        if populated.empty:
            errors.append(f"{forecast_label} target '{target_column}' has no populated values.")
            continue
        numeric_ratio = float(pd.to_numeric(populated, errors="coerce").notna().mean())
        if numeric_ratio < 0.8:
            errors.append(
                f"{forecast_label} target '{target_column}' is not numeric enough for model training."
            )

    if not feature_columns:
        errors.append("Choose at least one feature column after excluding target variables.")

    return errors


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
    starting_rows = len(filtered)
    filtered = filtered.dropna(subset=[target_column])
    dropped_target_rows = starting_rows - len(filtered)

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
    explainability_models: list[dict[str, Any]] = []

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
        explainability = build_tree_explainability(pipeline, model_name, forecast_label)
        if explainability is not None:
            explainability_models.append(explainability)

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
    explainability_models.sort(key=lambda item: (item["model_name"] != best_model, item["model_name"]))

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
        "explainability_models": explainability_models,
        "dropped_target_rows": dropped_target_rows,
    }


def default_feature_columns(columns: list[str], selected_targets: list[str]) -> list[str]:
    excluded = set(selected_targets)
    return [column for column in columns if column not in excluded]


@app.get("/")
def agribusiness_index() -> str:
    sample_summary = summarize_dataset(SAMPLE_DATA_PATH, "agribusiness_sample.csv", "sample")
    return render_template(
        "index.html",
        sample_summary=sample_summary,
        storage_status=current_storage_status(),
        recent_analyses=recent_saved_analyses(),
    )


@app.get("/analysis/<analysis_id>")
def view_saved_analysis(analysis_id: str) -> str:
    snapshot = load_analysis_snapshot(analysis_id)
    if snapshot is None:
        flash("That saved analysis is no longer available in the current storage mode.", "error")
        return redirect(url_for("agribusiness_index"))

    return render_template(
        "results.html",
        filename=snapshot["filename"],
        results=snapshot["results"],
        use_holdout=snapshot["use_holdout"],
        use_cv=snapshot["use_cv"],
        feature_columns=snapshot["feature_columns"],
        analysis_summary={
            "analysis_id": snapshot["analysis_id"],
            "saved_at_label": humanize_timestamp(snapshot["saved_at"]),
            "storage_label": snapshot["storage_label"],
        },
    )


@app.post("/upload")
def upload() -> str:
    source = request.form.get("source", "sample")

    try:
        if source == "sample":
            summary = summarize_dataset(SAMPLE_DATA_PATH, "agribusiness_sample.csv", "sample")
        else:
            uploaded_file = request.files.get("dataset")
            if uploaded_file is None or uploaded_file.filename == "":
                flash("Please choose a dataset file to continue.", "error")
                return redirect(url_for("agribusiness_index"))

            source_kind, reference, dataframe, safe_name = persist_uploaded_dataset(uploaded_file)
            summary = build_dataset_summary(dataframe, safe_name, reference, source_kind)
    except Exception as exc:
        flash(f"Unable to read the dataset: {exc}", "error")
        return redirect(url_for("agribusiness_index"))

    return render_template("configure.html", summary=summary, storage_status=current_storage_status())


@app.post("/analyze")
def analyze() -> str:
    dataset_reference = request.form.get("dataset_path", "")
    dataset_source_kind = request.form.get("dataset_source_kind", "sample")
    filename = request.form.get("filename", "dataset")

    if not dataset_reference:
        flash("The dataset reference is missing. Upload the file again.", "error")
        return redirect(url_for("agribusiness_index"))

    try:
        dataframe = load_dataset_from_reference(dataset_source_kind, dataset_reference, filename)
        selected_targets = {
            "Crop Yield": request.form.get("yield_target", "").strip(),
            "Market Price": request.form.get("price_target", "").strip(),
            "Demand": request.form.get("demand_target", "").strip(),
        }
        feature_columns = request.form.getlist("feature_columns")
        targets_in_use = [value for value in selected_targets.values() if value]

        if not feature_columns:
            feature_columns = default_feature_columns(dataframe.columns.tolist(), targets_in_use)

        feature_columns = [column for column in feature_columns if column not in targets_in_use]
        validation_errors = validate_selected_columns(dataframe, selected_targets, feature_columns)
        if validation_errors:
            flash(" ".join(validation_errors), "error")
            return redirect(url_for("agribusiness_index"))

        duplicate_rows = int(dataframe.duplicated().sum())
        if duplicate_rows:
            flash(
                f"Dataset quality note: {duplicate_rows} duplicate row{'s' if duplicate_rows != 1 else ''} were included in training.",
                "warning",
            )

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
    except FileNotFoundError:
        flash("The dataset is no longer available. Upload it again to continue.", "error")
        return redirect(url_for("agribusiness_index"))
    except Exception as exc:
        flash(f"Analysis could not be completed: {exc}", "error")
        return redirect(url_for("agribusiness_index"))

    analysis_summary: dict[str, str] | None = None
    try:
        snapshot = save_analysis_snapshot(
            {
                "filename": filename,
                "dataset_reference": dataset_reference,
                "dataset_source_kind": dataset_source_kind,
                "feature_columns": feature_columns,
                "use_holdout": use_holdout,
                "use_cv": use_cv,
                "results": results,
            }
        )
        analysis_summary = {
            "analysis_id": snapshot["analysis_id"],
            "saved_at_label": humanize_timestamp(snapshot["saved_at"]),
            "storage_label": snapshot["storage_label"],
        }
    except Exception as exc:
        flash(f"Analysis finished, but the save step failed: {exc}", "error")

    return render_template(
        "results.html",
        filename=filename,
        results=results,
        use_holdout=use_holdout,
        use_cv=use_cv,
        feature_columns=feature_columns,
        analysis_summary=analysis_summary,
    )


if __name__ == "__main__":
    app.run(debug=True)
