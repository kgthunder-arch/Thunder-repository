from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import secrets
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from functools import wraps
from io import BytesIO
from pathlib import Path
from typing import Any
from urllib.error import HTTPError
from urllib.parse import quote, urlencode, urlparse
from urllib.request import Request, urlopen

os.environ.setdefault("JOBLIB_MULTIPROCESSING", "0")

import pandas as pd
from flask import Flask, flash, g, redirect, render_template, request, session, url_for
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
PERSISTENT_AUTH_DIR = os.path.join(PERSISTENT_STORAGE_DIR, "auth")
PERSISTENT_AUTH_USERS_DIR = os.path.join(PERSISTENT_AUTH_DIR, "users")
PERSISTENT_AUTH_PENDING_DIR = os.path.join(PERSISTENT_AUTH_DIR, "pending")
SAMPLE_DATA_PATH = os.path.join(BASE_DIR, "sample_data", "agribusiness_sample.csv")
ALLOWED_EXTENSIONS = {"csv", "xlsx", "xls"}
RECENT_ANALYSIS_LIMIT = 4
RESEND_API_KEY = os.getenv("RESEND_API_KEY")
EMAIL_FROM = os.getenv("EMAIL_FROM") or os.getenv("AUTH_EMAIL_FROM")
VERIFICATION_CODE_TTL_MINUTES = 10
VERIFICATION_RESEND_COOLDOWN_SECONDS = 60
PROTECTED_ENDPOINTS = {"view_saved_analysis"}
EMAIL_PATTERN = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
if not IS_VERCEL:
    os.makedirs(PERSISTENT_UPLOADS_DIR, exist_ok=True)
    os.makedirs(PERSISTENT_ANALYSES_DIR, exist_ok=True)
    os.makedirs(PERSISTENT_AUTH_USERS_DIR, exist_ok=True)
    os.makedirs(PERSISTENT_AUTH_PENDING_DIR, exist_ok=True)

app = Flask(__name__)
app.config["SECRET_KEY"] = "agribusiness-predictive-analysis-secret"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


def login_required(view: Any) -> Any:
    @wraps(view)
    def wrapped_view(*args: Any, **kwargs: Any) -> Any:
        if not auth_email_enabled():
            return view(*args, **kwargs)
        if not current_user_email():
            target = request.full_path if request.query_string else request.path
            return redirect(url_for("login", next=verification_redirect_target(target)))
        return view(*args, **kwargs)

    return wrapped_view


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


def auth_storage_mode() -> str:
    if blob_storage_enabled():
        return "blob"
    return "local"


def auth_email_enabled() -> bool:
    return bool(RESEND_API_KEY and EMAIL_FROM)


def auth_status() -> dict[str, str]:
    if auth_email_enabled():
        return {
            "enabled": "true",
            "label": "Verified email login is active",
            "description": "A real verification code and sign-in link will be sent to each user email address.",
        }
    return {
        "enabled": "false",
        "label": "Email delivery still needs sender setup",
        "description": "Configure RESEND_API_KEY and EMAIL_FROM with a verified sender address to activate real email verification.",
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
            payload = response.read().decode("utf-8")
            return json.loads(payload) if payload else {}
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


def delete_blob(pathname: str) -> None:
    blob_api_json_request("DELETE", query={"pathname": pathname})


def auth_user_key(email: str) -> str:
    return hashlib.sha256(email.encode("utf-8")).hexdigest()


def auth_user_record_path(email: str) -> str:
    return os.path.join(PERSISTENT_AUTH_USERS_DIR, f"{auth_user_key(email)}.json")


def auth_pending_record_path(email: str) -> str:
    return os.path.join(PERSISTENT_AUTH_PENDING_DIR, f"{auth_user_key(email)}.json")


def auth_blob_user_path(email: str) -> str:
    return f"auth/users/{auth_user_key(email)}.json"


def auth_blob_pending_path(email: str) -> str:
    return f"auth/pending/{auth_user_key(email)}.json"


def save_json_record(record_path: str, payload: dict[str, Any], *, content_type: str = "application/json") -> None:
    serialized = json.dumps(payload, ensure_ascii=True).encode("utf-8")
    if auth_storage_mode() == "blob":
        put_blob(
            record_path,
            serialized,
            access="private",
            add_random_suffix=False,
            content_type=content_type,
            overwrite=True,
        )
        return

    os.makedirs(os.path.dirname(record_path), exist_ok=True)
    with open(record_path, "wb") as output_file:
        output_file.write(serialized)


def load_json_record(record_path: str) -> dict[str, Any] | None:
    try:
        if auth_storage_mode() == "blob":
            raw_bytes = load_blob_bytes(record_path)
        else:
            if not os.path.exists(record_path):
                return None
            with open(record_path, "rb") as input_file:
                raw_bytes = input_file.read()
    except FileNotFoundError:
        return None

    return json.loads(raw_bytes.decode("utf-8"))


def delete_json_record(record_path: str) -> None:
    if auth_storage_mode() == "blob":
        try:
            delete_blob(record_path)
        except RuntimeError:
            return
        return

    if os.path.exists(record_path):
        os.remove(record_path)


def normalize_email(email: str) -> str:
    return email.strip().lower()


def valid_email_address(email: str) -> bool:
    return bool(EMAIL_PATTERN.match(email))


def hash_secret(secret_value: str) -> str:
    return hashlib.sha256(secret_value.encode("utf-8")).hexdigest()


def auth_user_record(email: str) -> dict[str, Any] | None:
    record_path = auth_blob_user_path(email) if auth_storage_mode() == "blob" else auth_user_record_path(email)
    return load_json_record(record_path)


def save_auth_user(email: str, payload: dict[str, Any]) -> dict[str, Any]:
    existing = auth_user_record(email) or {}
    created_at = existing.get("created_at") or utc_now().isoformat()
    record = {
        "email": email,
        "created_at": created_at,
        "verified_at": existing.get("verified_at") or payload.get("verified_at") or utc_now().isoformat(),
        "last_login_at": payload.get("last_login_at") or utc_now().isoformat(),
        "login_count": int(existing.get("login_count", 0)) + 1,
    }
    record_path = auth_blob_user_path(email) if auth_storage_mode() == "blob" else auth_user_record_path(email)
    save_json_record(record_path, record)
    return record


def pending_auth_record(email: str) -> dict[str, Any] | None:
    record_path = auth_blob_pending_path(email) if auth_storage_mode() == "blob" else auth_pending_record_path(email)
    return load_json_record(record_path)


def save_pending_auth(email: str, payload: dict[str, Any]) -> None:
    record_path = auth_blob_pending_path(email) if auth_storage_mode() == "blob" else auth_pending_record_path(email)
    save_json_record(record_path, payload)


def clear_pending_auth(email: str) -> None:
    record_path = auth_blob_pending_path(email) if auth_storage_mode() == "blob" else auth_pending_record_path(email)
    delete_json_record(record_path)


def verification_expiry_iso() -> str:
    return utc_now().replace(microsecond=0).isoformat()


def current_user_email() -> str | None:
    return getattr(g, "current_user_email", None)


def verification_redirect_target(raw_target: str | None) -> str:
    if not raw_target:
        return url_for("agribusiness_index")
    parsed = urlparse(raw_target)
    if parsed.scheme or parsed.netloc or not parsed.path.startswith("/"):
        return url_for("agribusiness_index")
    return raw_target


def current_app_base_url() -> str:
    configured = os.getenv("APP_BASE_URL")
    if configured:
        return configured.rstrip("/")
    if request.host_url:
        return request.host_url.rstrip("/")
    return "http://127.0.0.1:5000"


def resend_email_request(to_email: str, subject: str, html: str, text: str) -> None:
    if not auth_email_enabled():
        raise RuntimeError(
            "Real email delivery is not configured yet. Set RESEND_API_KEY and EMAIL_FROM with a verified sender address."
        )

    request = Request(
        "https://api.resend.com/emails",
        data=json.dumps(
            {
                "from": EMAIL_FROM,
                "to": [to_email],
                "subject": subject,
                "html": html,
                "text": text,
            }
        ).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {RESEND_API_KEY}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urlopen(request, timeout=30) as response:
            response.read()
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"Verification email could not be sent ({exc.code}): {detail}") from exc


def build_verification_email(email: str, code: str, link: str) -> tuple[str, str]:
    html = f"""
    <div style="background:#f3ecdd;padding:32px 16px;font-family:Trebuchet MS,Segoe UI,sans-serif;color:#182218;">
        <div style="max-width:560px;margin:0 auto;background:#fffaf3;border-radius:24px;padding:32px;border:1px solid rgba(79,52,34,0.12);">
            <p style="margin:0 0 10px;font-size:12px;letter-spacing:0.16em;text-transform:uppercase;color:#355e3b;">AgriCast Intelligence</p>
            <h1 style="margin:0 0 14px;font-family:Georgia,Times New Roman,serif;font-size:32px;line-height:1.1;color:#182218;">Verify your email to enter the forecasting workspace</h1>
            <p style="margin:0 0 18px;font-size:16px;line-height:1.7;color:#4f3422;">Use the code below or tap the verification button. This email was requested for {email}.</p>
            <div style="margin:0 0 20px;padding:18px 22px;border-radius:18px;background:#edf4ea;border:1px solid rgba(53,94,59,0.12);font-size:30px;font-weight:700;letter-spacing:0.18em;text-align:center;color:#355e3b;">{code}</div>
            <p style="margin:0 0 24px;text-align:center;">
                <a href="{link}" style="display:inline-block;padding:14px 24px;border-radius:999px;background:#355e3b;color:#f8f6f0;text-decoration:none;font-weight:700;">Verify Email</a>
            </p>
            <p style="margin:0;font-size:14px;line-height:1.6;color:#6b5b4d;">This link and code expire in {VERIFICATION_CODE_TTL_MINUTES} minutes.</p>
        </div>
    </div>
    """
    text = (
        f"Verify your AgriCast Intelligence login.\n\n"
        f"Verification code: {code}\n"
        f"Verification link: {link}\n\n"
        f"This code expires in {VERIFICATION_CODE_TTL_MINUTES} minutes."
    )
    return html, text


def create_pending_verification(email: str, next_target: str) -> dict[str, Any]:
    existing = pending_auth_record(email)
    now = utc_now()
    if existing is not None:
        last_sent_at = datetime.fromisoformat(existing["last_sent_at"])
        elapsed_seconds = (now - last_sent_at).total_seconds()
        if elapsed_seconds < VERIFICATION_RESEND_COOLDOWN_SECONDS:
            wait_seconds = int(VERIFICATION_RESEND_COOLDOWN_SECONDS - elapsed_seconds)
            raise RuntimeError(f"Please wait {wait_seconds} second(s) before requesting another code.")

    code = f"{secrets.randbelow(1_000_000):06d}"
    token = secrets.token_urlsafe(24)
    expires_at = now.replace(microsecond=0) + timedelta(minutes=VERIFICATION_CODE_TTL_MINUTES)
    record = {
        "email": email,
        "code_hash": hash_secret(code),
        "token_hash": hash_secret(token),
        "created_at": now.isoformat(),
        "last_sent_at": now.isoformat(),
        "expires_at": expires_at.isoformat(),
        "next_target": verification_redirect_target(next_target),
    }
    save_pending_auth(email, record)
    return {"code": code, "token": token, "record": record}


def pending_verification_valid(record: dict[str, Any]) -> bool:
    return utc_now() <= datetime.fromisoformat(record["expires_at"])


def complete_verified_login(email: str) -> str:
    pending = pending_auth_record(email)
    if pending is None or not pending_verification_valid(pending):
        clear_pending_auth(email)
        raise RuntimeError("That verification request has expired. Request a new code and try again.")

    user_record = save_auth_user(
        email,
        {
            "verified_at": utc_now().isoformat(),
            "last_login_at": utc_now().isoformat(),
        },
    )
    clear_pending_auth(email)
    session["user_email"] = user_record["email"]
    session["user_verified_at"] = user_record["verified_at"]
    return pending.get("next_target", url_for("agribusiness_index"))


@app.before_request
def load_authenticated_user() -> Any:
    g.current_user_email = None
    user_email = normalize_email(session.get("user_email", "")) if session.get("user_email") else None
    if user_email:
        record = auth_user_record(user_email)
        if record is None:
            session.clear()
        else:
            g.current_user_email = record["email"]
            g.current_user = record

    if request.endpoint is None or request.endpoint.startswith("static"):
        return None
    if request.endpoint not in PROTECTED_ENDPOINTS:
        return None
    if not auth_email_enabled():
        return None
    if g.current_user_email:
        return None

    target = request.full_path if request.query_string else request.path
    return redirect(url_for("login", next=verification_redirect_target(target)))


@app.context_processor
def auth_template_context() -> dict[str, Any]:
    return {
        "current_user_email": current_user_email(),
        "auth_status": auth_status(),
    }


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
        "user_email": current_user_email(),
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

    snapshot = json.loads(raw_bytes.decode("utf-8"))
    if auth_email_enabled() and snapshot.get("user_email") != current_user_email():
        return None
    return snapshot


def recent_saved_analyses(limit: int = RECENT_ANALYSIS_LIMIT) -> list[SavedAnalysisSummary]:
    storage_mode = active_storage_mode()
    records: list[dict[str, Any]] = []
    signed_in_user = current_user_email()
    if auth_email_enabled() and not signed_in_user:
        return []

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

    if auth_email_enabled():
        records = [record for record in records if record.get("user_email") == signed_in_user]
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


@app.route("/login", methods=["GET", "POST"])
def login() -> str:
    if current_user_email():
        return redirect(url_for("agribusiness_index"))

    next_target = verification_redirect_target(request.values.get("next"))
    email_prefill = normalize_email(request.values.get("email", ""))
    if request.method == "POST":
        if not auth_email_enabled():
            flash(auth_status()["description"], "warning")
            return render_template("login.html", next_target=next_target, email=email_prefill)

        email = normalize_email(request.form.get("email", ""))
        if not valid_email_address(email):
            flash("Enter a valid email address to receive a real verification message.", "error")
            return render_template("login.html", next_target=next_target, email=email)

        try:
            verification = create_pending_verification(email, next_target)
            verification_link = (
                f"{current_app_base_url()}{url_for('verify_email')}"
                f"?email={quote(email)}&token={quote(verification['token'])}"
            )
            html, text = build_verification_email(email, verification["code"], verification_link)
            resend_email_request(email, "Verify your AgriCast Intelligence login", html, text)
            flash(f"We sent a verification code to {email}. Check your inbox and spam folder.", "warning")
            return redirect(url_for("verify_email", email=email))
        except Exception as exc:
            clear_pending_auth(email)
            flash(str(exc), "error")

        return render_template("login.html", next_target=next_target, email=email)

    return render_template("login.html", next_target=next_target, email=email_prefill)


@app.route("/verify", methods=["GET", "POST"])
def verify_email() -> str:
    if current_user_email():
        return redirect(url_for("agribusiness_index"))
    if not auth_email_enabled():
        flash(auth_status()["description"], "warning")
        return redirect(url_for("login"))

    email = normalize_email(request.values.get("email", ""))
    pending = pending_auth_record(email) if email else None
    token = request.values.get("token", "").strip()

    if token and pending is not None:
        if not pending_verification_valid(pending):
            clear_pending_auth(email)
            flash("That verification link has expired. Request a fresh email code.", "error")
            return redirect(url_for("login"))
        if hmac.compare_digest(hash_secret(token), pending.get("token_hash", "")):
            try:
                flash("Email verified. You are now signed in.", "warning")
                return redirect(complete_verified_login(email))
            except Exception as exc:
                flash(str(exc), "error")
                return redirect(url_for("login"))

    if request.method == "POST":
        code = request.form.get("code", "").strip()
        if not email or pending is None:
            flash("Start from the login page so we can send you a valid verification code.", "error")
            return redirect(url_for("login"))
        if not pending_verification_valid(pending):
            clear_pending_auth(email)
            flash("That verification code has expired. Request a new one.", "error")
            return redirect(url_for("login"))
        if not hmac.compare_digest(hash_secret(code), pending.get("code_hash", "")):
            flash("That verification code does not match the latest email we sent.", "error")
            return render_template(
                "verify.html",
                email=email,
                expires_at=humanize_timestamp(pending.get("expires_at", "")),
            )
        flash("Email verified. You are now signed in.", "warning")
        return redirect(complete_verified_login(email))

    if not email or pending is None:
        flash("Request a verification email before trying to sign in.", "error")
        return redirect(url_for("login"))

    return render_template(
        "verify.html",
        email=email,
        expires_at=humanize_timestamp(pending.get("expires_at", "")),
    )


@app.get("/logout")
def logout() -> Any:
    session.clear()
    flash("You have been signed out.", "warning")
    return redirect(url_for("login"))


@app.get("/favicon.ico")
def favicon() -> Any:
    return redirect(url_for("static", filename="favicon.svg"), code=302)


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
