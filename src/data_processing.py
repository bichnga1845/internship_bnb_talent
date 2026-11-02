from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

REQUIRED_COLUMNS = [
    "Lead ID",
    "Lead Date",
    "Source",
    "Product",
    "Budget",
    "Status",
    "Owner",
    "Customer Concerns",
    "Next Action",
    "Follow-up Status",
    "Last Interaction",
    "Loss Reason",
    "Status Classification",
]

VALID_STATUSES = {
    "Mới hỏi",
    "Đã tư vấn sơ bộ",
    "Chờ ảnh bếp",
    "Chờ khảo sát",
    "Chờ kỹ thuật xem",
    "Đã báo giá",
    "Đã mua",
    "Mất",
}

VALID_CLASSIFICATIONS = {"Open", "Converted", "Lost"}
STATUS_CLASSIFICATION_MAP = {
    "Mới hỏi": "Open",
    "Đã tư vấn sơ bộ": "Open",
    "Chờ ảnh bếp": "Open",
    "Chờ khảo sát": "Open",
    "Chờ kỹ thuật xem": "Open",
    "Đã báo giá": "Open",
    "Đã mua": "Converted",
    "Mất": "Lost",
}

VALID_FOLLOW_UP = {"Complete", "Pending", "Not recorded"}


def load_lead_file(file_path: str | Path) -> pd.DataFrame:
    """Load a CSV or Excel file into a pandas DataFrame."""
    path = Path(file_path)
    suffix = path.suffix.lower()

    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)

    raise ValueError("Unsupported file type. Please upload a CSV or Excel file.")


def _canonicalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names by trimming whitespace and matching common input variations."""
    renamed = df.copy()
    renamed.columns = [str(col).strip() for col in renamed.columns]
    return renamed


def _get_column_name(df: pd.DataFrame, column_name: str) -> str | None:
    normalized = {str(col).strip().lower(): col for col in df.columns}
    return normalized.get(column_name.lower())


def _coerce_datetime(series: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(series, errors="coerce")
    return parsed


def validate_lead_data(df: pd.DataFrame) -> dict[str, Any]:
    """Validate lead data and return a structured summary plus invalid rows.

    The function does not silently discard records. It reports the number of valid,
    invalid, and duplicate rows and returns the invalid subset for review.
    """
    working_df = _canonicalize_columns(df.copy())
    report: dict[str, Any] = {
        "total_rows": len(working_df),
        "valid_rows": 0,
        "invalid_rows": 0,
        "excluded_rows": 0,
        "missing_columns": [],
        "duplicate_lead_ids": [],
        "invalid_statuses": [],
        "invalid_dates": [],
        "invalid_budget_rows": [],
        "follow_up_issues": [],
        "loss_reason_issues": [],
        "classification_issues": [],
        "missing_values": {},
        "problem_rows": pd.DataFrame(columns=working_df.columns),
    }

    missing_columns = []
    for required in REQUIRED_COLUMNS:
        if _get_column_name(working_df, required) is None:
            missing_columns.append(required)

    report["missing_columns"] = missing_columns

    if missing_columns:
        report["excluded_rows"] = len(working_df)
        report["valid_rows"] = 0
        report["invalid_rows"] = len(working_df)
        report["problem_rows"] = working_df.copy()
        return report

    # Standardize the required columns to the expected names.
    rename_map = {}
    for required in REQUIRED_COLUMNS:
        actual = _get_column_name(working_df, required)
        if actual and actual != required:
            rename_map[actual] = required
    working_df = working_df.rename(columns=rename_map)

    issue_map: dict[int, list[str]] = {idx: [] for idx in range(len(working_df))}

    lead_id_series = working_df["Lead ID"].astype(str).fillna("")
    duplicate_ids = lead_id_series[lead_id_series.str.strip() != ""].duplicated(keep=False)
    duplicate_indices = duplicate_ids[duplicate_ids].index.tolist()
    for idx in duplicate_indices:
        issue_map[idx].append("Duplicate Lead ID")
    report["duplicate_lead_ids"] = sorted(duplicate_indices)

    # Missing required analytical fields.
    for column in ["Lead ID", "Lead Date", "Source", "Product", "Budget", "Status", "Owner", "Follow-up Status", "Last Interaction", "Status Classification"]:
        missing = working_df[column].isna() | (working_df[column].astype(str).str.strip() == "")
        if missing.any():
            report["missing_values"][column] = int(missing.sum())
            for idx in missing[missing].index.tolist():
                issue_map[idx].append(f"Missing value in {column}")
        else:
            report["missing_values"][column] = 0

    # Status validation.
    valid_status_flag = working_df["Status"].astype(str).str.strip().isin(VALID_STATUSES)
    invalid_status_rows = valid_status_flag[~valid_status_flag].index.tolist()
    for idx in invalid_status_rows:
        issue_map[idx].append("Invalid Status")
    report["invalid_statuses"] = invalid_status_rows

    # Date validation.
    for column in ["Lead Date", "Last Interaction"]:
        parsed = _coerce_datetime(working_df[column])
        invalid_date_rows = parsed.isna()[working_df[column].notna() & parsed.isna()].index.tolist()
        for idx in invalid_date_rows:
            issue_map[idx].append(f"Invalid date in {column}")
    report["invalid_dates"] = sorted(
        set(
            idx
            for idx in range(len(working_df))
            if any("Invalid date" in issue for issue in issue_map.get(idx, []))
        )
    )

    # Budget validation.
    budget_series = pd.to_numeric(working_df["Budget"], errors="coerce")
    budget_issues = (budget_series.isna()) | (budget_series < 0)
    budget_issue_rows = budget_issues[budget_issues].index.tolist()
    for idx in budget_issue_rows:
        issue_map[idx].append("Invalid Budget")
    report["invalid_budget_rows"] = budget_issue_rows

    # Follow-up status validation.
    follow_up_values = working_df["Follow-up Status"].astype(str).str.strip()
    follow_up_valid = follow_up_values.isin(VALID_FOLLOW_UP)
    follow_up_rows = follow_up_valid[~follow_up_valid].index.tolist()
    for idx in follow_up_rows:
        issue_map[idx].append("Invalid Follow-up Status")
    report["follow_up_issues"] = follow_up_rows

    # Last Interaction should not be earlier than Lead Date.
    parsed_lead_dates = _coerce_datetime(working_df["Lead Date"])
    parsed_last_interaction = _coerce_datetime(working_df["Last Interaction"])
    inconsistent_date_rows = (parsed_last_interaction < parsed_lead_dates).fillna(False)
    date_inconsistency_indices = inconsistent_date_rows[inconsistent_date_rows].index.tolist()
    for idx in date_inconsistency_indices:
        issue_map[idx].append("Last Interaction earlier than Lead Date")

    # Status Classification validation.
    expected_classifications = working_df["Status"].map(STATUS_CLASSIFICATION_MAP)
    provided_classifications = working_df["Status Classification"].astype(str).str.strip()
    inconsistent_classification_rows = (provided_classifications != expected_classifications.fillna(""))
    classification_issue_rows = inconsistent_classification_rows[inconsistent_classification_rows].index.tolist()
    for idx in classification_issue_rows:
        issue_map[idx].append("Status Classification inconsistent with Status")
    report["classification_issues"] = classification_issue_rows

    # Loss Reason is only appropriate for Lost leads.
    loss_reason_values = working_df["Loss Reason"].fillna("").astype(str).str.strip()
    for idx, row in working_df.iterrows():
        if row["Status"] != "Mất" and loss_reason_values.loc[idx] != "":
            issue_map[idx].append("Loss Reason recorded for non-Lost lead")
        if row["Status"] == "Mất" and loss_reason_values.loc[idx] == "" and pd.notna(row["Loss Reason"]):
            issue_map[idx].append("Lost lead missing Loss Reason")
    report["loss_reason_issues"] = sorted(
        set(
            idx
            for idx in range(len(working_df))
            if any("Loss Reason" in issue for issue in issue_map.get(idx, []))
        )
    )

    invalid_rows = [idx for idx, issues in issue_map.items() if issues]
    report["invalid_rows"] = len(invalid_rows)
    report["valid_rows"] = len(working_df) - report["invalid_rows"]
    report["problem_rows"] = working_df.loc[invalid_rows].copy()
    report["problem_rows"]["Validation Issues"] = ["; ".join(issue_map[idx]) for idx in invalid_rows]

    return report


def summarize_validation(report: dict[str, Any]) -> str:
    """Return a human-readable validation summary."""
    summary = [
        f"Total rows: {report['total_rows']}",
        f"Valid rows: {report['valid_rows']}",
        f"Invalid rows: {report['invalid_rows']}",
        f"Excluded rows: {report['excluded_rows']}",
    ]

    if report["missing_columns"]:
        summary.append(f"Missing required columns: {', '.join(report['missing_columns'])}")
    if report["duplicate_lead_ids"]:
        summary.append(f"Duplicate Lead IDs found: {len(report['duplicate_lead_ids'])}")
    if report["invalid_statuses"]:
        summary.append(f"Rows with invalid statuses: {len(report['invalid_statuses'])}")
    if report["invalid_dates"]:
        summary.append(f"Rows with invalid dates: {len(report['invalid_dates'])}")
    if report["invalid_budget_rows"]:
        summary.append(f"Rows with invalid budgets: {len(report['invalid_budget_rows'])}")
    if report["follow_up_issues"]:
        summary.append(f"Rows with invalid follow-up status: {len(report['follow_up_issues'])}")
    if report["classification_issues"]:
        summary.append(f"Rows with inconsistent Status Classification: {len(report['classification_issues'])}")
    if report["loss_reason_issues"]:
        summary.append(f"Rows with loss-reason inconsistencies: {len(report['loss_reason_issues'])}")

    return "\n".join(summary)
