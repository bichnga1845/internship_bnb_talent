from __future__ import annotations

from typing import Any

import pandas as pd

from src.data_processing import STATUS_CLASSIFICATION_MAP


def _ensure_status_classification(df: pd.DataFrame) -> pd.DataFrame:
    """Add or normalize the approved Status Classification column.

    If the dataset already contains a valid Status Classification, it is retained.
    Otherwise it is derived from the project-approved mapping in data_processing.py.
    """
    working_df = df.copy()
    if "Status Classification" not in working_df.columns:
        working_df["Status Classification"] = working_df["Status"].map(STATUS_CLASSIFICATION_MAP)
    else:
        working_df["Status Classification"] = working_df["Status Classification"].astype(str).str.strip()
    return working_df


def _safe_percentage(numerator: int | float, denominator: int | float) -> float | None:
    """Return a percentage when the denominator is positive; otherwise return None."""
    if pd.isna(numerator) or pd.isna(denominator):
        return None
    if denominator == 0:
        return None
    return (numerator / denominator) * 100


def _budget_band(value: Any) -> str:
    """Convert budget into a simple business-friendly band."""
    if pd.isna(value):
        return "Missing"
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return "Missing"

    if numeric_value < 0:
        return "Invalid"
    if numeric_value < 10000000:
        return "Under 10M"
    if numeric_value < 30000000:
        return "10M - 30M"
    if numeric_value < 60000000:
        return "30M - 60M"
    if numeric_value < 100000000:
        return "60M - 100M"
    return "100M+"


def calculate_kpis(df: pd.DataFrame) -> dict[str, Any]:
    """Calculate headline lead metrics from the current validated dataset.

    The brief defines Lost Lead Rate with completed outcomes only, while the project
    requirements call for Conversion Rate as Converted / Total Leads and Lost Lead
    Share as Lost / Total Leads. This function exposes both the brief and the user-
    requested view so the analytics remain traceable and explicit.
    """
    working_df = _ensure_status_classification(df)

    total_leads = int(len(working_df))
    converted = int((working_df["Status Classification"] == "Converted").sum())
    lost = int((working_df["Status Classification"] == "Lost").sum())
    open_leads = int((working_df["Status Classification"] == "Open").sum())
    completed_leads = converted + lost

    kpis = {
        "total_leads": total_leads,
        "converted_leads": converted,
        "lost_leads": lost,
        "open_leads": open_leads,
        "completed_leads": completed_leads,
        "lost_lead_rate": _safe_percentage(lost, completed_leads),
        "conversion_rate": _safe_percentage(converted, total_leads),
        "lost_lead_share": _safe_percentage(lost, total_leads),
        "conversion_rate_completed": _safe_percentage(converted, completed_leads),
    }
    return kpis


def build_group_analysis(df: pd.DataFrame, group_column: str) -> pd.DataFrame:
    """Aggregate lost-lead metrics for a single grouping dimension.

    This function useful for source, product, owner, follow-up, and similar
    categorical views. It does not silently drop missing values: missing categories
    are shown as 'Missing' to make the data quality issue visible.
    """
    working_df = _ensure_status_classification(df)

    if group_column not in working_df.columns:
        return pd.DataFrame(
            columns=[group_column, "converted_leads", "lost_leads", "completed_leads", "lost_lead_rate"]
        )

    grouped_df = working_df.copy()
    grouped_df[group_column] = grouped_df[group_column].where(grouped_df[group_column].notna(), "Missing")
    grouped_df[group_column] = grouped_df[group_column].astype(str).str.strip()
    grouped_df[group_column] = grouped_df[group_column].replace("", "Missing")

    completed_df = grouped_df[grouped_df["Status Classification"].isin(["Converted", "Lost"])].copy()

    if completed_df.empty:
        return pd.DataFrame(
            columns=[group_column, "converted_leads", "lost_leads", "completed_leads", "lost_lead_rate"]
        )

    summary = (
        completed_df.groupby(group_column, dropna=False)
        .agg(
            converted_leads=("Status Classification", lambda s: int((s == "Converted").sum())),
            lost_leads=("Status Classification", lambda s: int((s == "Lost").sum())),
        )
        .reset_index()
    )

    summary["completed_leads"] = summary["converted_leads"] + summary["lost_leads"]
    summary["lost_lead_rate"] = summary.apply(
        lambda row: _safe_percentage(row["lost_leads"], row["completed_leads"]), axis=1
    )
    summary = summary.sort_values(["lost_lead_rate", "lost_leads"], ascending=[False, False], na_position="last")
    summary = summary.reset_index(drop=True)
    return summary


def build_lost_lead_segments(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Create all required segment tables used in lost-lead analysis."""
    working_df = _ensure_status_classification(df)
    for column in ["Source", "Product", "Owner", "Follow-up Status", "Loss Reason"]:
        if column not in working_df.columns:
            working_df[column] = "Missing"
    if "Budget" not in working_df.columns:
        working_df["Budget"] = pd.NA
    lost_df = working_df[working_df["Status Classification"] == "Lost"].copy()

    segments = {
        "loss_reason_distribution": (
            lost_df.groupby("Loss Reason", dropna=False)
            .size()
            .reset_index(name="lost_leads")
            .rename(columns={"Loss Reason": "Loss Reason"})
            if not lost_df.empty
            else pd.DataFrame(columns=["Loss Reason", "lost_leads"])
        ),
        "by_source": build_group_analysis(working_df, "Source"),
        "by_product": build_group_analysis(working_df, "Product"),
        "by_owner": build_group_analysis(working_df, "Owner"),
        "by_follow_up": build_group_analysis(working_df, "Follow-up Status"),
    }

    if not lost_df.empty:
        lost_df["Budget Band"] = lost_df["Budget"].apply(_budget_band)
        segments["by_budget_band"] = (
            lost_df.groupby("Budget Band", dropna=False)
            .size()
            .reset_index(name="lost_leads")
            .sort_values("lost_leads", ascending=False)
            .reset_index(drop=True)
        )
    else:
        segments["by_budget_band"] = pd.DataFrame(columns=["Budget Band", "lost_leads"])

    return segments


def investigate_lead(df: pd.DataFrame, lead_id: str) -> dict[str, Any]:
    """Return a structured lead-investigation record.

    The result clearly separates recorded facts from analytical observations.
    It does not invent interaction history or hidden business context.
    """
    working_df = _ensure_status_classification(df)
    match = working_df[working_df["Lead ID"].astype(str).str.strip() == str(lead_id).strip()]

    if match.empty:
        return {
            "lead_id": lead_id,
            "found": False,
            "recorded_facts": {},
            "analytical_observations": {
                "status_classification": "Unknown",
                "is_lost_lead": False,
                "is_open_lead": False,
                "missing_fields": [],
                "follow_up_risk": "Not available",
            },
            "message": "No lead was found with that Lead ID.",
        }

    lead_row = match.iloc[0].to_dict()
    missing_fields = []
    for field in [
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
    ]:
        value = lead_row.get(field)
        if pd.isna(value) or str(value).strip() == "":
            missing_fields.append(field)

    recorded_facts = {
        "Lead ID": lead_row.get("Lead ID"),
        "Lead Date": lead_row.get("Lead Date"),
        "Source": lead_row.get("Source"),
        "Product": lead_row.get("Product"),
        "Budget": lead_row.get("Budget"),
        "Status": lead_row.get("Status"),
        "Status Classification": lead_row.get("Status Classification"),
        "Owner": lead_row.get("Owner"),
        "Customer Concerns": lead_row.get("Customer Concerns"),
        "Next Action": lead_row.get("Next Action"),
        "Follow-up Status": lead_row.get("Follow-up Status"),
        "Last Interaction": lead_row.get("Last Interaction"),
        "Loss Reason": lead_row.get("Loss Reason"),
    }

    analytical_observations = {
        "status_classification": lead_row.get("Status Classification", "Unknown"),
        "is_lost_lead": lead_row.get("Status Classification") == "Lost",
        "is_open_lead": lead_row.get("Status Classification") == "Open",
        "missing_fields": missing_fields,
        "follow_up_risk": lead_row.get("Follow-up Status", "Not recorded"),
    }

    return {
        "lead_id": lead_id,
        "found": True,
        "recorded_facts": recorded_facts,
        "analytical_observations": analytical_observations,
        "message": "Lead record found. Recorded facts are shown below; analytical observations are separate.",
    }
