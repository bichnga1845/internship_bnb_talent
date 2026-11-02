from __future__ import annotations

import json
from typing import Any

import pandas as pd

from src.analytics import build_lost_lead_segments, calculate_kpis


def summarize_analytics_context(df: pd.DataFrame, lead_id: str | None = None) -> dict[str, Any]:
    """Prepare a compact analytical context object for AI prompting."""
    if df.empty:
        return {
            "synthetic_data_notice": "This is demonstration data only.",
            "total_leads": 0,
            "converted_leads": 0,
            "lost_leads": 0,
            "open_leads": 0,
            "conversion_rate": "N/A",
            "lost_lead_rate": "N/A",
            "lost_lead_share": "N/A",
            "top_patterns": "No data available.",
            "loss_reason_distribution": "No data available.",
            "by_source": "No data available.",
            "by_product": "No data available.",
            "by_follow_up": "No data available.",
            "limitations": "No analysis can be performed because the dataset is empty.",
            "lead_id": lead_id,
        }

    kpis = calculate_kpis(df)
    segments = build_lost_lead_segments(df)

    source_summary = segments.get("by_source", pd.DataFrame())
    product_summary = segments.get("by_product", pd.DataFrame())
    reason_summary = segments.get("loss_reason_distribution", pd.DataFrame())
    follow_up_summary = segments.get("by_follow_up", pd.DataFrame())

    top_source = source_summary.sort_values("lost_leads", ascending=False).head(3).to_dict("records") if not source_summary.empty else []
    top_product = product_summary.sort_values("lost_leads", ascending=False).head(3).to_dict("records") if not product_summary.empty else []
    top_reasons = reason_summary.sort_values("lost_leads", ascending=False).head(5).to_dict("records") if not reason_summary.empty else []
    top_follow_up = follow_up_summary.sort_values("lost_leads", ascending=False).head(5).to_dict("records") if not follow_up_summary.empty else []

    context = {
        "synthetic_data_notice": "This is demonstration data only; it is not real company operational data.",
        "total_leads": int(kpis.get("total_leads", 0)),
        "converted_leads": int(kpis.get("converted_leads", 0)),
        "lost_leads": int(kpis.get("lost_leads", 0)),
        "open_leads": int(kpis.get("open_leads", 0)),
        "conversion_rate": kpis.get("conversion_rate"),
        "lost_lead_rate": kpis.get("lost_lead_rate"),
        "lost_lead_share": kpis.get("lost_lead_share"),
        "top_patterns": json.dumps({
            "top_sources": top_source,
            "top_products": top_product,
        }, ensure_ascii=False),
        "loss_reason_distribution": json.dumps(top_reasons, ensure_ascii=False),
        "by_source": json.dumps(top_source, ensure_ascii=False),
        "by_product": json.dumps(top_product, ensure_ascii=False),
        "by_follow_up": json.dumps(top_follow_up, ensure_ascii=False),
        "limitations": "This analysis is limited to the supplied filtered dataset and does not prove causation.",
        "lead_id": lead_id,
    }
    return context


def prepare_lead_context(df: pd.DataFrame, lead_id: str) -> dict[str, Any]:
    """Prepare a single-lead record for AI investigation."""
    if df.empty:
        return {
            "lead_id": lead_id,
            "status": "Not available",
            "found": False,
            "recorded_facts": {},
            "missing_information": ["No lead data was available in the selected dataset."],
        }

    match = df[df["Lead ID"].astype(str).str.strip() == str(lead_id).strip()]
    if match.empty:
        return {
            "lead_id": lead_id,
            "status": "Not found",
            "found": False,
            "recorded_facts": {},
            "missing_information": ["The requested Lead ID is not present in the loaded dataset."],
        }

    lead = match.iloc[0].to_dict()
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
        value = lead.get(field)
        if pd.isna(value) or str(value).strip() == "":
            missing_fields.append(field)

    return {
        "lead_id": lead_id,
        "status": lead.get("Status Classification", "Unknown"),
        "found": True,
        "recorded_facts": {
            key: value for key, value in lead.items() if key in [
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
        },
        "missing_information": missing_fields,
    }
