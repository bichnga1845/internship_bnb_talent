import pandas as pd

from src.analytics import (
    build_group_analysis,
    build_lost_lead_segments,
    calculate_kpis,
    investigate_lead,
)


VALID_DF = pd.DataFrame(
    {
        "Lead ID": ["L1", "L2", "L3", "L4", "L5"],
        "Lead Date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]),
        "Source": ["Website", "Website", "Facebook", "Zalo", "Facebook"],
        "Product": ["Máy hút bụi", "Máy hút bụi", "Máy sấy", "Bếp điện từ", "Máy sấy"],
        "Budget": [10000000, 25000000, 50000000, 15000000, 20000000],
        "Status": ["Mới hỏi", "Đã mua", "Mất", "Đã mua", "Mất"],
        "Owner": ["Lan", "Minh", "Huy", "Lan", "Minh"],
        "Customer Concerns": ["A", "B", "C", "D", "E"],
        "Next Action": ["Call", "Quote", "", "Follow-up", ""],
        "Follow-up Status": ["Pending", "Complete", "Not recorded", "Complete", "Pending"],
        "Last Interaction": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-06", "2024-01-07"]),
        "Loss Reason": ["", "", "Không đáp ứng ngân sách", "", "Khách không phản hồi"],
        "Status Classification": ["Open", "Converted", "Lost", "Converted", "Lost"],
    }
)


def test_calculate_kpis():
    result = calculate_kpis(VALID_DF)
    assert result["total_leads"] == 5
    assert result["converted_leads"] == 2
    assert result["lost_leads"] == 2
    assert result["open_leads"] == 1
    assert result["lost_lead_rate"] == 50.0
    assert result["conversion_rate"] == 40.0
    assert result["lost_lead_share"] == 40.0


def test_group_analysis_by_source():
    result = build_group_analysis(VALID_DF, "Source")
    assert set(result.columns) >= {"Source", "converted_leads", "lost_leads", "completed_leads", "lost_lead_rate"}
    assert result["completed_leads"].sum() == 4


def test_lost_lead_segments_structure():
    segments = build_lost_lead_segments(VALID_DF)
    assert "loss_reason_distribution" in segments
    assert "by_source" in segments
    assert "by_product" in segments
    assert "by_budget_band" in segments
    assert "by_owner" in segments
    assert "by_follow_up" in segments
    assert segments["loss_reason_distribution"]["lost_leads"].sum() == 2


def test_investigate_lead_found_and_missing_fields():
    result = investigate_lead(VALID_DF, "L3")
    assert result["found"] is True
    assert result["recorded_facts"]["Status"] == "Mất"
    assert "Next Action" in result["analytical_observations"]["missing_fields"]


def test_investigate_lead_missing_id():
    result = investigate_lead(VALID_DF, "NOT_FOUND")
    assert result["found"] is False
    assert result["message"].startswith("No lead was found")


def test_zero_denominator_handling():
    empty_df = pd.DataFrame(
        {
            "Lead ID": ["A1"],
            "Lead Date": [pd.Timestamp("2024-01-01")],
            "Source": ["Website"],
            "Product": ["Máy hút bụi"],
            "Budget": [10000000],
            "Status": ["Mới hỏi"],
            "Owner": ["Lan"],
            "Customer Concerns": ["A"],
            "Next Action": [""],
            "Follow-up Status": ["Pending"],
            "Last Interaction": [pd.Timestamp("2024-01-05")],
            "Loss Reason": [""],
            "Status Classification": ["Open"],
        }
    )
    result = calculate_kpis(empty_df)
    assert result["lost_lead_rate"] is None
    assert result["conversion_rate"] == 0.0
    assert result["lost_lead_share"] == 0.0


def test_missing_values_do_not_crash():
    df = VALID_DF.copy()
    df.loc[2, "Source"] = None
    df.loc[1, "Follow-up Status"] = None
    result = build_group_analysis(df, "Source")
    assert result is not None
    assert "Missing" in result["Source"].astype(str).values
