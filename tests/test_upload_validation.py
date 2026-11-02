from pathlib import Path

import pandas as pd

from src.data_processing import validate_lead_data


def test_validation_accepts_clean_dataset():
    df = pd.DataFrame(
        {
            "Lead ID": ["L1", "L2"],
            "Lead Date": pd.to_datetime(["2024-01-01", "2024-01-05"]),
            "Source": ["Website", "Facebook"],
            "Product": ["Máy hút bụi", "Máy xay sinh tố"],
            "Budget": [50000000, 15000000],
            "Status": ["Mới hỏi", "Đã mua"],
            "Owner": ["Lan", "Minh"],
            "Customer Concerns": ["Cần tư vấn", "Quan tâm đến bảo hành"],
            "Next Action": ["Gọi lại", "Gửi báo giá"],
            "Follow-up Status": ["Pending", "Complete"],
            "Last Interaction": pd.to_datetime(["2024-01-03", "2024-01-10"]),
            "Loss Reason": ["", ""],
            "Status Classification": ["Open", "Converted"],
        }
    )
    report = validate_lead_data(df)
    assert report["valid_rows"] == 2
    assert report["invalid_rows"] == 0


def test_validation_detects_duplicate_and_invalid_status():
    df = pd.DataFrame(
        {
            "Lead ID": ["L1", "L1", "L3"],
            "Lead Date": ["2024-01-01", "2024-01-02", "2024-01-03"],
            "Source": ["Website", "Website", "Zalo"],
            "Product": ["Máy hút bụi", "Máy hút bụi", "Máy rửa bát"],
            "Budget": [50000000, 60000000, 70000000],
            "Status": ["Mới hỏi", "Mới hỏi", "Sai trạng thái"],
            "Owner": ["Lan", "Lan", "Huy"],
            "Customer Concerns": ["A", "B", "C"],
            "Next Action": ["Gọi lại", "Gọi lại", ""],
            "Follow-up Status": ["Pending", "Pending", "Pending"],
            "Last Interaction": ["2024-01-04", "2024-01-05", "2024-01-06"],
            "Loss Reason": ["", "", ""],
            "Status Classification": ["Open", "Open", "Open"],
        }
    )
    report = validate_lead_data(df)
    assert report["duplicate_lead_ids"] == [0, 1]
    assert report["invalid_statuses"] == [2]


def test_validation_detects_bad_dates_and_budget():
    df = pd.DataFrame(
        {
            "Lead ID": ["A1", "A2"],
            "Lead Date": ["2024-01-02", "bad-date"],
            "Source": ["Website", "Website"],
            "Product": ["Máy sấy", "Máy sấy"],
            "Budget": [1000000, -50],
            "Status": ["Đã mua", "Mất"],
            "Owner": ["Lan", "Nhi"],
            "Customer Concerns": ["Concern", "Concern"],
            "Next Action": ["", ""],
            "Follow-up Status": ["Complete", "Unknown"],
            "Last Interaction": ["2024-01-01", "2024-01-05"],
            "Loss Reason": ["", "Mất do giá cao"],
            "Status Classification": ["Converted", "Lost"],
        }
    )
    report = validate_lead_data(df)
    assert report["invalid_budget_rows"] == [1]
    assert 1 in report["invalid_dates"] or 0 in report["invalid_dates"]
    assert report["follow_up_issues"] == [1]


def test_validation_detects_inconsistent_status_classification():
    df = pd.DataFrame(
        {
            "Lead ID": ["B1", "B2"],
            "Lead Date": ["2024-02-01", "2024-02-02"],
            "Source": ["Website", "Referral"],
            "Product": ["Bếp điện từ", "Máy rửa bát"],
            "Budget": [30000000, 25000000],
            "Status": ["Đã mua", "Mất"],
            "Owner": ["Lan", "Minh"],
            "Customer Concerns": ["Concern", "Concern"],
            "Next Action": ["", ""],
            "Follow-up Status": ["Complete", "Pending"],
            "Last Interaction": ["2024-02-04", "2024-02-05"],
            "Loss Reason": ["", "Khách chưa quyết định"],
            "Status Classification": ["Open", "Converted"],
        }
    )
    report = validate_lead_data(df)
    assert report["classification_issues"] == [0, 1]
    assert report["problem_rows"].shape[0] == 2
