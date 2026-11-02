from pathlib import Path

import pandas as pd

from src.data_generator import generate_synthetic_leads


def test_synthetic_dataset_has_expected_columns():
    df = generate_synthetic_leads(n_rows=200, seed=7)
    required_columns = {
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
    }
    assert set(df.columns) >= required_columns


def test_synthetic_dataset_has_valid_status_mapping():
    df = generate_synthetic_leads(n_rows=200, seed=7)
    valid_statuses = {"Mới hỏi", "Đã tư vấn sơ bộ", "Chờ ảnh bếp", "Chờ khảo sát", "Chờ kỹ thuật xem", "Đã báo giá", "Đã mua", "Mất"}
    assert set(df["Status"].unique()) <= valid_statuses
    assert set(df["Status Classification"].unique()) <= {"Open", "Converted", "Lost"}


def test_synthetic_dataset_has_valid_dates_and_budget():
    df = generate_synthetic_leads(n_rows=250, seed=9)
    assert pd.api.types.is_datetime64_any_dtype(df["Lead Date"])
    assert pd.api.types.is_datetime64_any_dtype(df["Last Interaction"])
    assert (df["Budget"] >= 0).all()
    assert (df["Last Interaction"] >= df["Lead Date"]).all()


def test_generated_csv_written_to_disk():
    output_path = Path("data/synthetic_leads.csv")
    df = generate_synthetic_leads(n_rows=120, seed=3, output_path=output_path)
    assert output_path.exists()
    saved_df = pd.read_csv(output_path)
    assert len(saved_df) == len(df)
    assert "Status Classification" in saved_df.columns
