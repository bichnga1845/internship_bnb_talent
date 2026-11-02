from __future__ import annotations

import random
from pathlib import Path

import pandas as pd


STATUS_MAPPING = {
    "Mới hỏi": "Open",
    "Đã tư vấn sơ bộ": "Open",
    "Chờ ảnh bếp": "Open",
    "Chờ khảo sát": "Open",
    "Chờ kỹ thuật xem": "Open",
    "Đã báo giá": "Open",
    "Đã mua": "Converted",
    "Mất": "Lost",
}

VALID_STATUSES = set(STATUS_MAPPING.keys())
VALID_FOLLOW_UP = {"Complete", "Pending", "Not recorded"}
PRODUCT_OPTIONS = [
    "Máy hút bụi", "Máy rửa bát", "Máy xay sinh tố", "Máy làm đá", "Máy sấy", "Bếp điện từ"
]
SOURCES = ["Website", "Facebook", "Zalo", "Referral", "Google Ads", "Showroom"]
OWNERS = ["Lan", "Minh", "Huy", "Tram", "Khoa", "Nhi"]
CONCERNS = [
    "Cần sản phẩm tiết kiệm điện",
    "So sánh thương hiệu và giá",
    "Cần tư vấn kích thước không gian",
    "Chưa quyết định mẫu",
    "Cần hỗ trợ lắp đặt",
    "Quan tâm đến bảo hành",
    "Cần báo giá chi tiết",
]
LOSS_REASONS = [
    "Đã đặt hàng ở đối thủ",
    "Không đáp ứng ngân sách",
    "Chưa cần mua trong thời gian này",
    "Tìm hiểu thêm nhưng chưa quyết định",
    "Khách hàng không phản hồi",
    "Chưa có nhu cầu rõ ràng",
]


def _random_date_range(start_date: pd.Timestamp, end_date: pd.Timestamp, rng: random.Random) -> pd.Timestamp:
    delta_days = (end_date - start_date).days
    return start_date + pd.Timedelta(days=rng.randint(0, delta_days))


def generate_synthetic_leads(
    n_rows: int = 750,
    seed: int = 42,
    output_path: str | Path | None = None,
) -> pd.DataFrame:
    """Generate a reproducible synthetic dataset for lead analysis.

    The data is designed to be clearly synthetic and aligned with the project brief.
    """
    rng = random.Random(seed)

    lead_ids = [f"BNB-{rng.randint(1000, 9999)}-{index:04d}" for index in range(1, n_rows + 1)]
    lead_dates = [
        _random_date_range(pd.Timestamp("2024-01-01"), pd.Timestamp("2025-06-30"), rng)
        for _ in range(n_rows)
    ]

    statuses = []
    for _ in range(n_rows):
        status_roll = rng.random()
        if status_roll < 0.42:
            statuses.append("Mới hỏi")
        elif status_roll < 0.60:
            statuses.append("Đã báo giá")
        elif status_roll < 0.74:
            statuses.append("Đã tư vấn sơ bộ")
        elif status_roll < 0.82:
            statuses.append("Chờ khảo sát")
        elif status_roll < 0.90:
            statuses.append("Đã mua")
        elif status_roll < 0.96:
            statuses.append("Mất")
        elif status_roll < 0.98:
            statuses.append("Chờ ảnh bếp")
        else:
            statuses.append("Chờ kỹ thuật xem")

    sources = [rng.choice(SOURCES) for _ in range(n_rows)]
    products = [rng.choice(PRODUCT_OPTIONS) for _ in range(n_rows)]
    budgets = [
        float(rng.randint(2500000, 125000000))
        for _ in range(n_rows)
    ]
    owners = [rng.choice(OWNERS) for _ in range(n_rows)]
    concerns = [rng.choice(CONCERNS) for _ in range(n_rows)]
    next_actions = [
        rng.choice([
            "Gọi lại khách hàng",
            "Gửi báo giá chi tiết",
            "Hẹn tư vấn thêm",
            "Yêu cầu khách hàng gửi ảnh",
            "Theo dõi thêm",
            "Chờ phản hồi từ khách",
            "Lập kế hoạch khảo sát",
            "Không cần hành động ngay",
            "",
        ])
        for _ in range(n_rows)
    ]

    follow_up_status = [
        rng.choice(["Complete", "Pending", "Not recorded"]) for _ in range(n_rows)
    ]

    last_interactions = []
    for lead_date, status in zip(lead_dates, statuses):
        if status == "Đã mua":
            offset_days = rng.randint(1, 35)
        elif status == "Mất":
            offset_days = rng.randint(0, 20)
        else:
            offset_days = rng.randint(0, 60)
        last_interactions.append(lead_date + pd.Timedelta(days=offset_days))

    loss_reasons = []
    for status in statuses:
        if status == "Mất":
            if rng.random() < 0.72:
                loss_reasons.append(rng.choice(LOSS_REASONS))
            else:
                loss_reasons.append("")
        else:
            loss_reasons.append("")

    df = pd.DataFrame(
        {
            "Lead ID": lead_ids,
            "Lead Date": lead_dates,
            "Source": sources,
            "Product": products,
            "Budget": budgets,
            "Status": statuses,
            "Owner": owners,
            "Customer Concerns": concerns,
            "Next Action": next_actions,
            "Follow-up Status": follow_up_status,
            "Last Interaction": last_interactions,
            "Loss Reason": loss_reasons,
        }
    )

    df["Status Classification"] = df["Status"].map(STATUS_MAPPING)

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)

    return df


def generate_default_dataset(output_path: str | Path = "data/synthetic_leads.csv") -> pd.DataFrame:
    """Convenience function to generate the default synthetic dataset for the project."""
    return generate_synthetic_leads(output_path=output_path)
