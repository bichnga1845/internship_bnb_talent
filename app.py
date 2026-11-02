from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import streamlit as st
import plotly.io as pio

# Set Plotly default dark theme
pio.templates.default = "plotly_dark"

from src.ai_analysis import prepare_lead_context, summarize_analytics_context
from src.ai_prompts import AI_SYSTEM_INSTRUCTIONS, build_lead_prompt, build_question_prompt, build_recommendation_prompt, build_report_prompt
from src.ai_simulation import (
    answer_simulated_question,
    build_simulated_lead_analysis,
    generate_simulated_recommendations,
    generate_simulated_report,
    summarize_simulated_lead,
)
from src.analytics import (
    build_group_analysis,
    build_lost_lead_segments,
    calculate_kpis,
    investigate_lead,
)
from src.data_generator import generate_synthetic_leads
from src.data_processing import validate_lead_data

DEFAULT_DATASET_PATH = Path("data/synthetic_leads.csv")


def _ai_available() -> bool:
    """Always operate in offline AI simulation mode for this project version."""
    return False


def _generate_ai_response(prompt: str, context_label: str) -> dict[str, str]:
    """Generate structured AI output using simulation only; no external request is allowed."""
    return {"status": "error", "message": "Chế độ mô phỏng AI – Phân tích ngoại tuyến dựa trên dữ liệu hiện có đang được sử dụng."}


def _generate_report(df: pd.DataFrame) -> dict[str, str]:
    """Generate an investigation report from the current filtered data."""
    if df.empty:
        return {"status": "error", "message": "Không có dữ liệu phù hợp để tạo báo cáo điều tra."}
    return {"status": "success", "label": "Báo cáo mô phỏng AI", "text": generate_simulated_report(df)}


def _generate_question_answer(df: pd.DataFrame, question: str) -> dict[str, str]:
    """Answer a business question using the supplied dataset context."""
    if not question.strip():
        return {"status": "error", "message": "Vui lòng nhập câu hỏi trước khi hỏi AI."}
    if df.empty:
        return {"status": "error", "message": "Không có dữ liệu để trả lời câu hỏi."}
    return {"status": "success", "label": "Trả lời mô phỏng AI", "text": answer_simulated_question(df, question)}


def _generate_lead_summary(df: pd.DataFrame, lead_id: str) -> dict[str, str]:
    """Generate an AI summary for a single selected lead."""
    if not lead_id.strip():
        return {"status": "error", "message": "Vui lòng nhập mã Lead trước khi tạo tóm tắt."}
    if df.empty:
        return {"status": "error", "message": "Không có dữ liệu Lead để điều tra."}
    return {"status": "success", "label": "Tổng quan lead mô phỏng AI", "text": summarize_simulated_lead(df, lead_id)}


def _generate_recommendations(df: pd.DataFrame) -> dict[str, str]:
    """Generate recommendations for the active filtered data."""
    if df.empty:
        return {"status": "error", "message": "Không có dữ liệu để tạo khuyến nghị."}
    return {"status": "success", "label": "Khuyến nghị mô phỏng AI", "text": generate_simulated_recommendations(df)}


def _render_ai_output(result: dict[str, str]) -> None:
    """Render AI output in a consistent, professional read-only format."""
    if result.get("status") == "error":
        st.warning(result["message"])
        return

    st.markdown(
        f"""
        <div class="panel">
            <h4>{result.get('label', 'AI output')}</h4>
            <div>{result['text']}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _inject_custom_css() -> None:
    """Apply a light premium SaaS-inspired design."""
    st.markdown(
        """
        <style>
        :root {
            --bg: #111827;
            --panel: #1F2937;
            --panel-alt: #111827;
            --line: #374151;
            --heading: #F3F4F6;  /* bright near‑white for all text */
            --body: #D1D5DB;    /* lighter gray */
            --primary: #2563EB;
            --primary-soft: rgba(37, 99, 235, 0.1);
            --secondary: #0EA5A4;
            --success: #10B981;
            --warning: #FBBF24;
            --danger: #EF4444;
            --shadow: rgba(15, 23, 42, 0.06);
        }

        .stApp {
            background: var(--bg);
            color: var(--heading);
        }

        .block-container {
            padding-top: 1.4rem;
            padding-bottom: 2.2rem;
            max-width: 1500px;
        }

        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #1F2937, #111827);
            border-right: 1px solid var(--line);
            box-shadow: inset -1px 0 0 rgba(15, 23, 42, 0.02);
        }

        .sidebar-section {
            padding: 0.7rem 0.2rem;
            border-bottom: 1px solid var(--line);
            margin-bottom: 0.5rem;
        }

        .sidebar-section h3 {
            color: var(--heading);
            font-size: 0.78rem;
            font-weight: 800;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            margin: 0;
        }

        .metric-card {
            background: linear-gradient(180deg, #FFFFFF, #F8FAFC);
            border: 1px solid var(--line);
            border-radius: 18px;
            box-shadow: 0 10px 24px var(--shadow);
            padding: 1rem 1.1rem;
            height: 100%;
        }

        .metric-card .label {
            color: var(--heading);
            font-size: 0.7rem;
            font-weight: 800;
            letter-spacing: 0.08em;
            text-transform: uppercase;
        }

        .metric-card .value {
            color: var(--primary);
            font-size: 2rem;
            font-weight: 800;
            margin-top: 0.35rem;
            line-height: 1.1;
        }

        .metric-card .delta {
            margin-top: 0.4rem;
            color: var(--primary);
            font-size: 0.76rem;
            font-weight: 800;
        }

        .section-header {
            color: var(--heading);
            font-size: 1rem;
            font-weight: 800;
            letter-spacing: 0.09em;
            text-transform: uppercase;
            margin: 1.15rem 0 0.8rem 0;
        }

        .panel {
            background: #FFFFFF;
            border: 1px solid var(--line);
            border-radius: 16px;
            padding: 1rem 1.1rem;
            box-shadow: 0 6px 18px rgba(15, 23, 42, 0.04);
            color: var(--heading); /* ensure panel text is bright */
        }

        .pill {
            display: inline-flex;
            align-items: center;
            padding: 0.38rem 0.7rem;
            border-radius: 999px;
            background: var(--primary-soft);
            border: 1px solid rgba(79, 70, 229, 0.16);
            color: var(--primary);
            font-size: 0.72rem;
            font-weight: 700;
            margin-right: 0.6rem;
        }

        .placeholder-box {
            background: var(--panel-alt);
            border: 1px dashed #D7DFEB;
            border-radius: 14px;
            padding: 1rem 1.1rem;
            color: var(--heading);
        }

        .stButton > button {
            border-radius: 12px;
            background: linear-gradient(180deg, var(--primary), #4338CA);
            color: white;
            border: none;
            font-weight: 700;
            padding: 0.72rem 1rem;
            box-shadow: 0 8px 18px rgba(79, 70, 229, 0.18);
        }

        .stButton > button:disabled {
            background: rgba(148, 163, 184, 0.22);
            color: var(--body);
            border: 1px solid var(--line);
            box-shadow: none;
        }

        h1 {
            color: var(--heading);
            font-weight: 800;
            letter-spacing: -0.04em;
            margin-bottom: 0.2rem;
        }

        h2, h3, h4 {
            color: var(--heading);
            letter-spacing: -0.02em;
        }

        p, li, label, span, div {
            color: var(--heading);
        }

        .stDataFrame {
            border: 1px solid var(--line);
            border-radius: 12px;
            overflow: hidden;
        }

        .stTabs [role="tablist"] {
            gap: 0.5rem;
        }

        .stTabs [role="tab"] {
            border-radius: 12px 12px 0 0;
            padding: 0.6rem 0.9rem;
            background: #FFFFFF;
            border: 1px solid var(--line);
            color: var(--heading);
        }

        .stTabs [role="tab"][aria-selected="true"] {
            background: rgba(79, 70, 229, 0.06);
            border-color: rgba(79, 70, 229, 0.24);
            color: var(--heading);
            font-weight: 700;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def ensure_default_dataset() -> Path:
    """Create the default synthetic dataset when it does not exist."""
    DEFAULT_DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not DEFAULT_DATASET_PATH.exists():
        generate_synthetic_leads(n_rows=750, seed=42, output_path=DEFAULT_DATASET_PATH)
    return DEFAULT_DATASET_PATH


def load_dataset_from_source(uploaded_file=None, use_synthetic: bool = True) -> tuple[pd.DataFrame, dict]:
    """Load the active dataset and validate it without silently dropping invalid rows."""
    if uploaded_file is not None:
        if uploaded_file.name.lower().endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
    elif use_synthetic:
        path = ensure_default_dataset()
        df = pd.read_csv(path)
    else:
        return pd.DataFrame(), {"total_rows": 0, "valid_rows": 0, "invalid_rows": 0, "problem_rows": pd.DataFrame()}

    validation = validate_lead_data(df)
    valid_df = df.copy()
    if validation["invalid_rows"] > 0:
        invalid_indexes = set(validation["problem_rows"].index.tolist())
        valid_df = df.loc[[idx for idx in df.index if idx not in invalid_indexes]].copy()
    return valid_df, validation


def filter_dataframe(
    df: pd.DataFrame,
    date_start: pd.Timestamp | None = None,
    date_end: pd.Timestamp | None = None,
    selected_sources: list[str] | None = None,
    selected_products: list[str] | None = None,
    selected_owners: list[str] | None = None,
    selected_statuses: list[str] | None = None,
) -> pd.DataFrame:
    """Apply consistent filters to the current dataset."""
    if df.empty:
        return df.copy()

    filtered = df.copy()

    if "Lead Date" in filtered.columns:
        filtered["Lead Date"] = pd.to_datetime(filtered["Lead Date"], errors="coerce")
        if date_start is not None:
            filtered = filtered[filtered["Lead Date"] >= pd.Timestamp(date_start)]
        if date_end is not None:
            filtered = filtered[filtered["Lead Date"] <= pd.Timestamp(date_end)]

    if selected_sources:
        filtered = filtered[filtered["Source"].astype(str).isin(selected_sources)]
    if selected_products:
        filtered = filtered[filtered["Product"].astype(str).isin(selected_products)]
    if selected_owners:
        filtered = filtered[filtered["Owner"].astype(str).isin(selected_owners)]
    if selected_statuses:
        filtered = filtered[filtered["Status Classification"].astype(str).isin(selected_statuses)]

    return filtered.reset_index(drop=True)


def format_pct(value: Any, fallback: str = "N/A") -> str:
    """Format percentages consistently while handling zero denominators."""
    if value is None or pd.isna(value):
        return fallback
    return f"{float(value):.2f}%"


def format_count(value: Any) -> str:
    """Format metric counts for compact KPI display."""
    if value is None or pd.isna(value):
        return "0"
    return f"{int(value):,}"


def build_monthly_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize leads by month."""
    if df.empty or "Lead Date" not in df.columns:
        return pd.DataFrame(columns=["Month", "Converted", "Lost", "Open"])

    monthly = df.copy()
    monthly["Month"] = pd.to_datetime(monthly["Lead Date"], errors="coerce").dt.to_period("M").astype(str)
    monthly = monthly.groupby("Month", dropna=False).agg(
        Converted=("Status Classification", lambda s: (s == "Converted").sum()),
        Lost=("Status Classification", lambda s: (s == "Lost").sum()),
        Open=("Status Classification", lambda s: (s == "Open").sum()),
    ).reset_index()
    return monthly.sort_values("Month").reset_index(drop=True)


def build_funnel_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Create funnel values for the current filtered dataset."""
    if df.empty:
        return pd.DataFrame(columns=["Stage", "Leads"])

    return pd.DataFrame(
        {
            "Stage": ["Total Leads", "Open", "Converted", "Lost"],
            "Leads": [
                len(df),
                int((df["Status Classification"] == "Open").sum()),
                int((df["Status Classification"] == "Converted").sum()),
                int((df["Status Classification"] == "Lost").sum()),
            ],
        }
    )


def render_kpi_cards(kpis: dict[str, Any]) -> None:
    """Render KPI cards in a premium analytics layout."""
    st.markdown('<div class="section-header">Tổng quan</div>', unsafe_allow_html=True)
    metric_layout = [
        ("Tổng số Lead", kpis.get("total_leads", 0), "Quy mô pipeline"),
        ("Lead thành công", kpis.get("converted_leads", 0), "Doanh số đã đóng"),
        ("Lead thất bại", kpis.get("lost_leads", 0), "Đã chốt thất bại"),
        ("Lead đang xử lý", kpis.get("open_leads", 0), "Vẫn đang hoạt động"),
        ("Tỷ lệ Lead thất bại", format_pct(kpis.get("lost_lead_rate")), "Chỉ tính kết quả hoàn tất"),
        ("Tỷ lệ chuyển đổi", format_pct(kpis.get("conversion_rate")), "Tất cả Lead"),
        ("Phần chia Lead thất bại", format_pct(kpis.get("lost_lead_share")), "Chia theo tổng pipeline"),
    ]

    col_count = 4
    metric_cols = st.columns(col_count)
    for idx, (label, value, note) in enumerate(metric_layout):
        with metric_cols[idx % col_count]:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="label">{label}</div>
                    <div class="value">{value}</div>
                    <div class="delta">{note}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_executive_overview(df: pd.DataFrame) -> None:
    """Executive overview dashboard with KPI cards and business charts."""
    if df.empty:
        st.info("Không có dữ liệu Lead hợp lệ trong bộ lọc hiện tại.")
        return

    kpis = calculate_kpis(df)
    render_kpi_cards(kpis)

    status_data = df["Status Classification"].fillna("Missing").astype(str).str.strip().replace("", "Missing")
    status_summary = pd.DataFrame(status_data.value_counts().reset_index())
    status_summary.columns = ["Status Classification", "Count"]
    status_summary = status_summary.sort_values("Count", ascending=False).reset_index(drop=True)

    monthly_summary = build_monthly_summary(df)
    status_chart = px.bar(
        status_summary,
        x="Status Classification",
        y="Count",
        color="Status Classification",
        title="Phân bổ trạng thái Lead",
        labels={"Status Classification": "Trạng thái", "Count": "Số Lead"},
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    status_chart.update_layout(
        template="plotly_white",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=30, r=10, b=40, l=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    monthly_chart = px.line(
        monthly_summary,
        x="Month",
        y=["Converted", "Lost", "Open"],
        title="Pipeline Lead theo tháng",
        labels={"value": "Số Lead", "Month": "Tháng"},
        line_shape="spline",
        color_discrete_sequence=["#4F46E5", "#0EA5A4", "#F59E0B"],
    )
    monthly_chart.update_layout(
        template="plotly_white",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        legend_title_text="Trạng thái Lead",
        margin=dict(t=30, r=10, b=20, l=10),
    )

    col_a, col_b = st.columns(2)
    with col_a:
        st.caption("Bối cảnh kinh doanh: theo dõi tỷ trọng Lead đang hoạt động, đã chuyển đổi và thất bại.")
        st.plotly_chart(status_chart, width="stretch")
    with col_b:
        st.caption("Bối cảnh kinh doanh: theo dõi động lực và tỷ lệ kết quả qua timeline pipeline.")
        st.plotly_chart(monthly_chart, width="stretch")

    funnel_data = build_funnel_summary(df)
    # Use a horizontal bar chart to display funnel stages more clearly
    funnel_chart = px.bar(
        funnel_data,
        x="Leads",
        y="Stage",
        orientation="h",
        title="Funnel chuyển đổi",
        color="Stage",
        color_discrete_sequence=["#4F46E5", "#0EA5A4", "#F59E0B", "#EF4444"],
    )
    funnel_chart.update_layout(
        template="plotly_white",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=30, r=10, b=20, l=10),
        yaxis=dict(categoryorder="total descending"),
    )

    st.caption("Bối cảnh kinh doanh: xem xét cách Lead tiến triển từ khi quan tâm đến khi có kết quả cuối cùng.")
    st.plotly_chart(funnel_chart, width="stretch")

    lost_segments = build_lost_lead_segments(df)
    source_summary = lost_segments.get("by_source", pd.DataFrame())
    source_name = "N/A"
    source_value = 0
    if not source_summary.empty:
        top_source = source_summary.sort_values("lost_leads", ascending=False).iloc[0]
        source_name = top_source.get("Source", "N/A")
        source_value = top_source.get("lost_leads", 0)

    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown(f"<span class='pill' style='background:#e6e6e6;color:#333;'>Tín hiệu ưu tiên</span> Nguồn có rủi ro cao nhất: <strong>{source_name}</strong> với <strong>{source_value}</strong> Lead thất bại.", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("### Tóm tắt tín hiệu kinh doanh")
    col1, col2, col3 = st.columns(3)
    col1.metric("Kết quả hoàn tất", format_count(kpis.get("completed_leads", 0)), "Lead thành công + thất bại")
    col2.metric("Phần chia Lead thất bại", format_pct(kpis.get("lost_lead_share")), f"{kpis.get('lost_leads', 0)} Lead thất bại")
    col3.metric("Pipeline đang mở", format_count(kpis.get("open_leads", 0)), "Vẫn đang hoạt động")


def render_ai_placeholder() -> None:
    """Render the AI investigation center using the configured OpenAI service when available or simulation when not."""
    st.markdown('<div class="section-header">Trung tâm phân tích AI</div>', unsafe_allow_html=True)

    current_df = st.session_state.get("current_df", pd.DataFrame())

    if not _ai_available():
        st.info("Chế độ mô phỏng AI đang hoạt động vì OpenAI chưa khả dụng trong môi trường hiện tại.")
    else:
        st.success("OpenAI đã được cấu hình để chạy phân tích AI trực tiếp trong không gian làm việc hiện tại.")

    left_col, right_col = st.columns([1.7, 1])
    with left_col:
        st.markdown(
            """
            <div class="panel">
                <h3>Khả năng AI</h3>
                <ul>
                    <li>Tóm tắt điều tra dựa trên bằng chứng</li>
                    <li>Giải thích chi tiết theo Lead với giới hạn dữ liệu rõ ràng</li>
                    <li>Hỏi đáp bằng ngôn ngữ tự nhiên trên dữ liệu đã lọc</li>
                    <li>Khuyến nghị dựa trên chỉ số và phân khúc</li>
                    <li>Gắn nhãn rõ ràng về độ không chắc chắn và giới hạn</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with right_col:
        st.markdown(
            """
            <div class="placeholder-box">
                <strong>Trạng thái runtime</strong><br><br>
                - Chế độ mô phỏng AI đang hoạt động khi OpenAI không khả dụng<br>
                - Thông báo dữ liệu mô phỏng vẫn hiển thị trong mỗi câu trả lời<br>
                - Không được suy luận gây nhân quả không được chứng minh<br>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("### Tạo insight")
    if current_df.empty:
        st.info("Tải lên hoặc lọc dữ liệu để kích hoạt các tính năng AI.")

    report_col, rec_col = st.columns(2)
    with report_col:
        if st.button("Tạo báo cáo điều tra", use_container_width=True):
            result = _generate_report(current_df)
            _render_ai_output(result)
    with rec_col:
        if st.button("Tạo khuyến nghị", use_container_width=True):
            result = _generate_recommendations(current_df)
            _render_ai_output(result)

    question = st.text_input("Hỏi câu hỏi về tập dữ liệu đã lọc", placeholder="Ví dụ: Nguồn nào có nhiều Lead thất bại nhất?")
    if question.strip():
        if st.button("Hỏi dữ liệu", use_container_width=True):
            result = _generate_question_answer(current_df, question)
            _render_ai_output(result)

    lead_id = st.text_input("Mã Lead để tóm tắt chi tiết", placeholder="Ví dụ: BNB-1234-0001")
    if lead_id.strip():
        if st.button("Tóm tắt Lead", use_container_width=True):
            result = _generate_lead_summary(current_df, lead_id)
            _render_ai_output(result)

    st.caption("Bảng điều khiển AI sử dụng tập dữ liệu đang lọc và luôn giữ thông báo dữ liệu mô phỏng trong mỗi câu trả lời.")


def render_lost_intelligence(df: pd.DataFrame) -> None:
    """Render the lost-lead intelligence section with meaningful business charts."""
    if df.empty:
        st.info("No data is available to analyze.")
        return

    st.markdown('<div class="section-header">Phân tích Lead thất bại</div>', unsafe_allow_html=True)
    segments = build_lost_lead_segments(df)

    source_summary = segments.get("by_source", pd.DataFrame())
    product_summary = segments.get("by_product", pd.DataFrame())
    reason_summary = segments.get("loss_reason_distribution", pd.DataFrame())
    budget_summary = segments.get("by_budget_band", pd.DataFrame())
    owner_summary = segments.get("by_owner", pd.DataFrame())
    follow_up_summary = segments.get("by_follow_up", pd.DataFrame())

    chart_col_a, chart_col_b = st.columns(2)
    if not source_summary.empty:
        source_chart = px.bar(
            source_summary.sort_values("lost_leads", ascending=False),
            x="Source",
            y="lost_leads",
            color="lost_leads",
            title="Lead thất bại theo nguồn",
            labels={"Source": "Nguồn", "lost_leads": "Lead thất bại"},
            color_continuous_scale="Teal",
        )
        source_chart.update_layout(template="plotly_white", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", margin=dict(t=30, r=10, b=40, l=10))
        chart_col_a.caption("Bối cảnh kinh doanh: xác định nguồn tiếp cận tạo ra nhiều doanh nghiệp thất bại nhất.")
        chart_col_a.plotly_chart(source_chart, width="stretch")

    if not product_summary.empty:
        product_chart = px.bar(
            product_summary.sort_values("lost_leads", ascending=False),
            x="Product",
            y="lost_leads",
            color="lost_leads",
            title="Lead thất bại theo sản phẩm",
            labels={"Product": "Sản phẩm", "lost_leads": "Lead thất bại"},
            color_continuous_scale="Sunset",
        )
        product_chart.update_layout(template="plotly_white", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", margin=dict(t=30, r=10, b=40, l=10))
        chart_col_b.caption("Bối cảnh kinh doanh: xác định danh mục sản phẩm có nguy cơ mất khách hàng cao nhất.")
        chart_col_b.plotly_chart(product_chart, width="stretch")

    chart_col_c, chart_col_d = st.columns(2)
    if not reason_summary.empty:
        reason_chart = px.bar(
            reason_summary.fillna("Missing").sort_values("lost_leads", ascending=False),
            x="lost_leads",
            y="Loss Reason",
            orientation="h",
            title="Phân bổ nguyên nhân mất Lead",
            labels={"lost_leads": "Lead thất bại", "Loss Reason": "Nguyên nhân"},
            color="lost_leads",
            color_continuous_scale="Blues",
        )
        reason_chart.update_layout(template="plotly_white", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", margin=dict(t=30, r=10, b=20, l=20))
        chart_col_c.caption("Bối cảnh kinh doanh: làm rõ nguyên nhân chính đứng sau các deal bị mất.")
        chart_col_c.plotly_chart(reason_chart, width="stretch")

    if not budget_summary.empty:
        budget_chart = px.bar(
            budget_summary,
            x="Budget Band",
            y="lost_leads",
            color="lost_leads",
            title="Lead thất bại theo ngân sách",
            labels={"Budget Band": "Ngân sách", "lost_leads": "Lead thất bại"},
            color_continuous_scale="Viridis",
        )
        budget_chart.update_layout(template="plotly_white", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", margin=dict(t=30, r=10, b=20, l=10))
        chart_col_d.caption("Bối cảnh kinh doanh: so sánh mức độ mất khách hàng giữa các phân khúc ngân sách.")
        chart_col_d.plotly_chart(budget_chart, width="stretch")

    st.markdown("### So sánh phân khúc")
    segment_tabs = st.tabs(["Nguồn", "Nhân sự phụ trách", "Theo dõi", "Nguyên nhân"])
    with segment_tabs[0]:
        if not source_summary.empty:
            st.dataframe(source_summary, use_container_width=True)
    with segment_tabs[1]:
        if not owner_summary.empty:
            st.dataframe(owner_summary, use_container_width=True)
    with segment_tabs[2]:
        if not follow_up_summary.empty:
            st.dataframe(follow_up_summary, use_container_width=True)
    with segment_tabs[3]:
        if not reason_summary.empty:
            st.dataframe(reason_summary, use_container_width=True)


# --- lead investigation view ---

def render_lead_investigation(df: pd.DataFrame) -> None:
    """Render a functional lead investigation workflow using the active dataset."""
    st.markdown('<div class="section-header">Điều tra khách hàng</div>', unsafe_allow_html=True)

    lead_options = _lead_search_options(df)
    selected_lead_id = _coerce_selected_lead_id(lead_options, st.session_state.get("lead_selection"))
    if selected_lead_id is None:
        st.session_state["lead_selection"] = None

    search_term = st.text_input(
        "Tìm theo mã Lead",
        value=(selected_lead_id or ""),
        placeholder="Ví dụ: BNB-1234-0001",
        help="Tìm kiếm trong dữ liệu hiện tại sau khi áp dụng bộ lọc hoặc tải lên tệp mới.",
        key="lead_search_input",
    )

    matches = _lead_search_matches(lead_options, search_term)
    if not matches:
        st.warning("Không tìm thấy Lead phù hợp với từ khoá hiện tại trong tập dữ liệu đang hoạt động.")
        st.session_state["lead_selection"] = None
        return

    selected_lead_id = st.selectbox(
        "Chọn Lead từ kết quả tìm kiếm",
        options=matches,
        index=matches.index(selected_lead_id) if selected_lead_id in matches else 0,
        key="lead_search_result",
    )

    st.session_state["lead_selection"] = selected_lead_id
    lead = _get_selected_lead(df, selected_lead_id)
    if lead is None:
        st.warning("Lead không còn có sẵn trong tập dữ liệu đang lọc. Vui lòng chọn lại.")
        return

    _render_customer_info_panel(lead)
    _render_lead_edit_form(df, selected_lead_id)
    _render_investigation_analysis(df, selected_lead_id)


def render_sidebar_controls() -> tuple[pd.DataFrame, dict, Any, Any, Any, Any, Any, Any, Any]:
    """Render the sidebar controls and dataset filters."""
    with st.sidebar:
        # Navigation section
        st.markdown('<div class="sidebar-section"><h3>Điều hướng</h3></div>', unsafe_allow_html=True)
        section = st.radio(
            "Phần",
            [
                "Tổng quan",
                "Trung tâm phân tích AI",
                "Phân tích Lead thất bại",
                "Điều tra khách hàng",
                "Khuyến nghị và hành động",
            ],
            index=0,
            label_visibility="collapsed",
        )

        # Data management section
        st.markdown('<div class="sidebar-section"><h3>Quản lý dữ liệu</h3></div>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Tải lên CSV hoặc Excel", type=["csv", "xlsx", "xls"])
        use_synthetic = st.checkbox("Dùng tập dữ liệu mẫu mô phỏng", value=True)

        active_df, validation = load_dataset_from_source(uploaded_file=uploaded_file, use_synthetic=use_synthetic)

        # Show validation summary if data is loaded (even if some rows are invalid)
        if not active_df.empty:
            try:
                from src.data_processing import summarize_validation
                summary_text = summarize_validation(validation)
                st.info(summary_text)
            except Exception:
                st.info(f"Tổng dòng: {validation.get('total_rows', 0)} | Dòng hợp lệ: {validation.get('valid_rows', 0)} | Dòng không hợp lệ: {validation.get('invalid_rows', 0)}")
            # Show invalid rows preview if any
            if validation.get('invalid_rows', 0) > 0 and not validation.get('problem_rows').empty:
                with st.expander("Xem các dòng lỗi (invalid rows)"):
                    st.dataframe(validation['problem_rows'])
            # Missing columns warning
            missing = validation.get('missing_columns', [])
            if missing:
                st.warning(f"Các cột bắt buộc thiếu: {', '.join(missing)}. Vui lòng tải lại tệp với đầy đủ các cột.")

        # Filters section
        st.markdown('<div class="sidebar-section"><h3>Bộ lọc</h3></div>', unsafe_allow_html=True)
        if active_df.empty:
            st.info("Chưa có dữ liệu được tải.")
            return active_df, validation, section, None, None, [], [], [], []

        df_for_filters = active_df.copy()
        if "Lead Date" in df_for_filters.columns:
            date_start = st.date_input("Ngày bắt đầu", pd.to_datetime(df_for_filters["Lead Date"]).min().date())
            date_end = st.date_input("Ngày kết thúc", pd.to_datetime(df_for_filters["Lead Date"]).max().date())
        else:
            date_start = None
            date_end = None

        source_options = sorted(df_for_filters["Source"].dropna().astype(str).str.strip().unique().tolist())
        product_options = sorted(df_for_filters["Product"].dropna().astype(str).str.strip().unique().tolist())
        owner_options = sorted(df_for_filters["Owner"].dropna().astype(str).str.strip().unique().tolist())
        status_options = sorted(df_for_filters["Status Classification"].dropna().astype(str).str.strip().unique().tolist())

        selected_sources = st.multiselect("Nguồn", source_options, default=source_options)
        selected_products = st.multiselect("Sản phẩm", product_options, default=product_options)
        selected_owners = st.multiselect("Nhân viên phụ trách", owner_options, default=owner_options)
        selected_statuses = st.multiselect("Trạng thái", status_options, default=status_options)

        return active_df, validation, section, date_start, date_end, selected_sources, selected_products, selected_owners, selected_statuses

    return pd.DataFrame(), {"valid_rows": 0, "invalid_rows": 0, "problem_rows": pd.DataFrame()}, "Tổng quan", None, None, [], [], [], []


def _normalize_lead_value(value: Any) -> str:
    """Render a consistent display value for lead fields."""
    if value is None or pd.isna(value):
        return "Chưa có ghi nhận"
    return str(value).strip() if str(value).strip() else "Chưa có ghi nhận"


def _get_lead_records(df: pd.DataFrame) -> pd.DataFrame:
    """Return a normalized dataset of lead records with a stable Lead ID index."""
    if df.empty:
        return pd.DataFrame(columns=["Lead ID", "Lead Date", "Source", "Product", "Budget", "Status", "Owner", "Customer Concerns", "Next Action", "Follow-up Status", "Last Interaction", "Loss Reason", "Status Classification"])
    working_df = df.copy()
    for column in ["Lead ID", "Source", "Product", "Status", "Owner", "Follow-up Status", "Loss Reason", "Status Classification"]:
        if column in working_df.columns:
            working_df[column] = working_df[column].astype(str).fillna("")
    return working_df.reset_index(drop=True)


def _lead_search_options(df: pd.DataFrame) -> list[str]:
    """Return valid, unique Lead IDs from the active dataset while ignoring blanks and nulls."""
    if df is None or df.empty:
        return []
    if "Lead ID" not in df.columns:
        return []

    options: list[str] = []
    for value in df["Lead ID"]:
        text = str(value).strip() if value is not None and not pd.isna(value) else ""
        if text == "":
            continue
        if text not in options:
            options.append(text)
    return options


def _lead_search_matches(lead_options: list[str], search_term: str) -> list[str]:
    """Return all Lead IDs matching the user input, with case-insensitive partial matching."""
    term = str(search_term or "").strip().lower()
    if not term:
        return list(lead_options)
    return [lead_id for lead_id in lead_options if term in str(lead_id).strip().lower()]


def _coerce_selected_lead_id(lead_options: list[str], value: Any) -> str | None:
    """Normalize and validate a persisted Lead ID selection against the active dataset."""
    if value is None or value == "":
        return None
    normalized = str(value).strip()
    if not normalized:
        return None
    if normalized in lead_options:
        return normalized
    return None


def _get_selected_lead(df: pd.DataFrame, lead_id: str | None) -> pd.Series | None:
    """Fetch a selected lead by Lead ID from the active dataset."""
    if df.empty or not lead_id or "Lead ID" not in df.columns:
        return None
    matches = df[df["Lead ID"].astype(str).str.strip() == str(lead_id).strip()]
    if matches.empty:
        return None
    return matches.iloc[0]


def _update_lead_record(df: pd.DataFrame, lead_id: str, updates: dict[str, Any]) -> pd.DataFrame:
    """Update a single lead record in the working dataframe and keep all other rows unchanged."""
    if df.empty or "Lead ID" not in df.columns:
        return df.copy()

    updated = df.copy()
    mask = updated["Lead ID"].astype(str).str.strip() == str(lead_id).strip()
    if not mask.any():
        return updated

    for key, value in updates.items():
        if key not in updated.columns:
            continue

        if key == "Last Interaction":
            updated[key] = pd.to_datetime(updated[key], errors="coerce")
            updated.loc[mask, key] = pd.Timestamp(value)
        elif key == "Budget":
            updated[key] = pd.to_numeric(updated[key], errors="coerce").astype(float)
            updated.loc[mask, key] = float(value)
        else:
            updated.loc[mask, key] = value
    return updated.reset_index(drop=True)


def _render_customer_info_panel(lead: pd.Series) -> None:
    """Render the customer identity and lead details in structured sections."""
    st.markdown("### Thông tin khách hàng")
    info_items = [
        ("Lead ID", lead.get("Lead ID", "Chưa có ghi nhận")),
        ("Ngày tạo Lead", lead.get("Lead Date", "Chưa có ghi nhận")),
        ("Nguồn Lead", lead.get("Source", "Chưa có ghi nhận")),
        ("Sản phẩm quan tâm", lead.get("Product", "Chưa có ghi nhận")),
        ("Ngân sách", lead.get("Budget", "Chưa có ghi nhận")),
        ("Nhân viên phụ trách", lead.get("Owner", "Chưa có ghi nhận")),
    ]

    cols = st.columns(3)
    for idx, (label, value) in enumerate(info_items):
        with cols[idx % 3]:
            st.markdown(f"<div class='panel'><strong>{label}</strong><br>{_normalize_lead_value(value)}</div>", unsafe_allow_html=True)

    st.markdown("### Tình trạng bán hàng")
    sales_items = [
        ("Trạng thái Lead", lead.get("Status", "Chưa có ghi nhận")),
        ("Trạng thái theo dõi", lead.get("Follow-up Status", "Chưa có ghi nhận")),
        ("Lần tương tác gần nhất", lead.get("Last Interaction", "Chưa có ghi nhận")),
        ("Hành động tiếp theo", lead.get("Next Action", "Chưa có ghi nhận")),
        ("Mối quan tâm của khách hàng", lead.get("Customer Concerns", "Chưa có ghi nhận")),
        ("Nguyên nhân mất Lead", lead.get("Loss Reason", "Chưa có ghi nhận")),
    ]
    sales_cols = st.columns(3)
    for idx, (label, value) in enumerate(sales_items):
        with sales_cols[idx % 3]:
            st.markdown(f"<div class='panel'><strong>{label}</strong><br>{_normalize_lead_value(value)}</div>", unsafe_allow_html=True)


def _render_lead_edit_form(df: pd.DataFrame, lead_id: str) -> None:
    """Render the editable lead form and persist updates into the session data."""
    lead = _get_selected_lead(df, lead_id)
    if lead is None:
        st.warning("Không tìm thấy Lead phù hợp để chỉnh sửa.")
        return

    with st.form(key=f"edit_lead_form_{lead_id}", clear_on_submit=False):
        st.markdown("### Chỉnh sửa thông tin Lead")
        st.caption("Các thay đổi được lưu trong phiên làm việc hiện tại; không ghi đè lên tệp gốc nếu bạn chưa xuất dữ liệu mới.")

        status_options = ["Mới hỏi", "Đã tư vấn sơ bộ", "Chờ ảnh bếp", "Chờ khảo sát", "Chờ kỹ thuật xem", "Đã báo giá", "Đã mua", "Mất"]
        follow_up_options = ["Complete", "Pending", "Not recorded"]

        edited_status = st.selectbox("Status", status_options, index=status_options.index(str(lead.get("Status", "Mới hỏi"))), key=f"lead_status_{lead_id}")
        edited_owner = st.text_input("Owner", value=str(lead.get("Owner", "")).strip() if not pd.isna(lead.get("Owner")) else "", key=f"lead_owner_{lead_id}")
        edited_budget = st.number_input("Budget", min_value=0.0, value=float(lead.get("Budget", 0) or 0), step=1000000.0, key=f"lead_budget_{lead_id}")
        edited_concerns = st.text_area("Customer Concerns", value=str(lead.get("Customer Concerns", "")).strip() if not pd.isna(lead.get("Customer Concerns")) else "", key=f"lead_concerns_{lead_id}")
        edited_next_action = st.text_input("Next Action", value=str(lead.get("Next Action", "")).strip() if not pd.isna(lead.get("Next Action")) else "", key=f"lead_next_action_{lead_id}")
        edited_follow_up = st.selectbox("Follow-up Status", follow_up_options, index=follow_up_options.index(str(lead.get("Follow-up Status", "Not recorded"))), key=f"lead_follow_up_{lead_id}")
        edited_last_interaction = st.date_input("Last Interaction", value=pd.to_datetime(lead.get("Last Interaction", pd.Timestamp.now())).date(), key=f"lead_last_interaction_{lead_id}")
        edited_loss_reason = st.text_area("Loss Reason", value=str(lead.get("Loss Reason", "")).strip() if not pd.isna(lead.get("Loss Reason")) else "", key=f"lead_loss_reason_{lead_id}")

        submitted = st.form_submit_button("Lưu thay đổi")
        if submitted:
            normalized_status = str(edited_status).strip()
            normalized_owner = str(edited_owner).strip()
            if not normalized_status:
                st.error("Trạng thái Lead không được để trống.")
                return
            if not normalized_owner:
                st.error("Nhân viên phụ trách không được để trống.")
                return
            if edited_budget < 0:
                st.error("Ngân sách không được âm.")
                return

            updates = {
                "Status": normalized_status,
                "Owner": normalized_owner,
                "Budget": float(edited_budget),
                "Customer Concerns": edited_concerns.strip(),
                "Next Action": edited_next_action.strip(),
                "Follow-up Status": edited_follow_up,
                "Last Interaction": pd.Timestamp(edited_last_interaction),
                "Loss Reason": edited_loss_reason.strip(),
                "Status Classification": {
                    "Mới hỏi": "Open",
                    "Đã tư vấn sơ bộ": "Open",
                    "Chờ ảnh bếp": "Open",
                    "Chờ khảo sát": "Open",
                    "Chờ kỹ thuật xem": "Open",
                    "Đã báo giá": "Open",
                    "Đã mua": "Converted",
                    "Mất": "Lost",
                }.get(normalized_status, "Open"),
            }
            new_df = _update_lead_record(df, lead_id, updates)
            st.session_state["current_df"] = new_df
            st.session_state["lead_selection"] = lead_id
            st.session_state["edited_lead_success"] = True
            st.success("Cập nhật thông tin Lead thành công trong phiên làm việc hiện tại.")
            st.rerun()


def _render_investigation_analysis(df: pd.DataFrame, lead_id: str) -> None:
    """Render the lead investigation analysis using simulation-backed explanations and dataset evidence."""
    result = build_simulated_lead_analysis(df, lead_id)
    if result.get("status") != "success":
        st.warning(result.get("message", "Không thể tạo phân tích cho Lead đang chọn."))
        return

    st.markdown("### Phân tích điều tra Lead")
    st.caption("Chế độ mô phỏng AI – Phân tích ngoại tuyến dựa trên dữ liệu hiện có")

    analysis = result.get("summary", "")
    st.markdown(analysis)

    facts = result.get("recorded_facts", {})
    observations = result.get("observations", {})

    st.markdown("#### 1. Tóm tắt tình trạng Lead")
    if facts:
        summary_text = ", ".join(f"{key}: {value}" for key, value in facts.items() if value not in [None, "", pd.NaT])[:200]
        st.write("- Dữ liệu ghi nhận: " + summary_text)
    else:
        st.write("- Không có dữ liệu ghi nhận đủ để tóm tắt.")

    st.markdown("#### 2. Các thông tin đã ghi nhận")
    if facts:
        for key, value in facts.items():
            if value not in [None, "", pd.NaT]:
                st.write(f"- {key}: {value}")
    else:
        st.info("Chưa có thông tin nào được ghi nhận cho Lead này.")

    st.markdown("#### 3. Những thông tin còn thiếu")
    missing = observations.get("missing_fields", []) if isinstance(observations, dict) else []
    if missing:
        for item in missing:
            st.write(f"- {item}")
    else:
        st.success("Không phát hiện thông tin nào còn thiếu trong hồ sơ hiện tại.")

    st.markdown("#### 4. Các dấu hiệu cần chú ý")
    risk = observations.get("follow_up_risk", "Không có dấu hiệu đặc biệt") if isinstance(observations, dict) else "Không có dấu hiệu đặc biệt"
    status = observations.get("status_classification", "Unknown") if isinstance(observations, dict) else "Unknown"
    st.write(f"- Trạng thái: {status}")
    st.write(f"- Rủi ro theo dõi: {risk}")

    st.markdown("#### 5. Nhận định dựa trên dữ liệu")
    st.write("- Đây là nhận định dựa trên dữ liệu có sẵn, không phải suy đoán về hành vi khách hàng ngoài hồ sơ.")
    st.write("- Nếu Lead đang ở trạng thái chờ theo dõi or thiếu thông tin mối quan tâm, cần ưu tiên cập nhật và xác minh lại.")

    st.markdown("#### 6. Hành động tiếp theo được đề xuất")
    if observations.get("is_lost_lead"):
        st.write("- Xác định nguyên nhân mất Lead và kiểm tra lại cam kết theo dõi đã thực hiện.")
    elif observations.get("is_open_lead"):
        st.write("- Tiếp tục theo dõi, cập nhật tiến độ và nhắc phản hồi trong thời hạn đã đặt.")
    else:
        st.write("- Đánh giá tiến độ hiện tại và điều chỉnh chiến lược theo dõi cho Lead này.")


def render_recommendations(df: pd.DataFrame) -> None:
    """Render recommendation cards using rule-based analytics only."""
    st.markdown('<div class="section-header">Khuyến nghị và hành động</div>', unsafe_allow_html=True)
    if df.empty:
        st.info("Không có dữ liệu hợp lệ để tạo khuyến nghị.")
        return

    segments = build_lost_lead_segments(df)
    by_source = segments.get("by_source", pd.DataFrame())
    by_product = segments.get("by_product", pd.DataFrame())
    pending_follow_up = int((df["Follow-up Status"].astype(str).str.strip().str.lower() == "pending").sum()) if "Follow-up Status" in df.columns else 0

    recs = []
    if not by_source.empty:
        top_source = by_source.sort_values("lost_leads", ascending=False).iloc[0]
        recs.append(
            {
                "title": f"Ưu tiên rà soát theo dõi cho {top_source.get('Source', 'nguồn')}",
                "evidence": f"{top_source.get('lost_leads', 0)} Lead thất bại được ghi nhận trong nguồn này.",
                "action": "Xem xét chất lượng đầu vào, kịch bản bán hàng và thời điểm phản hồi cho phân khúc này.",
            }
        )
    if not by_product.empty:
        top_product = by_product.sort_values("lost_leads", ascending=False).iloc[0]
        recs.append(
            {
                "title": f"Kiểm tra rủi ro sản phẩm cho {top_product.get('Product', 'sản phẩm')}",
                "evidence": f"{top_product.get('lost_leads', 0)} Lead thất bại trong nhóm sản phẩm này.",
                "action": "Kiểm tra giá, sự phù hợp và các mối quan tâm khách hàng ảnh hưởng đến tỷ lệ chuyển đổi.",
            }
        )
    if pending_follow_up > 0:
        recs.append(
            {
                "title": "Giảm khoảng trống theo dõi",
                "evidence": f"Hiện có {pending_follow_up} hồ sơ đang ở trạng thái chờ theo dõi.",
                "action": "Thiết lập SLA rõ ràng để giảm thời gian chậm trễ và phục hồi cơ hội đang bị treo.",
            }
        )

    if not recs:
        st.info("Không xác định được mô hình hành động nào phù hợp với góc nhìn dữ liệu hiện tại.")
        return

    for rec in recs:
        st.markdown(
            f"""
            <div class="panel" style="margin-bottom: 1rem;">
                <h4>{rec['title']}</h4>
                <p><strong>Bằng chứng:</strong> {rec['evidence']}</p>
                <p><strong>Hành động đề xuất:</strong> {rec['action']}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.caption("Khu vực hành động này dựa trên quy tắc và phân tích dữ liệu. Chế độ AI mô phỏng được bật khi OpenAI không khả dụng.")

def main() -> None:
    """Entry point for the Streamlit app.
    Sets page config, injects CSS, loads data, and routes UI.
    """
    st.set_page_config(page_title="BNB Talent Dashboard", page_icon="🚀", layout="wide")
    _inject_custom_css()
    active_df, validation, section, date_start, date_end, selected_sources, selected_products, selected_owners, selected_statuses = render_sidebar_controls()
    if active_df.empty:
        st.stop()
    filtered_df = filter_dataframe(
        active_df,
        date_start=date_start,
        date_end=date_end,
        selected_sources=selected_sources,
        selected_products=selected_products,
        selected_owners=selected_owners,
        selected_statuses=selected_statuses,
    )
    if section == "Tổng quan":
        render_executive_overview(filtered_df)
    elif section == "Trung tâm phân tích AI":
        render_ai_placeholder()
    elif section == "Phân tích Lead thất bại":
        render_lost_intelligence(filtered_df)
    elif section == "Điều tra khách hàng":
        render_lead_investigation(filtered_df)
    elif section == "Khuyến nghị và hành động":
        render_recommendations(filtered_df)
    else:
        st.info("Chưa có nội dung cho phần này.")

if __name__ == "__main__":
    main()