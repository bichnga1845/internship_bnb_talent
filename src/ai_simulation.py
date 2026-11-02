from __future__ import annotations

from typing import Any

import pandas as pd

from src.analytics import build_lost_lead_segments, calculate_kpis, investigate_lead


def _fmt_pct(value: Any) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"{float(value):.2f}%"


def _safe_value(series: pd.Series, field: str, default: str = "N/A") -> Any:
    if series.empty:
        return default
    value = series.iloc[0].get(field, default)
    return value if value is not None and str(value).strip() != "" else default


def _build_loss_segments_safe(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Return segment summaries, falling back to direct aggregation when optional columns are missing."""
    try:
        return build_lost_lead_segments(df)
    except KeyError:
        working_df = df.copy()
        working_df["Status Classification"] = working_df.get("Status Classification", pd.Series(["" for _ in range(len(working_df))], index=working_df.index))
        working_df["Status Classification"] = working_df["Status Classification"].astype(str).str.strip()
        lost_df = working_df[working_df["Status Classification"] == "Lost"].copy()

        def safe_counts(column: str, label: str) -> pd.DataFrame:
            if column not in working_df.columns or working_df.empty:
                return pd.DataFrame(columns=[label, "lost_leads"])
            grouped = lost_df.groupby(column, dropna=False).size().reset_index(name="lost_leads") if not lost_df.empty else pd.DataFrame(columns=[label, "lost_leads"])
            grouped = grouped.rename(columns={column: label}) if not grouped.empty else grouped
            return grouped

        return {
            "loss_reason_distribution": safe_counts("Loss Reason", "Loss Reason"),
            "by_source": safe_counts("Source", "Source"),
            "by_product": safe_counts("Product", "Product"),
            "by_owner": safe_counts("Owner", "Owner"),
            "by_follow_up": safe_counts("Follow-up Status", "Follow-up Status"),
            "by_budget_band": pd.DataFrame(columns=["Budget Band", "lost_leads"]),
        }


def build_simulated_lead_analysis(df: pd.DataFrame, lead_id: str) -> dict[str, Any]:
    """Build a structured lead summary using the current dataset as the only source of truth."""
    if df.empty:
        return {
            "status": "error",
            "lead_id": lead_id,
            "message": "Không có dữ liệu để điều tra Lead hiện tại.",
            "recorded_facts": {},
            "observations": {},
            "summary": "### Tổng quan lead mô phỏng AI\n\nKhông có dữ liệu phù hợp để kiểm tra Lead này.",
        }

    result = investigate_lead(df, lead_id)
    if not result.get("found"):
        return {
            "status": "error",
            "lead_id": lead_id,
            "message": result.get("message", "Không tìm thấy Lead phù hợp."),
            "recorded_facts": {},
            "observations": {},
            "summary": "### Tổng quan lead mô phỏng AI\n\nKhông tìm thấy Lead này trong tập dữ liệu đang lọc.",
        }

    facts = result.get("recorded_facts", {})
    observations = result.get("analytical_observations", {})
    summary = summarize_simulated_lead(df, lead_id)

    return {
        "status": "success",
        "lead_id": lead_id,
        "message": result.get("message", "Lead được tìm thấy."),
        "recorded_facts": facts,
        "observations": observations,
        "summary": summary,
    }


def generate_simulated_report(df: pd.DataFrame) -> str:
    """Return a Vietnamese simulated investigation report derived from the current dataset."""
    if df.empty:
        return """### Báo cáo mô phỏng AI

Không có dữ liệu phù hợp để tạo báo cáo. Vui lòng tải lên một tập dữ liệu hoặc áp dụng bộ lọc khác.

> Chế độ mô phỏng AI: Đây là kết quả dựa trên dữ liệu hiện tại, không phải phản hồi từ mô hình AI thực tế."""

    kpis = calculate_kpis(df)
    segments = _build_loss_segments_safe(df)
    source_summary = segments.get("by_source", pd.DataFrame())
    product_summary = segments.get("by_product", pd.DataFrame())
    reason_summary = segments.get("loss_reason_distribution", pd.DataFrame())
    follow_up_summary = segments.get("by_follow_up", pd.DataFrame())

    top_source = source_summary.sort_values("lost_leads", ascending=False).iloc[0] if not source_summary.empty else None
    top_product = product_summary.sort_values("lost_leads", ascending=False).iloc[0] if not product_summary.empty else None
    top_reason = reason_summary.sort_values("lost_leads", ascending=False).iloc[0] if not reason_summary.empty else None
    pending_follow_up = int((df["Follow-up Status"].astype(str).str.strip().str.lower() == "pending").sum()) if "Follow-up Status" in df.columns else 0

    source_text = f"{top_source.get('Source', 'N/A')} với {top_source.get('lost_leads', 0)} Lead thất bại" if top_source is not None else "Không đủ dữ liệu để xác định"
    product_text = f"{top_product.get('Product', 'N/A')} với {top_product.get('lost_leads', 0)} Lead thất bại" if top_product is not None else "Không đủ dữ liệu để xác định"
    reason_text = f"{top_reason.get('Loss Reason', 'N/A')} với {top_reason.get('lost_leads', 0)} Lead" if top_reason is not None else "Không đủ dữ liệu để xác định"

    source_recommendation = (
        f"- Tập trung rà soát nguồn {top_source.get('Source', 'N/A')} nếu giá trị này có số Lead thất bại cao nhất."
        if top_source is not None
        else "- Không đủ dữ liệu nguồn để gợi ý phân tích sâu hơn."
    )
    product_recommendation = (
        f"- Kiểm tra lại cách trình bày và định giá cho sản phẩm {top_product.get('Product', 'N/A')}."
        if top_product is not None
        else "- Không đủ dữ liệu sản phẩm để gợi ý phân tích sâu hơn."
    )

    report = f"""### Báo cáo điều tra Lead thất bại
> Chế độ mô phỏng AI: Kết quả này được tạo dựa trên dữ liệu hiện tại và chỉ mang tính hỗ trợ phân tích. Không coi đây là kết luận từ mô hình AI thật.

## 1. Tóm tắt điều hành
- Tổng số Lead: {kpis.get('total_leads', 0)}
- Lead thành công: {kpis.get('converted_leads', 0)}
- Lead thất bại: {kpis.get('lost_leads', 0)}
- Lead đang xử lý: {kpis.get('open_leads', 0)}
- Tỷ lệ chuyển đổi: {_fmt_pct(kpis.get('conversion_rate'))}
- Tỷ lệ Lead thất bại: {_fmt_pct(kpis.get('lost_lead_rate'))}
- Phần chia Lead thất bại trong tổng pipeline: {_fmt_pct(kpis.get('lost_lead_share'))}

## 2. Các phát hiện chính
1. Nguồn lead có nhiều thất bại nhất là {source_text}.
2. Nhóm sản phẩm có nhiều thất bại nhất là {product_text}.
3. Nguyên nhân mất Lead phổ biến nhất là {reason_text}.
4. Có {pending_follow_up} Lead đang ở trạng thái chờ theo dõi.

## 3. Phân tích xu hướng Lead thất bại
Dựa trên dữ liệu đã lọc, các nhóm lead thất bại tập trung ở các kênh và sản phẩm có tỷ lệ mất khách hàng cao hơn mức trung bình. Nhóm nguồn và sản phẩm có số lượng thất bại cao nhất cần được xem xét trước tiên để tìm nguyên nhân thực sự.

## 4. Các nguyên nhân tiềm năng
- Nguồn lead có mức thất bại cao cho thấy chất lượng đầu vào hoặc hiệu quả kênh bán hàng cần được rà soát.
- Sản phẩm có tỷ lệ thất bại cao có thể phản ánh vấn đề về giá, đối tượng khách hàng hoặc thông điệp bán hàng.
- Các Lead đang ở trạng thái chờ theo dõi có thể bị nghẽn trong quy trình xử lý nếu không có SLA rõ ràng.

## 5. Tác động đến quy trình bán hàng
- Phân bổ nguồn lực cho các kênh và sản phẩm có tỷ lệ thất bại cao giúp giảm mất cơ hội.
- Tăng tốc độ phản hồi và theo dõi lead có thể nâng tỷ lệ chuyển đổi ở nhóm đang ở trạng thái chờ.
- Việc phân tích nguyên nhân mất Lead giúp cải thiện chính sách giá, tư vấn và nội dung tiếp cận khách hàng.

## 6. Khuyến nghị cải thiện
{source_recommendation}
{product_recommendation}
- Thiết lập SLA cho trạng thái theo dõi và ưu tiên xử lý các lead đang chờ.
- Nâng chất lượng câu hỏi và tư vấn trong giai đoạn đầu để giảm đối tượng khách hàng do dự.

## 7. Giới hạn dữ liệu
- Dữ liệu hiện tại là dữ liệu mô phỏng phục vụ demo.
- Kết quả không chứng minh nguyên nhân gây ra mất lead ở mức causality.
- Chỉ có thể suy luận dựa trên các số liệu và phân khúc hiện có trong tập dữ liệu đang lọc.
"""
    return report.strip()


def generate_simulated_recommendations(df: pd.DataFrame) -> str:
    """Build evidence-based recommendation text in Vietnamese from the current dataset."""
    if df.empty:
        return "### Khuyến nghị mô phỏng AI\n\nKhông có dữ liệu phù hợp để tạo khuyến nghị."

    segments = build_lost_lead_segments(df)
    by_source = segments.get("by_source", pd.DataFrame())
    by_product = segments.get("by_product", pd.DataFrame())
    pending_follow_up = int((df["Follow-up Status"].astype(str).str.strip().str.lower() == "pending").sum()) if "Follow-up Status" in df.columns else 0

    recommendations = []
    if not by_source.empty:
        top_source = by_source.sort_values("lost_leads", ascending=False).iloc[0]
        recommendations.append(
            {
                "issue": f"Nguồn {top_source.get('Source', 'N/A')} đang tạo nhiều Lead thất bại nhất.",
                "evidence": f"{top_source.get('lost_leads', 0)} Lead thất bại được ghi nhận từ nguồn này.",
                "action": "Rà soát kênh tiếp cận, cách chào hàng và tốc độ phản hồi với nguồn này.",
                "benefit": "Giảm tỷ lệ mất khách hàng và nâng chất lượng đầu vào.",
                "priority": "High",
            }
        )
    if not by_product.empty:
        top_product = by_product.sort_values("lost_leads", ascending=False).iloc[0]
        recommendations.append(
            {
                "issue": f"Sản phẩm {top_product.get('Product', 'N/A')} ghi nhận mức thất bại cao.",
                "evidence": f"{top_product.get('lost_leads', 0)} Lead thất bại được ghi nhận trong nhóm sản phẩm này.",
                "action": "Kiểm tra giá, cách mô tả sản phẩm và các mối quan tâm của khách hàng đối với danh mục này.",
                "benefit": "Cải thiện khả năng chuyển đổi và giảm nguy cơ mất lead.",
                "priority": "High",
            }
        )
    if pending_follow_up > 0:
        recommendations.append(
            {
                "issue": "Có Lead đang chờ theo dõi nhưng chưa được xử lý kịp thời.",
                "evidence": f"Hiện có {pending_follow_up} Lead ở trạng thái chờ theo dõi.",
                "action": "Thiết lập SLA theo dõi rõ ràng và ưu tiên xử lý nhóm lead đang bị treo.",
                "benefit": "Tăng khả năng hồi phục cơ hội và giảm cạnh tranh thời gian.",
                "priority": "Medium",
            }
        )

    if not recommendations:
        return "### Khuyến nghị mô phỏng AI\n\nHiện chưa phát hiện ra mô hình nào đủ mạnh để đề xuất hành động cụ thể trong tập dữ liệu đang lọc."

    sections = ["### Khuyến nghị mô phỏng AI\n> Chế độ mô phỏng AI: các khuyến nghị được tạo từ dữ liệu hiện tại, không phải từ mô hình AI thật."]
    for idx, rec in enumerate(recommendations, start=1):
        sections.append(
            f"{idx}. **{rec['issue']}**\n"
            f"- Mô tả bằng chứng: {rec['evidence']}\n"
            f"- Hành động đề xuất: {rec['action']}\n"
            f"- Lợi ích kỳ vọng: {rec['benefit']}\n"
            f"- Mức độ ưu tiên: {rec['priority']}"
        )
    return "\n\n".join(sections)


def answer_simulated_question(df: pd.DataFrame, question: str) -> str:
    """Answer common Vietnamese sales questions from the current dataset without external services."""
    if df.empty:
        return "### Câu trả lời mô phỏng AI\n\nKhông có dữ liệu phù hợp để trả lời câu hỏi hiện tại."

    question_text = (question or "").strip().lower()
    if not question_text:
        return "### Câu trả lời mô phỏng AI\n\nVui lòng nhập câu hỏi để hệ thống tìm câu trả lời từ dữ liệu hiện tại."

    try:
        kpis = calculate_kpis(df)
        segments = _build_loss_segments_safe(df)
    except KeyError:
        kpis = {}
        segments = {}

    source_summary = segments.get("by_source", pd.DataFrame()) if isinstance(segments, dict) else pd.DataFrame()
    product_summary = segments.get("by_product", pd.DataFrame()) if isinstance(segments, dict) else pd.DataFrame()
    reason_summary = segments.get("loss_reason_distribution", pd.DataFrame()) if isinstance(segments, dict) else pd.DataFrame()

    if any(keyword in question_text for keyword in ["tỷ lệ chuyển đổi", "conversion", "chuyen doi"]):
        if not kpis:
            return "### Câu trả lời mô phỏng AI\n\nKhông đủ dữ liệu để tính tỷ lệ chuyển đổi trên tập dữ liệu hiện tại."
        return f"### Câu trả lời mô phỏng AI\n\nTỷ lệ chuyển đổi hiện tại là {_fmt_pct(kpis.get('conversion_rate'))}. Đây là tỷ số giữa số Lead thành công và tổng số Lead trong tập dữ liệu đang lọc."

    if any(keyword in question_text for keyword in ["nguồn nào", "source", "nhieu lead that bai", "mất lead nhiều"]):
        if source_summary.empty:
            return "### Câu trả lời mô phỏng AI\n\nHiện không có dữ liệu đủ để xác định nguồn nào gây ra nhiều Lead thất bại nhất."
        top_source = source_summary.sort_values("lost_leads", ascending=False).iloc[0]
        return f"### Câu trả lời mô phỏng AI\n\n> Chế độ mô phỏng AI: Nguồn có số Lead thất bại cao nhất là {top_source.get('Source', 'N/A')} với {top_source.get('lost_leads', 0)} Lead thất bại."

    if any(keyword in question_text for keyword in ["nguyên nhân", "loss reason", "mat lead", "phổ biến nhất"]):
        if reason_summary.empty:
            return "### Câu trả lời mô phỏng AI\n\nHiện chưa có đủ dữ liệu phần nguyên nhân mất Lead để xác định vận hành phổ biến nhất."
        top_reason = reason_summary.sort_values("lost_leads", ascending=False).iloc[0]
        return f"### Câu trả lời mô phỏng AI\n\n> Chế độ mô phỏng AI: Nguyên nhân mất Lead phổ biến nhất là {top_reason.get('Loss Reason', 'N/A')} với {top_reason.get('lost_leads', 0)} Lead."

    if any(keyword in question_text for keyword in ["đang chờ theo dõi", "chờ theo dõi", "pending", "dang cho"]):
        pending = int((df["Follow-up Status"].astype(str).str.strip().str.lower() == "pending").sum()) if "Follow-up Status" in df.columns else 0
        return f"### Câu trả lời mô phỏng AI\n\n> Chế độ mô phỏng AI: Hiện có {pending} Lead đang ở trạng thái chờ theo dõi trong tập dữ liệu đang lọc."

    if any(keyword in question_text for keyword in ["sản phẩm nào", "product", "nhieu lead that bai nhat"]):
        if product_summary.empty:
            return "### Câu trả lời mô phỏng AI\n\nHiện chưa đủ dữ liệu để xác định sản phẩm nào có nhiều Lead thất bại nhất."
        top_product = product_summary.sort_values("lost_leads", ascending=False).iloc[0]
        return f"### Câu trả lời mô phỏng AI\n\n> Chế độ mô phỏng AI: Sản phẩm có số Lead thất bại cao nhất là {top_product.get('Product', 'N/A')} với {top_product.get('lost_leads', 0)} Lead thất bại."

    return "### Câu trả lời mô phỏng AI\n\nHệ thống hiện chỉ hỗ trợ các câu hỏi phổ biến về nguồn Lead, nguyên nhân mất Lead, trạng thái theo dõi, sản phẩm và tỷ lệ chuyển đổi. Vui lòng đặt câu hỏi phù hợp với dữ liệu hiện có."


def summarize_simulated_lead(df: pd.DataFrame, lead_id: str) -> str:
    """Create a Vietnamese lead summary using the dataset as the sole source of truth."""
    if not lead_id or not str(lead_id).strip():
        return "### Tổng quan lead mô phỏng AI\n\nVui lòng nhập mã Lead để hệ thống kiểm tra thông tin."

    result = investigate_lead(df, lead_id)
    if not result.get("found"):
        return "### Tổng quan lead mô phỏng AI\n\nKhông tìm thấy Lead này trong tập dữ liệu đang lọc."

    facts = result.get("recorded_facts", {})
    observations = result.get("analytical_observations", {})
    missing_fields = observations.get("missing_fields", [])

    status = facts.get("Status Classification", observations.get("status_classification", "Unknown"))
    source = facts.get("Source", "N/A")
    product = facts.get("Product", "N/A")
    owner = facts.get("Owner", "N/A")
    customer_concerns = facts.get("Customer Concerns", "Không có ghi nhận")
    next_action = facts.get("Next Action", "Không có ghi nhận")
    follow_up = facts.get("Follow-up Status", "Không có ghi nhận")
    loss_reason = facts.get("Loss Reason", "Không có ghi nhận")

    lines = [
        "### Tổng quan lead mô phỏng AI",
        f"- Mã Lead: {lead_id}",
        f"- Trạng thái: {status}",
        f"- Nguồn: {source}",
        f"- Sản phẩm: {product}",
        f"- Nhân viên phụ trách: {owner}",
        f"- Mối quan tâm của khách hàng: {customer_concerns}",
        f"- Hành động tiếp theo: {next_action}",
        f"- Trạng thái theo dõi: {follow_up}",
        f"- Nguyên nhân mất Lead: {loss_reason}",
        "",
        "### Thông tin còn thiếu",
    ]

    if missing_fields:
        lines.append("- " + "; ".join(missing_fields))
    else:
        lines.append("- Không phát hiện thiếu thông tin trong hồ sơ hiện có.")

    lines.extend([
        "",
        "### Giải thích khả dĩ",
        "- Chỉ dựa trên dữ liệu đã ghi nhận trong hồ sơ lead hiện tại.",
        "- Nếu thiếu thông tin về mối quan tâm hoặc nguyên nhân mất lead, hệ thống sẽ không suy đoán thêm.",
        "- Hành động tiếp theo nên tập trung vào việc cập nhật thông tin còn thiếu và xử lý trạng thái theo dõi kịp thời.",
    ])
    return "\n".join(lines)
