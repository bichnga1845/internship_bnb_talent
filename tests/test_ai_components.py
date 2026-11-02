from __future__ import annotations

import pandas as pd

from src.ai_analysis import prepare_lead_context, summarize_analytics_context
from src.ai_prompts import build_lead_prompt, build_question_prompt, build_recommendation_prompt, build_report_prompt
from src.ai_simulation import (
    answer_simulated_question,
    generate_simulated_recommendations,
    generate_simulated_report,
    summarize_simulated_lead,
)
from src.llm_service import LLMService


def test_summarize_analytics_context_returns_expected_structure():
    df = pd.DataFrame(
        {
            "Lead ID": ["A1", "A2", "A3"],
            "Lead Date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
            "Source": ["Website", "Website", "Zalo"],
            "Product": ["Máy hút bụi", "Máy sấy", "Bếp điện từ"],
            "Budget": [10000000, 20000000, 30000000],
            "Status": ["Mới hỏi", "Đã mua", "Mất"],
            "Owner": ["Lan", "Minh", "Nhi"],
            "Customer Concerns": ["A", "B", "C"],
            "Next Action": ["Call", "Follow-up", ""],
            "Follow-up Status": ["Pending", "Complete", "Pending"],
            "Last Interaction": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]),
            "Loss Reason": ["", "", "Không đáp ứng ngân sách"],
            "Status Classification": ["Open", "Converted", "Lost"],
        }
    )
    context = summarize_analytics_context(df)
    assert context["total_leads"] == 3
    assert context["lost_leads"] == 1
    assert "synthetic_data_notice" in context
    assert "top_patterns" in context


def test_prepare_lead_context_handles_missing_information():
    df = pd.DataFrame(
        {
            "Lead ID": ["L1"],
            "Lead Date": [pd.Timestamp("2024-01-01")],
            "Source": ["Website"],
            "Product": ["Máy hút bụi"],
            "Budget": [10000000],
            "Status": ["Mất"],
            "Owner": ["Lan"],
            "Customer Concerns": [""],
            "Next Action": [""],
            "Follow-up Status": ["Pending"],
            "Last Interaction": [pd.Timestamp("2024-01-03")],
            "Loss Reason": [""],
            "Status Classification": ["Lost"],
        }
    )
    context = prepare_lead_context(df, "L1")
    assert context["found"] is True
    assert "Customer Concerns" in context["recorded_facts"]
    assert "Customer Concerns" in context["missing_information"]


def test_prompt_builders_return_contextual_text():
    report_prompt = build_report_prompt({"total_leads": 10, "lost_leads": 2})
    rec_prompt = build_recommendation_prompt({"lost_leads": 2})
    question_prompt = build_question_prompt("Which source had the highest losses?", {"top_patterns": "source data"})
    lead_prompt = build_lead_prompt({"lead_id": "L1", "status": "Lost"})

    assert "Executive summary" in report_prompt
    assert "recommendations" in rec_prompt.lower()
    assert "Question" in question_prompt
    assert "Lead context" in lead_prompt


def test_llm_service_works_without_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    service = LLMService(api_key=None)

    assert service.api_key is None
    assert service.model == "gpt-4o-mini"
    assert service.configured is False

    df = pd.DataFrame(
        {
            "Lead ID": ["A1"],
            "Status Classification": ["Lost"],
            "Source": ["Website"],
            "Follow-up Status": ["Pending"],
            "Loss Reason": ["Giá cao"],
        }
    )
    response = service.generate("Tạo báo cáo cho dữ liệu hiện tại", "system", data=df)
    assert "Chế độ mô phỏng AI" in response


def test_llm_service_does_not_trigger_openai_requests(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    df = pd.DataFrame(
        {
            "Lead ID": ["A1", "A2"],
            "Source": ["Website", "Website"],
            "Product": ["Máy hút bụi", "Máy sấy"],
            "Status Classification": ["Lost", "Lost"],
            "Follow-up Status": ["Pending", "Pending"],
            "Loss Reason": ["Giá cao", "Giá cao"],
        }
    )

    called = {"value": False}

    def fake_openai_call(*args, **kwargs):
        called["value"] = True
        raise AssertionError("External OpenAI endpoint should not be called")

    monkeypatch.setattr("src.llm_service.openai", type("Dummy", (), {"OpenAI": staticmethod(fake_openai_call)}), raising=False)
    service = LLMService(api_key=None)
    answer = service.generate("Nguồn nào có nhiều Lead thất bại nhất?", "system", data=df)
    assert "Website" in answer
    assert called["value"] is False
    assert service.configured is False


def test_simulated_report_uses_dataset_values():
    df = pd.DataFrame(
        {
            "Lead ID": ["A1", "A2", "A3"],
            "Lead Date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
            "Source": ["Website", "Website", "Zalo"],
            "Product": ["Máy hút bụi", "Máy sấy", "Bếp điện từ"],
            "Budget": [10000000, 20000000, 30000000],
            "Status": ["Mới hỏi", "Đã mua", "Mất"],
            "Owner": ["Lan", "Minh", "Nhi"],
            "Customer Concerns": ["A", "B", "C"],
            "Next Action": ["Call", "Follow-up", ""],
            "Follow-up Status": ["Pending", "Complete", "Pending"],
            "Last Interaction": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]),
            "Loss Reason": ["", "", "Không đáp ứng ngân sách"],
            "Status Classification": ["Open", "Converted", "Lost"],
        }
    )
    report = generate_simulated_report(df)
    assert "Chế độ mô phỏng AI" in report
    assert "Lead thành công" in report
    assert "Tổng số Lead" in report


def test_simulated_question_answer_uses_dataset():
    df = pd.DataFrame(
        {
            "Lead ID": ["A1", "A2"],
            "Status Classification": ["Lost", "Lost"],
            "Source": ["Website", "Website"],
            "Product": ["Máy hút bụi", "Máy sấy"],
            "Follow-up Status": ["Pending", "Pending"],
            "Loss Reason": ["Giá cao", "Giá cao"],
        }
    )
    answer = answer_simulated_question(df, "Nguồn nào có nhiều Lead thất bại nhất?")
    assert "Website" in answer
    assert "Chế độ mô phỏng AI" in answer


def test_simulated_recommendations_and_lead_summary():
    df = pd.DataFrame(
        {
            "Lead ID": ["L1"],
            "Lead Date": [pd.Timestamp("2024-01-01")],
            "Source": ["Website"],
            "Product": ["Máy hút bụi"],
            "Budget": [10000000],
            "Status": ["Mất"],
            "Owner": ["Lan"],
            "Customer Concerns": ["Giá cao"],
            "Next Action": ["Gọi lại"],
            "Follow-up Status": ["Pending"],
            "Last Interaction": [pd.Timestamp("2024-01-02")],
            "Loss Reason": ["Giá cao"],
            "Status Classification": ["Lost"],
        }
    )
    recs = generate_simulated_recommendations(df)
    summary = summarize_simulated_lead(df, "L1")
    assert "Khuyến nghị mô phỏng AI" in recs
    assert "Tổng quan lead mô phỏng AI" in summary


def test_simulated_outputs_use_dataset_context():
    df = pd.DataFrame(
        {
            "Lead ID": ["A1", "A2"],
            "Status Classification": ["Open", "Lost"],
            "Source": ["Website", "Website"],
            "Product": ["Máy hút bụi", "Máy sấy"],
            "Follow-up Status": ["Pending", "Pending"],
            "Loss Reason": ["", "Giá cao"],
        }
    )

    report = generate_simulated_report(df)
    recs = generate_simulated_recommendations(df)
    answer = answer_simulated_question(df, "Nguồn nào có nhiều Lead thất bại nhất?")

    assert "Website" in report or "Website" in recs or "Website" in answer
    assert "Chế độ mô phỏng AI" in report
    assert "Khuyến nghị mô phỏng AI" in recs
    assert "Câu trả lời mô phỏng AI" in answer


def test_failures_in_one_feature_do_not_break_other_features():
    df = pd.DataFrame(
        {
            "Lead ID": ["A1", "A2"],
            "Status Classification": ["Open", "Converted"],
            "Source": ["Website", "Zalo"],
            "Product": ["Máy hút bụi", "Máy sấy"],
            "Follow-up Status": ["Pending", "Complete"],
            "Loss Reason": ["", ""],
        }
    )

    report = generate_simulated_report(df)
    summary = summarize_simulated_lead(df, "A1")
    answer = answer_simulated_question(df, "Tỷ lệ chuyển đổi hiện tại là bao nhiêu?")

    assert "Chế độ mô phỏng AI" in report
    assert "Tổng quan lead mô phỏng AI" in summary
    assert "Câu trả lời mô phỏng AI" in answer
