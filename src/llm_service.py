from __future__ import annotations

import os
from typing import Any

from src.ai_simulation import (
    answer_simulated_question,
    generate_simulated_recommendations,
    generate_simulated_report,
)


class LLMConfigurationError(RuntimeError):
    """Raised when the OpenAI configuration is missing or invalid."""


class LLMService:
    """Offline AI simulation wrapper used by this project.

    The app intentionally never calls a live external model. This service keeps the
    original interface but resolves requests through dataset-grounded simulation.
    """

    def __init__(self, api_key: str | None = None, model: str | None = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.client = None

    def _safe_payload(self, prompt: str, system_prompt: str) -> dict[str, Any]:
        return {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.2,
            "max_tokens": 800,
        }

    def generate(self, prompt: str, system_prompt: str, *, data=None) -> str:
        """Return an offline dataset-grounded answer. No external request is made."""
        if not isinstance(prompt, str) or not prompt.strip():
            return "Chế độ mô phỏng AI – Phân tích ngoại tuyến dựa trên dữ liệu hiện có\n\nVui lòng nhập câu hỏi hoặc ngữ cảnh dữ liệu trước khi tạo phản hồi."

        if data is not None:
            question = prompt.strip()
            lower_prompt = question.lower()
            if "báo cáo" in lower_prompt or "report" in lower_prompt:
                return generate_simulated_report(data)
            if "khuyến nghị" in lower_prompt or "recommend" in lower_prompt:
                return generate_simulated_recommendations(data)
            return answer_simulated_question(data, question)

        return (
            "Chế độ mô phỏng AI – Phân tích ngoại tuyến dựa trên dữ liệu hiện có\n\n"
            "Phản hồi này được tạo ra từ dữ liệu hiện có trong ứng dụng. Không có yêu cầu API bên ngoài nào được thực hiện."
        )

    @property
    def configured(self) -> bool:
        return False


def get_llm_service() -> LLMService:
    """Create a safe offline simulation service without relying on an external API."""
    return LLMService()
