from __future__ import annotations

AI_SYSTEM_INSTRUCTIONS = """
You are an evidence-based sales investigation assistant for a kitchen-appliance business.

Your job is to analyze the supplied analytics context, not raw customer data.
Use only numbers, categories, and records provided by the application.
Do not invent missing statistics, customer details, or business outcomes.
Never claim that correlation proves causation.

Requirements:
- Clearly separate facts from interpretations.
- Label hypotheses as hypotheses, not proven causes.
- If the evidence is weak or incomplete, say so explicitly.
- Reference synthetic data clearly as demonstration data only.
- Do not present synthetic findings as real company findings.
- Keep recommendations realistic for a kitchen-appliance sales process.
- If a question cannot be answered from the data, explain the limitation.
"""


def build_report_prompt(context: dict) -> str:
    """Build the prompt for an investigation report."""
    return f"""
You are preparing an executive investigation report for a lost-lead analysis review.

Use the following structured evidence only.

Dataset summary:
- synthetic_data_notice: {context.get('synthetic_data_notice', 'Not provided')}
- total_leads: {context.get('total_leads', 'N/A')}
- converted_leads: {context.get('converted_leads', 'N/A')}
- lost_leads: {context.get('lost_leads', 'N/A')}
- open_leads: {context.get('open_leads', 'N/A')}
- conversion_rate: {context.get('conversion_rate', 'N/A')}
- lost_lead_rate: {context.get('lost_lead_rate', 'N/A')}
- lost_lead_share: {context.get('lost_lead_share', 'N/A')}

Top patterns:
{context.get('top_patterns', 'No pattern summary supplied.')}

Loss reason distribution:
{context.get('loss_reason_distribution', 'No loss reason data supplied.')}

Lost leads by source:
{context.get('by_source', 'No source summary supplied.')}

Lost leads by product:
{context.get('by_product', 'No product summary supplied.')}

Follow-up status summary:
{context.get('by_follow_up', 'No follow-up summary supplied.')}

Important limitations:
{context.get('limitations', 'No limitations supplied.')}

Return a concise but professional report in markdown with sections:
1. Executive summary
2. Key findings
3. Evidence-based observations
4. Hypotheses and possible explanations
5. Business implications
6. Recommended actions
7. Data limitations

Keep each claim tied to the supplied numbers and do not speculate beyond the data.
"""


def build_recommendation_prompt(context: dict) -> str:
    """Build the prompt for action recommendations."""
    return f"""
Based on the following analytical context, propose practical recommendations for improving sales performance in a kitchen-appliance business.

Context:
{context}

Requirements:
- Give 3 to 5 actionable recommendations.
- Each recommendation must include: issue, supporting evidence, recommended action, expected business benefit, priority (High/Medium/Low)
- Use only the supplied metrics and segment summaries.
- Clearly label suggestions as recommendations, not proven causes.
- If the evidence is weak, say so.
"""


def build_question_prompt(question: str, context: dict) -> str:
    """Prepare a prompt for natural-language dataset Q&A."""
    return f"""
Answer the following business question using only the structured data below.

Question: {question}

Analytical context:
{context}

Rules:
- Use numbers only from the supplied data.
- If the data cannot support a confident answer, say that clearly.
- Avoid guessing or inventing missing business context.
- Keep the answer concise, professional, and evidence-based.
"""


def build_lead_prompt(lead_context: dict) -> str:
    """Prepare a prompt for a single-lead investigation summary."""
    return f"""
Investigate the following lead using only the records available in the dataset.

Lead context:
{lead_context}

Instructions:
- Summarize the current sales situation.
- List recorded facts and missing information separately.
- Highlight potential reasons for non-conversion based on the provided record only.
- Suggest next actions consistent with the available evidence.
- State clearly if information is missing from the data.
- Do not invent interactions, preferences, or guarantees that are not present.
"""
