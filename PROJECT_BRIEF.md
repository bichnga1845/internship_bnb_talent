# Project Brief: AI-Assisted Lost Lead Investigation Tool

## 1. Project identity

**Title:** Development of an AI-Assisted Lost Lead Investigation Tool for Sales Process Improvement

**Business context:** Bếp Ngọc Bảo (BNB), under Peaki Company Limited, in the kitchen-appliance sales context.

**Purpose:** Build a lightweight analytical prototype that helps a sales manager or analyst explore lost leads, inspect individual lead records, identify patterns in available data, and optionally receive AI-assisted summaries and investigation suggestions.

This is an investigation and decision-support tool. It is **not** a full CRM, predictive ML system, or automated causal-diagnosis system.

## 2. Business problem and objectives

The project is designed around potential lead-management challenges:
- Leads do not progress through the sales pipeline.
- Follow-up is delayed or incomplete.
- Customer information is incomplete.
- Managers have limited visibility into where leads become inactive or are lost.
- Individual lost leads are difficult to investigate consistently.

These are project problem areas, not proven company-wide findings unless supported by authorized company evidence.

Objectives:
1. Structure lead data for analysis.
2. Calculate lead-funnel and lost-lead metrics.
3. Explore patterns by source, product, budget, sales owner, and follow-up completeness.
4. Allow investigation of an individual lead by Lead ID.
5. Optionally use AI to summarize calculated findings and suggest further investigation.
6. Produce a working local prototype and test evidence for the internship report.

## 3. Expected outcome

A locally runnable web prototype built with Python and Streamlit. Users should be able to:
- Upload CSV or Excel lead data.
- Review data-quality checks.
- View KPI cards and interactive charts.
- Filter lead records.
- Analyze lost-lead patterns.
- Search for a Lead ID and inspect available details.
- Optionally generate evidence-grounded AI insights.
- Export filtered records or summaries where practical.

## 4. Scope boundaries

### In scope
- Synthetic data generation for development and demonstration.
- CSV/Excel upload.
- Data validation and basic cleaning.
- Lead-status classification.
- KPI calculations and grouped analysis.
- Interactive filters and charts.
- Lead-level investigation.
- Optional AI-assisted summaries and recommendations.
- Testing, documentation, and screenshots.

### Out of scope for version 1
- Full CRM functionality.
- User accounts, authentication, and role-based permissions.
- Automated email, SMS, Zalo, or sales follow-up.
- Live integration with company systems.
- Predictive machine learning.
- Automated causal inference or definitive identification of why a customer did not purchase.
- Complex database or multi-user deployment.
- Claims of improved real-world sales performance.

A pattern is an observation, not proof of causation. For example, a higher lost-lead rate in one source does not prove that the source caused the loss.

## 5. Recommended technology

Use **Visual Studio Code as the only IDE**.

| Component | Technology | Purpose |
|---|---|---|
| IDE | Visual Studio Code | Develop and run the project |
| Language | Python | Application and analytical logic |
| Web interface | Streamlit | Interactive local web app |
| Data processing | Pandas | Load, validate, transform, aggregate |
| Charts | Plotly | Interactive visualizations |
| Input files | CSV and Excel | Lead data |
| Optional AI | OpenAI API | Evidence-grounded summaries |
| Secrets | Environment variable or Streamlit secrets | Store API key securely |

Do not add a database in version 1. Do not require an AI API key for the core app. Build and test the analytics first.

## 6. Lead stages and classification

Use these illustrative stages as the initial project model:

| Status | English label | Classification |
|---|---|---|
| Mới hỏi | New Inquiry | Open |
| Đã tư vấn sơ bộ | Initial Consultation Completed | Open |
| Chờ ảnh bếp | Waiting for Kitchen Photos | Open |
| Chờ khảo sát | Waiting for Site Survey | Open |
| Chờ kỹ thuật xem | Waiting for Technical Review | Open |
| Đã báo giá | Quotation Provided | Open |
| Đã mua | Purchased | Converted |
| Mất | Lost | Lost |

Centralize this mapping in one place in the code. Open leads must not be treated as failed leads because they may still convert.

## 7. Suggested dataset schema

| Field | Type | Description |
|---|---|---|
| Lead ID | Text | Unique lead identifier |
| Lead Date | Date | Date the lead entered the process |
| Source | Category | Lead acquisition source |
| Product | Category | Product of interest |
| Budget | Numeric | Estimated budget in VND |
| Status | Category | Lead stage or outcome |
| Owner | Category | Assigned sales representative |
| Customer Concerns | Text/category | Customer questions or concerns |
| Next Action | Text | Planned next action, if any |
| Follow-up Status | Category | Complete, Pending, Not recorded |
| Last Interaction | Date | Latest recorded interaction date |
| Loss Reason | Category/text, optional | Recorded reason for a Lost lead, if available |

The app must document the expected schema. Do not imply that every field is confirmed to exist in the company's real system.

### Synthetic data rules
- Generate approximately 500–1,000 records initially; the app must also work with smaller files.
- Use a fixed random seed so the dataset is reproducible.
- Include Open, Converted, and Lost leads.
- Include varied sources, products, budgets, owners, and concerns.
- Include complete and incomplete follow-up records.
- Use unique fictional Lead IDs.
- Ensure Last Interaction is not earlier than Lead Date.
- Populate Loss Reason only for Lost leads, while allowing some Lost leads to have no recorded reason.
- Do not use real customer personal data.
- Clearly label all synthetic patterns as simulated design choices, not company findings.

## 8. KPI and analytical definitions

Let:
- `N_all` = all valid leads.
- `N_converted` = converted leads.
- `N_lost` = lost leads.
- `N_open` = open leads.

Counts:
- Total Leads = `N_all`
- Converted Leads = `N_converted`
- Lost Leads = `N_lost`
- Open Leads = `N_open`

**Lost Lead Rate**

`N_lost / (N_converted + N_lost) * 100`

Only completed outcomes (Converted and Lost) are included in the denominator. Open leads are excluded.

**Conversion Rate among completed leads**

`N_converted / (N_converted + N_lost) * 100`

**Lost Lead Share**

`N_lost / N_all * 100`

Do not confuse Lost Lead Rate with Lost Lead Share.

For a group `g`:

`Lost Lead Rate_g = N_lost,g / (N_converted,g + N_lost,g) * 100`

Show the numerator and denominator near the percentage, e.g. `12 / 30 completed leads`. If the denominator is zero, show `N/A`, not `0%`. Warn users when group sample sizes are small.

Recommended analysis dimensions:
- Lead source.
- Product.
- Budget range.
- Sales owner.
- Follow-up status/completeness.
- Loss reason, where recorded.
- Lead month/date, if date coverage supports it.

## 9. Data-quality requirements

Validate:
- Required columns exist.
- Lead ID is not blank.
- Duplicate Lead IDs.
- Status values are recognized.
- Dates can be parsed.
- Last Interaction is not earlier than Lead Date.
- Budget is numeric and non-negative.
- Follow-up Status uses allowed values.
- Loss Reason is only applicable to Lost leads; flag inconsistencies rather than silently rewriting them.
- Missing values in analytical fields.

Do not silently discard records. Report valid, invalid, and excluded row counts and explain why. Let users inspect problematic rows where practical.

## 10. Functional requirements

**FR-01 Data upload:** Accept CSV and Excel files with clear error messages.

**FR-02 Data validation:** Check required fields, data types, duplicates, statuses, dates, and numeric constraints. Show validation summary and problematic rows.

**FR-03 Dashboard:** Display Total Leads, Converted Leads, Lost Leads, Open Leads, Lost Lead Rate, and optionally Conversion Rate and Lost Lead Share.

**FR-04 Filters:** Filter by date, source, product, owner, and status. All relevant metrics, tables, and charts must update consistently.

**FR-05 Lost-lead analysis:** Show counts and rates by relevant dimensions. Include sample sizes and denominators.

**FR-06 Lead investigation:** Search by Lead ID and show available details: status, source, product, budget, concerns, follow-up status, next action, last interaction, and loss reason if recorded. Highlight missing information. Never invent an activity history that is not in the data.

**FR-07 AI insights (optional):** Return these sections:
1. Observed Finding.
2. Supporting Evidence.
3. Possible Interpretation.
4. Recommended Investigation.
5. Data Limitations.

AI must use only supplied metrics and details, distinguish observation from hypothesis, never invent missing evidence, and disclose incomplete or small samples. Do not send unnecessary personal or confidential data to an external API. The app must work if the API key is absent or the API fails.

**FR-08 Export:** Allow download of filtered data or summaries as CSV where practical. Never modify the original uploaded file.

## 11. Suggested user interface

Use a clean, professional, business-oriented interface. Keep it simple and readable.

1. **Overview**
   - KPI cards.
   - Lead status distribution.
   - Lost-lead trend if dates support it.
   - Data-quality notice.

2. **Lost Lead Analysis**
   - Lost counts and rates by source, product, owner, budget, and follow-up.
   - Interactive filters.
   - Tables showing sample sizes and denominators.

3. **Lead Investigation**
   - Lead ID search.
   - Lead detail panel.
   - Missing-information and follow-up indicators.

4. **AI Insights**
   - Generate evidence-grounded summary.
   - Display findings, evidence, interpretations, recommendations, and limitations.
   - Label AI-generated content clearly.

5. **Data Quality / Upload**
   - File upload.
   - Schema and validation results.
   - Invalid/incomplete rows.

Use Streamlit sidebar navigation or tabs—whichever is simpler and more maintainable.

## 12. Recommended project structure

```text
lost-lead-investigation/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── .env.example
├── data/
│   └── synthetic_leads.csv
├── src/
│   ├── __init__.py
│   ├── data_generator.py
│   ├── data_processing.py
│   ├── analytics.py
│   ├── lead_investigation.py
│   └── ai_insights.py
├── tests/
│   ├── test_data_processing.py
│   └── test_analytics.py
└── outputs/
```

Module responsibilities:
- `app.py`: Streamlit entry point, navigation, and page composition.
- `data_generator.py`: reproducible synthetic data.
- `data_processing.py`: loading, schema checks, validation, and cleaning.
- `analytics.py`: KPI and grouped calculations.
- `lead_investigation.py`: Lead ID lookup and detail preparation.
- `ai_insights.py`: optional AI prompt and response handling.
- `tests/`: tests for validation and calculations.

Keep the architecture beginner-friendly. Avoid unnecessary abstractions and excessive files.

## 13. Development sequence

### Phase 1 — Environment and setup
1. Confirm Python and VS Code.
2. Install the Python extension in VS Code.
3. Create and open the project folder.
4. Create a virtual environment.
5. Install minimum dependencies.
6. Create a minimal Streamlit app and run it locally.

**Milestone:** a minimal Streamlit page opens successfully in the browser.

### Phase 2 — Synthetic dataset
1. Implement a fixed-seed data generator.
2. Generate the CSV.
3. Inspect row count, fields, categories, missing values, and date consistency.
4. Document that it is synthetic.

**Milestone:** the CSV exists and Pandas can read it.

### Phase 3 — Upload and validation
1. Implement CSV and Excel upload.
2. Validate required columns.
3. Check duplicates, invalid categories, missing values, dates, and numeric fields.
4. Display validation summary and problematic rows.

**Milestone:** valid and invalid sample files produce understandable results.

### Phase 4 — Analytics
1. Implement centralized lead classification.
2. Implement KPI calculations.
3. Implement grouped lost-lead rates.
4. Verify calculations manually using a small known dataset.
5. Add filters and charts.

**Milestone:** KPI results match manual calculations and filters update all dependent views.

### Phase 5 — Lead investigation
1. Add Lead ID search.
2. Display available fields.
3. Highlight missing follow-up information.
4. Handle missing or duplicated IDs safely.

**Milestone:** valid and invalid Lead ID searches work correctly.

### Phase 6 — AI-assisted insights
1. Start only after core analytics are stable.
2. Send aggregated, non-sensitive evidence to the AI service.
3. Require structured output and limitations.
4. Handle missing API key, rate limits, and API errors.
5. Check that statements are traceable to supplied evidence.

**Milestone:** AI can summarize supplied evidence; the app remains functional without AI.

### Phase 7 — Testing and documentation
Test:
- Valid CSV and Excel uploads.
- Missing required columns.
- Duplicate Lead IDs.
- Unknown status.
- Invalid dates.
- Negative or nonnumeric budget.
- Missing follow-up information.
- Zero completed leads (rate must be N/A).
- KPI calculations against a manually checked example.
- Filters and charts.
- Valid and invalid Lead ID lookup.
- AI disabled or API failure.

Capture screenshots from the actual running app. Record test input, expected result, actual result, and pass/fail.

## 14. Coding-AI working rules

1. Work incrementally. Inspect the workspace before changing anything.
2. Propose a short plan and implement one phase at a time.
3. Never overwrite existing files blindly.
4. Do not claim code works unless it has actually been run.
5. Write readable beginner-friendly Python with clear names and straightforward functions.
6. Keep business logic separate from Streamlit UI code.
7. Centralize status mappings and required-column definitions.
8. Handle errors with clear user-facing messages.
9. Do not silently change or discard source data.
10. Calculate all KPIs and charts from the currently loaded dataset; never hardcode displayed results.
11. Clearly label synthetic data and simulated findings.
12. Keep API keys out of source code and version control.
13. Keep AI optional; core features must work without it.
14. Avoid scope creep: no CRM, authentication, database, or predictive ML unless explicitly approved.
15. After each phase, report: files created/changed, commands to run, expected result, tests performed, known limitations, and the next exact step.
16. Ask before major architecture or scope changes. If a requirement is unclear, ask one focused question.

## 15. Report alignment

### Chapter 1 — Introduction
Business context, problem statement, objectives, scope, limitations, methodology.

### Chapter 2 — Business Requirements and Solution Design
Existing lead process, potential investigation points, functional/non-functional requirements, proposed workflow, architecture, technology stack, data model, and analytical framework.

### Chapter 3 — Prototype Development
- 3.1 Synthetic Dataset Generation.
- 3.2 Data Processing and Exploratory Analysis.
- 3.3 Lost Lead Pattern Identification.
- 3.4 Lead Investigation Module.
- 3.5 AI-Assisted Findings and Recommendations.
- 3.6 User Interface and Application Integration.

Write this chapter based on features actually implemented. Include real screenshots, data fields, formulas, code explanations, and implementation decisions.

### Chapter 4 — Testing and Results
- 4.1 Testing Approach and Test Cases.
- 4.2 Functional Testing Results.
- 4.3 Prototype Demonstration and Analytical Findings.
- 4.4 Limitations and Evaluation.

Separate software test results from analytical observations. Label findings from synthetic data as illustrative.

### Chapter 5 — Conclusion and Recommendations
- 5.1 Project Outcomes.
- 5.2 Recommendations for Further Development.
- 5.3 Conclusion.

Only claim implemented features and verified results. Do not claim improved conversion, reduced lead loss, or real operational impact without deployment evidence.

## 16. Limitations to acknowledge

- The prototype uses synthetic data unless authorized real company data is provided.
- Synthetic distributions cannot establish actual company performance or customer behavior.
- Recorded loss reasons may be incomplete or subjective.
- The tool identifies patterns and supports investigation; it does not prove causes.
- AI interpretations depend on input quality and completeness.
- This is not a production CRM and does not automate sales operations.
- Local prototype tests do not prove measurable sales improvement.

## 17. First task

Begin by checking whether Python and VS Code are installed and confirming the Python version. Then guide the user to create the project folder, open it in VS Code, create a virtual environment, and install only the minimum dependencies.

Do not start with AI integration. The first milestone is a minimal Streamlit application that launches successfully.

## 18. Guiding principle

Build a small, reliable, explainable investigation tool—not an oversized system. Every metric must have a definition, every chart must use uploaded data, every lead-level statement must come from an available field, every AI recommendation must be tied to evidence, and every limitation must be stated honestly.
