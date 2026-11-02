# BNB Lost Lead Investigation Prototype

This project is a local Streamlit prototype for investigating lost leads in a kitchen appliance sales context.

## Project status

Phase 1 is complete: Python, VS Code environment, and a minimal Streamlit app have been verified.

## Quick start

1. Open the project folder in VS Code.
2. Create and activate a virtual environment.
3. Install dependencies:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements.txt
   ```
4. Run the app:
   ```bash
   streamlit run app.py
   ```

## Project structure

- `app.py` — minimal Streamlit entry point
- `src/` — future analytics and processing modules
- `data/` — generated and uploaded datasets
- `tests/` — validation and analytics tests
- `outputs/` — export and demonstration files

## Notes

- The project uses synthetic data only.
- OpenAI integration is optional and not required for the core app.
- The app is a prototype, not a full CRM or deployment-ready system.
