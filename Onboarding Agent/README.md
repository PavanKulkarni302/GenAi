# Local AI Customer Onboarding Assistant (Starter)

This is a starter project for the **Local AI Customer Onboarding Assistant** capstone.
Everything runs locally using Hugging Face transformers (no paid APIs or tokens required).

## What is included
- app/ai_local.py - Hugging Face pipeline wrapper (summarize/generate).
- app/core.py - CSV loader, validation, summarizer & onboarding plan/email generator.
- ui/streamlit_app.py - Streamlit app to upload CSV and view outputs.
- tests/test_core.py - pytest tests (includes one intentionally failing test for debugging practice).
- requirements.txt - open-source dependencies.
- architecture.png - simple architecture diagram.

## Quickstart

1. Create a virtualenv and install requirements:
```
python -m venv venv
source venv/bin/activate      # on Windows: venv\Scripts\activate
pip install -r requirements.txt
```

2. Run Streamlit:
```
streamlit run ui/streamlit_app.py
```

3. Run tests:
```
pytest -q
```
