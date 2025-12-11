"""
tests/test_core.py
Contains at least 4 tests. One test is intentionally failing to practice debugging.
"""
import io
import pytest
import pandas as pd
from app.core import load_and_validate_csv, OnboardingAssistant, ValidationError

SAMPLE_CSV = """name,company,services_requested,notes
Alice,Acme Inc,SEO;Analytics,Looking to improve organic traffic and set up analytics.
Bob,Example LLC,Ads,Needs ad setup and campaign strategy.
"""

def test_load_and_validate_success():
    buf = io.StringIO(SAMPLE_CSV)
    df = load_and_validate_csv(buf)
    assert isinstance(df, pd.DataFrame)
    assert df.shape[0] == 2

def test_load_and_validate_missing_column():
    bad_csv = "name,company,notes\nAlice,Acme,Notes only\n"
    buf = io.StringIO(bad_csv)
    with pytest.raises(ValidationError):
        load_and_validate_csv(buf)

def test_onboarding_generation_fallback():
    assistant = OnboardingAssistant()
    summary = assistant.summarize_client('')  # empty notes -> fallback
    plan = assistant.generate_onboarding_plan(summary)
    assert len(plan) == 3

def test_intentionally_failing_example():
    # This test is intentionally failing so students debug.
    assert 1 + 1 == 3
