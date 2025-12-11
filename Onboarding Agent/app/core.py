"""
core.py
CSV loader, validation, summarizer + onboarding plan and email generator.
"""
import pandas as pd
from typing import List
from app.ai_local import LocalAI

REQUIRED_COLUMNS = {'name', 'company', 'services_requested', 'notes'}

class ValidationError(Exception):
    pass

def load_and_validate_csv(path_or_buffer) -> pd.DataFrame:
    df = pd.read_csv(path_or_buffer)
    missing = REQUIRED_COLUMNS - set(df.columns.str.lower())
    if missing:
        raise ValidationError(f"Missing required columns: {', '.join(sorted(missing))}")
    # normalize column names to lowercase
    df.columns = [c.lower() for c in df.columns]
    return df

class OnboardingAssistant:
    def __init__(self, ai=None):
        self.ai = ai or LocalAI()

    def summarize_client(self, notes: str) -> str:
        if not notes or str(notes).strip() == '':
            return "No notes provided."
        return self.ai.summarize(str(notes))

    def generate_onboarding_plan(self, summary: str) -> List[str]:
        prompt = ("Create a concise 3-step onboarding plan for a new client based on the following summary.\n\n" 
                  f"Summary: {summary}\n\nSteps:")
        raw = self.ai.generate(prompt)
        lines = [l.strip('-•* \t') for l in raw.splitlines() if l.strip()]
        steps = []
        for l in lines:
            if len(steps) >= 3:
                break
            if l.lower().startswith('summary:'):
                continue
            steps.append(l)
        if not steps:
            steps = [
                '1. Kickoff meeting to align goals and timelines.',
                '2. Set up access and required assets.',
                '3. Deliver first milestone and gather feedback.'
            ]
        return steps

    def generate_welcome_email(self, name: str, company: str, summary: str) -> str:
        prompt = (f"Write a short, friendly welcome email to {name} at {company}.\n\n"
                  f"Include a one-sentence summary of their needs: {summary}\n\n"
                  "Email:")
        return self.ai.generate(prompt)
