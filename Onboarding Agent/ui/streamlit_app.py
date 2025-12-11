"""
streamlit_app.py
Simple Streamlit UI to upload CSV and show summaries, onboarding plans, and email drafts.
Run with: streamlit run ui/streamlit_app.py
"""

import streamlit as st
import pandas as pd
import sys, os
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from app.core import load_and_validate_csv, OnboardingAssistant
st.set_page_config(page_title="Local Onboarding Assistant", layout="wide")

st.title("Local AI — Customer Onboarding Assistant")

uploaded = st.file_uploader("Upload a client CSV", type=['csv'])
use_preview = st.checkbox("Show raw CSV preview", value=True)

if uploaded:
    try:
        df = load_and_validate_csv(uploaded)
    except Exception as e:
        st.error(f"CSV validation error: {e}")
        st.stop()

    if use_preview:
        st.subheader("CSV Preview")
        st.dataframe(df.head())

    assistant = OnboardingAssistant()
    st.subheader("Generated outputs")

    for idx, row in df.iterrows():
        with st.expander(f"{row.get('name', 'Unknown')} — {row.get('company', '')}"):
            notes = row.get('notes', '')
            summary = assistant.summarize_client(notes)
            plan = assistant.generate_onboarding_plan(summary)
            email = assistant.generate_welcome_email(row.get('name', ''), row.get('company', ''), summary)

            st.markdown("**Summary**")
            st.write(summary)
            st.markdown("**3-step onboarding plan**")
            for s in plan:
                st.write(f"- {s}")
            st.markdown("**Welcome email draft**")
            st.code(email)
