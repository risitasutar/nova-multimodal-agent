"""Evaluation page inside the Nova Streamlit app (reads evaluation/results/latest.json)."""

import streamlit as st

from evaluation.dashboard import render

st.set_page_config(page_title="Nova · Evaluation", page_icon="📊", layout="wide")
render()
