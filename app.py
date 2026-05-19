from __future__ import annotations

import streamlit as st

from src.display.pages.home import render_home
from src.display.pages.basics import render_basics
from src.display.pages.MAB import render_mab


st.set_page_config(
    page_title="MAB Research Lab",
    page_icon="📊",
    layout="centered",
    initial_sidebar_state="expanded",
)


def apply_theme() -> None:
    """Apply one fixed clean dark scientific Streamlit theme."""
    st.markdown(
        """
        <style>
        html {
            font-size: 150%;
        }

        .stApp {
            background: radial-gradient(circle at top left, #1E3A8A 0, #0F172A 32%, #020617 100%);
            color: #E5E7EB;
        }

        section[data-testid="stSidebar"] {
            background: rgba(2, 6, 23, 0.98);
            border-right: 1px solid rgba(255, 255, 255, 0.08);
        }

        h1 {
            color: #FFFFFF;
            font-weight: 850;
            letter-spacing: -0.035em;
        }

        h2, h3 {
            color: #F8FAFC;
            font-weight: 750;
        }

        p, li, div {
            color: #E5E7EB;
        }

        .hero-card {
            background: linear-gradient(135deg, rgba(59, 130, 246, 0.22), rgba(20, 184, 166, 0.12));
            border: 1px solid rgba(147, 197, 253, 0.35);
            border-radius: 24px;
            padding: 1.45rem 1.6rem;
            margin-bottom: 1.1rem;
            box-shadow: 0 18px 45px rgba(0, 0, 0, 0.32);
        }

        .hero-card h1 {
            margin-bottom: 0.35rem;
        }

        .hero-card p {
            color: #D1D5DB;
            font-size: 1.05rem;
            margin-bottom: 0;
        }

        .scientific-card {
            background: rgba(15, 23, 42, 0.84);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 20px;
            padding: 1.05rem 1.25rem;
            margin: 0.75rem 0;
            box-shadow: 0 14px 34px rgba(0, 0, 0, 0.26);
        }

        div[data-testid="stMetric"] {
            background: rgba(15, 23, 42, 0.82);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 16px;
            padding: 0.85rem;
        }

        div[data-testid="stMetricLabel"] {
            color: #93C5FD;
        }

        div[data-testid="stMetricValue"] {
            color: #FFFFFF;
        }

        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
        }

        .stTabs [data-baseweb="tab"] {
            background: rgba(15, 23, 42, 0.72);
            border-radius: 12px;
            padding: 8px 14px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


apply_theme()

with st.sidebar:
    st.markdown("## MAB Research Lab")
    st.caption("Theory · Simulation · Regret · UCB")
    st.divider()

    page = st.radio(
        "Navigation",
        [
            "Home",
            "Basics",
            "MAB models",
        ],
        label_visibility="collapsed",
    )

    st.divider()
    st.caption("First version: stochastic MAB, random allocation, and UCB.")


if page == "Home":
    render_home()
elif page == "Basics":
    render_basics()
elif page == "MAB models":
    render_mab()
