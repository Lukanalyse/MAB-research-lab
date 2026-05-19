from __future__ import annotations

import streamlit as st

from src.display.pages.home import render_home
from src.display.pages.basics import render_basics
from src.display.pages.MAB import render_mab


st.set_page_config(
    page_title="MAB Research Lab",
    layout="wide",
    initial_sidebar_state="expanded",
)


def apply_theme() -> None:
    """Apply a clean dark scientific Streamlit theme."""

    st.markdown(
        """
        <style>

        /* ===== GLOBAL LAYOUT ===== */

        html {
            font-size: 100%;
        }

        .main .block-container {
            max-width: 1450px;
            margin-left: auto;
            margin-right: auto;
            padding-top: 1.5rem;
            padding-bottom: 1rem;
        }

        .stApp {
            background:
                radial-gradient(
                    circle at top left,
                    #1E3A8A 0%,
                    #0F172A 32%,
                    #020617 100%
                );
            color: #E5E7EB;
        }

        /* ===== SIDEBAR ===== */

        section[data-testid="stSidebar"] {
            background: rgba(2, 6, 23, 0.98);
            border-right: 1px solid rgba(255, 255, 255, 0.08);
        }

        /* ===== TYPOGRAPHY ===== */

        h1 {
            color: #FFFFFF;
            font-weight: 800;
            letter-spacing: -0.03em;
            font-size: 3rem !important;
        }

        h2 {
            color: #F8FAFC;
            font-weight: 700;
            font-size: 2rem !important;
        }

        h3 {
            color: #F8FAFC;
            font-weight: 650;
            font-size: 1.35rem !important;
        }

        p, li, div {
            color: #E5E7EB;
            font-size: 1rem;
        }

        /* ===== HERO CARD ===== */

        .hero-card {
            background:
                linear-gradient(
                    135deg,
                    rgba(59, 130, 246, 0.20),
                    rgba(20, 184, 166, 0.10)
                );

            border: 1px solid rgba(147, 197, 253, 0.25);
            border-radius: 22px;

            padding: 1.25rem 1.45rem;
            margin-bottom: 1rem;

            box-shadow:
                0 12px 32px rgba(0, 0, 0, 0.28);
        }

        .hero-card h1 {
            margin-bottom: 0.3rem;
        }

        .hero-card p {
            color: #D1D5DB;
            font-size: 1rem;
            margin-bottom: 0;
        }

        /* ===== SCIENTIFIC CARDS ===== */

        .scientific-card {
            background: rgba(15, 23, 42, 0.82);

            border: 1px solid rgba(255, 255, 255, 0.07);

            border-radius: 18px;

            padding: 1rem 1.15rem;
            margin: 0.65rem 0;

            box-shadow:
                0 10px 28px rgba(0, 0, 0, 0.24);
        }

        /* ===== METRICS ===== */

        div[data-testid="stMetric"] {
            background: rgba(15, 23, 42, 0.80);

            border: 1px solid rgba(255, 255, 255, 0.07);

            border-radius: 14px;

            padding: 0.75rem;

            box-shadow:
                0 6px 20px rgba(0, 0, 0, 0.18);
        }

        div[data-testid="stMetricLabel"] {
            color: #93C5FD;
            font-size: 0.95rem;
        }

        div[data-testid="stMetricValue"] {
            color: #FFFFFF;
        }

        /* ===== TABS ===== */

        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
        }

        .stTabs [data-baseweb="tab"] {
            background: rgba(15, 23, 42, 0.72);

            border-radius: 12px;

            padding: 0.55rem 1rem;

            transition: all 0.2s ease;
        }

        .stTabs [aria-selected="true"] {
            background: rgba(59, 130, 246, 0.25);
            border: 1px solid rgba(147, 197, 253, 0.25);
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
