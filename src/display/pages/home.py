

from __future__ import annotations

import base64
from pathlib import Path

import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[3]
PAPERS_DIR = PROJECT_ROOT / "papers"

PAPER_ORDER = [
    {
        "display_title": "Robbins and Monro (1951) — A Stochastic Approximation Method",
        "match_terms": ["stochastic approximation method", "robbins monro", "monro"],
    },
    {
        "display_title": "Robbins (1952) — Some aspects of the sequential design of experiments",
        "match_terms": ["some aspects", "sequential design", "robbins"],
    },
    {
        "display_title": "Lai and Robbins (1985) — Asymptotically efficient adaptive allocation rules",
        "match_terms": ["asymptotically efficient", "adaptive allocation", "lai"],
    },
    {
        "display_title": "Auer, Cesa-Bianchi and Fischer (2002) — Finite-time Analysis of the Multiarmed Bandit Problem",
        "match_terms": ["finite-time", "finite time", "auer", "multiarmed bandit problem"],
    },
    {
        "display_title": "Audibert, Bubeck and Munos (2010) — Best Arm Identification in Multi-Armed Bandits",
        "match_terms": ["best arm identification", "audibert", "munos"],
    },
    {
        "display_title": "Agrawal and Goyal (2012) — Analysis of Thompson Sampling for the Multi-armed Bandit Problem",
        "match_terms": ["analysis of thompson", "thompson sampling", "agrawal", "goyal"],
    },
    {
        "display_title": "Bubeck and Cesa-Bianchi (2012) — Regret Analysis of Stochastic and Nonstochastic Multi-armed Bandit Problems",
        "match_terms": ["regret analysis", "stochastic and nonstochastic", "bubeck", "cesa"],
    },
]


def hero(title: str, subtitle: str) -> None:
    """Render a large header card."""
    st.markdown(
        f"""
        <div class="hero-card">
            <h1>{title}</h1>
            <p>{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def card(html: str) -> None:
    """Render a small scientific card."""
    st.markdown(f'<div class="scientific-card">{html}</div>', unsafe_allow_html=True)


def paper_order_index(pdf_path: Path) -> int | None:
    """Match a local PDF to the chronological bibliography order when possible."""
    normalized_name = pdf_path.stem.lower().replace("_", " ").replace("-", " ")

    for index, paper in enumerate(PAPER_ORDER):
        if any(term in normalized_name for term in paper["match_terms"]):
            return index

    return None


def paper_display_title(pdf_path: Path) -> str:
    """Return the canonical paper title if the PDF can be matched."""
    matched_index = paper_order_index(pdf_path)
    if matched_index is None:
        return pdf_path.stem.replace("_", " ").replace("-", " ").strip()
    return PAPER_ORDER[matched_index]["display_title"]


def list_project_pdfs() -> list[Path]:
    """List local PDF files in the papers folder, ordered like the bibliography."""
    if not PAPERS_DIR.exists():
        return []

    pdfs = list(PAPERS_DIR.glob("*.pdf"))

    def sort_key(pdf_path: Path) -> tuple[int, str]:
        matched_index = paper_order_index(pdf_path)
        if matched_index is None:
            return (10_000, pdf_path.name.lower())
        return (matched_index, pdf_path.name.lower())

    return sorted(pdfs, key=sort_key)


def display_pdf_viewer(pdf_path: Path, height: int = 850) -> None:
    """Display a local PDF directly in Streamlit."""
    if not pdf_path.exists():
        st.warning(f"PDF not found: {pdf_path}")
        return

    encoded_pdf = base64.b64encode(pdf_path.read_bytes()).decode("utf-8")
    st.markdown(
        f"""
        <iframe
            src="data:application/pdf;base64,{encoded_pdf}"
            width="100%"
            height="{height}px"
            type="application/pdf"
            style="border: 1px solid rgba(255,255,255,0.12); border-radius: 16px; background: #111827;"
        ></iframe>
        """,
        unsafe_allow_html=True,
    )


def render_pdf_library() -> None:
    """Render the local PDF library on the home page."""
    pdfs = list_project_pdfs()

    st.markdown("### Local paper library")
    st.markdown(
        """
        The PDFs below are stored locally in the `papers/` folder. They are ordered
        to match the **Scientific anchors** bibliography above.
        """
    )

    if not pdfs:
        st.warning("No PDF found in the local `papers/` folder yet.")
        return

    pdf_titles = [paper_display_title(pdf_path) for pdf_path in pdfs]
    selected_title = st.selectbox("Select a paper to preview", pdf_titles)
    selected_pdf = pdfs[pdf_titles.index(selected_title)]

    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown(f"**Selected paper:** {paper_display_title(selected_pdf)}")
        st.caption(f"Local file: `{selected_pdf.name}`")

    with col2:
        st.download_button(
            label="Download selected PDF",
            data=selected_pdf.read_bytes(),
            file_name=selected_pdf.name,
            mime="application/pdf",
            width="stretch",
        )

    with st.expander("Show all local PDFs", expanded=False):
        for pdf_path in pdfs:
            st.markdown(f"- **{paper_display_title(pdf_path)}** — `{pdf_path.name}`")

    display_pdf_viewer(selected_pdf)


def render_home() -> None:
    """Render the home page of the MAB Research Lab."""
    hero(
        "MAB Research Lab",
        "A simple scientific workspace for Multi-Armed Bandits: theory, simulations, figures, and research notes.",
    )

    card(
        """
        <h3>Project objective</h3>
        <p>
        This application is built to understand Multi-Armed Bandits step by step.
        This is still in production. It is not finish at all.
        </p>
        """
    )

    st.markdown("### Workflow")
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("1", "Theory")
    col2.metric("2", "Policy")
    col3.metric("3", "Simulation")
    col4.metric("4", "Figures")
    col5.metric("5", "Research note")

    st.markdown("### Scientific anchors")
    st.markdown(
        """
        This lab is built on a small set of foundational papers in sequential decision-making,
        stochastic approximation, regret analysis, UCB, Thompson Sampling, and best-arm identification.
        The references below are ordered chronologically.

        1. **Robbins and Monro (1951)** — *A Stochastic Approximation Method*.  
           Used for online stochastic approximation and incremental mean updates.

        2. **Robbins (1952)** — *Some aspects of the sequential design of experiments*.  
           Used as one of the historical foundations of sequential experimental design and bandit allocation.

        3. **Lai and Robbins (1985)** — *Asymptotically efficient adaptive allocation rules*.  
           Used for regret lower bounds and the theoretical motivation for efficient adaptive sampling.

        4. **Auer, Cesa-Bianchi and Fischer (2002)** — *Finite-time Analysis of the Multiarmed Bandit Problem*.  
           Used as the main reference for UCB and finite-time regret analysis.

        5. **Audibert, Bubeck and Munos (2010)** — *Best Arm Identification in Multi-Armed Bandits*.  
           Used for the pure exploration and best-arm identification perspective.

        6. **Agrawal and Goyal (2012)** — *Analysis of Thompson Sampling for the Multi-armed Bandit Problem*.  
           Used for the Bayesian posterior-sampling view of bandit algorithms.

        7. **Bubeck and Cesa-Bianchi (2012)** — *Regret Analysis of Stochastic and Nonstochastic Multi-armed Bandit Problems*.  
           Used as the main modern survey for regret analysis and the general MAB framework.
        """
    )

    st.divider()
    render_pdf_library()