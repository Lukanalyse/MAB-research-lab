from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.algorithms.metrics import summary_table
from src.algorithms.simulation import (
    default_gaussian_bandit_parameters,
    run_comparison,
    run_simulation,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
RESEARCH_DIR = PROJECT_ROOT / "src" / "recherche"

PLOTLY_LAYOUT = {
    "template": "plotly_dark",
    "paper_bgcolor": "rgba(0,0,0,0)",
    "plot_bgcolor": "rgba(15,23,42,0.92)",
    "font": {"color": "#E5E7EB"},
    "margin": {"l": 45, "r": 25, "t": 55, "b": 40},
}


def hero(title: str, subtitle: str) -> None:
    st.markdown(
        f"""
        <div class="hero-card">
            <h1>{title}</h1>
            <p>{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def read_markdown_note(filename: str) -> str:
    path = RESEARCH_DIR / filename
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def plot_cumulative_regret(df: pd.DataFrame, t_max: int | None = None) -> go.Figure:
    if t_max is not None:
        df = df[df["t"] <= t_max]

    fig = go.Figure()
    for policy, group in df.groupby("policy", sort=False):
        fig.add_trace(
            go.Scatter(
                x=group["t"],
                y=group["cumulative_regret"],
                mode="lines",
                name=str(policy),
                line={"width": 3},
            )
        )

    fig.update_layout(
        title="Cumulative regret",
        xaxis_title="Time",
        yaxis_title="Regret",
        hovermode="x unified",
        **PLOTLY_LAYOUT,
    )
    return fig


def plot_cumulative_reward(df: pd.DataFrame, t_max: int | None = None) -> go.Figure:
    if t_max is not None:
        df = df[df["t"] <= t_max]

    fig = go.Figure()
    for policy, group in df.groupby("policy", sort=False):
        fig.add_trace(
            go.Scatter(
                x=group["t"],
                y=group["cumulative_reward"],
                mode="lines",
                name=str(policy),
                line={"width": 3},
            )
        )

    fig.update_layout(
        title="Cumulative reward",
        xaxis_title="Time",
        yaxis_title="Reward",
        hovermode="x unified",
        **PLOTLY_LAYOUT,
    )
    return fig


def plot_arm_counts(df: pd.DataFrame, t_max: int | None = None) -> go.Figure:
    if t_max is not None:
        df = df[df["t"] <= t_max]

    last = df.iloc[-1]
    count_cols = [col for col in df.columns if col.startswith("count_arm_")]
    counts = [int(last[col]) for col in count_cols]
    arms = [col.replace("count_arm_", "Arm ") for col in count_cols]

    fig = go.Figure(
        data=[go.Bar(x=arms, y=counts, text=counts, textposition="outside")]
    )
    fig.update_layout(
        title="Arm pull counts",
        xaxis_title="Arm",
        yaxis_title="Number of pulls",
        **PLOTLY_LAYOUT,
    )
    return fig


def plot_empirical_means(df: pd.DataFrame, t_max: int | None = None) -> go.Figure:
    if t_max is not None:
        df = df[df["t"] <= t_max]

    fig = go.Figure()
    mean_cols = [col for col in df.columns if col.startswith("empirical_mean_arm_")]

    for col in mean_cols:
        arm = col.replace("empirical_mean_arm_", "")
        fig.add_trace(
            go.Scatter(
                x=df["t"],
                y=df[col],
                mode="lines",
                name=f"Arm {arm}",
                line={"width": 2.5},
            )
        )

    fig.update_layout(
        title="Empirical means over time",
        xaxis_title="Time",
        yaxis_title="Empirical mean",
        hovermode="x unified",
        **PLOTLY_LAYOUT,
    )
    return fig


@st.cache_data(show_spinner=False)
def cached_simulation(policy_name: str, n_arms: int, horizon: int, seed: int) -> pd.DataFrame:
    means, stds = default_gaussian_bandit_parameters(n_arms)
    return run_simulation(
        policy_name=policy_name,
        means=means,
        stds=stds,
        horizon=horizon,
        seed=seed,
    )


@st.cache_data(show_spinner=False)
def cached_random_vs_ucb(n_arms: int, horizon: int, seed: int) -> pd.DataFrame:
    means, stds = default_gaussian_bandit_parameters(n_arms)
    return run_comparison(
        policy_names=["random", "ucb"],
        means=means,
        stds=stds,
        horizon=horizon,
        seed=seed,
    )


def render_random_baseline() -> None:
    hero(
        "Random Allocation Baseline",
        "A non-adaptive benchmark that samples arms uniformly at random.",
    )

    note = read_markdown_note("random_baseline.md")
    if note:
        st.markdown(note)

    st.latex(r"A_t \sim \mathrm{Uniform}\{1,\dots,K\}")

    col1, col2, col3 = st.columns(3)
    with col1:
        n_arms = st.slider("Number of arms K", 2, 10, 6, key="random_k")
    with col2:
        horizon = st.slider("Horizon T", 50, 5000, 1000, step=50, key="random_horizon")
    with col3:
        seed = st.number_input("Seed", value=42, step=1, key="random_seed")

    df = cached_simulation("random", n_arms, horizon, int(seed))

    st.subheader("Final summary")
    st.dataframe(summary_table(df), use_container_width=True)

    tab1, tab2, tab3, tab4 = st.tabs(
        ["Regret", "Reward", "Pull counts", "Empirical means"]
    )
    with tab1:
        st.plotly_chart(plot_cumulative_regret(df), use_container_width=True)
    with tab2:
        st.plotly_chart(plot_cumulative_reward(df), use_container_width=True)
    with tab3:
        st.plotly_chart(plot_arm_counts(df), use_container_width=True)
    with tab4:
        st.plotly_chart(plot_empirical_means(df), use_container_width=True)

    st.info(
        "Random allocation is useful as a neutral benchmark, but it does not exploit past observations."
    )


def render_ucb() -> None:
    hero(
        "Upper Confidence Bound",
        "An adaptive policy based on optimism under uncertainty.",
    )

    note = read_markdown_note("ucb.md")
    if note:
        st.markdown(note)

    st.latex(r"UCB_a(t) = \hat{\mu}_a(t) + \sqrt{\frac{2\log(t)}{N_a(t)}}")

    col1, col2, col3 = st.columns(3)
    with col1:
        n_arms = st.slider("Number of arms K", 2, 10, 6, key="ucb_k")
    with col2:
        horizon = st.slider("Horizon T", 50, 5000, 1000, step=50, key="ucb_horizon")
    with col3:
        seed = st.number_input("Seed", value=42, step=1, key="ucb_seed")

    df = cached_simulation("ucb", n_arms, horizon, int(seed))

    st.subheader("Final summary")
    st.dataframe(summary_table(df), use_container_width=True)

    st.subheader("Interactive replay")
    t_selected = st.slider("Time step", 1, horizon, min(100, horizon), key="ucb_time")

    current = df[df["t"] == t_selected].iloc[0]
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("t", int(current["t"]))
    m2.metric("Selected arm", int(current["selected_arm"]))
    m3.metric("Reward", round(float(current["reward"]), 4))
    m4.metric("Cumulative regret", round(float(current["cumulative_regret"]), 4))

    tab1, tab2, tab3, tab4 = st.tabs(
        ["Regret", "Reward", "Pull counts", "Empirical means"]
    )
    with tab1:
        st.plotly_chart(plot_cumulative_regret(df, t_selected), use_container_width=True)
    with tab2:
        st.plotly_chart(plot_cumulative_reward(df, t_selected), use_container_width=True)
    with tab3:
        st.plotly_chart(plot_arm_counts(df, t_selected), use_container_width=True)
    with tab4:
        st.plotly_chart(plot_empirical_means(df, t_selected), use_container_width=True)

    st.info(
        "UCB first explores uncertain arms. As observations accumulate, the confidence bonus decreases and the policy concentrates more on promising arms."
    )


def render_random_vs_ucb_comparison() -> None:
    hero(
        "Random Allocation vs UCB",
        "A precise comparison between a non-adaptive baseline and an adaptive UCB policy.",
    )

    st.markdown(
        """
        This comparison is intentionally narrow: it only compares **Random Allocation** and **UCB**.
        Later, the lab can include other comparison pages such as UCB vs Thompson Sampling,
        UCB vs epsilon-greedy, or pure exploration algorithms.
        """
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        n_arms = st.slider("Number of arms K", 2, 10, 6, key="cmp_k")
    with col2:
        horizon = st.slider("Horizon T", 50, 5000, 1000, step=50, key="cmp_horizon")
    with col3:
        seed = st.number_input("Seed", value=42, step=1, key="cmp_seed")

    df = cached_random_vs_ucb(n_arms, horizon, int(seed))

    st.subheader("Final summary")
    st.dataframe(summary_table(df), use_container_width=True)

    tab1, tab2 = st.tabs(["Regret comparison", "Reward comparison"])
    with tab1:
        st.plotly_chart(plot_cumulative_regret(df), use_container_width=True)
    with tab2:
        st.plotly_chart(plot_cumulative_reward(df), use_container_width=True)

    st.info(
        "This comparison shows why adaptive allocation matters: UCB uses uncertainty to guide exploration, while random allocation ignores all past information."
    )


def render_mab() -> None:
    section = st.sidebar.selectbox(
        "MAB section",
        [
            "Random allocation baseline",
            "UCB",
            "Random allocation vs UCB comparison",
        ],
    )

    if section == "Random allocation baseline":
        render_random_baseline()
    elif section == "UCB":
        render_ucb()
    elif section == "Random allocation vs UCB comparison":
        render_random_vs_ucb_comparison()
