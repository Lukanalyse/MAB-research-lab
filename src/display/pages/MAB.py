from __future__ import annotations

import base64
from pathlib import Path

import numpy as np
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
OUTPUTS_PDF_DIR = PROJECT_ROOT / "outputs" / "pdf"

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


def find_pdf_note(filename: str) -> Path | None:
    candidate = OUTPUTS_PDF_DIR / filename
    if candidate.exists():
        return candidate
    return None


def display_pdf_note(pdf_path: Path, height: int = 850) -> None:
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


def render_pdf_research_note(filename: str, caption: str) -> None:
    with st.expander("PDF research note", expanded=False):
        st.caption(caption)
        pdf_note = find_pdf_note(filename)

        if pdf_note is None:
            st.info(f"PDF note not found yet. Expected path: `outputs/pdf/{filename}`")
            return

        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown(f"**Loaded file:** `outputs/pdf/{filename}`")
        with col2:
            st.download_button(
                label=f"Download {filename}",
                data=pdf_note.read_bytes(),
                file_name=filename,
                mime="application/pdf",
                width="stretch",
            )

        display_pdf_note(pdf_note)


def true_means_table(means: tuple[float, ...], stds: tuple[float, ...]) -> pd.DataFrame:
    means_array = np.asarray(means, dtype=float)
    stds_array = np.asarray(stds, dtype=float)
    optimal_arm = int(means_array.argmax())
    n_arms = len(means_array)
    return pd.DataFrame(
        {
            "arm": list(range(n_arms)),
            "true_mean": means_array,
            "true_std": stds_array,
            "is_optimal": [arm == optimal_arm for arm in range(n_arms)],
        }
    )


def validate_environment_table(
    editor_df: pd.DataFrame,
    n_arms: int,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    if len(editor_df) != n_arms:
        st.error(f"The environment table must contain exactly {n_arms} rows.")
        st.stop()

    means = pd.to_numeric(editor_df["true_mean"], errors="coerce")
    stds = pd.to_numeric(editor_df["true_std"], errors="coerce")

    if means.isna().any():
        st.error("All true_mean values must be numeric.")
        st.stop()

    if stds.isna().any():
        st.error("All true_std values must be numeric.")
        st.stop()

    if (stds <= 0).any():
        st.error("All true_std values must be strictly positive.")
        st.stop()

    return tuple(float(value) for value in means), tuple(float(value) for value in stds)


def render_environment_initialization(
    n_arms: int,
    key_prefix: str,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    st.subheader("Environment initialization")
    st.markdown(
        """
        The true arm means define the hidden environment. In real bandit problems,
        these values are unknown to the algorithm. Here they are exposed only for
        pedagogical purposes, so that the user can understand how allocation reacts
        to different reward landscapes.

        Ordered means create an easy pedagogical case. Random means create less
        artificial environments. Manual means allow controlled experiments, for
        example small gaps between arms.
        """
    )

    mode = st.radio(
        "Environment initialization mode",
        ["Default ordered means", "Random means", "Manual means"],
        horizontal=True,
        key=f"{key_prefix}_environment_mode",
    )

    if mode == "Default ordered means":
        means, stds = default_gaussian_bandit_parameters(n_arms)
        return tuple(float(value) for value in means), tuple(float(value) for value in stds)

    if mode == "Random means":
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            min_mean = st.number_input(
                "Minimum true mean",
                value=0.20,
                step=0.01,
                key=f"{key_prefix}_min_mean",
            )
        with col2:
            max_mean = st.number_input(
                "Maximum true mean",
                value=0.90,
                step=0.01,
                key=f"{key_prefix}_max_mean",
            )
        with col3:
            reward_std = st.number_input(
                "Reward standard deviation",
                min_value=0.001,
                value=0.15,
                step=0.01,
                key=f"{key_prefix}_reward_std",
            )
        with col4:
            environment_seed = st.number_input(
                "Environment seed",
                value=123,
                step=1,
                key=f"{key_prefix}_environment_seed",
            )

        if float(min_mean) >= float(max_mean):
            st.error("Minimum true mean must be strictly smaller than maximum true mean.")
            st.stop()

        rng = np.random.default_rng(int(environment_seed))
        means = rng.uniform(float(min_mean), float(max_mean), size=n_arms)
        stds = np.full(n_arms, float(reward_std))
        return tuple(float(value) for value in means), tuple(float(value) for value in stds)

    means, stds = default_gaussian_bandit_parameters(n_arms)
    editor_df = pd.DataFrame(
        {
            "arm": list(range(n_arms)),
            "true_mean": means,
            "true_std": stds,
        }
    )
    edited_df = st.data_editor(
        editor_df,
        width="stretch",
        num_rows="fixed",
        disabled=["arm"],
        key=f"{key_prefix}_manual_environment_editor",
        column_config={
            "arm": st.column_config.NumberColumn("arm", disabled=True),
            "true_mean": st.column_config.NumberColumn("true_mean", step=0.01),
            "true_std": st.column_config.NumberColumn(
                "true_std",
                min_value=0.001,
                step=0.01,
            ),
        },
    )
    return validate_environment_table(edited_df, n_arms)


def add_warmup_end_marker(fig: go.Figure, df: pd.DataFrame, t_max: int | None = None) -> None:
    if "is_warmup" not in df.columns:
        return

    warmup_rows = df[df["is_warmup"]]
    if warmup_rows.empty:
        return

    warmup_end = int(warmup_rows["t"].max())
    visible_end = int(t_max if t_max is not None else df["t"].max())
    horizon = int(df["t"].max())

    if warmup_end <= 0 or warmup_end >= horizon or warmup_end > visible_end:
        return

    fig.add_vline(
        x=warmup_end,
        line_dash="dash",
        line_color="#FBBF24",
        annotation_text="End of initial uniform allocation",
        annotation_position="top left",
    )


def plot_cumulative_regret(df: pd.DataFrame, t_max: int | None = None) -> go.Figure:
    original_df = df
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
    add_warmup_end_marker(fig, original_df, t_max)
    return fig


def plot_cumulative_reward(df: pd.DataFrame, t_max: int | None = None) -> go.Figure:
    original_df = df
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
    add_warmup_end_marker(fig, original_df, t_max)
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
def cached_simulation(
    policy_name: str,
    means: tuple[float, ...],
    stds: tuple[float, ...],
    horizon: int,
    seed: int,
    initial_pulls_per_arm: int,
) -> pd.DataFrame:
    return run_simulation(
        policy_name=policy_name,
        means=np.asarray(means, dtype=float),
        stds=np.asarray(stds, dtype=float),
        horizon=horizon,
        seed=seed,
        initial_pulls_per_arm=initial_pulls_per_arm,
    )


@st.cache_data(show_spinner=False)
def cached_random_vs_ucb(
    means: tuple[float, ...],
    stds: tuple[float, ...],
    horizon: int,
    seed: int,
    initial_pulls_per_arm: int,
) -> pd.DataFrame:
    return run_comparison(
        policy_names=["random", "ucb"],
        means=np.asarray(means, dtype=float),
        stds=np.asarray(stds, dtype=float),
        horizon=horizon,
        seed=seed,
        initial_pulls_per_arm=initial_pulls_per_arm,
    )


def render_random_baseline() -> None:
    hero(
        "Random Allocation Baseline",
        "A non-adaptive benchmark that samples arms uniformly at random.",
    )
    render_pdf_research_note(
        "MAB_Random_Allocation.pdf",
        "Short research note explaining uniform allocation as a non-adaptive baseline for stochastic multi-armed bandits.",
    )

    note = read_markdown_note("random_baseline.md")
    if note:
        st.markdown(note)

    st.latex(r"A_t \sim \mathrm{Uniform}\{1,\dots,K\}")

    st.markdown(
        """
        The same initial uniform phase is available here for comparability with
        adaptive policies. After that controlled initialization, Random Allocation
        remains non-adaptive and ignores all past observations.
        """
    )

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        n_arms = st.slider("Number of arms K", 2, 10, 6, key="random_k")
    with col2:
        horizon = st.slider("Horizon T", 50, 5000, 1000, step=50, key="random_horizon")
    with col3:
        seed = st.number_input("Seed", value=42, step=1, key="random_seed")
    with col4:
        initial_pulls_per_arm = st.slider(
            "Initial uniform pulls per arm",
            min_value=0,
            max_value=20,
            value=1,
            step=1,
            help="Number of forced uniform pulls per arm before the adaptive policy starts.",
            key="random_initial_pulls",
        )

    means, stds = render_environment_initialization(n_arms, "random")

    df = cached_simulation(
        "random",
        means,
        stds,
        horizon,
        int(seed),
        int(initial_pulls_per_arm),
    )

    st.subheader("Hidden Gaussian environment")
    st.dataframe(true_means_table(means, stds), width="stretch")

    st.subheader("Final summary")
    st.dataframe(summary_table(df), width="stretch")

    tab1, tab2, tab3, tab4 = st.tabs(
        ["Regret", "Reward", "Pull counts", "Empirical means"]
    )
    with tab1:
        st.plotly_chart(plot_cumulative_regret(df), width="stretch")
    with tab2:
        st.plotly_chart(plot_cumulative_reward(df), width="stretch")
    with tab3:
        st.plotly_chart(plot_arm_counts(df), width="stretch")
    with tab4:
        st.plotly_chart(plot_empirical_means(df), width="stretch")

    st.info(
        "Random allocation is useful as a neutral benchmark, but it does not exploit past observations."
    )


def render_ucb() -> None:
    hero(
        "Upper Confidence Bound",
        "An adaptive policy based on optimism under uncertainty.",
    )
    render_pdf_research_note(
        "MAB_UCB.pdf",
        "Short research note explaining optimism under uncertainty, confidence bounds, and the UCB allocation rule.",
    )

    note = read_markdown_note("ucb.md")
    if note:
        st.markdown(note)

    st.latex(r"UCB_a(t) = \hat{\mu}_a(t) + \left(\frac{2\log(t)}{N_a(t)}\right)^{1/2}")

    st.markdown(
        """
        The initial uniform allocation phase is a controlled initialization step,
        not the final algorithmic objective. It ensures each arm has a minimum
        number of observations. After this phase, UCB uses the empirical mean plus
        an uncertainty bonus.
        """
    )

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        n_arms = st.slider("Number of arms K", 2, 10, 6, key="ucb_k")
    with col2:
        horizon = st.slider("Horizon T", 50, 5000, 1000, step=50, key="ucb_horizon")
    with col3:
        seed = st.number_input("Seed", value=42, step=1, key="ucb_seed")
    with col4:
        initial_pulls_per_arm = st.slider(
            "Initial uniform pulls per arm",
            min_value=0,
            max_value=20,
            value=1,
            step=1,
            help="Number of forced uniform pulls per arm before the adaptive policy starts.",
            key="ucb_initial_pulls",
        )

    means, stds = render_environment_initialization(n_arms, "ucb")

    df = cached_simulation(
        "ucb",
        means,
        stds,
        horizon,
        int(seed),
        int(initial_pulls_per_arm),
    )

    st.subheader("Hidden Gaussian environment")
    st.dataframe(true_means_table(means, stds), width="stretch")

    st.subheader("Final summary")
    st.dataframe(summary_table(df), width="stretch")

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
        st.plotly_chart(plot_cumulative_regret(df, t_selected), width="stretch")
    with tab2:
        st.plotly_chart(plot_cumulative_reward(df, t_selected), width="stretch")
    with tab3:
        st.plotly_chart(plot_arm_counts(df, t_selected), width="stretch")
    with tab4:
        st.plotly_chart(plot_empirical_means(df, t_selected), width="stretch")

    st.info(
        "UCB first explores uncertain arms. As observations accumulate, the confidence bonus decreases and the policy concentrates more on promising arms."
    )


def render_random_vs_ucb_comparison() -> None:
    hero(
        "Random Allocation vs UCB",
        "A precise comparison between a non-adaptive baseline and an adaptive UCB policy.",
    )
    render_pdf_research_note(
        "MAB_Random_vs_UCB.pdf",
        "Short research note explaining why adaptive allocation can outperform non-adaptive allocation in cumulative regret.",
    )

    st.markdown(
        """
        This comparison is intentionally narrow: it only compares **Random Allocation** and **UCB**.
        Later, the lab can include other comparison pages such as UCB vs Thompson Sampling,
        UCB vs epsilon-greedy, or pure exploration algorithms.

        Both policies face the same true arm means. A single simulation trajectory is
        illustrative; robust comparisons should average regret over many independent
        Monte Carlo replications. This will be added later.
        """
    )

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        n_arms = st.slider("Number of arms K", 2, 10, 6, key="cmp_k")
    with col2:
        horizon = st.slider("Horizon T", 50, 5000, 1000, step=50, key="cmp_horizon")
    with col3:
        seed = st.number_input("Seed", value=42, step=1, key="cmp_seed")
    with col4:
        initial_pulls_per_arm = st.slider(
            "Initial uniform pulls per arm",
            min_value=0,
            max_value=20,
            value=1,
            step=1,
            help="Number of forced uniform pulls per arm before the adaptive policy starts.",
            key="cmp_initial_pulls",
        )

    means, stds = render_environment_initialization(n_arms, "cmp")

    df = cached_random_vs_ucb(
        means,
        stds,
        horizon,
        int(seed),
        int(initial_pulls_per_arm),
    )

    st.subheader("Hidden Gaussian environment")
    st.dataframe(true_means_table(means, stds), width="stretch")

    st.subheader("Final summary")
    st.dataframe(summary_table(df), width="stretch")

    tab1, tab2 = st.tabs(["Regret comparison", "Reward comparison"])
    with tab1:
        st.plotly_chart(plot_cumulative_regret(df), width="stretch")
    with tab2:
        st.plotly_chart(plot_cumulative_reward(df), width="stretch")

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
