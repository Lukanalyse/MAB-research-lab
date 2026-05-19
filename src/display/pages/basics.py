from __future__ import annotations

import base64
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[3]
RESEARCH_DIR = PROJECT_ROOT / "src" / "recherche"

PLOTLY_LAYOUT = {
    "template": "plotly_dark",
    "paper_bgcolor": "rgba(0,0,0,0)",
    "plot_bgcolor": "rgba(15,23,42,0.92)",
    "font": {"color": "#E5E7EB"},
    "margin": {"l": 45, "r": 25, "t": 55, "b": 40},
}

ARM_COLORS = [
    "#60A5FA",  # blue
    "#34D399",  # green
    "#FBBF24",  # amber
    "#F87171",  # red
    "#A78BFA",  # purple
    "#22D3EE",  # cyan
    "#FB7185",  # rose
    "#C084FC",  # violet
    "#4ADE80",  # light green
    "#F97316",  # orange
    "#38BDF8",  # sky
    "#E879F9",  # fuchsia
]


def arm_color(arm: int) -> str:
    """Return the stable color associated with one arm."""
    return ARM_COLORS[arm % len(ARM_COLORS)]

def find_pdf_note(filename: str) -> Path | None:
    """Find a PDF note in the local outputs/pdf folder."""
    candidate = PROJECT_ROOT / "outputs" / "pdf" / filename

    if candidate.exists():
        return candidate

    return None


def display_pdf_note(pdf_path: Path, height: int = 850) -> None:
    """Display a local PDF directly inside Streamlit."""
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

def hero(title: str, subtitle: str) -> None:
    """Render a large page header."""
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
    """Render a text block inside the shared scientific-card CSS class."""
    st.markdown(f'<div class="scientific-card">{html}</div>', unsafe_allow_html=True)


def read_markdown_note(filename: str) -> str:
    """Read a Markdown research note from src/recherche if it exists."""
    path = RESEARCH_DIR / filename
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def plot_true_means(means: np.ndarray) -> go.Figure:
    """Plot hidden true expectations, using one stable color per arm."""
    arms = [f"Arm {i}" for i in range(len(means))]

    fig = go.Figure(
        data=[
            go.Bar(
                x=arms,
                y=means,
                text=[round(float(value), 3) for value in means],
                textposition="outside",
                name="True expected reward",
                marker_color=[arm_color(i) for i in range(len(means))],
            )
        ]
    )
    fig.update_layout(
        title="True expected reward by arm",
        xaxis_title="Arm",
        yaxis_title="True expected reward",
        yaxis_range=[0, 1],
        **PLOTLY_LAYOUT,
    )
    return fig


def simulate_random_learning(
    means: np.ndarray,
    std: float,
    horizon: int,
    seed: int,
    initial_pulls_per_arm: int = 0,
) -> dict[str, np.ndarray]:
    """Simulate random sampling to show how empirical means are learned.

    This is not an optimal MAB policy. It is a pedagogical simulation used to
    understand the difference between hidden true expectations and observed
    empirical estimates.
    """
    rng = np.random.default_rng(seed)
    n_arms = len(means)
    warmup_steps = n_arms * initial_pulls_per_arm

    counts = np.zeros(n_arms, dtype=int)
    empirical_means = np.zeros(n_arms, dtype=float)

    counts_history = []
    empirical_history = []
    selected_arms = []
    rewards = []

    for t in range(1, horizon + 1):
        if t <= warmup_steps:
            arm = (t - 1) % n_arms
        else:
            arm = int(rng.integers(0, n_arms))
        reward = float(rng.normal(means[arm], std))

        counts[arm] += 1
        empirical_means[arm] += (reward - empirical_means[arm]) / counts[arm]

        selected_arms.append(arm)
        rewards.append(reward)
        counts_history.append(counts.copy())
        empirical_history.append(empirical_means.copy())

    return {
        "counts_history": np.asarray(counts_history),
        "empirical_history": np.asarray(empirical_history),
        "selected_arms": np.asarray(selected_arms),
        "rewards": np.asarray(rewards),
    }


def plot_learning_curves(
    true_means: np.ndarray,
    empirical_history: np.ndarray,
    t: int,
) -> go.Figure:
    """Plot empirical mean estimates over time with true means as dotted lines.

    Each arm keeps the same color across empirical and true-mean traces. The
    difference between empirical and true values is shown by line style: empirical
    lines are solid and true means are dotted.
    """
    time_index = np.arange(1, t + 1)
    n_arms = len(true_means)

    fig = go.Figure()

    for arm in range(n_arms):
        color = arm_color(arm)

        fig.add_trace(
            go.Scatter(
                x=time_index,
                y=empirical_history[:t, arm],
                mode="lines",
                name=f"Empirical mean — Arm {arm}",
                line={"width": 3.0, "color": color},
            )
        )

        fig.add_trace(
            go.Scatter(
                x=time_index,
                y=np.full(t, true_means[arm]),
                mode="lines",
                name=f"True mean — Arm {arm}",
                line={"dash": "dot", "width": 1.8, "color": color},
                opacity=0.65,
            )
        )

    fig.update_layout(
        title=f"Empirical mean estimates over time — up to t = {t}",
        xaxis_title="Time step",
        yaxis_title="Reward mean",
        yaxis_range=[0, 1],
        hovermode="x unified",
        **PLOTLY_LAYOUT,
    )
    return fig


def plot_allocation_curves(
    counts_history: np.ndarray,
    t: int,
) -> go.Figure:
    """Plot cumulative number of pulls per arm over time."""
    time_index = np.arange(1, t + 1)
    n_arms = counts_history.shape[1]

    fig = go.Figure()

    for arm in range(n_arms):
        fig.add_trace(
            go.Scatter(
                x=time_index,
                y=counts_history[:t, arm],
                mode="lines",
                name=f"Arm {arm}",
                line={"width": 3.0, "color": arm_color(arm)},
            )
        )

    fig.update_layout(
        title=f"Cumulative allocation by arm — up to t = {t}",
        xaxis_title="Time step",
        yaxis_title="Number of pulls",
        hovermode="x unified",
        **PLOTLY_LAYOUT,
    )
    return fig


def plot_state_snapshot(
    true_means: np.ndarray,
    empirical_means: np.ndarray,
    counts: np.ndarray,
    t: int,
) -> go.Figure:
    """Show the current learning state at selected time t.

    True means and empirical means use the same color by arm. True means are
    transparent bars, while empirical means are patterned bars.
    """
    arms = [f"Arm {i}" for i in range(len(true_means))]
    colors = [arm_color(i) for i in range(len(true_means))]

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=arms,
            y=true_means,
            name="True expected reward",
            opacity=0.42,
            text=[round(float(x), 3) for x in true_means],
            textposition="outside",
            marker_color=colors,
        )
    )

    fig.add_trace(
        go.Bar(
            x=arms,
            y=empirical_means,
            name="Empirical mean at selected t",
            opacity=0.95,
            text=[round(float(x), 3) for x in empirical_means],
            textposition="outside",
            marker_color=colors,
            marker_pattern_shape="/",
            marker_line_color="#E5E7EB",
            marker_line_width=0.6,
        )
    )

    fig.add_trace(
        go.Scatter(
            x=arms,
            y=counts,
            name="Number of pulls",
            mode="lines+markers+text",
            yaxis="y2",
            text=[int(x) for x in counts],
            textposition="top center",
            line={"color": "#E5E7EB", "width": 2.4},
            marker={"color": "#E5E7EB", "size": 8},
        )
    )

    fig.update_layout(
        title=f"State of learning at t = {t}",
        xaxis_title="Arm",
        yaxis={"title": "Reward mean", "range": [0, 1]},
        yaxis2={
            "title": "Pull count",
            "overlaying": "y",
            "side": "right",
            "showgrid": False,
        },
        barmode="group",
        hovermode="x unified",
        **PLOTLY_LAYOUT,
    )
    return fig


def render_basics() -> None:
    hero(
        "MAB Basics",
        "A first theoretical and visual introduction to stochastic multi-armed bandits.",
    )

    note = read_markdown_note("mab_basics.md")
    if note:
        st.markdown(note)
    else:
        card(
            """
            <h3>Definition</h3>
            <p>
            A multi-armed bandit is a sequential decision problem. At each time step,
            an agent chooses one arm among K possible arms and observes a random reward.
            The objective is to learn which arm performs best while collecting rewards.
            </p>
            """
        )

    st.markdown("### PDF research note — MAB_Basics")

    st.markdown(
        """
        This section contains the theoretical note associated with the MAB Basics module.
        The PDF is written separately in LaTeX/Overleaf to keep the Streamlit interface
        focused on interactive simulations.
        """
    )

    pdf_note = find_pdf_note("MAB_Basics.pdf")

    if pdf_note is not None:
        col_pdf_1, col_pdf_2 = st.columns([2, 1])

        with col_pdf_1:
            st.markdown("**Loaded file:** `outputs/pdf/MAB_Basics.pdf`")

        with col_pdf_2:
            st.download_button(
                label="Download MAB_Basics.pdf",
                data=pdf_note.read_bytes(),
                file_name="MAB_Basics.pdf",
                mime="application/pdf",
                width="stretch",
            )

        display_pdf_note(pdf_note)

    else:
        st.warning(
            "PDF note not found. Please place your Overleaf output here: "
            "`outputs/pdf/MAB_Basics.pdf`."
        )

    st.divider()
    st.subheader("Hidden true expectations")

    st.markdown(
        """
        The graph below shows the true expected reward of each arm. In a real bandit
        problem, these values are hidden. The learner only sees noisy rewards.
        """
    )

    st.markdown("### Arm mean configuration")

    st.markdown(
        """
        In this basic simulation, the arms are still selected randomly. However, you can
        choose the hidden true mean of each arm to test different environments.
        """
    )

    col1, col2 = st.columns(2)

    with col1:
        n_arms = st.slider("Number of arms", 2, 12, 6)

    with col2:
        mean_mode = st.radio(
            "Mean configuration",
            ["Random means", "Custom means"],
            horizontal=True,
        )

    if mean_mode == "Random means":
        seed = st.number_input("Environment seed", value=123, step=1)
        rng = np.random.default_rng(int(seed))
        means = np.sort(rng.uniform(0.20, 0.90, n_arms))

    else:
        st.markdown("#### Choose the hidden true mean of each arm")

        mean_values = []
        mean_cols = st.columns(min(n_arms, 4))

        for arm in range(n_arms):
            with mean_cols[arm % len(mean_cols)]:
                value = st.slider(
                    f"μ Arm {arm}",
                    min_value=0.00,
                    max_value=1.00,
                    value=float(0.20 + 0.60 * arm / max(1, n_arms - 1)),
                    step=0.01,
                    key=f"custom_mean_arm_{arm}",
                )
                mean_values.append(value)

        means = np.asarray(mean_values, dtype=float)

    st.caption(
        "Selection rule in this Basics page: arms are still sampled randomly. "
        "The custom means only change the hidden environment."
    )

    st.plotly_chart(plot_true_means(means), width="stretch")

    m1, m2 = st.columns(2)
    m1.metric("Optimal arm", int(np.argmax(means)))
    m2.metric("Optimal mean", round(float(np.max(means)), 4))

    st.info(
        "The optimal arm is visible here for teaching purposes. A real policy does not observe it directly."
    )

    st.divider()
    st.subheader("Replay simulation: learning hidden expectations")

    st.markdown(
        """
        This section shows how empirical means evolve as rewards are sampled. We avoid
        automatic live animation because Streamlit re-renders the page and can force the
        screen to jump upward. Instead, use the replay slider to move through time.
        """
    )

    sim_col1, sim_col2, sim_col3, sim_col4 = st.columns(4)

    with sim_col1:
        sim_horizon = st.slider("Simulation horizon", 20, 500, 120, step=10)

    with sim_col2:
        reward_noise = st.slider("Reward noise (sigma)", 0.02, 0.40, 0.12, step=0.01)

    with sim_col3:
        sim_seed = st.number_input("Simulation seed", value=123, step=1)

    with sim_col4:
        initial_uniform_pulls_per_arm = st.slider(
            "Initial uniform pulls per arm",
            min_value=0,
            max_value=20,
            value=0,
            step=1,
        )

    if initial_uniform_pulls_per_arm > 0:
        st.caption(
            "Initialization phase: each arm is pulled uniformly before random sampling continues."
        )

    simulation = simulate_random_learning(
        means=means,
        std=float(reward_noise),
        horizon=int(sim_horizon),
        seed=int(sim_seed),
        initial_pulls_per_arm=int(initial_uniform_pulls_per_arm),
    )

    t_live = st.slider(
        "Replay time step",
        1,
        int(sim_horizon),
        min(30, int(sim_horizon)),
    )

    idx = t_live - 1
    counts = simulation["counts_history"][idx]
    empirical_means = simulation["empirical_history"][idx]
    selected_arm = int(simulation["selected_arms"][idx])
    reward = float(simulation["rewards"][idx])

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("t", t_live)
    c2.metric("Selected arm", selected_arm)
    c3.metric("Observed reward", round(reward, 4))
    c4.metric("Most sampled arm", int(np.argmax(counts)))

    tab1, tab2, tab3 = st.tabs(
        [
            "Mean learning curves",
            "Allocation curves",
            "State at selected time",
        ]
    )

    with tab1:
        st.plotly_chart(
            plot_learning_curves(
                true_means=means,
                empirical_history=simulation["empirical_history"],
                t=t_live,
            ),
            width="stretch",
        )

    with tab2:
        st.plotly_chart(
            plot_allocation_curves(
                counts_history=simulation["counts_history"],
                t=t_live,
            ),
            width="stretch",
        )

    with tab3:
        st.plotly_chart(
            plot_state_snapshot(
                true_means=means,
                empirical_means=empirical_means,
                counts=counts,
                t=t_live,
            ),
            width="stretch",
        )

    st.info(
        "Reading the graphs: each arm keeps the same color across all panels. The mean-learning curves show how empirical estimates move over time toward the hidden true expectations. The allocation curves show how many times each arm has been sampled. This basic random-learning example is not strategic yet; it is designed to make the difference between true expectations and learned empirical means visible."
    )
