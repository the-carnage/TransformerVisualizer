from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from transformer_core import (
    attention_entropy,
    classify_token,
    cosine_similarity_matrix,
    pca_project,
    resolve_head_count,
    run_transformer_pipeline,
)
from components.attention_arcs import attention_arc_figure
from components.architecture_diagram import architecture_diagram
from components.flow_animations import (
    dimension_flow_figure,
    distribution_histogram,
    gelu_curve_with_activations,
    sinusoidal_wave_figure,
    token_journey_figure,
)


# ── Page config ──────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Transformer Visualizer — Interactive Lab",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Premium dark-mode CSS ────────────────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

    :root {
        --bg-deep: #0a0e1a;
        --bg-card: rgba(17, 24, 39, 0.65);
        --bg-card-hover: rgba(17, 24, 39, 0.85);
        --border: rgba(99, 102, 241, 0.12);
        --border-glow: rgba(99, 102, 241, 0.35);
        --text-primary: #e2e8f0;
        --text-secondary: #94a3b8;
        --text-muted: #64748b;
        --accent-indigo: #6366f1;
        --accent-violet: #8b5cf6;
        --accent-amber: #f59e0b;
        --accent-cyan: #0ea5e9;
        --accent-emerald: #10b981;
        --gradient-hero: linear-gradient(135deg, #6366f1 0%, #8b5cf6 40%, #d946ef 100%);
    }

    html, body, [class*="css"] {
        font-family: "Inter", -apple-system, sans-serif !important;
    }

    .stApp {
        background: var(--bg-deep);
        color: var(--text-primary);
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #111827 100%) !important;
        border-right: 1px solid var(--border) !important;
    }

    section[data-testid="stSidebar"] .stMarkdown h2,
    section[data-testid="stSidebar"] .stMarkdown h3 {
        background: var(--gradient-hero);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        background: rgba(15, 23, 42, 0.6);
        border-radius: 16px;
        padding: 6px;
        border: 1px solid var(--border);
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 12px;
        padding: 8px 16px;
        color: var(--text-secondary);
        font-weight: 500;
        font-size: 0.82rem;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }

    .stTabs [aria-selected="true"] {
        background: rgba(99, 102, 241, 0.15) !important;
        color: #a5b4fc !important;
        border-bottom: 2px solid var(--accent-indigo) !important;
        box-shadow: 0 0 20px rgba(99, 102, 241, 0.15);
    }

    /* Cards */
    .glass-card {
        background: var(--bg-card);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid var(--border);
        border-radius: 20px;
        padding: 1.5rem;
        margin-bottom: 1.2rem;
        position: relative;
        overflow: hidden;
        transition: all 0.3s ease;
    }

    .glass-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(99,102,241,0.3), transparent);
    }

    .glass-card:hover {
        border-color: var(--border-glow);
        box-shadow: 0 8px 40px rgba(99, 102, 241, 0.08);
    }

    /* Hero */
    .hero-container {
        background: linear-gradient(135deg, rgba(99,102,241,0.08) 0%, rgba(139,92,246,0.06) 50%, rgba(217,70,239,0.04) 100%);
        border: 1px solid var(--border);
        border-radius: 24px;
        padding: 2rem 2.2rem;
        margin-bottom: 1.5rem;
        position: relative;
        overflow: hidden;
    }

    .hero-container::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle at 30% 30%, rgba(99,102,241,0.06), transparent 50%),
                    radial-gradient(circle at 70% 70%, rgba(217,70,239,0.04), transparent 40%);
        animation: hero-glow 8s ease-in-out infinite alternate;
    }

    @keyframes hero-glow {
        0% { transform: translate(0, 0); }
        100% { transform: translate(-5%, -5%); }
    }

    .hero-kicker {
        text-transform: uppercase;
        letter-spacing: 0.22em;
        font-size: 0.72rem;
        font-weight: 600;
        color: var(--accent-amber);
        margin-bottom: 0.6rem;
        position: relative;
        z-index: 1;
    }

    .hero-title {
        font-size: 2.6rem;
        font-weight: 800;
        line-height: 1.05;
        margin: 0 0 0.8rem 0;
        background: var(--gradient-hero);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        position: relative;
        z-index: 1;
    }

    .hero-subtitle {
        color: var(--text-secondary);
        font-size: 1rem;
        line-height: 1.6;
        max-width: 720px;
        position: relative;
        z-index: 1;
    }

    /* Metric chips */
    .metrics-row {
        display: flex;
        flex-wrap: wrap;
        gap: 0.75rem;
        margin-top: 1.2rem;
        position: relative;
        z-index: 1;
    }

    .metric-pill {
        background: rgba(15, 23, 42, 0.6);
        backdrop-filter: blur(10px);
        border: 1px solid var(--border);
        border-radius: 14px;
        padding: 0.75rem 1.1rem;
        min-width: 130px;
        transition: all 0.3s ease;
    }

    .metric-pill:hover {
        border-color: var(--border-glow);
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(99, 102, 241, 0.12);
    }

    .metric-pill .label {
        font-size: 0.68rem;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: var(--text-muted);
        margin-bottom: 0.25rem;
    }

    .metric-pill .value {
        font-size: 1.45rem;
        font-weight: 700;
        color: var(--text-primary);
    }

    /* Section intro cards */
    .intro-card {
        background: rgba(15, 23, 42, 0.5);
        border: 1px solid var(--border);
        border-left: 3px solid var(--accent-indigo);
        border-radius: 16px;
        padding: 1.2rem 1.4rem;
        margin-bottom: 1.2rem;
    }

    .intro-card h3 {
        color: #c7d2fe;
        margin: 0 0 0.4rem 0;
        font-size: 1.15rem;
        font-weight: 600;
    }

    .intro-card p {
        color: var(--text-secondary);
        margin: 0 0 0.7rem 0;
        line-height: 1.55;
        font-size: 0.92rem;
    }

    .takeaway-box {
        background: rgba(245, 158, 11, 0.06);
        border: 1px solid rgba(245, 158, 11, 0.15);
        border-radius: 12px;
        padding: 0.7rem 0.9rem;
        font-size: 0.88rem;
        color: #fbbf24;
    }

    .takeaway-box strong {
        color: var(--accent-amber);
    }

    /* Token chip */
    .token-chip {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 6px 14px;
        border-radius: 10px;
        font-family: "JetBrains Mono", monospace;
        font-size: 0.82rem;
        font-weight: 500;
        margin: 3px;
        transition: all 0.2s ease;
        cursor: default;
    }

    .token-chip:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
    }

    .token-chip.word {
        background: rgba(99, 102, 241, 0.12);
        border: 1px solid rgba(99, 102, 241, 0.25);
        color: #a5b4fc;
    }

    .token-chip.punctuation {
        background: rgba(245, 158, 11, 0.12);
        border: 1px solid rgba(245, 158, 11, 0.25);
        color: #fbbf24;
    }

    .token-chip.special {
        background: rgba(239, 68, 68, 0.12);
        border: 1px solid rgba(239, 68, 68, 0.25);
        color: #fca5a5;
    }

    .token-index {
        font-size: 0.65rem;
        color: var(--text-muted);
        font-weight: 400;
    }

    /* Dataframes */
    .stDataFrame {
        border: 1px solid var(--border) !important;
        border-radius: 12px !important;
        overflow: hidden !important;
    }

    /* Plotly charts */
    .stPlotlyChart {
        border-radius: 16px;
        overflow: hidden;
    }

    /* Architecture panel */
    .arch-panel {
        background: rgba(15, 23, 42, 0.4);
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 0.2rem;
    }

    /* Code */
    code {
        font-family: "JetBrains Mono", monospace !important;
        background: rgba(15, 23, 42, 0.6) !important;
        color: #c7d2fe !important;
        padding: 2px 6px;
        border-radius: 6px;
    }

    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 6px;
        height: 6px;
    }
    ::-webkit-scrollbar-track {
        background: transparent;
    }
    ::-webkit-scrollbar-thumb {
        background: rgba(99,102,241,0.3);
        border-radius: 3px;
    }

    /* Hide streamlit branding */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }

    /* Expander styling */
    .streamlit-expanderHeader {
        background: rgba(15, 23, 42, 0.5) !important;
        border-radius: 12px !important;
        color: var(--text-secondary) !important;
        font-weight: 500 !important;
    }

    /* Animation for content */
    @keyframes fadeSlideIn {
        from {
            opacity: 0;
            transform: translateY(10px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    .stTabs [data-baseweb="tab-panel"] {
        animation: fadeSlideIn 0.4s ease-out;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ── Helper functions ─────────────────────────────────────────────────────

_chart_counter = {"n": 0}

def plot_chart(figure: go.Figure, **kwargs) -> None:
    """Wrapper around st.plotly_chart that auto-generates a unique key."""
    _chart_counter["n"] += 1
    # Translate legacy use_container_width to the new 'width' param
    if kwargs.pop("use_container_width", None):
        kwargs.setdefault("width", "stretch")
    kwargs.setdefault("width", "stretch")
    st.plotly_chart(figure, key=f"_pc_{_chart_counter['n']}", **kwargs)


def dim_labels(size: int) -> list[str]:
    return [f"d{index}" for index in range(size)]


def render_intro(title: str, description: str, takeaway: str) -> None:
    st.markdown(
        f"""
        <div class="intro-card">
            <h3>{title}</h3>
            <p>{description}</p>
            <div class="takeaway-box"><strong>🔍 Key takeaway:</strong> {takeaway}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def style_figure(figure: go.Figure, height: int = 420) -> go.Figure:
    figure.update_layout(
        height=height,
        margin=dict(l=20, r=20, t=60, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color="#e2e8f0"),
        title_font=dict(size=16, color="#e2e8f0"),
        xaxis=dict(gridcolor="rgba(148,163,184,0.08)", zerolinecolor="rgba(148,163,184,0.12)"),
        yaxis=dict(gridcolor="rgba(148,163,184,0.08)", zerolinecolor="rgba(148,163,184,0.12)"),
        legend=dict(
            bgcolor="rgba(17,24,39,0.6)",
            bordercolor="rgba(148,163,184,0.15)",
            font=dict(size=11),
        ),
    )
    return figure


def signed_heatmap(matrix: np.ndarray, x_labels: list[str], y_labels: list[str], title: str) -> go.Figure:
    figure = px.imshow(
        matrix,
        x=x_labels,
        y=y_labels,
        aspect="auto",
        color_continuous_scale="RdBu",
        color_continuous_midpoint=0.0,
        labels={"x": "Dimensions", "y": "Tokens", "color": "Value"},
        title=title,
    )
    return style_figure(figure)


def positive_heatmap(
    matrix: np.ndarray,
    x_labels: list[str],
    y_labels: list[str],
    title: str,
    x_axis_label: str = "Columns",
    y_axis_label: str = "Rows",
    color_label: str = "Value",
    colorscale: str = "Viridis",
) -> go.Figure:
    figure = px.imshow(
        matrix,
        x=x_labels,
        y=y_labels,
        aspect="auto",
        color_continuous_scale=colorscale,
        labels={"x": x_axis_label, "y": y_axis_label, "color": color_label},
        title=title,
    )
    return style_figure(figure)


def vector_bar(vector: np.ndarray, title: str, legend_label: str) -> go.Figure:
    n = len(vector)
    colors = [f"hsl({int(220 + i * 140 / max(n, 1))}, 70%, 60%)" for i in range(n)]
    figure = go.Figure()
    figure.add_bar(
        x=dim_labels(n),
        y=vector,
        marker=dict(color=colors, line=dict(color="rgba(255,255,255,0.1)", width=0.5)),
        name=legend_label,
    )
    figure.update_layout(
        title=dict(text=title, font=dict(size=15, color="#e2e8f0")),
        xaxis_title="Embedding dimensions",
        yaxis_title="Activation",
    )
    return style_figure(figure, height=340)


def compare_vectors(series: dict[str, np.ndarray], title: str) -> go.Figure:
    figure = go.Figure()
    palette = ["#6366f1", "#f59e0b", "#10b981", "#ef4444", "#8b5cf6", "#0ea5e9"]
    for idx, (label, values) in enumerate(series.items()):
        figure.add_trace(
            go.Scatter(
                x=dim_labels(len(values)),
                y=values,
                mode="lines+markers",
                name=label,
                line=dict(color=palette[idx % len(palette)], width=2),
                marker=dict(size=4),
            )
        )
    figure.update_layout(
        title=dict(text=title, font=dict(size=15, color="#e2e8f0")),
        xaxis_title="Embedding dimensions",
        yaxis_title="Activation",
    )
    return style_figure(figure, height=360)


def projection_figure(matrix: np.ndarray, tokens: list[str], title: str) -> go.Figure:
    coordinates = pca_project(matrix, n_components=2)
    frame = pd.DataFrame(
        {
            "x": coordinates[:, 0],
            "y": coordinates[:, 1],
            "token": tokens,
            "position": list(range(len(tokens))),
        }
    )
    figure = px.scatter(
        frame,
        x="x",
        y="y",
        text="token",
        color="position",
        color_continuous_scale="Viridis",
        title=title,
    )
    figure.update_traces(
        marker=dict(size=14, line=dict(width=1.5, color="rgba(255,255,255,0.3)")),
        textposition="top center",
        textfont=dict(size=9, color="#e2e8f0"),
    )
    figure.update_layout(xaxis_title="PC 1", yaxis_title="PC 2")
    return style_figure(figure, height=420)


def cosine_drift(representations: list[np.ndarray]) -> np.ndarray:
    baseline = representations[0]
    baseline_norm = baseline / np.clip(np.linalg.norm(baseline, axis=1, keepdims=True), 1e-9, None)
    drift_rows = []
    for representation in representations:
        current_norm = representation / np.clip(np.linalg.norm(representation, axis=1, keepdims=True), 1e-9, None)
        cosine = np.sum(baseline_norm * current_norm, axis=1)
        drift_rows.append(1.0 - cosine)
    return np.vstack(drift_rows)


def render_token_chips(tokens: list[str], token_types: list[str]) -> None:
    """Render interactive token chips with color-coded types."""
    chips_html = ""
    for idx, (token, ttype) in enumerate(zip(tokens, token_types)):
        chips_html += f'<span class="token-chip {ttype}"><span class="token-index">{idx}</span> {token}</span>'
    st.markdown(f'<div style="line-height: 2.4;">{chips_html}</div>', unsafe_allow_html=True)


# ── Sidebar ──────────────────────────────────────────────────────────────

default_text = (
    "When the curious student revisited the first idea at the end of the lesson, "
    "the transformer could still connect the distant words."
)

with st.sidebar:
    st.markdown("### ⚡ Controls")
    input_text = st.text_area("Input text", value=default_text, height=120)

    st.markdown("##### 🏗️ Architecture")
    num_layers = st.slider("Layers", min_value=1, max_value=6, value=3)
    requested_heads = st.slider("Attention heads", min_value=1, max_value=8, value=4)
    embedding_dim = st.slider("Embedding dim", min_value=16, max_value=128, value=64, step=16)

    st.markdown("##### 🎛️ Display")
    show_attention_maps = st.toggle("Attention maps", value=True)
    show_positional_encoding = st.toggle("Positional encoding", value=True)
    show_architecture = st.toggle("Architecture diagram", value=True)
    random_seed = st.number_input("Simulation seed", min_value=1, max_value=999, value=7, step=1)

# ── Pipeline ─────────────────────────────────────────────────────────────

resolved_heads = resolve_head_count(embedding_dim, requested_heads)
pipeline = run_transformer_pipeline(
    text=input_text,
    num_layers=num_layers,
    requested_heads=requested_heads,
    embedding_dim=embedding_dim,
    seed=int(random_seed),
)

tokens = pipeline["tokens"]
token_types = pipeline["token_types"]

with st.sidebar:
    st.markdown("##### 🔬 Inspection")
    selected_token_index = st.selectbox(
        "Tracked token",
        options=list(range(len(tokens))),
        format_func=lambda idx: f"{idx}: {tokens[idx]}",
    )
    selected_layer_index = st.slider("Inspection layer", min_value=1, max_value=num_layers, value=1)
    selected_head_index = st.slider("Inspection head", min_value=1, max_value=resolved_heads, value=1)

    if resolved_heads != requested_heads:
        st.caption(
            f"⚙️ Adjusted to **{resolved_heads}** heads so {embedding_dim}-D embeddings split evenly."
        )

selected_layer = pipeline["layers"][selected_layer_index - 1]
selected_head = selected_head_index - 1
tracked_token = tokens[selected_token_index]
token_labels = [f"{i}: {t}" for i, t in enumerate(tokens)]
dimension_labels = dim_labels(embedding_dim)

# ── Hero ─────────────────────────────────────────────────────────────────

st.markdown(
    f"""
    <div class="hero-container">
        <div class="hero-kicker">⚡ Interactive Transformer Lab</div>
        <h1 class="hero-title">Watch tokens build context<br>layer by layer.</h1>
        <p class="hero-subtitle">
            Explore tokenization, positional encoding, self-attention, multi-head attention,
            feedforward transforms, residual pathways, and contextual representation drift —
            all driven by a deterministic mini-Transformer simulator.
        </p>
        <div class="metrics-row">
            <div class="metric-pill">
                <div class="label">Tokens</div>
                <div class="value">{len(tokens)}</div>
            </div>
            <div class="metric-pill">
                <div class="label">Layers</div>
                <div class="value">{num_layers}</div>
            </div>
            <div class="metric-pill">
                <div class="label">Heads</div>
                <div class="value">{resolved_heads}</div>
            </div>
            <div class="metric-pill">
                <div class="label">Embed dim</div>
                <div class="value">{embedding_dim}</div>
            </div>
            <div class="metric-pill">
                <div class="label">Head dim</div>
                <div class="value">{embedding_dim // resolved_heads}</div>
            </div>
            <div class="metric-pill">
                <div class="label">FFN hidden</div>
                <div class="value">{embedding_dim * 2}</div>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Architecture diagram ─────────────────────────────────────────────────

# We'll track which tab is active via session state
if "active_tab_idx" not in st.session_state:
    st.session_state.active_tab_idx = 0

# ── Tabs ─────────────────────────────────────────────────────────────────

tab_token, tab_position, tab_attention, tab_multihead, tab_ffn, tab_residual, tab_layers = st.tabs(
    [
        "🔤 Tokenization",
        "📍 Positional Encoding",
        "🎯 Self-Attention",
        "🧠 Multi-Head Attention",
        "⚡ Feedforward Network",
        "🔄 Residual + Norm",
        "📊 Layer-wise Repr.",
    ]
)


# ── Tab 1: Tokenization ─────────────────────────────────────────────────

with tab_token:
    if show_architecture:
        with st.expander("🏗️ Architecture — where are we?", expanded=False):
            plot_chart(architecture_diagram(active_tab=0), use_container_width=True)

    render_intro(
        "Tokenization & Input Representation",
        "Text → tokens → dense vectors. The sentence is split into model tokens and each is "
        "mapped to a numerical embedding before any context is added.",
        "Compare whitespace words to model tokens. Repeated tokens share the same base embedding — "
        "position alone can't distinguish them yet.",
    )

    # Token chips
    st.markdown("#### Token Stream")
    render_token_chips(tokens, token_types)

    legend_html = (
        '<div style="display:flex;gap:16px;margin:8px 0 16px;">'
        '<span class="token-chip word" style="font-size:0.72rem;">word</span>'
        '<span class="token-chip punctuation" style="font-size:0.72rem;">punctuation</span>'
        '<span class="token-chip special" style="font-size:0.72rem;">special</span>'
        '</div>'
    )
    st.markdown(legend_html, unsafe_allow_html=True)

    # Tables + heatmap
    col_table, col_heat = st.columns([1, 1.3])

    with col_table:
        comparison_length = max(len(pipeline["whitespace_tokens"]), len(tokens))
        comparison_frame = pd.DataFrame(
            {
                "Whitespace words": pipeline["whitespace_tokens"]
                + [""] * (comparison_length - len(pipeline["whitespace_tokens"])),
                "Model tokens": tokens + [""] * (comparison_length - len(tokens)),
            }
        )
        st.markdown("##### Words vs Tokens")
        st.dataframe(comparison_frame, use_container_width=True, hide_index=True)

        token_frame = pd.DataFrame(
            {"Position": list(range(len(tokens))), "Token": tokens, "Token ID": pipeline["token_ids"]}
        )
        st.markdown("##### Token → Index Mapping")
        st.dataframe(token_frame, use_container_width=True, hide_index=True)

    with col_heat:
        plot_chart(
            signed_heatmap(
                pipeline["embeddings"],
                x_labels=dimension_labels,
                y_labels=token_labels,
                title="Initial Token Embeddings",
            ),
            use_container_width=True,
        )

    # Tracked token embedding profile
    plot_chart(
        vector_bar(
            pipeline["embeddings"][selected_token_index],
            title=f"Embedding profile — '{tracked_token}'",
            legend_label="Initial embedding",
        ),
        use_container_width=True,
    )

    # 3D PCA scatter
    with st.expander("🌐 Embedding Space (PCA Projection)", expanded=True):
        plot_chart(
            projection_figure(
                pipeline["embeddings"],
                tokens=tokens,
                title="Token embeddings in 2D (before position)",
            ),
            use_container_width=True,
        )

# ── Tab 2: Positional Encoding ──────────────────────────────────────────

with tab_position:
    if show_architecture:
        with st.expander("🏗️ Architecture — where are we?", expanded=False):
            plot_chart(architecture_diagram(active_tab=1), use_container_width=True)

    render_intro(
        "Positional Encoding",
        "Self-attention treats tokens as a set — no order! Sinusoidal positional encoding "
        "injects a structured wave pattern so the model knows who came first and how far apart tokens are.",
        "Look at the before-and-after vectors: the same word gets a different combined representation "
        "depending on where it appears in the sequence.",
    )

    if show_positional_encoding:
        # Sinusoidal wave figure
        max_positions = min(len(tokens), 8)
        selected_positions = list(range(0, len(tokens), max(1, len(tokens) // max_positions)))[:max_positions]

        plot_chart(
            sinusoidal_wave_figure(pipeline["positional_encoding"], selected_positions, tokens),
            use_container_width=True,
        )

        col_pe, col_cmp = st.columns([1.2, 1])
        with col_pe:
            plot_chart(
                signed_heatmap(
                    pipeline["positional_encoding"],
                    x_labels=dimension_labels,
                    y_labels=token_labels,
                    title="Positional Encoding Matrix",
                ),
                use_container_width=True,
            )
        with col_cmp:
            plot_chart(
                compare_vectors(
                    {
                        "Embedding only": pipeline["embeddings"][selected_token_index],
                        "Position signal": pipeline["positional_encoding"][selected_token_index],
                        "Combined input": pipeline["input_with_position"][selected_token_index],
                    },
                    title=f"How position changes '{tracked_token}'",
                ),
                use_container_width=True,
            )

        st.caption(
            "💡 Try repeating the same token in different positions — "
            "the base embedding stays fixed while the combined representation changes."
        )
    else:
        st.info("Enable **Positional encoding** in the sidebar to reveal position patterns.")


# ── Tab 3: Self-Attention ────────────────────────────────────────────────

with tab_attention:
    if show_architecture:
        with st.expander("🏗️ Architecture — where are we?", expanded=False):
            plot_chart(architecture_diagram(active_tab=2), use_container_width=True)

    render_intro(
        "Self-Attention Mechanism",
        "Each token projects into Query (Q), Key (K), and Value (V) vectors. The attention score "
        "is Q·Kᵀ / √d — then softmax normalizes these into weights that describe how strongly "
        "each token reads from every other.",
        "Select a token and see which tokens it pulls information from. Compare raw scores with "
        "normalized weights — softmax sharpens the strongest connections.",
    )

    selected_scores = selected_layer["attention_scores"][selected_head]
    selected_weights = selected_layer["attention_weights"][selected_head]
    attended_index = int(np.argmax(selected_weights[selected_token_index]))
    attended_weight = float(selected_weights[selected_token_index, attended_index])

    # Metrics row
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Tracked", f"'{tracked_token}'")
    m2.metric("Strongest target", f"'{tokens[attended_index]}'")
    m3.metric("Attention weight", f"{attended_weight:.3f}")
    entropy_val = float(attention_entropy(selected_weights)[selected_token_index])
    m4.metric("Entropy", f"{entropy_val:.2f}")

    # Attention arc diagram
    st.markdown("#### Attention Flow Arcs")
    plot_chart(
        attention_arc_figure(
            tokens=tokens,
            attention_weights=selected_weights,
            query_index=selected_token_index,
            title=f"Attention from '{tracked_token}' — L{selected_layer_index} H{selected_head_index}",
        ),
        use_container_width=True,
    )

    # Heatmaps side by side
    col_scores, col_weights = st.columns([1, 1])
    with col_scores:
        plot_chart(
            signed_heatmap(
                selected_scores,
                x_labels=token_labels,
                y_labels=token_labels,
                title=f"Q·K scores — L{selected_layer_index} H{selected_head_index}",
            ),
            use_container_width=True,
        )
    with col_weights:
        if show_attention_maps:
            plot_chart(
                positive_heatmap(
                    selected_weights,
                    x_labels=token_labels,
                    y_labels=token_labels,
                    title=f"Attention weights (softmax) — L{selected_layer_index} H{selected_head_index}",
                    x_axis_label="Key tokens",
                    y_axis_label="Query tokens",
                    color_label="Weight",
                ),
                use_container_width=True,
            )
        else:
            st.info("Enable **Attention maps** in sidebar.")

    # Bar chart for tracked token
    plot_chart(
        vector_bar(
            selected_weights[selected_token_index],
            title=f"Where '{tracked_token}' attends — L{selected_layer_index} H{selected_head_index}",
            legend_label="Attention weight",
        ),
        use_container_width=True,
    )


# ── Tab 4: Multi-Head Attention ──────────────────────────────────────────

with tab_multihead:
    if show_architecture:
        with st.expander("🏗️ Architecture — where are we?", expanded=False):
            plot_chart(architecture_diagram(active_tab=3), use_container_width=True)

    render_intro(
        "Multi-Head Attention",
        "Different heads learn different patterns in parallel — one may track nearby syntax while "
        "another captures long-range semantic links, giving the model multiple relational views.",
        "Compare head-by-head maps. Does the tracked token shift focus across heads? "
        "Higher head diversity means the model captures richer relationships.",
    )

    # Head diversity metric
    diversity = selected_layer["head_diversity"]
    importance = selected_layer["token_importance"]

    d1, d2, d3 = st.columns(3)
    d1.metric("Head diversity (JSD)", f"{diversity:.4f}")
    d2.metric("Most important token", f"'{tokens[int(np.argmax(importance))]}'")
    d3.metric("Importance score", f"{float(np.max(importance)):.2f}")

    # Summary table
    summary_rows = []
    for head_index in range(resolved_heads):
        head_weights = selected_layer["attention_weights"][head_index]
        focus_index = int(np.argmax(head_weights[selected_token_index]))
        ent = float(attention_entropy(head_weights)[selected_token_index])
        summary_rows.append(
            {
                "Head": f"H{head_index + 1}",
                "Top focus": tokens[focus_index],
                "Weight": round(float(head_weights[selected_token_index, focus_index]), 3),
                "Entropy": round(ent, 3),
                "Focus type": "Focused 🎯" if ent < 1.5 else "Diffuse 🌊",
            }
        )
    st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

    # Token importance bar
    fig_imp = go.Figure()
    imp_colors = [
        "#f59e0b" if i == int(np.argmax(importance)) else "#6366f1"
        for i in range(len(tokens))
    ]
    fig_imp.add_bar(
        x=token_labels,
        y=importance,
        marker=dict(color=imp_colors, line=dict(color="rgba(255,255,255,0.1)", width=0.5)),
    )
    fig_imp.update_layout(
        title=dict(text="Token Importance (attention received)", font=dict(size=15, color="#e2e8f0")),
        xaxis_title="Tokens",
        yaxis_title="Total attention received",
    )
    plot_chart(style_figure(fig_imp, height=320), use_container_width=True)

    # Head comparison grid
    if show_attention_maps:
        head_cols = st.columns(2)
        for head_index in range(resolved_heads):
            with head_cols[head_index % 2]:
                plot_chart(
                    positive_heatmap(
                        selected_layer["attention_weights"][head_index],
                        x_labels=token_labels,
                        y_labels=token_labels,
                        title=f"L{selected_layer_index} | Head {head_index + 1}",
                        x_axis_label="Key",
                        y_axis_label="Query",
                        color_label="Weight",
                    ),
                    use_container_width=True,
                )
    else:
        st.info("Enable **Attention maps** in sidebar to compare heads visually.")


# ── Tab 5: Feedforward Network ───────────────────────────────────────────

with tab_ffn:
    if show_architecture:
        with st.expander("🏗️ Architecture — where are we?", expanded=False):
            plot_chart(architecture_diagram(active_tab=4), use_container_width=True)

    render_intro(
        "Feedforward Network (FFN)",
        "After attention mixes information across tokens, the FFN transforms each token "
        "independently: expand to a wider hidden layer, apply GELU non-linearity, then compress back.",
        "Watch the tracked token change after the FFN. Notice the expand→compress dimension flow "
        "and how GELU selectively activates features.",
    )

    # Dimension flow diagram
    col_dim, col_gelu = st.columns([1, 1])
    with col_dim:
        plot_chart(
            dimension_flow_figure(
                embedding_dim,
                embedding_dim * 2,
                embedding_dim,
                title="FFN Dimension Flow",
            ),
            use_container_width=True,
        )
    with col_gelu:
        # GELU activation with actual values
        ffn_input_vals = selected_layer["norm_1"][selected_token_index]
        plot_chart(
            gelu_curve_with_activations(
                ffn_input_vals[:20],  # Show first 20 dims for clarity
                title=f"GELU activations — '{tracked_token}'",
            ),
            use_container_width=True,
        )

    # Before/after comparison
    col_hidden, col_compare = st.columns([1.1, 0.9])
    with col_hidden:
        plot_chart(
            signed_heatmap(
                selected_layer["ffn_hidden"],
                x_labels=dim_labels(selected_layer["ffn_hidden"].shape[1]),
                y_labels=token_labels,
                title=f"L{selected_layer_index} — expanded FFN hidden activations",
            ),
            use_container_width=True,
        )
    with col_compare:
        plot_chart(
            compare_vectors(
                {
                    "FFN input": selected_layer["norm_1"][selected_token_index],
                    "FFN output": selected_layer["ffn_output"][selected_token_index],
                    "Block output": selected_layer["norm_2"][selected_token_index],
                },
                title=f"'{tracked_token}' through FFN — L{selected_layer_index}",
            ),
            use_container_width=True,
        )

    plot_chart(
        signed_heatmap(
            selected_layer["ffn_output"],
            x_labels=dimension_labels,
            y_labels=token_labels,
            title=f"L{selected_layer_index} — FFN output by token",
        ),
        use_container_width=True,
    )


# ── Tab 6: Residual + Normalization ──────────────────────────────────────

with tab_residual:
    if show_architecture:
        with st.expander("🏗️ Architecture — where are we?", expanded=False):
            plot_chart(architecture_diagram(active_tab=5), use_container_width=True)

    render_intro(
        "Residual Connections & Layer Normalization",
        "Skip connections add the block input back into the output, preserving earlier information. "
        "Layer normalization then stabilizes representations to a consistent scale for the next block.",
        "Watch two things: residual steps preserve magnitude and structure, while normalization "
        "compresses mean→0 and variance→1 range.",
    )

    stage_sequence = [
        ("Block input", selected_layer["block_input"]),
        ("Attention output", selected_layer["multi_head_output"]),
        ("Residual 1 (input + attn)", selected_layer["residual_1"]),
        ("LayerNorm 1", selected_layer["norm_1"]),
        ("FFN output", selected_layer["ffn_output"]),
        ("Residual 2 (norm1 + ffn)", selected_layer["residual_2"]),
        ("LayerNorm 2", selected_layer["norm_2"]),
    ]

    # Distribution histograms
    col_dist1, col_dist2 = st.columns([1, 1])
    with col_dist1:
        plot_chart(
            distribution_histogram(
                selected_layer["residual_1"],
                selected_layer["norm_1"],
                before_label="Before Norm (Residual 1)",
                after_label="After LayerNorm 1",
                title="Distribution shift — LayerNorm 1",
            ),
            use_container_width=True,
        )
    with col_dist2:
        plot_chart(
            distribution_histogram(
                selected_layer["residual_2"],
                selected_layer["norm_2"],
                before_label="Before Norm (Residual 2)",
                after_label="After LayerNorm 2",
                title="Distribution shift — LayerNorm 2",
            ),
            use_container_width=True,
        )

    # Norm heatmap and stats table
    norm_matrix = np.array(
        [[np.linalg.norm(token_vec) for token_vec in vals] for _, vals in stage_sequence]
    )
    stats_frame = pd.DataFrame(
        [
            {
                "Stage": name,
                "Tracked norm": round(float(np.linalg.norm(vals[selected_token_index])), 3),
                "Mean": round(float(np.mean(vals)), 4),
                "Std": round(float(np.std(vals)), 4),
            }
            for name, vals in stage_sequence
        ]
    )

    col_norms, col_stats = st.columns([1, 1])
    with col_norms:
        plot_chart(
            positive_heatmap(
                norm_matrix,
                x_labels=token_labels,
                y_labels=[name for name, _ in stage_sequence],
                title=f"L{selected_layer_index} — token norms across residual path",
                x_axis_label="Tokens",
                y_axis_label="Block stages",
                color_label="Norm",
                colorscale="Inferno",
            ),
            use_container_width=True,
        )
    with col_stats:
        st.dataframe(stats_frame, use_container_width=True, hide_index=True)


# ── Tab 7: Layer-wise Representation ─────────────────────────────────────

with tab_layers:
    if show_architecture:
        with st.expander("🏗️ Architecture — where are we?", expanded=False):
            plot_chart(architecture_diagram(active_tab=6), use_container_width=True)

    render_intro(
        "Stacked Layers & Contextual Representation",
        "As layers stack, token vectors stop representing just identity and start encoding "
        "contextual meaning. Tracking a single token reveals how context gradually reshapes it.",
        "Use the drift map and token journey to spot where the model makes the biggest "
        "representational changes vs. just passing information through.",
    )

    layer_labels = ["Input + Pos"] + [f"Layer {i}" for i in range(1, num_layers + 1)]
    tracked_history = np.vstack(
        [rep[selected_token_index] for rep in pipeline["representations"]]
    )
    drift = cosine_drift(pipeline["representations"])

    # Token journey — animated embedding evolution
    plot_chart(
        token_journey_figure(
            tracked_history,
            layer_labels,
            dimension_labels,
            tracked_token,
        ),
        use_container_width=True,
    )

    # Tracked token heatmap + drift
    col_tracked, col_drift = st.columns([1, 1])
    with col_tracked:
        plot_chart(
            signed_heatmap(
                tracked_history,
                x_labels=dimension_labels,
                y_labels=layer_labels,
                title=f"'{tracked_token}' representation across layers",
            ),
            use_container_width=True,
        )
    with col_drift:
        plot_chart(
            positive_heatmap(
                drift,
                x_labels=token_labels,
                y_labels=layer_labels,
                title="Representation drift from input",
                x_axis_label="Tokens",
                y_axis_label="Layers",
                color_label="Drift",
                colorscale="Magma",
            ),
            use_container_width=True,
        )

    # Before/after geometry
    col_before, col_after = st.columns([1, 1])
    with col_before:
        plot_chart(
            projection_figure(
                pipeline["representations"][0],
                tokens=tokens,
                title="Token geometry — before layers",
            ),
            use_container_width=True,
        )
    with col_after:
        plot_chart(
            projection_figure(
                pipeline["final_representation"],
                tokens=tokens,
                title="Token geometry — after final layer",
            ),
            use_container_width=True,
        )

    # Similarity matrices
    col_sim1, col_sim2 = st.columns([1, 1])
    with col_sim1:
        plot_chart(
            signed_heatmap(
                cosine_similarity_matrix(pipeline["representations"][0]),
                x_labels=token_labels,
                y_labels=token_labels,
                title="Token similarity — at input",
            ),
            use_container_width=True,
        )
    with col_sim2:
        plot_chart(
            signed_heatmap(
                cosine_similarity_matrix(pipeline["final_representation"]),
                x_labels=token_labels,
                y_labels=token_labels,
                title="Token similarity — after final layer",
            ),
            use_container_width=True,
        )
