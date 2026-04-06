"""Flow animation and specialized chart builders."""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go


def gelu_curve_with_activations(
    activations: np.ndarray,
    title: str = "GELU Activation Function",
) -> go.Figure:
    """Plot the GELU curve with a token's actual activations highlighted."""
    x = np.linspace(-4, 4, 300)
    y = 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)))

    fig = go.Figure()

    # GELU curve
    fig.add_trace(
        go.Scatter(
            x=x, y=y,
            mode="lines",
            line=dict(color="#6366f1", width=2.5),
            name="GELU(x)",
            hoverinfo="skip",
        )
    )

    # Actual activations
    act_y = 0.5 * activations * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (activations + 0.044715 * activations**3)))
    fig.add_trace(
        go.Scatter(
            x=activations,
            y=act_y,
            mode="markers",
            marker=dict(
                size=8,
                color="#f59e0b",
                line=dict(color="rgba(255,255,255,0.4)", width=1),
            ),
            name="Token activations",
            hovertemplate="Input: %{x:.3f}<br>Output: %{y:.3f}<extra></extra>",
        )
    )

    # Zero reference
    fig.add_hline(y=0, line_dash="dot", line_color="rgba(148,163,184,0.3)")
    fig.add_vline(x=0, line_dash="dot", line_color="rgba(148,163,184,0.3)")

    fig.update_layout(
        title=dict(text=title, font=dict(size=15, color="#e2e8f0")),
        height=340,
        margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color="#e2e8f0"),
        xaxis=dict(
            title="Input",
            gridcolor="rgba(148,163,184,0.1)",
            zerolinecolor="rgba(148,163,184,0.2)",
        ),
        yaxis=dict(
            title="Output",
            gridcolor="rgba(148,163,184,0.1)",
            zerolinecolor="rgba(148,163,184,0.2)",
        ),
        legend=dict(
            bgcolor="rgba(17,24,39,0.6)",
            bordercolor="rgba(148,163,184,0.2)",
            font=dict(size=11),
        ),
    )
    return fig


def distribution_histogram(
    before: np.ndarray,
    after: np.ndarray,
    before_label: str = "Before",
    after_label: str = "After",
    title: str = "Distribution Shift",
) -> go.Figure:
    """Overlay histograms showing distribution before and after a transformation."""
    fig = go.Figure()

    fig.add_trace(
        go.Histogram(
            x=before.flatten(),
            name=before_label,
            marker_color="rgba(99,102,241,0.5)",
            nbinsx=40,
        )
    )
    fig.add_trace(
        go.Histogram(
            x=after.flatten(),
            name=after_label,
            marker_color="rgba(245,158,11,0.5)",
            nbinsx=40,
        )
    )

    fig.update_layout(
        barmode="overlay",
        title=dict(text=title, font=dict(size=15, color="#e2e8f0")),
        height=340,
        margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color="#e2e8f0"),
        xaxis=dict(
            title="Activation value",
            gridcolor="rgba(148,163,184,0.1)",
        ),
        yaxis=dict(
            title="Count",
            gridcolor="rgba(148,163,184,0.1)",
        ),
        legend=dict(
            bgcolor="rgba(17,24,39,0.6)",
            bordercolor="rgba(148,163,184,0.2)",
        ),
    )
    return fig


def dimension_flow_figure(
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    title: str = "FFN Dimension Flow",
) -> go.Figure:
    """Visualize the expand-then-compress dimension flow of the FFN."""
    fig = go.Figure()

    stages = ["Input", "Hidden (expanded)", "Output"]
    dims = [input_dim, hidden_dim, output_dim]
    colors = ["#6366f1", "#f59e0b", "#10b981"]

    # Bars
    fig.add_trace(
        go.Bar(
            x=stages,
            y=dims,
            marker=dict(
                color=colors,
                line=dict(color="rgba(255,255,255,0.2)", width=1),
            ),
            text=[f"{d}D" for d in dims],
            textposition="outside",
            textfont=dict(color="#e2e8f0", size=13),
            hovertemplate="%{x}: %{y} dimensions<extra></extra>",
            showlegend=False,
        )
    )

    # Arrows between bars
    for i in range(len(stages) - 1):
        label = f"×{dims[i+1]/dims[i]:.0f}" if dims[i+1] > dims[i] else f"÷{dims[i]/dims[i+1]:.0f}"
        fig.add_annotation(
            x=i + 0.5, y=max(dims) * 0.85,
            text=f"<b>{label}</b>",
            showarrow=False,
            font=dict(size=14, color="#94a3b8"),
        )

    fig.update_layout(
        title=dict(text=title, font=dict(size=15, color="#e2e8f0")),
        height=300,
        margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color="#e2e8f0"),
        yaxis=dict(
            title="Dimensions",
            gridcolor="rgba(148,163,184,0.1)",
        ),
        xaxis=dict(
            gridcolor="rgba(148,163,184,0.1)",
        ),
    )
    return fig


def token_journey_figure(
    tracked_history: np.ndarray,
    layer_labels: list[str],
    dimension_labels: list[str],
    token_name: str,
) -> go.Figure:
    """Animated line chart showing how a token's embedding evolves across layers."""
    fig = go.Figure()

    palette = [
        "#6366f1", "#8b5cf6", "#a855f7", "#d946ef",
        "#f59e0b", "#10b981", "#0ea5e9", "#ef4444",
    ]

    for layer_idx, label in enumerate(layer_labels):
        color = palette[layer_idx % len(palette)]
        alpha = 0.3 + 0.7 * (layer_idx / max(len(layer_labels) - 1, 1))
        fig.add_trace(
            go.Scatter(
                x=dimension_labels,
                y=tracked_history[layer_idx],
                mode="lines",
                name=label,
                line=dict(color=color, width=1.5),
                opacity=alpha,
            )
        )

    fig.update_layout(
        title=dict(
            text=f"Embedding evolution of '{token_name}' across layers",
            font=dict(size=15, color="#e2e8f0"),
        ),
        height=380,
        margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color="#e2e8f0"),
        xaxis=dict(
            title="Embedding dimensions",
            gridcolor="rgba(148,163,184,0.1)",
        ),
        yaxis=dict(
            title="Activation",
            gridcolor="rgba(148,163,184,0.1)",
        ),
        legend=dict(
            bgcolor="rgba(17,24,39,0.6)",
            bordercolor="rgba(148,163,184,0.2)",
            font=dict(size=10),
        ),
    )
    return fig


def sinusoidal_wave_figure(
    positional_encoding: np.ndarray,
    selected_positions: list[int],
    tokens: list[str],
) -> go.Figure:
    """Visualize sinusoidal positional encoding for selected positions."""
    fig = go.Figure()
    dims = positional_encoding.shape[1]
    x = list(range(dims))

    palette = [
        "#6366f1", "#f59e0b", "#10b981", "#ef4444",
        "#8b5cf6", "#0ea5e9", "#d946ef", "#84cc16",
    ]

    for i, pos in enumerate(selected_positions):
        if pos >= positional_encoding.shape[0]:
            continue
        color = palette[i % len(palette)]
        label = f"Pos {pos}" if pos >= len(tokens) else f"Pos {pos}: '{tokens[pos]}'"
        fig.add_trace(
            go.Scatter(
                x=x, y=positional_encoding[pos],
                mode="lines",
                name=label,
                line=dict(color=color, width=2),
            )
        )

    fig.update_layout(
        title=dict(
            text="Sinusoidal Positional Encoding Waves",
            font=dict(size=15, color="#e2e8f0"),
        ),
        height=340,
        margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color="#e2e8f0"),
        xaxis=dict(
            title="Embedding dimension",
            gridcolor="rgba(148,163,184,0.1)",
        ),
        yaxis=dict(
            title="Encoding value",
            gridcolor="rgba(148,163,184,0.1)",
        ),
        legend=dict(
            bgcolor="rgba(17,24,39,0.6)",
            bordercolor="rgba(148,163,184,0.2)",
            font=dict(size=10),
        ),
    )
    return fig
