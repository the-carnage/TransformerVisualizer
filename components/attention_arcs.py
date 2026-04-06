"""Attention arc diagram — curved arcs connecting tokens based on attention weights."""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go


def _bezier_arc(x0: float, x1: float, n_points: int = 50) -> tuple[np.ndarray, np.ndarray]:
    """Generate a smooth quadratic Bezier arc between x0 and x1."""
    t = np.linspace(0, 1, n_points)
    mid_x = (x0 + x1) / 2.0
    height = abs(x1 - x0) * 0.45  # arc height proportional to distance
    # Quadratic Bezier: P(t) = (1-t)²P0 + 2(1-t)tP1 + t²P2
    bx = (1 - t) ** 2 * x0 + 2 * (1 - t) * t * mid_x + t**2 * x1
    by = (1 - t) ** 2 * 0 + 2 * (1 - t) * t * height + t**2 * 0
    return bx, by


def attention_arc_figure(
    tokens: list[str],
    attention_weights: np.ndarray,
    query_index: int | None = None,
    title: str = "Attention Flow",
    threshold: float = 0.05,
) -> go.Figure:
    """Build an arc diagram showing attention flow from a query token.

    If query_index is None, show the strongest connection for every token.
    """
    seq_len = len(tokens)
    fig = go.Figure()

    # Token positions along x-axis
    x_positions = list(range(seq_len))

    if query_index is not None:
        # Show arcs from one query to all keys
        weights = attention_weights[query_index]
        for key_idx in range(seq_len):
            w = float(weights[key_idx])
            if w < threshold:
                continue
            bx, by = _bezier_arc(query_index, key_idx)
            opacity = min(0.2 + w * 0.8, 1.0)
            width = 1 + w * 6
            fig.add_trace(
                go.Scatter(
                    x=bx,
                    y=by,
                    mode="lines",
                    line=dict(
                        color=f"rgba(99,102,241,{opacity})",
                        width=width,
                    ),
                    hoverinfo="text",
                    hovertext=f"{tokens[query_index]} → {tokens[key_idx]}: {w:.3f}",
                    showlegend=False,
                )
            )
    else:
        # Show top connection per query
        for q_idx in range(seq_len):
            k_idx = int(np.argmax(attention_weights[q_idx]))
            w = float(attention_weights[q_idx, k_idx])
            if w < threshold or q_idx == k_idx:
                continue
            bx, by = _bezier_arc(q_idx, k_idx)
            opacity = min(0.3 + w * 0.7, 1.0)
            width = 1 + w * 5
            fig.add_trace(
                go.Scatter(
                    x=bx,
                    y=by,
                    mode="lines",
                    line=dict(
                        color=f"rgba(139,92,246,{opacity})",
                        width=width,
                    ),
                    hoverinfo="text",
                    hovertext=f"{tokens[q_idx]} → {tokens[k_idx]}: {w:.3f}",
                    showlegend=False,
                )
            )

    # Draw token circles
    token_colors = ["#6366f1"] * seq_len
    if query_index is not None:
        token_colors[query_index] = "#f59e0b"

    fig.add_trace(
        go.Scatter(
            x=x_positions,
            y=[0] * seq_len,
            mode="markers+text",
            marker=dict(
                size=28,
                color=token_colors,
                line=dict(color="rgba(255,255,255,0.3)", width=2),
            ),
            text=[t[:8] for t in tokens],
            textposition="bottom center",
            textfont=dict(size=10, color="#e2e8f0"),
            hoverinfo="text",
            hovertext=[f"[{i}] {t}" for i, t in enumerate(tokens)],
            showlegend=False,
        )
    )

    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color="#e2e8f0")),
        height=max(280, 200 + seq_len * 3),
        margin=dict(l=10, r=10, t=50, b=60),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color="#e2e8f0"),
        xaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            range=[-1, seq_len],
        ),
        yaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            range=[-1.5, max(seq_len * 0.25, 3)],
        ),
    )

    return fig
