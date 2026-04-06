"""Interactive architecture diagram of a Transformer block."""
from __future__ import annotations

import plotly.graph_objects as go


# Block layout positions (x_center, y_center, width, height, label, color)
BLOCKS = [
    (0.5, 0.95, 0.35, 0.06, "Input Embeddings + Positional Encoding", "#6366f1"),
    (0.5, 0.82, 0.30, 0.06, "Multi-Head Self-Attention", "#8b5cf6"),
    (0.5, 0.70, 0.22, 0.04, "Add & Layer Norm", "#0ea5e9"),
    (0.5, 0.58, 0.30, 0.06, "Feed-Forward Network", "#f59e0b"),
    (0.5, 0.46, 0.22, 0.04, "Add & Layer Norm", "#0ea5e9"),
    (0.5, 0.34, 0.30, 0.06, "Output Representation", "#10b981"),
]

# Map tab index (0-6) to the block index that should be highlighted
TAB_TO_BLOCK = {
    0: 0,  # Tokenization → Input Embeddings
    1: 0,  # Positional Encoding → Input Embeddings
    2: 1,  # Self-Attention → Multi-Head Self-Attention
    3: 1,  # Multi-Head Attention → Multi-Head Self-Attention
    4: 3,  # FFN → Feed-Forward Network
    5: 2,  # Residual & Norm → Add & Layer Norm (first one)
    6: 5,  # Layer-wise → Output Representation
}


def architecture_diagram(active_tab: int = 0) -> go.Figure:
    """Render a vertical architecture block diagram with the active component highlighted."""
    fig = go.Figure()

    highlighted = TAB_TO_BLOCK.get(active_tab, -1)

    # Draw connecting arrows
    for i in range(len(BLOCKS) - 1):
        x0 = BLOCKS[i][0]
        y0 = BLOCKS[i][1] - BLOCKS[i][3] / 2
        y1 = BLOCKS[i + 1][1] + BLOCKS[i + 1][3] / 2
        fig.add_annotation(
            x=x0,
            y=y0,
            ax=x0,
            ay=y1,
            xref="x",
            yref="y",
            axref="x",
            ayref="y",
            showarrow=True,
            arrowhead=3,
            arrowsize=1.2,
            arrowwidth=1.5,
            arrowcolor="rgba(148,163,184,0.5)",
        )

    # Draw residual skip connection (attention block)
    fig.add_shape(
        type="line",
        x0=0.18, y0=BLOCKS[1][1],
        x1=0.18, y1=BLOCKS[2][1],
        line=dict(color="rgba(14,165,233,0.4)", width=2, dash="dot"),
    )
    fig.add_annotation(
        x=0.15, y=(BLOCKS[1][1] + BLOCKS[2][1]) / 2,
        text="skip", showarrow=False,
        font=dict(size=8, color="#94a3b8"),
    )

    # Draw residual skip connection (FFN block)
    fig.add_shape(
        type="line",
        x0=0.82, y0=BLOCKS[3][1],
        x1=0.82, y1=BLOCKS[4][1],
        line=dict(color="rgba(14,165,233,0.4)", width=2, dash="dot"),
    )
    fig.add_annotation(
        x=0.85, y=(BLOCKS[3][1] + BLOCKS[4][1]) / 2,
        text="skip", showarrow=False,
        font=dict(size=8, color="#94a3b8"),
    )

    # Draw blocks
    for idx, (cx, cy, w, h, label, color) in enumerate(BLOCKS):
        is_active = idx == highlighted
        opacity = 1.0 if is_active else 0.5
        border_width = 3 if is_active else 1
        fill_opacity = 0.25 if is_active else 0.08
        glow = f"rgba({int(color[1:3], 16)},{int(color[3:5], 16)},{int(color[5:7], 16)},{fill_opacity})"

        fig.add_shape(
            type="rect",
            x0=cx - w / 2, y0=cy - h / 2,
            x1=cx + w / 2, y1=cy + h / 2,
            fillcolor=glow,
            line=dict(color=color if is_active else "rgba(148,163,184,0.3)", width=border_width),
            layer="above",
        )

        fig.add_annotation(
            x=cx, y=cy,
            text=f"<b>{label}</b>" if is_active else label,
            showarrow=False,
            font=dict(
                size=11 if is_active else 9,
                color=color if is_active else "#94a3b8",
                family="Inter, sans-serif",
            ),
            opacity=opacity,
        )

    # Nx layer repeat indicator
    fig.add_shape(
        type="rect",
        x0=0.08, y0=BLOCKS[1][1] + BLOCKS[1][3] / 2 + 0.02,
        x1=0.92, y1=BLOCKS[4][1] - BLOCKS[4][3] / 2 - 0.02,
        fillcolor="rgba(99,102,241,0.03)",
        line=dict(color="rgba(99,102,241,0.15)", width=1, dash="dash"),
    )
    fig.add_annotation(
        x=0.92, y=BLOCKS[1][1] + BLOCKS[1][3] / 2 + 0.02,
        text="× N layers",
        showarrow=False,
        font=dict(size=9, color="#6366f1"),
        xanchor="right",
    )

    fig.update_layout(
        height=380,
        margin=dict(l=0, r=0, t=10, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(range=[0, 1], showgrid=False, zeroline=False, showticklabels=False, fixedrange=True),
        yaxis=dict(range=[0.25, 1.05], showgrid=False, zeroline=False, showticklabels=False, fixedrange=True),
        font=dict(family="Inter, sans-serif"),
    )

    return fig
