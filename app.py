from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from transformer_core import cosine_similarity_matrix, resolve_head_count, run_transformer_pipeline


st.set_page_config(
    page_title="Transformer Visualizer",
    page_icon="TF",
    layout="wide",
)


st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&family=IBM+Plex+Mono:wght@400;500&display=swap');

    html, body, [class*="css"]  {
        font-family: "Space Grotesk", sans-serif;
    }

    .stApp {
        background:
            radial-gradient(circle at 0% 0%, rgba(255, 181, 107, 0.26), transparent 28%),
            radial-gradient(circle at 100% 0%, rgba(59, 130, 246, 0.18), transparent 24%),
            linear-gradient(180deg, #f7f2e8 0%, #f1ede5 100%);
        color: #102038;
    }

    [data-testid="stSidebar"] {
        background: rgba(255, 255, 255, 0.74);
        border-right: 1px solid rgba(16, 32, 56, 0.08);
    }

    .hero-card, .note-card {
        background: rgba(255, 255, 255, 0.8);
        border: 1px solid rgba(16, 32, 56, 0.08);
        border-radius: 22px;
        padding: 1.25rem 1.4rem;
        box-shadow: 0 16px 40px rgba(16, 32, 56, 0.08);
        margin-bottom: 1rem;
    }

    .hero-kicker {
        text-transform: uppercase;
        letter-spacing: 0.18em;
        color: #b45309;
        font-size: 0.8rem;
        margin-bottom: 0.4rem;
    }

    .hero-title {
        font-size: 2.45rem;
        line-height: 1.05;
        margin: 0;
        color: #102038;
    }

    .hero-subtitle {
        color: #31445f;
        margin-top: 0.75rem;
        font-size: 1rem;
    }

    .takeaway {
        padding: 0.9rem 1rem;
        border-radius: 16px;
        background: rgba(255, 244, 230, 0.9);
        border: 1px solid rgba(180, 83, 9, 0.14);
        margin-bottom: 1rem;
    }

    .metric-strip {
        display: flex;
        flex-wrap: wrap;
        gap: 0.75rem;
        margin-top: 1rem;
    }

    .metric-chip {
        border-radius: 16px;
        padding: 0.85rem 1rem;
        background: rgba(255, 255, 255, 0.82);
        border: 1px solid rgba(16, 32, 56, 0.08);
        min-width: 150px;
    }

    .metric-chip-label {
        font-size: 0.78rem;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #6b7280;
    }

    .metric-chip-value {
        font-size: 1.5rem;
        color: #102038;
        font-weight: 700;
    }

    code {
        font-family: "IBM Plex Mono", monospace !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def dim_labels(size: int) -> list[str]:
    return [f"d{index}" for index in range(size)]


def render_intro(title: str, description: str, takeaway: str) -> None:
    st.markdown(
        f"""
        <div class="note-card">
            <h3 style="margin-top: 0; margin-bottom: 0.35rem;">{title}</h3>
            <p style="margin-bottom: 0.6rem; color: #31445f;">{description}</p>
            <div class="takeaway"><strong>Key takeaway:</strong> {takeaway}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def style_figure(figure: go.Figure, height: int = 420) -> go.Figure:
    figure.update_layout(
        height=height,
        margin=dict(l=20, r=20, t=60, b=20),
        paper_bgcolor="rgba(255,255,255,0.0)",
        plot_bgcolor="rgba(255,255,255,0.0)",
        font=dict(family="Space Grotesk, sans-serif", color="#102038"),
        title_font=dict(size=18),
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
) -> go.Figure:
    figure = px.imshow(
        matrix,
        x=x_labels,
        y=y_labels,
        aspect="auto",
        color_continuous_scale="YlOrRd",
        labels={"x": x_axis_label, "y": y_axis_label, "color": color_label},
        title=title,
    )
    return style_figure(figure)


def vector_bar(vector: np.ndarray, title: str, legend_label: str) -> go.Figure:
    figure = go.Figure()
    figure.add_bar(x=dim_labels(len(vector)), y=vector, marker_color="#f97316", name=legend_label)
    figure.update_layout(title=title, xaxis_title="Embedding dimensions", yaxis_title="Activation")
    return style_figure(figure, height=360)


def compare_vectors(series: dict[str, np.ndarray], title: str) -> go.Figure:
    figure = go.Figure()
    palette = ["#2563eb", "#f97316", "#0f766e", "#ef4444"]
    for color, (label, values) in zip(palette, series.items()):
        figure.add_trace(
            go.Scatter(
                x=dim_labels(len(values)),
                y=values,
                mode="lines+markers",
                name=label,
                line=dict(color=color, width=2),
            )
        )
    figure.update_layout(title=title, xaxis_title="Embedding dimensions", yaxis_title="Activation")
    return style_figure(figure, height=360)


def projection_figure(matrix: np.ndarray, tokens: list[str], title: str) -> go.Figure:
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    if centered.shape[0] == 1:
        coordinates = np.zeros((1, 2))
    else:
        _, _, vectors = np.linalg.svd(centered, full_matrices=False)
        coordinates = centered @ vectors[:2].T

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
        color_continuous_scale="Turbo",
        title=title,
    )
    figure.update_traces(marker=dict(size=13, line=dict(width=1, color="white")))
    figure.update_layout(xaxis_title="Principal component 1", yaxis_title="Principal component 2")
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


default_text = (
    "When the curious student revisited the first idea at the end of the lesson, "
    "the transformer could still connect the distant words."
)

with st.sidebar:
    st.header("Controls")
    input_text = st.text_area("Input text", value=default_text, height=140)
    num_layers = st.slider("Number of layers", min_value=1, max_value=6, value=3)
    requested_heads = st.slider("Number of attention heads", min_value=1, max_value=8, value=4)
    embedding_dim = st.slider("Embedding dimension", min_value=16, max_value=128, value=64, step=16)
    show_attention_maps = st.toggle("Show attention maps", value=True)
    show_positional_encoding = st.toggle("Show positional encoding", value=True)
    random_seed = st.number_input("Simulation seed", min_value=1, max_value=999, value=7, step=1)

resolved_heads = resolve_head_count(embedding_dim, requested_heads)
pipeline = run_transformer_pipeline(
    text=input_text,
    num_layers=num_layers,
    requested_heads=requested_heads,
    embedding_dim=embedding_dim,
    seed=int(random_seed),
)

tokens = pipeline["tokens"]
selected_token_index = st.sidebar.selectbox(
    "Tracked token",
    options=list(range(len(tokens))),
    format_func=lambda idx: f"{idx}: {tokens[idx]}",
)
selected_layer_index = st.sidebar.slider("Inspection layer", min_value=1, max_value=num_layers, value=1)
selected_head_index = st.sidebar.slider("Inspection head", min_value=1, max_value=resolved_heads, value=1)

if resolved_heads != requested_heads:
    st.sidebar.caption(
        f"Adjusted to {resolved_heads} heads so {embedding_dim}-D embeddings split evenly across heads."
    )

selected_layer = pipeline["layers"][selected_layer_index - 1]
selected_head = selected_head_index - 1
tracked_token = tokens[selected_token_index]
token_labels = [f"{index}: {token}" for index, token in enumerate(tokens)]
dimension_labels = dim_labels(embedding_dim)

st.markdown(
    f"""
    <div class="hero-card">
        <div class="hero-kicker">Interactive Transformer Lab</div>
        <h1 class="hero-title">Watch tokens build context layer by layer.</h1>
        <p class="hero-subtitle">
            This app uses a deterministic mini-Transformer simulator to expose the mechanics of
            tokenization, positional encoding, self-attention, multi-head attention, feedforward
            transforms, residual pathways, and contextual representation drift.
        </p>
        <div class="metric-strip">
            <div class="metric-chip">
                <div class="metric-chip-label">Sequence length</div>
                <div class="metric-chip-value">{len(tokens)}</div>
            </div>
            <div class="metric-chip">
                <div class="metric-chip-label">Layers</div>
                <div class="metric-chip-value">{num_layers}</div>
            </div>
            <div class="metric-chip">
                <div class="metric-chip-label">Heads</div>
                <div class="metric-chip-value">{resolved_heads}</div>
            </div>
            <div class="metric-chip">
                <div class="metric-chip-label">Embedding width</div>
                <div class="metric-chip-value">{embedding_dim}</div>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.caption(
    "The weights are fixed by the simulation seed so you can change the input sentence and see "
    "how the same architecture handles different token relationships."
)

tab_token, tab_position, tab_attention, tab_multihead, tab_ffn, tab_residual, tab_layers = st.tabs(
    [
        "1. Tokenization",
        "2. Positional Encoding",
        "3. Self-Attention",
        "4. Multi-Head Attention",
        "5. Feedforward Network",
        "6. Residual + Norm",
        "7. Layer-wise Representation",
    ]
)

with tab_token:
    render_intro(
        "Tokenization and Input Representation",
        "The sentence is split into model tokens, and each token is mapped to a dense vector before any context is added. Repeated tokens reuse the same base embedding, which is why order alone cannot distinguish them yet.",
        "Compare whitespace words to tokens, then inspect how each token starts as a position-free embedding profile.",
    )

    comparison_length = max(len(pipeline["whitespace_tokens"]), len(tokens))
    comparison_frame = pd.DataFrame(
        {
            "Whitespace words": pipeline["whitespace_tokens"] + [""] * (comparison_length - len(pipeline["whitespace_tokens"])),
            "Model tokens": tokens + [""] * (comparison_length - len(tokens)),
        }
    )
    token_frame = pd.DataFrame(
        {
            "Position": list(range(len(tokens))),
            "Token": tokens,
            "Token ID": pipeline["token_ids"],
        }
    )

    left, right = st.columns([1, 1])
    with left:
        st.subheader("Words vs tokens")
        st.dataframe(comparison_frame, use_container_width=True, hide_index=True)
        st.subheader("Token-to-index mapping")
        st.dataframe(token_frame, use_container_width=True, hide_index=True)
    with right:
        st.subheader("Embedding heatmap")
        st.plotly_chart(
            signed_heatmap(
                pipeline["embeddings"],
                x_labels=dimension_labels,
                y_labels=token_labels,
                title="Initial token embeddings",
            ),
            use_container_width=True,
        )

    st.plotly_chart(
        vector_bar(
            pipeline["embeddings"][selected_token_index],
            title=f"Embedding profile for token '{tracked_token}'",
            legend_label="Initial embedding",
        ),
        use_container_width=True,
    )

with tab_position:
    render_intro(
        "Positional Encoding",
        "Self-attention treats tokens as a set unless we inject order. Sinusoidal positional encoding adds a structured pattern to every token so the model can tell who came first and how far apart tokens are.",
        "Look at the before-and-after vectors for the tracked token and notice that the added position signal changes the same word differently depending on where it appears.",
    )

    if show_positional_encoding:
        left, right = st.columns([1.2, 1])
        with left:
            st.plotly_chart(
                signed_heatmap(
                    pipeline["positional_encoding"],
                    x_labels=dimension_labels,
                    y_labels=token_labels,
                    title="Positional encoding matrix",
                ),
                use_container_width=True,
            )
        with right:
            st.plotly_chart(
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
        st.caption("Try repeating the same token in different positions to see the base embedding stay fixed while the combined representation changes.")
    else:
        st.info("Enable `Show positional encoding` in the sidebar to reveal the position patterns.")

with tab_attention:
    render_intro(
        "Self-Attention Mechanism",
        "Each token projects into Query, Key, and Value vectors. Attention scores come from Query-Key compatibility, and softmax turns those scores into weights that describe how strongly a token reads from the rest of the sequence.",
        "Use the selected layer and head to see which tokens the tracked token pulls information from, then compare raw scores with normalized attention weights.",
    )

    selected_scores = selected_layer["attention_scores"][selected_head]
    selected_weights = selected_layer["attention_weights"][selected_head]
    attended_index = int(np.argmax(selected_weights[selected_token_index]))
    attended_weight = float(selected_weights[selected_token_index, attended_index])

    stats_a, stats_b, stats_c = st.columns(3)
    stats_a.metric("Tracked token", tracked_token)
    stats_b.metric("Strongest target", tokens[attended_index])
    stats_c.metric("Attention weight", f"{attended_weight:.3f}")

    left, right = st.columns([1, 1])
    with left:
        st.plotly_chart(
            signed_heatmap(
                selected_scores,
                x_labels=token_labels,
                y_labels=token_labels,
                title=f"Layer {selected_layer_index}, head {selected_head_index} Query-Key scores",
            ),
            use_container_width=True,
        )
    with right:
        if show_attention_maps:
            st.plotly_chart(
                positive_heatmap(
                    selected_weights,
                    x_labels=token_labels,
                    y_labels=token_labels,
                    title=f"Layer {selected_layer_index}, head {selected_head_index} attention weights",
                    x_axis_label="Key tokens",
                    y_axis_label="Query tokens",
                    color_label="Weight",
                ),
                use_container_width=True,
            )
        else:
            st.info("Attention map display is disabled in the sidebar.")

    st.plotly_chart(
        vector_bar(
            selected_weights[selected_token_index],
            title=f"Where '{tracked_token}' attends in layer {selected_layer_index}, head {selected_head_index}",
            legend_label="Attention weight",
        ),
        use_container_width=True,
    )

with tab_multihead:
    render_intro(
        "Multi-Head Attention",
        "Different heads can focus on different patterns at the same time. One head may lock onto nearby syntax, while another highlights longer-range links, giving the model multiple relational views of the same sentence.",
        "Compare the head-by-head maps and notice whether your tracked token shifts focus across heads even within the same layer.",
    )

    summary_rows = []
    for head_index in range(resolved_heads):
        head_weights = selected_layer["attention_weights"][head_index]
        focus_index = int(np.argmax(head_weights[selected_token_index]))
        entropy = float(-np.sum(head_weights[selected_token_index] * np.log(head_weights[selected_token_index] + 1e-9)))
        summary_rows.append(
            {
                "Head": f"H{head_index + 1}",
                "Top focus token": tokens[focus_index],
                "Weight": round(float(head_weights[selected_token_index, focus_index]), 3),
                "Entropy": round(entropy, 3),
            }
        )

    st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

    if show_attention_maps:
        head_columns = st.columns(2)
        for head_index in range(resolved_heads):
            with head_columns[head_index % 2]:
                st.plotly_chart(
                    positive_heatmap(
                        selected_layer["attention_weights"][head_index],
                        x_labels=token_labels,
                        y_labels=token_labels,
                        title=f"Layer {selected_layer_index} | Head {head_index + 1}",
                        x_axis_label="Key tokens",
                        y_axis_label="Query tokens",
                        color_label="Weight",
                    ),
                    use_container_width=True,
                )
    else:
        st.info("Enable `Show attention maps` in the sidebar to compare heads visually.")

with tab_ffn:
    render_intro(
        "Feedforward Network",
        "After attention mixes information across tokens, the feedforward block transforms each token independently with a wider hidden layer and a non-linear activation. This helps the model reshape features and amplify useful combinations.",
        "Watch how the tracked token changes after the FFN and notice that the network expands the feature space before compressing it back.",
    )

    left, right = st.columns([1.1, 0.9])
    with left:
        st.plotly_chart(
            signed_heatmap(
                selected_layer["ffn_hidden"],
                x_labels=dim_labels(selected_layer["ffn_hidden"].shape[1]),
                y_labels=token_labels,
                title=f"Layer {selected_layer_index} expanded FFN hidden activations",
            ),
            use_container_width=True,
        )
    with right:
        st.plotly_chart(
            compare_vectors(
                {
                    "Input to FFN": selected_layer["norm_1"][selected_token_index],
                    "FFN output": selected_layer["ffn_output"][selected_token_index],
                    "Block output": selected_layer["norm_2"][selected_token_index],
                },
                title=f"Tracked token through the FFN in layer {selected_layer_index}",
            ),
            use_container_width=True,
        )

    st.plotly_chart(
        signed_heatmap(
            selected_layer["ffn_output"],
            x_labels=dimension_labels,
            y_labels=token_labels,
            title=f"Layer {selected_layer_index} FFN output by token",
        ),
        use_container_width=True,
    )

with tab_residual:
    render_intro(
        "Residual Connections and Layer Normalization",
        "Residual paths preserve earlier information by adding the block input back into the transformed output. Layer normalization then stabilizes the representation so the next block sees a consistent scale.",
        "Look for two things: residual steps preserve magnitude and structure, while normalization compresses mean and variance into a steadier range.",
    )

    stage_sequence = [
        ("Block input", selected_layer["block_input"]),
        ("Attention output", selected_layer["multi_head_output"]),
        ("Residual 1", selected_layer["residual_1"]),
        ("Norm 1", selected_layer["norm_1"]),
        ("FFN output", selected_layer["ffn_output"]),
        ("Residual 2", selected_layer["residual_2"]),
        ("Norm 2", selected_layer["norm_2"]),
    ]

    norm_matrix = np.array([[np.linalg.norm(token_vector) for token_vector in values] for _, values in stage_sequence])
    stats_frame = pd.DataFrame(
        [
            {
                "Stage": name,
                "Tracked token norm": round(float(np.linalg.norm(values[selected_token_index])), 3),
                "Feature mean": round(float(np.mean(values)), 3),
                "Feature std": round(float(np.std(values)), 3),
            }
            for name, values in stage_sequence
        ]
    )

    left, right = st.columns([1, 1])
    with left:
        st.plotly_chart(
            positive_heatmap(
                norm_matrix,
                x_labels=token_labels,
                y_labels=[name for name, _ in stage_sequence],
                title=f"Layer {selected_layer_index} token norms across residual path",
                x_axis_label="Tokens",
                y_axis_label="Block stages",
                color_label="Norm",
            ),
            use_container_width=True,
        )
    with right:
        st.dataframe(stats_frame, use_container_width=True, hide_index=True)

with tab_layers:
    render_intro(
        "Stacked Layers and Contextual Representation",
        "As layers accumulate, token vectors stop representing just the token identity and start encoding contextual role. Tracking a single token across layers reveals how the surrounding sequence gradually reshapes its meaning.",
        "Use the tracked token heatmap and the drift map to spot where the model meaningfully changes a representation instead of simply passing it through.",
    )

    layer_labels = ["Input + Pos"] + [f"Layer {index}" for index in range(1, num_layers + 1)]
    tracked_history = np.vstack([representation[selected_token_index] for representation in pipeline["representations"]])
    drift = cosine_drift(pipeline["representations"])

    left, right = st.columns([1, 1])
    with left:
        st.plotly_chart(
            signed_heatmap(
                tracked_history,
                x_labels=dimension_labels,
                y_labels=layer_labels,
                title=f"Representation of '{tracked_token}' across layers",
            ),
            use_container_width=True,
        )
    with right:
        st.plotly_chart(
            positive_heatmap(
                drift,
                x_labels=token_labels,
                y_labels=layer_labels,
                title="Representation drift relative to the input state",
                x_axis_label="Tokens",
                y_axis_label="Layers",
                color_label="Drift",
            ),
            use_container_width=True,
        )

    bottom_left, bottom_right = st.columns([1, 1])
    with bottom_left:
        st.plotly_chart(
            projection_figure(
                pipeline["representations"][0],
                tokens=tokens,
                title="Token geometry before stacked layers",
            ),
            use_container_width=True,
        )
    with bottom_right:
        st.plotly_chart(
            projection_figure(
                pipeline["final_representation"],
                tokens=tokens,
                title="Token geometry after the final layer",
            ),
            use_container_width=True,
        )

    similarity_columns = st.columns([1, 1])
    with similarity_columns[0]:
        st.plotly_chart(
            signed_heatmap(
                cosine_similarity_matrix(pipeline["representations"][0]),
                x_labels=token_labels,
                y_labels=token_labels,
                title="Token similarity at input",
            ),
            use_container_width=True,
        )
    with similarity_columns[1]:
        st.plotly_chart(
            signed_heatmap(
                cosine_similarity_matrix(pipeline["final_representation"]),
                x_labels=token_labels,
                y_labels=token_labels,
                title="Token similarity after final layer",
            ),
            use_container_width=True,
        )
