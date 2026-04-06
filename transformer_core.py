from __future__ import annotations

import hashlib
import re

import numpy as np


TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]")

PUNCTUATION_CHARS = set(".,;:!?\"'()[]{}/-–—…")


def tokenize_text(text: str) -> list[str]:
    stripped = text.strip()
    if not stripped:
        return ["[EMPTY]"]

    tokens = TOKEN_PATTERN.findall(stripped)
    return tokens or ["[EMPTY]"]


def classify_token(token: str) -> str:
    """Classify a token as 'word', 'punctuation', or 'special'."""
    if token.startswith("[") and token.endswith("]"):
        return "special"
    if all(ch in PUNCTUATION_CHARS for ch in token):
        return "punctuation"
    return "word"


def resolve_head_count(embedding_dim: int, requested_heads: int) -> int:
    capped = min(requested_heads, embedding_dim)
    for head_count in range(capped, 0, -1):
        if embedding_dim % head_count == 0:
            return head_count
    return 1


def _stable_seed(*parts: object) -> int:
    payload = "::".join(str(part) for part in parts)
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % (2**32 - 1)


def _random_matrix(rows: int, cols: int, *seed_parts: object) -> np.ndarray:
    rng = np.random.default_rng(_stable_seed(*seed_parts))
    scale = 1.0 / np.sqrt(max(rows, 1))
    return rng.normal(loc=0.0, scale=scale, size=(rows, cols))


def _token_embedding(token: str, embedding_dim: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(_stable_seed("token", token.lower(), embedding_dim, seed))
    return rng.normal(loc=0.0, scale=0.7, size=embedding_dim)


def positional_encoding(sequence_length: int, embedding_dim: int) -> np.ndarray:
    positions = np.arange(sequence_length)[:, np.newaxis]
    dimensions = np.arange(embedding_dim)[np.newaxis, :]
    angle_rates = 1.0 / np.power(10000.0, (2 * (dimensions // 2)) / max(embedding_dim, 1))
    angles = positions * angle_rates

    encoding = np.zeros((sequence_length, embedding_dim))
    encoding[:, 0::2] = np.sin(angles[:, 0::2])
    encoding[:, 1::2] = np.cos(angles[:, 1::2])
    return encoding


def softmax(values: np.ndarray, axis: int = -1) -> np.ndarray:
    shifted = values - np.max(values, axis=axis, keepdims=True)
    exp_values = np.exp(shifted)
    return exp_values / np.sum(exp_values, axis=axis, keepdims=True)


def gelu(values: np.ndarray) -> np.ndarray:
    return 0.5 * values * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (values + 0.044715 * values**3)))


def layer_norm(values: np.ndarray, epsilon: float = 1e-5) -> np.ndarray:
    means = np.mean(values, axis=-1, keepdims=True)
    variances = np.var(values, axis=-1, keepdims=True)
    return (values - means) / np.sqrt(variances + epsilon)


def cosine_similarity_matrix(values: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    normalized = values / np.clip(norms, 1e-9, None)
    return normalized @ normalized.T


# ---------------------------------------------------------------------------
# New metrics for hackathon upgrade
# ---------------------------------------------------------------------------


def attention_entropy(weights: np.ndarray) -> np.ndarray:
    """Compute Shannon entropy per query token for an attention weight matrix.

    Higher entropy → more diffuse attention; lower → more focused.
    Shape: (seq_len,)
    """
    return -np.sum(weights * np.log(weights + 1e-9), axis=-1)


def head_diversity(head_weights: np.ndarray) -> float:
    """Average pairwise Jensen-Shannon divergence across heads.

    head_weights shape: (num_heads, seq_len, seq_len)
    Returns a scalar ∈ [0, 1] indicating how different the heads are.
    """
    num_heads = head_weights.shape[0]
    if num_heads < 2:
        return 0.0

    # Flatten each head's full attention matrix into a single distribution
    flat = head_weights.reshape(num_heads, -1)
    flat = flat / flat.sum(axis=1, keepdims=True)

    divergences = []
    for i in range(num_heads):
        for j in range(i + 1, num_heads):
            m = 0.5 * (flat[i] + flat[j])
            kl_i = np.sum(flat[i] * np.log(flat[i] / (m + 1e-12) + 1e-12))
            kl_j = np.sum(flat[j] * np.log(flat[j] / (m + 1e-12) + 1e-12))
            divergences.append(0.5 * (kl_i + kl_j))

    return float(np.mean(divergences))


def token_importance(attention_weights: np.ndarray) -> np.ndarray:
    """How much total attention each token *receives* (averaged across heads).

    attention_weights shape: (num_heads, seq_len, seq_len)
    Returns shape: (seq_len,)
    """
    # Sum over query dimension (axis=1) to get how much attention flows *to* each key
    received = attention_weights.sum(axis=1)  # (num_heads, seq_len)
    return received.mean(axis=0)  # average across heads


def distribution_stats(values: np.ndarray) -> dict:
    """Compute distribution statistics for a matrix (per-feature across tokens)."""
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "per_token_norms": np.linalg.norm(values, axis=-1).tolist(),
        "per_token_means": np.mean(values, axis=-1).tolist(),
        "per_token_stds": np.std(values, axis=-1).tolist(),
    }


def pca_project(matrix: np.ndarray, n_components: int = 2) -> np.ndarray:
    """Project rows of matrix onto top principal components via SVD."""
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    if centered.shape[0] <= 1:
        return np.zeros((centered.shape[0], n_components))
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    return centered @ vt[:n_components].T


# ---------------------------------------------------------------------------
# Core building blocks
# ---------------------------------------------------------------------------


def _build_embeddings(tokens: list[str], embedding_dim: int, seed: int) -> tuple[np.ndarray, dict[str, int], list[int]]:
    vocabulary: dict[str, int] = {}
    embeddings = []
    token_ids = []

    for token in tokens:
        if token not in vocabulary:
            vocabulary[token] = len(vocabulary)
        token_ids.append(vocabulary[token])
        embeddings.append(_token_embedding(token, embedding_dim, seed))

    return np.vstack(embeddings), vocabulary, token_ids


def transformer_layer(
    inputs: np.ndarray,
    layer_index: int,
    num_heads: int,
    embedding_dim: int,
    seed: int,
) -> dict[str, object]:
    sequence_length = inputs.shape[0]
    head_dim = embedding_dim // num_heads

    all_attention_scores = []
    all_attention_weights = []
    all_queries = []
    all_keys = []
    all_values = []
    head_outputs = []
    per_head_entropy = []

    for head_index in range(num_heads):
        query_projection = _random_matrix(
            embedding_dim, head_dim, "query", layer_index, head_index, seed,
        )
        key_projection = _random_matrix(
            embedding_dim, head_dim, "key", layer_index, head_index, seed,
        )
        value_projection = _random_matrix(
            embedding_dim, head_dim, "value", layer_index, head_index, seed,
        )

        query = inputs @ query_projection
        key = inputs @ key_projection
        value = inputs @ value_projection

        scores = (query @ key.T) / np.sqrt(max(head_dim, 1))
        weights = softmax(scores, axis=-1)
        head_output = weights @ value

        all_queries.append(query)
        all_keys.append(key)
        all_values.append(value)
        all_attention_scores.append(scores)
        all_attention_weights.append(weights)
        head_outputs.append(head_output)
        per_head_entropy.append(attention_entropy(weights))

    stacked_weights = np.stack(all_attention_weights)

    concatenated_heads = np.concatenate(head_outputs, axis=-1)
    output_projection = _random_matrix(
        embedding_dim, embedding_dim, "output", layer_index, seed,
    )
    multi_head_output = concatenated_heads @ output_projection
    residual_1 = inputs + multi_head_output
    norm_1 = layer_norm(residual_1)

    hidden_dim = embedding_dim * 2
    feedforward_in = _random_matrix(embedding_dim, hidden_dim, "ffn_in", layer_index, seed)
    feedforward_out = _random_matrix(hidden_dim, embedding_dim, "ffn_out", layer_index, seed)

    ffn_hidden = gelu(norm_1 @ feedforward_in)
    ffn_output = ffn_hidden @ feedforward_out
    residual_2 = norm_1 + ffn_output
    norm_2 = layer_norm(residual_2)

    return {
        "layer_index": np.array(layer_index),
        "block_input": inputs,
        "queries": np.stack(all_queries),
        "keys": np.stack(all_keys),
        "values": np.stack(all_values),
        "attention_scores": np.stack(all_attention_scores),
        "attention_weights": stacked_weights,
        "head_outputs": np.stack(head_outputs),
        "multi_head_output": multi_head_output,
        "residual_1": residual_1,
        "norm_1": norm_1,
        "ffn_hidden": ffn_hidden,
        "ffn_output": ffn_output,
        "residual_2": residual_2,
        "norm_2": norm_2,
        "attention_rollup": np.mean(stacked_weights, axis=0),
        "sequence_length": np.array(sequence_length),
        "head_dim": np.array(head_dim),
        # New metrics
        "per_head_entropy": np.stack(per_head_entropy),  # (num_heads, seq_len)
        "head_diversity": head_diversity(stacked_weights),
        "token_importance": token_importance(stacked_weights),
        "block_input_stats": distribution_stats(inputs),
        "norm_1_stats": distribution_stats(norm_1),
        "norm_2_stats": distribution_stats(norm_2),
        "residual_1_stats": distribution_stats(residual_1),
        "residual_2_stats": distribution_stats(residual_2),
    }


def run_transformer_pipeline(
    text: str,
    num_layers: int,
    requested_heads: int,
    embedding_dim: int,
    seed: int = 7,
) -> dict[str, object]:
    tokens = tokenize_text(text)
    token_types = [classify_token(t) for t in tokens]
    embeddings, vocabulary, token_ids = _build_embeddings(tokens, embedding_dim, seed)
    num_heads = resolve_head_count(embedding_dim, requested_heads)

    positions = positional_encoding(len(tokens), embedding_dim)
    input_with_position = embeddings + positions

    layers = []
    representations = [input_with_position.copy()]
    current = input_with_position.copy()

    for layer_index in range(num_layers):
        layer_data = transformer_layer(
            inputs=current,
            layer_index=layer_index,
            num_heads=num_heads,
            embedding_dim=embedding_dim,
            seed=seed,
        )
        layers.append(layer_data)
        current = layer_data["norm_2"]
        representations.append(current.copy())

    # PCA projections for all representation stages
    pca_stages = []
    for rep in representations:
        pca_stages.append(pca_project(rep, n_components=2))

    return {
        "raw_text": text,
        "whitespace_tokens": text.split(),
        "tokens": tokens,
        "token_types": token_types,
        "vocabulary": vocabulary,
        "token_ids": token_ids,
        "embeddings": embeddings,
        "positional_encoding": positions,
        "input_with_position": input_with_position,
        "layers": layers,
        "representations": representations,
        "pca_stages": pca_stages,
        "final_representation": current,
        "num_layers": num_layers,
        "num_heads": num_heads,
        "embedding_dim": embedding_dim,
        "head_dim": embedding_dim // num_heads,
        "seed": seed,
        "initial_similarity": cosine_similarity_matrix(input_with_position),
        "final_similarity": cosine_similarity_matrix(current),
    }
