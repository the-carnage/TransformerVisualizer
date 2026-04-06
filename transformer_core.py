from __future__ import annotations

import hashlib
import re

import numpy as np


TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]")


def tokenize_text(text: str) -> list[str]:
    stripped = text.strip()
    if not stripped:
        return ["[EMPTY]"]

    tokens = TOKEN_PATTERN.findall(stripped)
    return tokens or ["[EMPTY]"]


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
) -> dict[str, np.ndarray]:
    sequence_length = inputs.shape[0]
    head_dim = embedding_dim // num_heads

    attention_scores = []
    attention_weights = []
    queries = []
    keys = []
    values = []
    head_outputs = []

    for head_index in range(num_heads):
        query_projection = _random_matrix(
            embedding_dim,
            head_dim,
            "query",
            layer_index,
            head_index,
            seed,
        )
        key_projection = _random_matrix(
            embedding_dim,
            head_dim,
            "key",
            layer_index,
            head_index,
            seed,
        )
        value_projection = _random_matrix(
            embedding_dim,
            head_dim,
            "value",
            layer_index,
            head_index,
            seed,
        )

        query = inputs @ query_projection
        key = inputs @ key_projection
        value = inputs @ value_projection

        scores = (query @ key.T) / np.sqrt(max(head_dim, 1))
        weights = softmax(scores, axis=-1)
        head_output = weights @ value

        queries.append(query)
        keys.append(key)
        values.append(value)
        attention_scores.append(scores)
        attention_weights.append(weights)
        head_outputs.append(head_output)

    concatenated_heads = np.concatenate(head_outputs, axis=-1)
    output_projection = _random_matrix(
        embedding_dim,
        embedding_dim,
        "output",
        layer_index,
        seed,
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
        "queries": np.stack(queries),
        "keys": np.stack(keys),
        "values": np.stack(values),
        "attention_scores": np.stack(attention_scores),
        "attention_weights": np.stack(attention_weights),
        "head_outputs": np.stack(head_outputs),
        "multi_head_output": multi_head_output,
        "residual_1": residual_1,
        "norm_1": norm_1,
        "ffn_hidden": ffn_hidden,
        "ffn_output": ffn_output,
        "residual_2": residual_2,
        "norm_2": norm_2,
        "attention_rollup": np.mean(np.stack(attention_weights), axis=0),
        "sequence_length": np.array(sequence_length),
        "head_dim": np.array(head_dim),
    }


def run_transformer_pipeline(
    text: str,
    num_layers: int,
    requested_heads: int,
    embedding_dim: int,
    seed: int = 7,
) -> dict[str, object]:
    tokens = tokenize_text(text)
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

    return {
        "raw_text": text,
        "whitespace_tokens": text.split(),
        "tokens": tokens,
        "vocabulary": vocabulary,
        "token_ids": token_ids,
        "embeddings": embeddings,
        "positional_encoding": positions,
        "input_with_position": input_with_position,
        "layers": layers,
        "representations": representations,
        "final_representation": current,
        "num_layers": num_layers,
        "num_heads": num_heads,
        "embedding_dim": embedding_dim,
        "head_dim": embedding_dim // num_heads,
        "seed": seed,
        "initial_similarity": cosine_similarity_matrix(input_with_position),
        "final_similarity": cosine_similarity_matrix(current),
    }
