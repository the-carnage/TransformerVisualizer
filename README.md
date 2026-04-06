# ⚡ Transformer Visualizer — Interactive Lab

> **Watch tokens build context layer by layer.**

An interactive Streamlit application that lets you explore *exactly* how a Transformer model processes text — from raw tokenization to deep contextual representations across stacked layers.

Built as a **teaching-focused simulator** with deterministic weights so you can change the input, architecture, and seed to see how each component behaves.

---

## 🎯 What You'll Learn

| Tab | Concept | Key Visual |
|-----|---------|-----------|
| 🔤 Tokenization | Text → tokens → embeddings | Color-coded token chips + embedding heatmap |
| 📍 Positional Encoding | Sinusoidal position signals | Wave patterns + before/after overlay |
| 🎯 Self-Attention | Q·K scores → softmax weights | **Attention arc diagram** + heatmaps |
| 🧠 Multi-Head Attention | Parallel relationship capture | Head diversity score + comparison grid |
| ⚡ FFN | Expand → GELU → compress | GELU curve + dimension flow bars |
| 🔄 Residual + Norm | Skip connections + LayerNorm | Distribution histograms + norm tracking |
| 📊 Layer-wise | Embedding evolution | Token journey + drift map + PCA scatter |

---

## ✨ Features

- **Premium dark UI** — glassmorphism, gradient accents, micro-animations
- **7 interactive tabs** covering all core Transformer concepts
- **Attention arc diagrams** — curved Bezier arcs showing token-to-token attention flow
- **Architecture diagram** — collapsible Transformer block diagram highlighting the active component
- **GELU activation curve** with actual token activations highlighted
- **Distribution histograms** showing pre/post normalization shifts
- **Token journey** — watch an embedding evolve across layers
- **Head diversity metrics** — Jensen-Shannon divergence across attention heads
- **Token importance scores** — which tokens receive the most attention
- **Deterministic simulation** — fixed-seed weights for reproducible exploration
- **Fully configurable** — layers, heads, embedding dim, seed, display toggles

---

## 🏗️ Architecture

```
app.py                          # Streamlit UI + chart helpers
transformer_core.py             # Tokenization, attention, FFN, residual, metrics
components/
├── __init__.py
├── attention_arcs.py           # Bezier arc attention flow diagram
├── architecture_diagram.py     # Transformer block architecture visualization
└── flow_animations.py          # GELU curve, histograms, dimension flow, waves
.streamlit/
└── config.toml                 # Dark theme + production server settings
```

---

## 🚀 Quick Start

### 1. Clone & create environment

```bash
git clone <repo-url>
cd TransformerVisualizer
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Launch

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 🎛️ Controls

| Control | Range | Default | Effect |
|---------|-------|---------|--------|
| Input text | Any sentence | Long demo text | Changes the token sequence |
| Layers | 1–6 | 3 | Number of stacked Transformer blocks |
| Attention heads | 1–8 | 4 | Auto-adjusted to divide embedding dim evenly |
| Embedding dim | 16–128 | 64 | Width of token vectors (step: 16) |
| Simulation seed | 1–999 | 7 | Deterministic weight initialization |
| Tracked token | Any token | First | Token to follow across visualizations |
| Inspection layer | 1–N | 1 | Layer for detailed inspection |
| Inspection head | 1–H | 1 | Head for detailed inspection |

---

## 📦 Requirements

- Python 3.10+
- streamlit ≥ 1.44
- numpy ≥ 1.26
- pandas ≥ 2.2
- plotly ≥ 5.24
- scipy ≥ 1.13

---

## 🧪 How It Works

The simulator uses a **deterministic mini-Transformer** — no training, no GPU required. Weights are derived from the simulation seed via SHA-256 hashing, so:

- Same seed + same architecture = same weights
- Change the input text to see how the same model handles different sequences
- Compare different architectures (layers, heads, dim) to see structural effects

Each Transformer block applies:
1. **Multi-head self-attention** (Q·K·V projections, softmax, weighted sum)
2. **Residual + LayerNorm** (skip connection + normalization)
3. **FFN** (expand with GELU, compress back)
4. **Residual + LayerNorm** (second skip + normalization)

---

## 🏆 Built For

This project was designed as a hackathon-ready, interactive teaching tool for understanding Transformer architectures — covering all 7 core concepts from tokenization to deep contextual representation.
