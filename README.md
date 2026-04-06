# TransformerVisualizer

Interactive Streamlit app for exploring how Transformer models turn text into tokens, build positional awareness, compute attention, and refine contextual representations across layers.

This project is designed as a teaching-focused Transformer simulator rather than a trained language model. It uses deterministic weights so the same architecture can be explored repeatedly while users change the sentence, number of layers, number of heads, and embedding dimension.

## What the app covers

- Tokenization and embedding inspection
- Positional encoding overlays
- Self-attention and multi-head attention maps
- Feedforward transformation views
- Residual connection and layer normalization comparisons
- Layer-by-layer representation tracking

## Streamlit layout

- Sidebar controls for input text, layers, heads, embedding size, visualization toggles, and simulation seed
- Seven learning tabs that map directly to the core Transformer concepts in the assignment
- Short explanations and a key takeaway in every section so the app teaches while it visualizes

## Local setup

1. Create a virtual environment:

```bash
python3 -m venv .venv
```

2. Install dependencies:

```bash
.venv/bin/pip install -r requirements.txt
```

3. Launch the Streamlit app:

```bash
.venv/bin/streamlit run app.py
```

## Implementation notes

- `app.py` builds the interactive Streamlit interface and all visual components
- `transformer_core.py` contains tokenization, positional encoding, self-attention, multi-head attention, FFN, residual, normalization, and stacked-layer simulation
- The simulator automatically adjusts the number of heads so the embedding dimension splits evenly across attention heads
