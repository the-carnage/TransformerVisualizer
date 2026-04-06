import streamlit as st


st.set_page_config(
    page_title="Transformer Visualizer",
    page_icon="TF",
    layout="wide",
)

st.title("Transformer Visualizer")
st.write(
    "An interactive learning lab for tokenization, positional encoding, attention, "
    "feedforward blocks, residual paths, and layer-wise contextual representations."
)

with st.sidebar:
    st.header("Controls")
    st.text_area("Input text", value="Transformers connect distant words through attention.")
    st.slider("Number of layers", min_value=1, max_value=6, value=3)
    st.slider("Number of attention heads", min_value=1, max_value=8, value=4)
    st.slider("Embedding dimension", min_value=16, max_value=128, value=64, step=16)
    st.toggle("Show attention maps", value=True)
    st.toggle("Show positional encoding", value=True)

st.info("Project scaffold is ready. Interactive Transformer computations land in the next commit.")
