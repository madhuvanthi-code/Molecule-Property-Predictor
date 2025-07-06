import streamlit as st
import torch
from src.models import GCNModel

@st.cache_resource
def load_model():
    model = GCNModel(hidden_channels=64)
    model.load_state_dict(torch.load("model.pt", map_location="cpu"))
    model.eval()
    return model

@st.cache_resource
def load_example_data():
    return torch.load("example_graph.pt", map_location="cpu")

model = load_model()
data = load_example_data()
data.batch = torch.zeros(data.num_nodes, dtype=torch.long)

st.title("Molecular Property Predictor (Demo)")
if st.button("Predict Example Molecule"):
    pred = model(data)
    st.success(f"Predicted Property: {pred.item():.4f}")
