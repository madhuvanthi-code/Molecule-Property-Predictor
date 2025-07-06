import streamlit as st
from rdkit import Chem
from src.models import GCNModel
from src.rdkit_utils import mol_to_graph_data_obj
import torch

@st.cache_resource
def load_model():
    model = GCNModel(hidden_channels=64)
    model.load_state_dict(torch.load("model.pt", map_location="cpu"))
    model.eval()
    return model

model = load_model()

st.title("Molecular Property Predictor (QM9)")

smiles = st.text_input("Enter a SMILES string:", value="CCO")

if smiles:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        st.error("Invalid SMILES string.")
    else:
        data = mol_to_graph_data_obj(mol)
        data.batch = torch.zeros(data.num_nodes, dtype=torch.long)
        pred = model(data)
        st.success(f"Predicted Property: {pred.item():.4f}")