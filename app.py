import streamlit as st
import torch
import pandas as pd
from src.models import GCNModel

@st.cache_resource
def load_model():
    model = GCNModel(hidden_channels=64)
    model.load_state_dict(torch.load("model.pt", map_location="cpu"))
    model.eval()
    return model

model = load_model()

st.title("Molecular Property Predictor")

st.markdown("""
Upload a CSV file with molecular features. 
Example format:
```
feature1,feature2,feature3,...
0.12,0.34,0.56,...
```
""")

uploaded_file = st.file_uploader("Upload CSV file", type="csv")

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.write("Preview:", df.head())

        if st.button("Predict"):
            inputs = torch.tensor(df.values, dtype=torch.float32)
            preds = model(inputs).detach().numpy()
            df['Predicted Property'] = preds
            st.write("Predictions:", df)
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("Download Predictions", csv, "predictions.csv", "text/csv")
    except Exception as e:
        st.error(f"Error processing file: {e}")