
import pandas as pd
import streamlit as st

st.set_page_config(page_title="ROCO Classification", layout="wide")
st.title("ROCO Radiology Multimodal Classification")
st.caption("timm/resnet50_clip.openai vs CLIP ViT-B/32 | zero-shot vs few-shot classification")

results = pd.read_csv("outputs/model_comparison.csv")
st.subheader("Model comparison")
st.dataframe(results, use_container_width=True)

st.subheader("Accuracy and F1")
st.bar_chart(results.set_index(["model", "shots"])[["accuracy", "f1_macro", "f1_weighted"]])

st.subheader("Classification reports")
for _, row in results.iterrows():
    slug = row["model"].lower().replace("/", "_").replace("-", "_").replace(" ", "_")
    path = f"outputs/classification_report_{slug}_{int(row['shots'])}shot.csv"
    try:
        st.markdown(f"### {row['model']} | {int(row['shots'])}-shot")
        st.dataframe(pd.read_csv(path), use_container_width=True)
    except Exception:
        pass
