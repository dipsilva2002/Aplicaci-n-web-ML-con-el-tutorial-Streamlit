import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import os
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from src.utils import get_env_str

@st.cache_resource
def load_model_and_meta():
    model_path = get_env_str("MODEL_PATH", "models/iris_clf.joblib")
    meta_path = "models/iris_meta.json"
    if not os.path.exists(model_path):
        st.error(f"No existe el modelo en {model_path}. Entrena con: python -m src.train")
        st.stop()
    if not os.path.exists(meta_path):
        st.error(f"No existe el archivo de metadatos {meta_path}. Entrena con: python -m src.train")
        st.stop()
    model = joblib.load(model_path)
    with open(meta_path) as f:
        meta = json.load(f)
    return model, meta

def predict_one(model, features: dict, feature_order: list):
    x = np.array([features[f] for f in feature_order], dtype=float).reshape(1, -1)
    pred = model.predict(x)[0]
    proba = model.predict_proba(x)[0]
    return int(pred), proba

def page_home(model, meta):
    st.header("Predicción individual")
    st.caption("Iris dataset")

    fns = meta["feature_names"]
    defaults = {
        "sepal length (cm)": 5.1,
        "sepal width (cm)": 3.5,
        "petal length (cm)": 1.4,
        "petal width (cm)": 0.2
    }
    values = {}
    cols = st.columns(2)
    for i, fname in enumerate(fns):
        with cols[i % 2]:
            values[fname] = st.number_input(
                fname, value=float(defaults.get(fname, 1.0)), step=0.1, format="%.2f"
            )

    if st.button("Predecir"):
        pred, proba = predict_one(model, values, fns)
        classes = meta["target_names"]
        st.success(f"Predicción: {classes[pred]}")
        proba_df = pd.DataFrame([proba], columns=classes)
        st.bar_chart(proba_df.T)

def page_batch(model, meta):
    st.header("Predicción por archivo CSV")
    st.caption("El CSV debe tener las columnas en el mismo orden que el modelo.")

    fns = meta["feature_names"]
    st.write("Columnas esperadas:", ", ".join(fns))
    uploaded = st.file_uploader("Sube un CSV", type=["csv"])
    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded)
        except Exception as e:
            st.error(f"No se pudo leer el CSV: {e}")
            return
        missing = [c for c in fns if c not in df.columns]
        if missing:
            st.error(f"Faltan columnas: {missing}")
            return
        X = df[fns].astype(float).values
        y_pred = model.predict(X)
        y_proba = model.predict_proba(X)
        out = df.copy()
        out["prediction"] = [meta["target_names"][i] for i in y_pred]
        st.write("Resultados")
        st.dataframe(out.head())
        st.download_button(
            label="Descargar resultados",
            data=out.to_csv(index=False),
            file_name="predicciones.csv",
            mime="text/csv"
        )

def main():
    st.set_page_config(page_title="Streamlit ML App", page_icon="🔎", layout="centered")
    st.title("Aplicación Web de ML con Streamlit")
    model, meta = load_model_and_meta()

    tab1, tab2 = st.tabs(["Interactivo", "CSV"])
    with tab1:
        page_home(model, meta)
    with tab2:
        page_batch(model, meta)

if __name__ == "__main__":
    main()
