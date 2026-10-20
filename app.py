import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import shap
import streamlit as st

st.set_page_config(page_title="Heart Risk Explainer", page_icon="🫀")

MODELS = Path("models")
if not (MODELS / "best_model.joblib").exists():
    st.error("No trained model found. Run `python train.py` first.")
    st.stop()


@st.cache_resource
def load_assets():
    model = joblib.load(MODELS / "best_model.joblib")
    meta = json.load(open(MODELS / "meta.json"))
    bg = pd.read_csv(MODELS / "background.csv")
    f = lambda d: model.predict_proba(
        pd.DataFrame(d, columns=meta["features"]))[:, 1]
    return model, meta, shap.Explainer(f, bg)


model, meta, explainer = load_assets()

st.title("🫀 Explainable Heart Disease Risk Predictor")
st.caption(f"Model: {meta['best_model']}. Educational demo only, "
           "not medical advice.")

c1, c2 = st.columns(2)
with c1:
    age = st.slider("Age", 20, 90, 54)
    sex = st.selectbox("Sex", [1, 0], format_func=lambda v: "Male" if v else "Female")
    cp = st.selectbox("Chest pain type", [0, 1, 2, 3], format_func=lambda v: [
        "Typical angina", "Atypical angina", "Non-anginal", "Asymptomatic"][v])
    trestbps = st.slider("Resting blood pressure (mm Hg)", 80, 220, 130)
    chol = st.slider("Cholesterol (mg/dl)", 100, 600, 240)
    fbs = st.selectbox("Fasting blood sugar > 120 mg/dl", [0, 1],
                       format_func=lambda v: "Yes" if v else "No")
    restecg = st.selectbox("Resting ECG", [0, 1, 2], format_func=lambda v: [
        "Normal", "ST-T abnormality", "LV hypertrophy"][v])
with c2:
    thalach = st.slider("Max heart rate achieved", 60, 220, 150)
    exang = st.selectbox("Exercise-induced angina", [0, 1],
                         format_func=lambda v: "Yes" if v else "No")
    oldpeak = st.slider("ST depression (oldpeak)", 0.0, 6.5, 1.0, 0.1)
    slope = st.selectbox("ST slope", [0, 1, 2], format_func=lambda v: [
        "Upsloping", "Flat", "Downsloping"][v])
    ca = st.selectbox("Major vessels colored (0-4)", [0, 1, 2, 3, 4])
    thal = st.selectbox("Thalassemia (thal)", [1, 2, 3], format_func=lambda v: [
        "Normal", "Fixed defect", "Reversible defect"][v - 1])

row = pd.DataFrame([{
    "age": age, "sex": sex, "cp": cp, "trestbps": trestbps, "chol": chol,
    "fbs": fbs, "restecg": restecg, "thalach": thalach, "exang": exang,
    "oldpeak": oldpeak, "slope": slope, "ca": ca, "thal": thal,
}])[meta["features"]]

if st.button("Predict risk", type="primary"):
    p = float(model.predict_proba(row)[0, 1])
    st.metric("Estimated probability of heart disease", f"{p:.0%}")
    st.progress(min(max(p, 0.0), 1.0))
    (st.error if p >= 0.5 else st.success)(
        "Higher risk pattern" if p >= 0.5 else "Lower risk pattern")

    st.subheader("Why this prediction?")
    sv = explainer(row)
    fig = plt.figure()
    shap.plots.waterfall(sv[0], show=False)
    st.pyplot(fig, bbox_inches="tight")
    st.caption("Red bars push risk up, blue bars push it down, "
               "relative to the average patient.")
