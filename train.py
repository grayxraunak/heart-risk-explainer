"""Train, compare and explain heart disease risk models.

Usage:
    python train.py                      # uses data/heart.csv
    python train.py --data path/to.csv

Expects the standard 13-feature heart disease table (UCI / Kaggle "heart.csv")
with a binary `target` column (1 = disease present).
"""
import argparse
import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                             RocCurveDisplay)
from sklearn.model_selection import (StratifiedKFold, cross_val_score,
                                     train_test_split)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

FEATURES = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
            "thalach", "exang", "oldpeak", "slope", "ca", "thal"]
SEED = 42
OUT = Path("models")
FIG = Path("figures")


TEXT_CODES = {
    "sex": {"male": 1, "female": 0},
    "cp": {"typical angina": 0, "atypical angina": 1,
           "non-anginal": 2, "asymptomatic": 3},
    "restecg": {"normal": 0, "st-t abnormality": 1, "lv hypertrophy": 2},
    "slope": {"upsloping": 0, "flat": 1, "downsloping": 2},
    "thal": {"normal": 1, "fixed defect": 2, "reversable defect": 3,
             "reversible defect": 3},
}


def load(path):
    """Load either the classic numeric heart.csv or the Kaggle
    'heart_disease_uci.csv' (text categories, `thalch`, `num`, missing values)."""
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    df = df.rename(columns={"thalch": "thalach", "num": "target"})
    for col, mapping in TEXT_CODES.items():
        if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].astype("string").str.strip().str.lower().map(mapping)
    for col in ["fbs", "exang"]:
        if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].astype("string").str.strip().str.lower().map(
                {"true": 1, "false": 0, "1": 1, "0": 0})
    missing = [c for c in FEATURES + ["target"] if c not in df.columns]
    if missing:
        raise SystemExit(f"Missing columns in CSV: {missing}")
    df = (df[FEATURES + ["target"]].apply(pd.to_numeric, errors="coerce")
          .astype(float))
    all_nan = [c for c in FEATURES if df[c].isna().all()]
    if all_nan:
        raise SystemExit(f"Could not parse these columns (all values missing "
                         f"after conversion): {all_nan}. Check their values.")
    # 0 cholesterol / blood pressure are missing-value placeholders in this data
    df.loc[df["chol"] == 0, "chol"] = np.nan
    df.loc[df["trestbps"] == 0, "trestbps"] = np.nan
    df = df.dropna(subset=["target"]).drop_duplicates()
    df["target"] = (df["target"] > 0).astype(int)  # any disease grade -> 1
    print("Missing values per column (imputed inside the model pipeline):")
    print(df[FEATURES].isna().sum()[lambda s: s > 0].to_string() or "none")
    return df[FEATURES], df["target"]


def main(path):
    OUT.mkdir(exist_ok=True)
    FIG.mkdir(exist_ok=True)
    X, y = load(path)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED)
    print(f"Rows: {len(X)} | train {len(X_tr)} | test {len(X_te)}")

    imp = lambda: SimpleImputer(strategy="median")
    models = {
        "Logistic Regression": make_pipeline(
            imp(), StandardScaler(), LogisticRegression(max_iter=1000)),
        "Random Forest": make_pipeline(imp(), RandomForestClassifier(
            n_estimators=300, random_state=SEED)),
        "Gradient Boosting": make_pipeline(imp(), HistGradientBoostingClassifier(
            max_depth=3, learning_rate=0.05, max_iter=200, random_state=SEED)),
    }

    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    rows = []
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, model in models.items():
        cv_auc = cross_val_score(model, X_tr, y_tr, cv=cv, scoring="roc_auc")
        model.fit(X_tr, y_tr)
        proba = model.predict_proba(X_te)[:, 1]
        pred = (proba >= 0.5).astype(int)
        rows.append({
            "Model": name,
            "CV AUC (mean)": round(cv_auc.mean(), 3),
            "CV AUC (std)": round(cv_auc.std(), 3),
            "Test Accuracy": round(accuracy_score(y_te, pred), 3),
            "Test F1": round(f1_score(y_te, pred), 3),
            "Test AUC": round(roc_auc_score(y_te, proba), 3),
        })
        RocCurveDisplay.from_predictions(y_te, proba, name=name, ax=ax)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set_title("ROC curves (held-out test set)")
    fig.tight_layout()
    fig.savefig(FIG / "roc_curves.png", dpi=150)
    plt.close(fig)

    results = pd.DataFrame(rows).sort_values("Test AUC", ascending=False)
    print("\n" + results.to_markdown(index=False))
    results.to_csv(OUT / "results.csv", index=False)
    (OUT / "results.md").write_text(results.to_markdown(index=False))

    # Pick the best model by cross-validated AUC (not the test set)
    best_name = results.sort_values("CV AUC (mean)", ascending=False).iloc[0]["Model"]
    best = models[best_name]
    print(f"\nBest by CV AUC: {best_name}")
    joblib.dump(best, OUT / "best_model.joblib")
    X_tr.sample(min(100, len(X_tr)), random_state=SEED).to_csv(
        OUT / "background.csv", index=False)
    json.dump({"best_model": best_name, "features": FEATURES},
              open(OUT / "meta.json", "w"))

    # Global explainability with SHAP (model-agnostic, works for any best model)
    bg = X_tr.sample(min(100, len(X_tr)), random_state=SEED)
    explainer = shap.Explainer(lambda d: best.predict_proba(
        pd.DataFrame(d, columns=FEATURES))[:, 1], bg)
    sv = explainer(X_te)
    plt.figure()
    shap.summary_plot(sv.values, X_te, show=False)
    plt.title(f"SHAP feature impact ({best_name})")
    plt.tight_layout()
    plt.savefig(FIG / "shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close()

    plt.figure()
    shap.summary_plot(sv.values, X_te, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(FIG / "shap_importance.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved models/ and figures/")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/heart.csv")
    main(ap.parse_args().data)
