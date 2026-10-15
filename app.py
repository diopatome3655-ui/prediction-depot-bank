"""
Application Streamlit — Classification sur Bank_Data (KNN / Régression logistique)
Adaptée de l'application "état d'un portable".

Lancement en local :  streamlit run app.py
Fichier requis dans le même dossier : bank_artifacts.joblib (créé depuis Colab)
"""

import numpy as np
import pandas as pd
import joblib as jb
import streamlit as st

st.set_page_config(page_title="Prédiction Bank_Data", page_icon="🏦", layout="centered")

DESCRIPTION = (
    "Ce modèle de machine learning (KNN ou régression logistique) prédit la classe "
    "d'un client à partir de ses caractéristiques."
)


@st.cache_resource
def load_artifacts():
    return jb.load("bank_artifacts.joblib")


art = load_artifacts()
features = art["features"]          # liste ordonnée des colonnes d'entrée
cat_encoders = art["encoders"]      # {colonne_catégorielle: LabelEncoder}
scaler = art["scaler"]              # normaliseur (ou None)
models = art["models"]              # {"KNN": modèle, "Régression logistique": modèle}
classnames = art["classes"]         # noms des classes


def predict_df(df, model):
    """Prédit pour un DataFrame contenant les colonnes `features`."""
    X = df[features].copy()
    for col, enc in cat_encoders.items():
        X[col] = enc.transform(X[col])
    X = X.values.astype(float)
    if scaler is not None:
        X = scaler.transform(X)
    y = model.predict(X)
    return [classnames[int(i)] for i in y]


st.title("🏦 Prédiction sur Bank_Data")
choix_modele = st.selectbox("Modèle", list(models.keys()))
model = models[choix_modele]

onglet1, onglet2 = st.tabs(["Prédiction simple", "Prédiction multiple"])

# ----------------------------- Onglet 1 -------------------------------
with onglet1:
    st.subheader("Prédiction avec une seule entrée")
    st.write(DESCRIPTION)

    with st.form("formulaire_simple"):
        valeurs = {}
        cols = st.columns(2)
        for i, f in enumerate(features):
            with cols[i % 2]:
                if f in cat_encoders:
                    valeurs[f] = st.selectbox(f, options=list(cat_encoders[f].classes_))
                else:
                    valeurs[f] = st.number_input(f, value=0.0, format="%.2f")
        soumettre = st.form_submit_button("Prédire", type="primary")

    if soumettre:
        try:
            resultat = predict_df(pd.DataFrame([valeurs]), model)[0]
            st.success(f"**Classe prédite :** {resultat}")
        except Exception as e:
            st.error(f"Erreur lors de la prédiction : {e}")

# ----------------------------- Onglet 2 -------------------------------
with onglet2:
    st.subheader("Prédiction avec plusieurs entrées")
    st.write(DESCRIPTION)
    st.caption("Le fichier CSV doit contenir les colonnes : " + ", ".join(features))

    fichier = st.file_uploader("Importer un fichier CSV", type=["csv"])

    if fichier is not None:
        try:
            df = pd.read_csv(fichier)
            manquantes = [c for c in features if c not in df.columns]
            if manquantes:
                st.error(f"Colonnes manquantes : {manquantes}")
            else:
                with st.spinner("Prédictions en cours…"):
                    df["prediction"] = predict_df(df, model)
                st.success(f"{len(df)} prédiction(s) effectuée(s).")
                st.dataframe(df, use_container_width=True)
                st.download_button(
                    "⬇️ Télécharger le fichier CSV",
                    data=df.to_csv(index=False).encode("utf-8"),
                    file_name="predictions.csv",
                    mime="text/csv",
                    type="primary",
                )
        except Exception as e:
            st.error(f"Erreur lors du traitement du fichier : {e}")
