import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Heart Disease Dashboard",
    layout="wide"
)

# =========================
# LOAD DATA
# =========================
@st.cache_data
def load_data():
    try:
        return pd.read_csv("heart_disease.csv")
    except Exception as e:
        st.error(f"Gagal load data: {e}")
        st.stop()

df = load_data()

# =========================
# FEATURE ENGINEERING
# =========================
df_fe = df.copy()

df_fe["rasio_tinggi_usia"] = df_fe["thalach"] / df_fe["age"]
df_fe["rasio_chol_usia"] = df_fe["chol"] / df_fe["age"]
df_fe["rasio_chol_thalach"] = df_fe["chol"] / df_fe["thalach"]

# =========================
# TRAIN MODEL
# =========================
@st.cache_resource
def train_model(data):
    X = data.drop("target", axis=1)
    y = data["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_scaled, y_train)

    accuracy = accuracy_score(y_test, model.predict(X_test_scaled))

    return (
        model,
        scaler,
        accuracy,
        X.columns,
        X_train.shape[0],
        X_test.shape[0]
    )

model, scaler, accuracy, feature_names, train_size, test_size = train_model(df)

# =========================
# SIDEBAR
# =========================
st.sidebar.title("📌 Navigasi")
menu = st.sidebar.radio(
    "Pilih Halaman",
    ["Overview", "EDA", "Prediksi"]
)

st.sidebar.markdown("---")
st.sidebar.caption("Streamlit Dashboard\nMachine Le
