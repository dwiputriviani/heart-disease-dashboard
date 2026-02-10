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
    return pd.read_csv("heart_disease.csv")

df = load_data()

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

    model = LogisticRegression()
    model.fit(X_train_scaled, y_train)

    accuracy = accuracy_score(y_test, model.predict(X_test_scaled))
    return model, scaler, accuracy, X.columns

model, scaler, accuracy, feature_names = train_model(df)

# =========================
# SIDEBAR
# =========================
st.sidebar.title("📌 Navigasi")
menu = st.sidebar.radio(
    "Pilih Halaman",
    ["Overview", "EDA", "Prediksi"]
)

st.sidebar.markdown("---")
st.sidebar.caption("Streamlit Dashboard\nMachine Learning Project")

# =========================
# OVERVIEW
# =========================
if menu == "Overview":
    st.title("💓 Heart Disease Dashboard")

    st.write(
        "Dashboard ini digunakan untuk menganalisis data kesehatan pasien "
        "dan memprediksi risiko penyakit jantung menggunakan Machine Learning."
    )

    st.markdown("### 📊 Ringkasan Data")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Jumlah Data", df.shape[0])

    with col2:
        st.metric("Jumlah Fitur", df.shape[1] - 1)

    with col3:
        st.metric("Rata-rata Usia", round(df["age"].mean(), 1))

    with col4:
        st.metric("Akurasi Model", f"{accuracy:.2f}")

    st.markdown("---")

    st.markdown(
        """
        **Tujuan Project:**
        - Memahami karakteristik pasien
        - Mengidentifikasi pola penyakit jantung
        - Membuat model prediksi sederhana
        """
    )

# =========================
# EDA
# =========================
elif menu == "EDA":
    st.title("📈 Exploratory Data Analysis")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Distribusi Usia Pasien")
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(df["age"], bins=20)
        ax.set_xlabel("Usia")
        ax.set_ylabel("Jumlah")
        st.pyplot(fig, use_container_width=True)

    with col2:
        st.subheader("Distribusi Penyakit Jantung")
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.countplot(x="target", data=df, ax=ax)
        ax.set_xlabel("Target (0 = Tidak, 1 = Ya)")
        ax.set_ylabel("Jumlah")
        st.pyplot(fig, use_container_width=True)

    st.markdown("---")

    st.subheader("Distribusi Jenis Kelamin")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.countplot(x="sex", hue="target", data=df, ax=ax)
    ax.set_xlabel("Jenis Kelamin")
    ax.set_ylabel("Jumlah")
    st.pyplot(fig, use_container_width=True)

# =========================
# PREDICTION
# =========================
elif menu == "Prediksi":
    st.title("🤖 Prediksi Penyakit Jantung")

    st.write(
        "Masukkan data pasien di bawah ini untuk melihat "
        "prediksi risiko penyakit jantung."
    )

    input_data = {}

    col1, col2 = st.columns(2)

    for i, feature in enumerate(feature_names):
        if i % 2 == 0:
            with col1:
                input_data[feature] = st.number_input(
                    feature,
                    float(df[feature].min()),
                    float(df[feature].max()),
                    float(df[feature].mean()),
                    step=0.5,
                    format="%d"
                )
        else:
            with col2:
                input_data[feature] = st.number_input(
                    feature,
                    float(df[feature].min()),
                    float(df[feature].max()),
                    float(df[feature].mean()),
                    step=0.5,
                    format="%.1d"
                )

    input_df = pd.DataFrame([input_data])
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)[0]

    st.markdown("---")

    if prediction == 1:
        st.error("⚠️ Pasien berisiko mengalami penyakit jantung")
    else:
        st.success("✅ Pasien tidak berisiko penyakit jantung")

# =========================
# FOOTER
# =========================
st.markdown("---")
st.caption("Project Data Analysis & Machine Learning | Streamlit")
