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
# FEATURE ENGINEERING (EDA ONLY)
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

    return model, scaler, accuracy, X.columns, X_train.shape[0], X_test.shape[0]

model, scaler, accuracy, feature_names, train_size, test_size = train_model(df)

# =========================
# FEATURE DESCRIPTIONS
# =========================
feature_desc = {
    "age": "Usia pasien (tahun)",
    "sex": "Jenis kelamin (0 = Perempuan, 1 = Laki-laki)",
    "cp": "Tipe nyeri dada (0 = tipikal, 1 = atipikal, 2 = non-angina, 3 = asimtomatik)",
    "trestbps": "Tekanan darah saat istirahat (mmHg)",
    "chol": "Kadar kolesterol serum (mg/dl)",
    "fbs": "Gula darah puasa > 120 mg/dl (1 = Ya, 0 = Tidak)",
    "restecg": "Hasil EKG saat istirahat",
    "thalach": "Detak jantung maksimum yang dicapai",
    "exang": "Nyeri dada akibat olahraga (1 = Ya, 0 = Tidak)",
    "oldpeak": "Penurunan segmen ST akibat aktivitas fisik",
    "slope": "Kemiringan segmen ST saat puncak olahraga",
    "ca": "Jumlah pembuluh darah utama (0–3)",
    "thal": "Kondisi thalassemia (1 = normal, 2 = cacat tetap, 3 = cacat reversibel)"
}

# =========================
# SIDEBAR
# =========================
st.sidebar.title("📌 Navigasi")
menu = st.sidebar.radio("Pilih Halaman", ["Overview", "EDA", "Prediksi"])
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

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Jumlah Data", df.shape[0])
    col2.metric("Jumlah Fitur", df.shape[1] - 1)
    col3.metric("Rata-rata Usia", round(df["age"].mean(), 1))
    col4.metric("Akurasi Model", f"{accuracy:.2f}")

    st.markdown("---")
    col5, col6 = st.columns(2)
    col5.metric("Data Training (80%)", train_size)
    col6.metric("Data Testing (20%)", test_size)

    st.markdown(
        "Dataset dibagi menjadi data latih dan data uji untuk memastikan "
        "evaluasi model dilakukan secara objektif."
    )

# =========================
# EDA
# =========================
elif menu == "EDA":
    st.title("📈 Exploratory Data Analysis")

    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots()
        ax.hist(df["age"], bins=20)
        ax.set_title("Distribusi Usia Pasien")
        st.pyplot(fig)

    with col2:
        fig, ax = plt.subplots()
        sns.countplot(x="target", data=df, ax=ax)
        ax.set_title("Distribusi Penyakit Jantung")
        st.pyplot(fig)

    st.markdown("---")
    fig, ax = plt.subplots()
    sns.countplot(x="sex", hue="target", data=df, ax=ax)
    ax.set_title("Distribusi Jenis Kelamin")
    st.pyplot(fig)

    st.markdown("---")
    st.subheader("🔧 Feature Engineering")
    st.write(
        "Feature engineering dilakukan dengan membuat fitur berbasis rasio "
        "untuk menangkap hubungan proporsional antar variabel medis."
    )
    st.dataframe(
        df_fe[
            ["age", "chol", "thalach",
             "rasio_tinggi_usia",
             "rasio_chol_usia",
             "rasio_chol_thalach"]
        ].head()
    )

# =========================
# PREDICTION
# =========================
elif menu == "Prediksi":
    st.title("🤖 Prediksi Penyakit Jantung")
    st.write("Masukkan data pasien untuk melihat prediksi risiko.")

    input_data = {}
    col1, col2 = st.columns(2)

    for i, feature in enumerate(feature_names):
        col = col1 if i % 2 == 0 else col2
        desc = feature_desc.get(feature, "")

        if df[feature].dtype == "int64":
            with col:
                input_data[feature] = st.number_input(
                    feature,
                    min_value=int(df[feature].min()),
                    max_value=int(df[feature].max()),
                    value=int(round(df[feature].mean())),
                    step=1,
                    format="%d",
                    help=desc
                )
        else:
            with col:
                input_data[feature] = st.number_input(
                    feature,
                    min_value=float(df[feature].min()),
                    max_value=float(df[feature].max()),
                    value=round(float(df[feature].mean()), 1),
                    step=0.5,
                    format="%.1f",
                    help=desc
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
