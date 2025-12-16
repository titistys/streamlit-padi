import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def format_id(angka):
    return f"{angka:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")


# ==================================================
# PAGE CONFIG
# ==================================================
st.set_page_config(
    page_title="Prediksi Produksi Padi",
    page_icon="🌾",
    layout="wide"
)

# ==================================================
# LOAD DATA
# ==================================================
@st.cache_data
def load_data():
    return pd.read_csv("produksi_padi_final.csv")

df = load_data()

# ==================================================
# MODEL
# ==================================================
X = df[['luas_panen', 'produktivitas', 'tadah_hujan', 'irigasi']]
y = df['produksi']

model = LinearRegression()
model.fit(X, y)

# ==================================================
# SIDEBAR NAVIGATION
# ==================================================
menu = st.sidebar.radio(
    "📌 MENU",
    ["🏠 Beranda", "📊 EDA", "🔮 Prediksi"]
)

# ==================================================
# BERANDA
# ==================================================
if menu == "🏠 Beranda":
    st.title("🌾 Prediksi Produksi Padi")
    st.markdown("""
    Aplikasi ini menggunakan **Regresi Linier Berganda** untuk memprediksi  
    **Produksi Padi Kabupaten Purwakarta** berdasarkan variabel:

    - **Luas Panen**
    - **Produktivitas**
    - **Tadah Hujan**
    - **Irigasi**

    ### Fitur Aplikasi:
    ✅ Exploratory Data Analysis (EDA) Interaktif  
    ✅ Visualisasi Tren Produksi  
    ✅ Prediksi Produksi Padi  
    ✅ Model Regresi Linier Berganda  

    > Dataset telah melalui tahap preprocessing dan siap digunakan.
    """)

# ==================================================
# EDA
# ==================================================
elif menu == "📊 EDA":
    st.title("📊 Exploratory Data Analysis (EDA)")

    eda_type = st.selectbox(
        "Pilih Jenis Visualisasi",
        [
            "Line Plot (vs Tahun)",
            "Scatter Plot",
            "Box Plot",
            "Correlation Matrix"
        ]
    )

    # ---------------- LINE PLOT ----------------
    if eda_type == "Line Plot (vs Tahun)":
        st.subheader("📈 Tren Variabel terhadap Tahun")

        fitur_line = st.selectbox(
            "Pilih Variabel",
            ['luas_panen', 'produktivitas', 'tadah_hujan', 'irigasi', 'produksi']
        )

        data_trend = df.groupby('tahun')[fitur_line].mean().reset_index()

        fig, ax = plt.subplots(figsize=(8,5))
        sns.lineplot(
            data=data_trend,
            x='tahun',
            y=fitur_line,
            marker='o',
            ax=ax
        )
        ax.set_title(f"Tren {fitur_line} terhadap Tahun")
        ax.set_xlabel("Tahun")
        ax.set_ylabel(fitur_line)
        st.pyplot(fig)

    # ---------------- SCATTER ----------------
    elif eda_type == "Scatter Plot":
        st.subheader("🔍 Scatter Plot terhadap Produksi")

        x_var = st.selectbox(
            "Pilih Variabel X",
            ['luas_panen', 'produktivitas', 'tadah_hujan', 'irigasi']
        )

        fig, ax = plt.subplots(figsize=(8,5))
        sns.scatterplot(
            data=df,
            x=x_var,
            y='produksi',
            ax=ax
        )
        ax.set_title(f"{x_var} vs Produksi")
        st.pyplot(fig)

    # ---------------- BOX PLOT ----------------
    elif eda_type == "Box Plot":
        st.subheader("📦 Distribusi Data")

        box_var = st.selectbox(
            "Pilih Variabel",
            ['luas_panen', 'produktivitas', 'tadah_hujan', 'irigasi', 'produksi']
        )

        fig, ax = plt.subplots(figsize=(6,5))
        sns.boxplot(y=df[box_var], ax=ax)
        ax.set_title(f"Distribusi {box_var}")
        st.pyplot(fig)

    # ---------------- CORRELATION ----------------
    elif eda_type == "Correlation Matrix":
        st.subheader("🧮 Korelasi Antar Variabel")

        numeric_df = df[['luas_panen','produktivitas','tadah_hujan','irigasi','produksi']]

        fig, ax = plt.subplots(figsize=(7,6))
        sns.heatmap(
            numeric_df.corr(),
            annot=True,
            cmap="viridis",
            fmt=".2f",
            ax=ax
        )
        st.pyplot(fig)

# ==================================================
# PREDIKSI
# ==================================================
elif menu == "🔮 Prediksi":
    st.title("🔮 Prediksi Produksi Padi")

    # default nilai prediksi = 0
    if "hasil_prediksi" not in st.session_state:
        st.session_state.hasil_prediksi = 0.0

    col1, col2 = st.columns(2)

    with col1:
        luas_panen = st.number_input("Luas Panen", min_value=0.0, value=0.0)
        produktivitas = st.number_input("Produktivitas", min_value=0.0, value=0.0)

    with col2:
        tadah_hujan = st.number_input("Tadah Hujan", min_value=0.0, value=0.0)
        irigasi = st.number_input("Irigasi", min_value=0.0, value=0.0)

    if st.button("🔍 Prediksi Produksi"):
        input_data = np.array([[luas_panen, produktivitas, tadah_hujan, irigasi]])
        st.session_state.hasil_prediksi = model.predict(input_data)[0]

    st.divider()

    # TAMPILAN HASIL (SELALU ADA, AWALNYA 0)
    st.success(
        f"🌾 **Prediksi Produksi Padi: {format_id(st.session_state.hasil_prediksi)} Ton**"
    )
