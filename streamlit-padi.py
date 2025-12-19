import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(
    page_title="Prediksi Produksi Padi",
    page_icon="🌾",
    layout="wide"
)

st.markdown("""
<style>
/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #0f172a;
}
.sidebar-btn {
    display: block;
    padding: 14px;
    margin-bottom: 10px;
    border-radius: 10px;
    background: #1e293b;
    color: white;
    text-align: left;
    font-size: 16px;
    cursor: pointer;
    border: none;
    width: 100%;
}
.sidebar-btn:hover {
    background: #2563eb;
}
.active {
    background: #2563eb !important;
}

/* Card */
.card {
    background: #ffffff;
    padding: 20px;
    border-radius: 14px;
    box-shadow: 0px 4px 14px rgba(0,0,0,0.08);
}

/* Metric */
.metric-box {
    background: #f8fafc;
    padding: 20px;
    border-radius: 14px;
    text-align: center;
}
.metric-box h2 {
    margin: 0;
}
</style>
""", unsafe_allow_html=True)

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

def format_id(x):
    return f"{x:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")

if "menu" not in st.session_state:
    st.session_state.menu = "Beranda"

with st.sidebar:
    if st.button("🏠 Beranda", key="beranda"):
        st.session_state.menu = "Beranda"
    if st.button("📊 EDA", key="eda"):
        st.session_state.menu = "EDA"
    if st.button("🔮 Prediksi", key="prediksi"):
        st.session_state.menu = "Prediksi"

menu = st.session_state.menu

# ==================================================
# BERANDA
# ==================================================
if menu == "Beranda":
    st.title("🌾 Prediksi Produksi Padi")

    st.markdown("""
    Aplikasi ini menggunakan **Regresi Linier Berganda**  
    untuk memprediksi **Produksi Padi Kabupaten Purwakarta**
    berdasarkan:
    - Luas Panen  
    - Produktivitas  
    - Tadah Hujan  
    - Irigasi  
    """)

    st.divider()

    # === EVALUASI MODEL ===
    y_pred = model.predict(X)
    mae = mean_absolute_error(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    r2 = r2_score(y, y_pred)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"<div class='metric-box'><h4>MAE</h4><h2>{format_id(mae)}</h2></div>", unsafe_allow_html=True)
    with col2:
        st.markdown(f"<div class='metric-box'><h4>RMSE</h4><h2>{format_id(rmse)}</h2></div>", unsafe_allow_html=True)
    with col3:
        st.markdown(f"<div class='metric-box'><h4>R² Score</h4><h2>{r2:.4f}</h2></div>", unsafe_allow_html=True)

    st.divider()

    st.markdown("### 📘 Dataset Produksi Padi di Kabupaten Purwakarta 2021-2022")
    st.dataframe(df, use_container_width=True)

# ==================================================
# EDA
# ==================================================
elif menu == "EDA":
    st.title("📊 Exploratory Data Analysis")

    eda_type = st.selectbox(
        "Pilih Visualisasi",
        ["Line Plot vs Tahun", "Scatter Plot", "Box Plot", "Correlation Matrix"]
    )

    # LINE PLOT
    if eda_type == "Line Plot vs Tahun":
        fitur = st.selectbox(
            "Pilih Variabel",
            ['luas_panen', 'produktivitas', 'tadah_hujan', 'irigasi', 'produksi']
        )

        trend = df.groupby("tahun")[fitur].mean().reset_index()

        fig, ax = plt.subplots(figsize=(8,5))
        sns.lineplot(data=trend, x="tahun", y=fitur, marker="o", ax=ax)
        ax.set_title(f"{fitur} terhadap Tahun")
        st.pyplot(fig)

    # SCATTER
    elif eda_type == "Scatter Plot":
        x_var = st.selectbox("Pilih X", X.columns)
        fig, ax = plt.subplots(figsize=(8,5))
        sns.scatterplot(data=df, x=x_var, y="produksi", ax=ax)
        st.pyplot(fig)

    # BOX
    elif eda_type == "Box Plot":
        box_var = st.selectbox("Pilih Variabel", X.columns.tolist() + ["produksi"])
        fig, ax = plt.subplots(figsize=(6,5))
        sns.boxplot(y=df[box_var], ax=ax)
        st.pyplot(fig)

    # CORRELATION
    elif eda_type == "Correlation Matrix":
        fig, ax = plt.subplots(figsize=(7,6))
        sns.heatmap(df[X.columns.tolist() + ["produksi"]].corr(), annot=True, cmap="viridis", fmt=".2f")
        st.pyplot(fig)

# ==================================================
# PREDIKSI
# ==================================================
elif menu == "Prediksi":
    st.title("🔮 Prediksi Produksi Padi")

    col1, col2 = st.columns(2)

    with col1:
        luas_panen = st.number_input("Luas Panen", value=0.0)
        produktivitas = st.number_input("Produktivitas", value=0.0)

    with col2:
        tadah_hujan = st.number_input("Tadah Hujan", value=0.0)
        irigasi = st.number_input("Irigasi", value=0.0)

    if st.button("🔍 Prediksi Produksi"):
        input_data = np.array([[luas_panen, produktivitas, tadah_hujan, irigasi]])
        hasil = model.predict(input_data)[0]

        st.success(f"🌾 **Prediksi Produksi Padi: {format_id(hasil)} Ton**")
