import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import numpy as np
import os
import pickle
from PIL import Image

# --- Load Pickle ---
def load_pickle(file):
    with open(file, 'rb') as f:
        return pickle.load(f)

model_rf = load_pickle('models/model_rf.pkl')
model_xgb = load_pickle('models/model_xgb.pkl')
scaler = load_pickle('models/scaler.pkl')
feature_columns = load_pickle('models/feature_columns.pkl')
le_category = load_pickle('models/label_encoder_category.pkl')
le_gender = load_pickle('models/label_encoder_gender.pkl')
le_job = load_pickle('models/label_encoder_job.pkl')

# --- Konfigurasi Halaman ---
st.set_page_config(page_title="Deteksi Penipuan Kartu Kredit", layout="wide")

# --- Custom CSS Profesional untuk Beranda ---
st.markdown("""
    <style>
    body {
        background-color: #ffffff;
    }
    .stApp {
        font-family: 'Segoe UI', sans-serif;
    }
    h1, h2, h3, .custom-sidebar h2 {
        color: #0d47a1;
        font-weight: 600;
    }
    .center-text {
        text-align: center;
        font-size: 26px;
        font-weight: 600;
        margin-bottom: 20px;
        color: #0d47a1;
    }
    .info-box {
        background-color: #e3f2fd;
        padding: 25px;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        margin-top: 25px;
    }
    ul.custom-list li {
        font-size: 16px;
        margin-bottom: 10px;
    }
    .highlight-box {
        background-color: #f0f8ff;
        border-left: 5px solid #0d47a1;
        padding: 18px;
        margin-top: 30px;
        border-radius: 8px;
        font-size: 16px;
    }
    </style>
""", unsafe_allow_html=True)

# --- Sidebar ---
with st.sidebar:
    logo_path = "logo.jpg"
    if os.path.exists(logo_path):
        logo = Image.open(logo_path)
        st.image(logo, width=220)

    st.markdown('<div class="custom-sidebar"><h2>Menu Utama</h2></div>', unsafe_allow_html=True)

    selected = option_menu(
        menu_title=None,
        options=["Beranda", "Prediksi", "Visualisasi"],
        icons=["house", "search", "bar-chart"],
        default_index=0,
        styles={
            "container": {
                "padding": "5px",
                "background-color": "#f0f2f6",
                "border-radius": "10px"
            },
            "icon": {
                "color": "#1976d2",
                "font-size": "18px"
            },
            "nav-link": {
                "font-size": "16px",
                "color": "#333",
                "padding": "10px 15px",
                "border-radius": "8px"
            },
            "nav-link-selected": {
                "background-color": "#1976d2",
                "color": "white"
            }
        }
    )

# --- Beranda ---
if selected == "Beranda":
    st.title("Beranda")

    if os.path.exists(logo_path):
        st.image(logo_path, use_container_width=True, width=400)

    st.markdown("<div class='center-text'>Aplikasi Deteksi Penipuan Kartu Kredit</div>", unsafe_allow_html=True)

    st.markdown("""
    <div class='info-box'>
        <p style='font-size:18px;'>
        Aplikasi ini membantu mengklasifikasikan transaksi kartu kredit apakah merupakan penipuan atau bukan.
        Dibangun menggunakan dua algoritma machine learning terbaik: <b>Random Forest</b> dan <b>XGBoost</b>.
        </p>
        <h4>🧭 Alur Penggunaan:</h4>
        <ul class='custom-list'>
            <li>Input data transaksi</li>
            <li>Prediksi penipuan otomatis</li>
            <li>Bandingkan hasil dua model</li>
            <li>Lihat visualisasi dan evaluasi</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class='highlight-box'>
        <b>🎯 Performa Model:</b><br>
        Random Forest: Akurasi 97.6% | F1-score: 97.7%<br>
        XGBoost: Akurasi 99.2% | F1-score: 99.2%
    </div>
    """, unsafe_allow_html=True)


# --- Prediksi / Klasifikasi ---
elif selected == "Prediksi":
    st.title("Prediksi Penipuan")
    st.markdown("Masukkan data transaksi untuk memprediksi apakah transaksi tersebut merupakan penipuan atau bukan.")

    col1, col2, col3 = st.columns(3)

    with col1:
        amt = st.number_input("Jumlah Transaksi (USD)", min_value=0.0, value=100.0, step=1.0)
        gender = st.selectbox("Jenis Kelamin", le_gender.classes_)
        city_pop = st.number_input("Populasi Kota", min_value=0, value=3000)

    with col2:
        category = st.selectbox("Kategori Transaksi", le_category.classes_)
        job = st.selectbox("Pekerjaan", le_job.classes_)
        hour = st.slider("Jam Transaksi", 0, 23, 12)

    with col3:
        age = st.slider("Usia Nasabah", 18, 100, 30)
        distance_km = st.number_input("Jarak ke Merchant (km)", min_value=0.0, value=10.0)

    st.markdown("---")

    if st.button("Prediksi"):
        # --- Persiapan Input ---
        input_dict = {
            'amt': amt,
            'category': le_category.transform([category])[0],
            'gender': le_gender.transform([gender])[0],
            'job': le_job.transform([job])[0],
            'city_pop': city_pop,
            'hour': hour,
            'age': age,
            'distance_km': distance_km
        }

        input_df = pd.DataFrame([input_dict])
        input_scaled = scaler.transform(input_df)

        # --- Prediksi ---
        pred_rf = model_rf.predict(input_scaled)[0]
        prob_rf = model_rf.predict_proba(input_scaled)[0][int(pred_rf)]

        pred_xgb = model_xgb.predict(input_scaled)[0]
        prob_xgb = model_xgb.predict_proba(input_scaled)[0][int(pred_xgb)]

        # --- Hasil Prediksi ---
        st.subheader("Hasil Prediksi")
        col_rf, col_xgb = st.columns(2)

        def display_prediction(model_name, pred, prob, acc):
            is_fraud = pred == 1
            box_color = "#ffe6e6" if is_fraud else "#e6f2ff"
            text_color = "#cc0000" if is_fraud else "#0066cc"
            label = "Penipuan" if is_fraud else "Bukan Penipuan"

            st.markdown(f"#### {model_name}")
            st.markdown(f"""
                <div style='
                    background-color:{box_color};
                    padding:16px;
                    border-radius:10px;
                    border-left: 6px solid {text_color};
                    '>
                    <b>Prediksi:</b> <span style='color:{text_color}; font-weight:bold;'>{label}</span><br>
                    <b>Probabilitas:</b> {prob:.2%}
                </div>
            """, unsafe_allow_html=True)

            st.caption(f"Akurasi model: {acc:.1f}%")

        with col_rf:
            display_prediction("Random Forest", pred_rf, prob_rf, 97.6)

        with col_xgb:
            display_prediction("XGBoost", pred_xgb, prob_xgb, 99.2)


# --- Visualisasi ---
elif selected == "Visualisasi":
    st.title("📊 Visualisasi Data")
    st.markdown("Bagian ini dapat digunakan untuk menampilkan visualisasi distribusi data, perbandingan hasil prediksi, dsb.")

    st.warning("⚠️ Visualisasi belum diimplementasikan. Silakan hubungi saya jika ingin menambahkan grafik atau insight lainnya.")

# --- Footer ---
st.markdown("---")
st.caption("© 2025 Aplikasi Deteksi Penipuan Kartu Kredit | Dibuat oleh Della Udya Khairah | Tugas Akhir")
