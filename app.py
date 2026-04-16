import streamlit as st
import pandas as pd
import google.generativeai as genai
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Konfigurasi Halaman
st.set_page_config(page_title="AI Archiver - Muna Barat", page_icon="🧠", layout="wide")

# --- STYLE (Koreksi pada unsafe_allow_html) ---
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #007bff; color: white; }
    .result-card { padding: 20px; border-radius: 10px; background-color: white; border-left: 5px solid #007bff; margin-bottom: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    </style>
    """, unsafe_allow_html=True) # <-- SUDAH DIPERBAIKI DI SINI

# --- SIDEBAR ---
with st.sidebar:
    st.title("⚙️ Pengaturan")
    api_key = st.text_input("Google API Key", type="password", help="Dapatkan di Google AI Studio")
    st.info("Aplikasi ini menggunakan dataset Klasifikasi Arsip yang sudah di-upgrade untuk akurasi maksimal.")

# --- LOAD DATA ---
@st.cache_data
def load_data():
    # Pastikan file ini sudah ada di repo GitHub kamu
    df = pd.read_csv('klasifikasi_arsip.csv')
    return df

try:
    df_arsip = load_data()
except Exception as e:
    st.error(f"Gagal memuat database arsip. Pastikan file CSV tersedia di repo. Error: {e}")
    st.stop()

# --- HEADER ---
st.title("🧠 AI Penentu Kode Klasifikasi Arsip")
st.subheader("Kabupaten Muna Barat")

# --- INPUT ---
user_input = st.text_area("✍️ Masukkan Perihal/Deskripsi Arsip:", placeholder="Contoh: Permohonan izin cuti tahunan...")

if st.button("Analisis Kontekstual"):
    if not api_key:
        st.warning("⚠️ Silakan masukkan API Key di sidebar terlebih dahulu.")
    elif not user_input:
        st.warning("⚠️ Masukkan deskripsi arsip terlebih dahulu.")
    else:
        try:
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel('gemini-1.5-flash')

            # Mengambil sampel context untuk referensi AI
            sample_context = df_arsip['ai_search_context'].head(500).tolist()
            
            prompt = f"""
            Anda adalah seorang Arsiparis Ahli. Tentukan 3 Kode Klasifikasi Arsip yang paling relevan.
            
            REFERENSI DATA:
            {sample_context[:30]}

            INPUT USER: "{user_input}"

            Berikan hasil dalam format daftar yang rapi dengan Kode, Uraian, dan Alasan mengapa kode tersebut dipilih.
            """

            with st.spinner('Sedang menganalisis...'):
                response = model.generate_content(prompt)
                st.success("✅ Analisis Selesai!")
                st.markdown(response.text)

        except Exception as e:
            st.error(f"Terjadi kendala: {e}")

# --- FOOTER ---
st.divider()
st.caption("Dinas Perpustakaan dan Kearsipan Kabupaten Muna Barat.")
