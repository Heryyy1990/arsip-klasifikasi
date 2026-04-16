import streamlit as st
import pandas as pd
import google.generativeai as genai
import os

# 1. Konfigurasi Halaman Dasar
st.set_page_config(page_title="AI Arsip Muna Barat", layout="centered")

st.title("📂 Penentu Klasifikasi Arsip")
st.caption("Dinas Perpustakaan dan Kearsipan Kabupaten Muna Barat")

# --- SISTEM CEK MANDIRI (Agar tidak langsung eror) ---

# Cek File CSV
data_file = 'klasifikasi_arsip.csv'
if not os.path.exists(data_file):
    st.error(f"❌ File '{data_file}' tidak ditemukan di GitHub kamu. Pastikan nama filenya sama persis.")
    st.stop()

# Cek API Key di Secrets
if "GEMINI_API_KEY" not in st.secrets:
    st.error("❌ API Key belum diatur! Masuk ke Settings > Secrets di Streamlit Cloud, lalu masukkan: GEMINI_API_KEY = 'KUNCI_KAMU'")
    st.stop()

# --- LOAD DATA ---
@st.cache_data
def load_data():
    return pd.read_csv(data_file)

df = load_data()

# --- INISIALISASI AI ---
try:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    # Menggunakan model 1.5-flash
    model = genai.GenerativeModel('gemini-1.5-flash')
except Exception as e:
    st.error(f"Gagal memuat AI: {e}")
    st.stop()

# --- TAMPILAN INPUT ---
query = st.text_area("Masukkan Perihal Surat:", placeholder="Contoh: Permohonan cuti tahunan...")

if st.button("Cek Kode Sekarang"):
    if not query:
        st.warning("Silakan tulis perihal suratnya dulu.")
    else:
        with st.spinner("AI sedang menganalisis..."):
            try:
                # Ambil sedikit sampel referensi dari data kamu
                ref = df['ai_search_context'].head(20).tolist()
                
                prompt = f"""Anda adalah Arsiparis Ahli. Gunakan pola ini: {ref}
                Tentukan 3 kode klasifikasi terbaik untuk perihal: {query}
                Berikan format: KODE - URAIAN (ALASAN)"""
                
                response = model.generate_content(prompt)
                st.success("Hasil Rekomendasi:")
                st.markdown(response.text)
            except Exception as e:
                st.error(f"Terjadi kendala saat analisis: {e}")
