import streamlit as st
import pandas as pd
import google.generativeai as genai
import os

# Konfigurasi Halaman
st.set_page_config(page_title="AI Arsip Muna Barat", layout="wide")

st.title("📂 Penentu Kode Klasifikasi Arsip")
st.caption("Dinas Perpustakaan dan Kearsipan Kabupaten Muna Barat")

# --- 1. VALIDASI API KEY DARI SECRETS ---
if "GEMINI_API_KEY" not in st.secrets:
    st.error("❌ API Key tidak ditemukan! Pastikan Anda sudah menulis GEMINI_API_KEY di menu Settings > Secrets.")
    st.stop()

# --- 2. VALIDASI DATA ---
# Mencari file CSV yang ada (baik yang asli maupun yang sudah di-upgrade)
data_file = 'klasifikasi_arsip.csv' if os.path.exists('klasifikasi_arsip_upgraded.csv') else 'klasifikasi_arsip.csv'

if not os.path.exists(data_file):
    st.error(f"❌ File '{data_file}' tidak ditemukan di GitHub Anda.")
    st.stop()

@st.cache_data
def load_data():
    return pd.read_csv(data_file)

df = load_data()

# --- 3. INISIALISASI MODEL (DENGAN FIX 404) ---
def get_recommendation(text_input):
    try:
        genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
        
        # Menggunakan model 1.5-flash (Versi stabil)
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        # Mengambil sampel context untuk membantu AI
        column_name = 'ai_search_context' if 'ai_search_context' in df.columns else 'uraian'
        sample = df[column_name].head(15).tolist()
        
        prompt = f"""Tugas: Berikan 3 rekomendasi kode klasifikasi arsip.
        Pola Data: {sample}
        Perihal: {text_input}
        Format: KODE - URAIAN (ALASAN)"""
        
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        # Jika error 404 muncul, tampilkan instruksi debug
        return f"Error: {str(e)}"

# --- 4. TAMPILAN UTAMA ---
uraian_surat = st.text_area("Masukkan Perihal/Uraian Surat:", height=150)

if st.button("Cek Kode Klasifikasi"):
    if not uraian_surat:
        st.warning("Mohon isi uraian surat terlebih dahulu.")
    else:
        with st.spinner("Sedang menganalisis..."):
            hasil = get_recommendation(uraian_surat)
            if "Error" in hasil:
                st.error(hasil)
                if "404" in hasil:
                    st.info("💡 **Tips Solusi 404:** Pastikan file requirements.txt Anda berisi: `google-generativeai>=0.5.0` lalu lakukan 'Reboot App' di dashboard Streamlit.")
            else:
                st.success("✅ Rekomendasi Berhasil!")
                st.markdown(hasil)
