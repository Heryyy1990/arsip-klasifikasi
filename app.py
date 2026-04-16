import streamlit as st
import pandas as pd
import google.generativeai as genai
import os

# Konfigurasi Halaman
st.set_page_config(page_title="AI Arsip Muna Barat", layout="centered")

st.title("📂 Penentu Klasifikasi Arsip")
st.caption("Dinas Perpustakaan dan Kearsipan Kabupaten Muna Barat")

# --- 1. CEK API KEY DI SECRETS ---
if "GEMINI_API_KEY" not in st.secrets:
    st.error("❌ API Key tidak ditemukan! Buka Settings > Secrets di Streamlit, lalu masukkan: GEMINI_API_KEY = 'KUNCI_ANDA'")
    st.stop()

# --- 2. CEK DATA CSV ---
# Mengecek file mana yang ada di GitHub Anda
file_upgraded = 'klasifikasi_arsip_upgraded.csv'
file_asli = 'klasifikasi_arsip.csv'

if os.path.exists(file_upgraded):
    csv_file = file_upgraded
    st.success(f"✅ Menggunakan database: {file_upgraded}")
elif os.path.exists(file_asli):
    csv_file = file_asli
    st.info(f"ℹ️ Menggunakan database asli: {file_asli}")
else:
    st.error("❌ File CSV tidak ditemukan! Pastikan file 'klasifikasi_arsip.csv' sudah di-upload ke GitHub.")
    st.stop()

@st.cache_data
def load_data():
    return pd.read_csv(csv_file)

df = load_data()

# --- 3. INISIALISASI MODEL ---
try:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    # Langsung panggil model tanpa embel-embel v1beta
    model = genai.GenerativeModel('gemini-1.5-flash')
except Exception as e:
    st.error(f"Gagal memuat AI: {e}")
    st.stop()

# --- 4. TAMPILAN UTAMA ---
perihal = st.text_area("✍️ Masukkan Perihal/Uraian Surat:", placeholder="Contoh: Permohonan cuti tahunan pegawai...", height=150)

if st.button("Mulai Analisis"):
    if not perihal:
        st.warning("Mohon isi deskripsi surat dulu.")
    else:
        with st.spinner("AI sedang menganalisis database..."):
            try:
                # Tentukan kolom pencarian
                search_col = 'ai_search_context' if 'ai_search_context' in df.columns else 'uraian'
                referensi = df[search_col].head(10).tolist()
                
                prompt = f"""Anda Pakar Arsiparis. Gunakan pola ini: {referensi}
                Tentukan 3 kode klasifikasi terbaik untuk perihal: {perihal}
                Format: KODE - URAIAN (ALASAN)"""
                
                response = model.generate_content(prompt)
                st.success("🎯 Rekomendasi AI:")
                st.markdown(response.text)
                
            except Exception as e:
                # Jika masih error 404, tampilkan instruksi debug yang tegas
                st.error(f"Analisis Gagal: {e}")
                if "404" in str(e):
                    st.info("⚠️ **SOLUSI TERAKHIR:** Masalah 404 v1beta berarti server Streamlit memakai library lama. Silakan masuk ke Dashboard Streamlit, klik tanda '...' di sebelah aplikasi Anda, pilih 'Delete', lalu 'New App' (Deploy ulang dari nol). Itu akan menghapus total semua error lama.")

st.divider()
st.caption("Dikembangkan untuk Dinas Perpustakaan dan Kearsipan Kabupaten Muna Barat.")
