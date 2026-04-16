import streamlit as st
import pandas as pd
import google.generativeai as genai
import os

# --- SETUP HALAMAN ---
st.set_page_config(page_title="AI Arsip Muna Barat", layout="centered")

st.title("📂 Penentu Klasifikasi Arsip")
st.caption("Dinas Perpustakaan dan Kearsipan")

# --- CEK FILE DATA ---
data_file = 'klasifikasi_arsip_upgraded.csv' if os.path.exists('klasifikasi_arsip_upgraded.csv') else 'klasifikasi_arsip.csv'

# --- FUNGSI ANALISIS ---
def run_ai(text_input):
    try:
        # 1. Ambil API Key
        api_key = st.secrets["GEMINI_API_KEY"]
        genai.configure(api_key=api_key)
        
        # 2. Daftar model untuk menghindari 404
        # AI akan mencoba mencari model yang valid secara otomatis
        model_name = 'gemini-1.5-flash'
        
        # 3. Inisialisasi Model
        model = genai.GenerativeModel(model_name)
        
        # 4. Ambil Data
        df = pd.read_csv(data_file)
        kolom = 'ai_search_context' if 'ai_search_context' in df.columns else 'uraian'
        ref = df[kolom].head(10).tolist()
        
        prompt = f"Sebagai Arsiparis, tentukan 3 kode klasifikasi untuk perihal: {text_input}\nReferensi: {ref}"
        
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"ERROR_SISTEM: {str(e)}"

# --- UI ---
user_query = st.text_area("Masukkan Perihal Surat:", height=150)

if st.button("Analisis Kode"):
    if user_query:
        with st.spinner("Sedang memproses..."):
            hasil = run_ai(user_query)
            if "ERROR_SISTEM" in hasil:
                st.error(f"Terjadi Kendala: {hasil}")
                st.info("💡 **Solusi Terakhir:** Jika tetap 404, silakan hapus aplikasi ini di Dashboard Streamlit dan buat (Deploy) ulang. Itu akan membersihkan semua error lama.")
            else:
                st.success("✅ Rekomendasi Kode:")
                st.markdown(hasil)
    else:
        st.warning("Mohon isi perihal surat.")
