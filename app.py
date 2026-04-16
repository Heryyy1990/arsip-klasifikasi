import streamlit as st
import pandas as pd
import google.generativeai as genai

# --- CONFIG ---
st.set_page_config(page_title="Arsip Muna Barat", layout="centered")

# --- LOAD DATA ---
@st.cache_data
def load_data():
    try:
        return pd.read_csv('klasifikasi_arsip_upgraded.csv')
    except:
        st.error("File 'klasifikasi_arsip.csv' tidak ditemukan!")
        return None

df = load_data()

# --- INIT AI ---
def start_ai():
    try:
        api_key = st.secrets["GEMINI_API_KEY"]
        genai.configure(api_key=api_key)
        
        # Cek model yang tersedia (Untuk Debugging)
        available_models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        
        # Coba gunakan nama model paling standar
        # Jika 'models/gemini-1.5-flash' tidak ada, dia akan coba yang lain
        target_model = 'gemini-1.5-flash'
        if f'models/{target_model}' not in available_models:
             # Jika 1.5 flash tidak ada, tampilkan daftar yang tersedia di log
             st.warning(f"Model {target_model} tidak terdeteksi. Model tersedia: {available_models}")
        
        return genai.GenerativeModel(target_model)
    except Exception as e:
        st.error(f"Gagal inisialisasi AI: {e}")
        return None

# --- UI ---
st.title("📂 Penentu Klasifikasi Arsip")
st.caption("Dinas Perpustakaan dan Kearsipan Kabupaten Muna Barat")

query = st.text_area("Masukkan Perihal Surat:", placeholder="Contoh: Surat permohonan cuti...")

if st.button("Cek Kode Sekarang"):
    if not query:
        st.warning("Isi perihal dulu!")
    else:
        model = start_ai()
        if model:
            with st.spinner("Menganalisis..."):
                try:
                    # Ambil sedikit sampel untuk panduan
                    ref = df['ai_search_context'].head(20).tolist() if df is not None else ""
                    
                    prompt = f"Anda pakar arsip. Berdasarkan data: {ref}\n\nTentukan 3 kode klasifikasi terbaik untuk: {query}"
                    
                    response = model.generate_content(prompt)
                    st.success("Hasil Analisis:")
                    st.markdown(response.text)
                except Exception as e:
                    st.error(f"Terjadi Kesalahan: {e}")
                    st.info("Catatan: Jika error 404 berlanjut, silakan lakukan REBOOT di Dashboard Streamlit.")
