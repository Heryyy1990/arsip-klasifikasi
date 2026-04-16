import streamlit as st
import pandas as pd
import google.generativeai as genai
import os

# --- 1. SETUP HALAMAN ---
st.set_page_config(page_title="Arsip Muna Barat", layout="centered")

st.title("📂 Penentu Klasifikasi Arsip")
st.caption("Dinas Perpustakaan dan Kearsipan Kabupaten Muna Barat")

# --- 2. CEK FILE DATA ---
csv_file = 'klasifikasi_arsip.csv'
if not os.path.exists(csv_file):
    # Cek apakah ada file cadangan jika nama berbeda
    csv_file = 'klasifikasi_arsip.csv'

if not os.path.exists(csv_file):
    st.error(f"❌ File database tidak ditemukan di GitHub!")
    st.stop()

# --- 3. INIT AI & HANDLING 404 ---
def init_model():
    try:
        if "GEMINI_API_KEY" not in st.secrets:
            st.error("❌ API Key (GEMINI_API_KEY) tidak ditemukan di Secrets!")
            return None
            
        genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
        
        # Menggunakan gemini-1.5-flash secara langsung
        # Versi library 0.7.2+ akan mengenali ini secara otomatis
        return genai.GenerativeModel('gemini-1.5-flash')
    except Exception as e:
        st.error(f"❌ Gagal konfigurasi AI: {e}")
        return None

# --- 4. LOAD DATA ---
@st.cache_data
def load_data(file):
    return pd.read_csv(file)

df = load_data(csv_file)

# --- 5. TAMPILAN ---
query = st.text_area("✍️ Masukkan Perihal/Uraian Surat:", height=150)

if st.button("Mulai Analisis"):
    if not query:
        st.warning("Mohon isi perihal surat.")
    else:
        model = init_model()
        if model:
            with st.spinner("AI sedang mencocokkan kode..."):
                try:
                    # Ambil contoh data untuk konteks
                    search_col = 'ai_search_context' if 'ai_search_context' in df.columns else 'uraian'
                    ref = df[search_col].head(20).tolist()
                    
                    prompt = f"""
                    Anda Pakar Arsiparis. Gunakan pola klasifikasi ini: {ref}
                    Tentukan 3 kode klasifikasi terbaik untuk perihal: {query}
                    Tampilkan hanya: KODE - URAIAN (ALASAN)
                    """
                    
                    response = model.generate_content(prompt)
                    st.success("✅ Rekomendasi Kode:")
                    st.markdown(response.text)
                except Exception as e:
                    # Jika masih 404, tampilkan instruksi debug
                    st.error(f"Kendala: {e}")
                    if "404" in str(e):
                        st.info("💡 Solusi: Silakan klik 'Manage App' > '...' > 'Reboot App' di dashboard Streamlit Anda.")
