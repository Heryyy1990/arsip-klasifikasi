import streamlit as st
import pandas as pd
import google.generativeai as genai

# Konfigurasi Halaman
st.set_page_config(page_title="AI Archiver - Muna Barat", page_icon="🧠", layout="wide")

# --- STYLE ---
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #007bff; color: white; }
    .result-card { padding: 20px; border-radius: 10px; background-color: white; border-left: 5px solid #007bff; margin-bottom: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    </style>
    """, unsafe_allow_html=True)

# --- LOAD DATA ---
@st.cache_data
def load_data():
    # Pastikan file ini ada di repo GitHub kamu
    df = pd.read_csv('klasifikasi_arsip.csv')
    return df

try:
    df_arsip = load_data()
except Exception as e:
    st.error(f"Gagal memuat database arsip. Pastikan file CSV tersedia. Error: {e}")
    st.stop()

# --- AMBIL API KEY DARI STREAMLIT SECRETS ---
# AI akan mencari kunci bernama 'GEMINI_API_KEY' di setting Streamlit Cloud
try:
    api_key = st.secrets["GEMINI_API_KEY"]
    genai.configure(api_key=api_key)
except Exception as e:
    st.error("⚠️ API Key tidak ditemukan di Secrets Streamlit. Silakan atur di 'Settings' -> 'Secrets'.")
    st.stop()

# --- HEADER ---
st.title("🧠 AI Penentu Kode Klasifikasi Arsip")
st.subheader("Kabupaten Muna Barat")

# --- INPUT ---
user_input = st.text_area("✍️ Masukkan Perihal/Deskripsi Arsip:", placeholder="Contoh: Permohonan izin cuti tahunan...")

if st.button("Analisis Kontekstual"):
    if not user_input:
        st.warning("⚠️ Masukkan deskripsi arsip terlebih dahulu.")
    else:
        try:
            # Gunakan penamaan model yang lebih stabil
            model = genai.GenerativeModel('gemini-1.5-flash')

            # Ambil data referensi (maksimalkan konteks)
            sample_context = df_arsip['ai_search_context'].head(1000).tolist()
            
            prompt = f"""
            Anda adalah Arsiparis Ahli. Berdasarkan dataset berikut:
            {sample_context[:50]} ...

            Tentukan 3 rekomendasi kode klasifikasi untuk: "{user_input}"
            
            Berikan jawaban dalam format:
            1. KODE - URAIAN
               Alasan: ...
            """

            with st.spinner('Menganalisis...'):
                response = model.generate_content(prompt)
                st.success("✅ Analisis Selesai!")
                st.markdown(response.text)

        except Exception as e:
            st.error(f"Terjadi kendala teknis: {e}. Pastikan API Key benar dan library terbaru.")

# --- FOOTER ---
st.divider()
st.caption("Dinas Perpustakaan dan Kearsipan Kabupaten Muna Barat.")
