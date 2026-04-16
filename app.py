import streamlit as st
import pandas as pd
import google.generativeai as genai

# --- Konfigurasi Halaman ---
st.set_page_config(page_title="AI Arsip Muna Barat", layout="wide")

# Custom CSS untuk tampilan lebih profesional
st.markdown("""
    <style>
    .main { background-color: #f9f9f9; }
    .stButton>button { background-color: #0d47a1; color: white; font-weight: bold; border-radius: 8px; }
    .result-box { padding: 20px; background-color: white; border-left: 6px solid #0d47a1; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
    </style>
""", unsafe_allow_html=True)

# --- Load Dataset ---
@st.cache_data
def load_data():
    try:
        return pd.read_csv('klasifikasi_arsip.csv')
    except:
        return None

df_arsip = load_data()

# --- Fungsi Inisialisasi AI ---
def get_ai_model():
    try:
        # Mengambil API Key dari Secrets Streamlit Cloud
        api_key = st.secrets["GEMINI_API_KEY"]
        genai.configure(api_key=api_key)
        # Menggunakan nama model standar tanpa awalan 'models/' untuk stabilitas
        return genai.GenerativeModel('gemini-1.5-flash')
    except Exception as e:
        st.error(f"Konfigurasi Secrets Gagal: {e}")
        return None

# --- Tampilan Utama ---
st.title("📂 Penentu Kode Klasifikasi Arsip")
st.markdown("##### Dinas Perpustakaan dan Kearsipan Kabupaten Muna Barat")

if df_arsip is None:
    st.error("File 'klasifikasi_arsip_upgraded.csv' tidak ditemukan di GitHub.")
    st.stop()

model = get_ai_model()

# Area Input
user_query = st.text_area("✍️ Masukkan Perihal/Uraian Arsip:", placeholder="Contoh: Surat permohonan izin cuti tahunan pegawai...", height=120)

if st.button("Mulai Analisis"):
    if not user_query:
        st.warning("Mohon masukkan deskripsi surat.")
    elif model:
        with st.spinner("AI sedang menganalisis database kearsipan..."):
            try:
                # Mengambil sampel data referensi agar AI memahami pola klasifikasi
                referensi = df_arsip['ai_search_context'].head(50).tolist()
                
                prompt = f"""
                Anda adalah Pakar Arsiparis. Tugas Anda memberikan 3 rekomendasi kode klasifikasi.
                
                REFERENSI STRUKTUR:
                {referensi}

                SURAT USER: "{user_query}"

                BERIKAN HASIL:
                1. Kode Klasifikasi - Uraian
                2. Alasan logis pemilihan kode tersebut.
                """
                
                response = model.generate_content(prompt)
                st.success("✅ Analisis Selesai")
                st.markdown(f'<div class="result-box">{response.text}</div>', unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"Error AI: {e}")
                st.info("Pastikan file requirements.txt sudah berisi 'google-generativeai>=0.5.0'")

st.divider()
st.caption("Dinas Perpustakaan dan Kearsipan Kabupaten Muna Barat")
