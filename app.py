import streamlit as st
import pandas as pd
import google.generativeai as genai
import os
from dotenv import load_dotenv

# Load environment variables (untuk API Key)
load_dotenv()

# Konfigurasi Halaman
st.set_page_config(page_title="AI Archiver - Muna Barat", page_icon="🧠", layout="wide")

# --- STYLE ---
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #007bff; color: white; }
    .result-card { padding: 20px; border-radius: 10px; background-color: white; border-left: 5px solid #007bff; margin-bottom: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    </style>
    """, unsafe_allow_dict=True)

# --- SIDEBAR ---
with st.sidebar:
    st.title("⚙️ Pengaturan")
    api_key = st.text_input("Google API Key", type="password", help="Dapatkan di Google AI Studio")
    st.info("Aplikasi ini menggunakan dataset Klasifikasi Arsip yang sudah di-upgrade untuk akurasi maksimal.")

# --- LOAD DATA ---
@st.cache_data
def load_data():
    # Pastikan file klasifikasi_arsip_upgraded.csv ada di folder yang sama
    df = pd.read_csv('klasifikasi_arsip_upgraded.csv')
    return df

try:
    df_arsip = load_data()
except Exception as e:
    st.error(f"Gagal memuat database arsip. Pastikan file CSV tersedia. Error: {e}")
    st.stop()

# --- HEADER ---
st.title("🧠 AI Penentu Kode Klasifikasi Arsip")
st.subheader("Kabupaten Muna Barat")
st.write("Masukkan perihal atau deskripsi surat, dan AI akan menentukan kode klasifikasi yang paling tepat secara kontekstual.")

# --- INPUT ---
user_input = st.text_area("✍️ Masukkan Perihal/Deskripsi Arsip:", placeholder="Contoh: Permohonan izin cuti tahunan karena alasan keluarga...")

if st.button("Analisis Kontekstual"):
    if not api_key:
        st.warning("⚠️ Silakan masukkan API Key di sidebar terlebih dahulu.")
    elif not user_input:
        st.warning("⚠️ Masukkan deskripsi arsip terlebih dahulu.")
    else:
        try:
            # 1. Inisialisasi AI
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel('gemini-1.5-flash')

            # 2. Filtering Level 2 (Strategi Akurasi)
            # Kita mengirimkan sampel data yang relevan ke AI sebagai referensi konteks
            # Untuk efisiensi token, kita hanya kirimkan kolom context
            sample_context = df_arsip['ai_search_context'].sample(min(len(df_arsip), 500)).tolist()
            
            prompt = f"""
            Anda adalah seorang Arsiparis Ahli. Tugas Anda adalah menentukan Kode Klasifikasi Arsip berdasarkan aturan nasional.
            
            DATASET REFERENSI (Contoh Format):
            {sample_context[:20]} ... (dan seterusnya)

            INPUT USER: "{user_input}"

            TUGAS:
            1. Pahami makna mendalam dari INPUT USER (Contoh: "cuti" berarti Kepegawaian, bukan SAR/Bencana).
            2. Berikan 3 rekomendasi kode klasifikasi terbaik dari dataset.
            3. Berikan alasan logis untuk setiap rekomendasi.

            FORMAT OUTPUT (JSON):
            [
              {{"kode": "...", "uraian": "...", "alasan": "..."}},
              ...
            ]
            """

            with st.spinner('Sedang menganalisis konteks arsip...'):
                response = model.generate_content(prompt)
                
                # Parsing hasil (Sederhana)
                st.success("✅ Analisis Selesai!")
                
                # Tampilkan Hasil
                st.markdown("### 🎯 Rekomendasi Kode Teratas")
                st.write(response.text) # Menampilkan output teks dari AI

        except Exception as e:
            st.error(f"Terjadi kendala pada server AI: {e}")

# --- FOOTER ---
st.divider()
st.caption("Dikembangkan untuk Dinas Perpustakaan dan Kearsipan Kabupaten Muna Barat.")
