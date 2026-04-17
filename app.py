import streamlit as st
import pandas as pd
import google.generativeai as genai
import os

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# =============================
# KONFIGURASI HALAMAN
# =============================
st.set_page_config(page_title="AI Arsip Muna Barat", layout="centered")

st.title("📂 Penentu Klasifikasi Arsip")
st.caption("Dinas Perpustakaan dan Kearsipan Kabupaten Muna Barat")

# =============================
# CEK API KEY
# =============================
if "GEMINI_API_KEY" not in st.secrets:
    st.error("❌ API Key tidak ditemukan!")
    st.stop()

# =============================
# CEK FILE CSV
# =============================
file_upgraded = 'klasifikasi_arsip_upgraded.csv'
file_asli = 'klasifikasi_arsip.csv'

if os.path.exists(file_upgraded):
    csv_file = file_upgraded
    st.success(f"✅ Menggunakan database: {file_upgraded}")
elif os.path.exists(file_asli):
    csv_file = file_asli
    st.info(f"ℹ️ Menggunakan database asli: {file_asli}")
else:
    st.error("❌ File CSV tidak ditemukan!")
    st.stop()

# =============================
# LOAD DATA
# =============================
@st.cache_data
def load_data():
    return pd.read_csv(csv_file)

df = load_data()

# =============================
# PREPARE TF-IDF (LOCAL ML)
# =============================
@st.cache_resource
def prepare_model(dataframe):
    kolom = 'ai_search_context' if 'ai_search_context' in dataframe.columns else 'uraian'
    
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(dataframe[kolom].astype(str))
    
    return vectorizer, tfidf_matrix, kolom

vectorizer, tfidf_matrix, search_col = prepare_model(df)

# =============================
# GEMINI SETUP
# =============================
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
model = genai.GenerativeModel('gemini-1.5-flash')

# =============================
# UI INPUT
# =============================
perihal = st.text_area(
    "✍️ Masukkan Perihal/Uraian Surat:",
    placeholder="Contoh: Permohonan cuti tahunan pegawai...",
    height=150
)

# =============================
# PROSES ANALISIS
# =============================
if st.button("Mulai Analisis"):

    if not perihal:
        st.warning("Mohon isi deskripsi surat dulu.")
    else:
        with st.spinner("🔍 Menganalisis dengan Local Model..."):

            # --- 1. TF-IDF Similarity ---
            input_vec = vectorizer.transform([perihal])
            similarity = cosine_similarity(input_vec, tfidf_matrix)

            # Ambil 3 terbaik
            top_idx = similarity[0].argsort()[-3:][::-1]
            hasil_lokal = df.iloc[top_idx]

        # =============================
        # TAMPILKAN HASIL LOKAL
        # =============================
        st.subheader("📊 Rekomendasi Awal (Local Model)")
        
        kandidat_list = []

        for i, row in hasil_lokal.iterrows():
            kode = row.get("kode", "Tidak ada kode")
            uraian = row.get("uraian", "Tidak ada uraian")
            skor = similarity[0][i]

            kandidat_list.append(f"{kode} - {uraian}")

            st.write(f"**{kode} - {uraian}**")
            st.caption(f"Kemiripan: {skor:.2f}")

        # =============================
        # AI ANALISIS
        # =============================
        with st.spinner("🤖 AI sedang memberikan analisis..."):
            try:
                prompt = f"""
Anda adalah Arsiparis Ahli.

Diberikan 3 kandidat klasifikasi arsip berikut:
{chr(10).join(kandidat_list)}

Perihal surat:
{perihal}

Tugas Anda:
1. Jelaskan secara singkat alasan tiap kandidat relevan
2. Pilih 1 kode terbaik
3. Berikan alasan paling kuat

Format:
1. KODE - PENJELASAN
2. KODE - PENJELASAN
3. KODE - PENJELASAN

REKOMENDASI UTAMA:
KODE TERPILIH - ALASAN
"""

                response = model.generate_content(prompt)

                st.subheader("🎯 Analisis AI")
                st.markdown(response.text)

            except Exception as e:
                st.error(f"AI Error: {e}")

st.divider()
st.caption("Dikembangkan untuk Dinas Perpustakaan dan Kearsipan Kabupaten Muna Barat.")
