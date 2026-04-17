import streamlit as st
import pandas as pd
import google.generativeai as genai
import os

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# =============================
# KONFIGURASI HALAMAN
# =============================
st.set_page_config(page_title="AI Arsip Muna Barat", layout="centered")

st.title("📂 Penentu Klasifikasi Arsip (Embedding + AI)")
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
file_optimized = 'klasifikasi_arsip_optimized.csv'
file_upgraded = 'klasifikasi_arsip_upgraded.csv'
file_asli = 'klasifikasi_arsip.csv'

if os.path.exists(file_optimized):
    csv_file = file_optimized
    st.success(f"✅ Menggunakan database terbaik: {file_optimized}")
elif os.path.exists(file_upgraded):
    csv_file = file_upgraded
    st.info(f"ℹ️ Menggunakan database upgraded")
elif os.path.exists(file_asli):
    csv_file = file_asli
    st.warning(f"⚠️ Menggunakan database dasar")
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
# LOAD EMBEDDING MODEL
# =============================
@st.cache_resource
def load_embedding():
    return SentenceTransformer('all-MiniLM-L6-v2')

embed_model = load_embedding()

# =============================
# PREPARE EMBEDDING DATABASE
# =============================
@st.cache_resource
def prepare_embeddings(dataframe):
    kolom = 'ai_context_final' if 'ai_context_final' in dataframe.columns else 'uraian'
    texts = dataframe[kolom].astype(str).tolist()
    embeddings = embed_model.encode(texts, show_progress_bar=False)
    return embeddings, kolom

embeddings_db, search_col = prepare_embeddings(df)

# =============================
# GEMINI SETUP
# =============================
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
model_ai = genai.GenerativeModel('gemini-1.5-flash')

# =============================
# INPUT USER
# =============================
perihal = st.text_area(
    "✍️ Masukkan Perihal/Uraian Surat:",
    placeholder="Contoh: Permohonan cuti tahunan pegawai...",
    height=150
)

# =============================
# PROSES
# =============================
if st.button("Mulai Analisis"):

    if not perihal:
        st.warning("Mohon isi deskripsi surat dulu.")
    else:

        with st.spinner("🔍 Mencari dengan Embedding..."):

            input_embedding = embed_model.encode([perihal])
            similarity = cosine_similarity(input_embedding, embeddings_db)

            top_idx = similarity[0].argsort()[-3:][::-1]
            hasil = df.iloc[top_idx]

        # =============================
        # HASIL EMBEDDING
        # =============================
        st.subheader("📊 Rekomendasi Awal (Embedding)")

        kandidat_list = []

        for i, row in hasil.iterrows():
            kode = row.get("kode", "Tidak ada kode")
            uraian = row.get("uraian", "Tidak ada uraian")
            skor = similarity[0][i]

            kandidat_list.append(f"{kode} - {uraian}")

            st.write(f"**{kode} - {uraian}**")
            st.caption(f"Similarity: {skor:.3f}")

        # =============================
        # AI ANALISIS (VERSI DIPERKUAT)
        # =============================
        with st.spinner("🤖 AI sedang menganalisis..."):
            try:
                prompt = f"""
Anda adalah Arsiparis Ahli di lingkungan pemerintah daerah.

Diberikan 3 kandidat klasifikasi arsip:
{chr(10).join(kandidat_list)}

Perihal surat:
{perihal}

Lakukan analisis sebagai berikut:

1. Jelaskan relevansi masing-masing kandidat terhadap perihal
2. Bandingkan perbedaan fokus antar kandidat
3. Tentukan 1 kode klasifikasi yang PALING TEPAT

Gunakan pendekatan:
- fungsi kegiatan
- objek arsip
- konteks administrasi

Hindari jawaban umum.

Format:

ANALISIS KANDIDAT:

1. KODE - PENJELASAN SPESIFIK
2. KODE - PENJELASAN SPESIFIK
3. KODE - PENJELASAN SPESIFIK

PERBANDINGAN:
Jelaskan perbedaan fokus masing-masing kandidat

REKOMENDASI UTAMA:
KODE TERPILIH

ALASAN:
Jelaskan secara logis:
- kegiatan apa
- objek apa
- kenapa paling tepat dibanding lainnya
"""

                response = model_ai.generate_content(prompt)

                st.subheader("🎯 Analisis AI (Final)")
                st.markdown(response.text)

            except Exception as e:
                st.error(f"AI Error: {e}")

st.divider()
st.caption("Sistem Hybrid: Embedding + AI Reasoning")
