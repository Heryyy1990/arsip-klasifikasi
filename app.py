import streamlit as st
import pandas as pd
import google.generativeai as genai
import os

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# =============================
# SETUP
# =============================
st.set_page_config(page_title="AI Arsip Muna Barat", layout="centered")

st.title("📂 Penentu Klasifikasi Arsip (Smart System)")

# API KEY
if "GEMINI_API_KEY" not in st.secrets:
    st.error("API Key tidak ditemukan!")
    st.stop()

# =============================
# LOAD DATA
# =============================
@st.cache_data
def load_data():
    return pd.read_csv("klasifikasi_arsip_upgraded.csv")

df = load_data()

# =============================
# EMBEDDING
# =============================
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model_embed = load_model()

@st.cache_resource
def encode_data(data):
    return model_embed.encode(data.tolist())

# =============================
# DETEKSI TOPIK (KUNCI PERBAIKAN)
# =============================
def detect_kode(perihal):
    p = perihal.lower()

    if "pegawai" in p or "cuti" in p:
        return "800"
    elif "keuangan" in p or "anggaran" in p:
        return "900"
    elif "rapat" in p or "undangan" in p:
        return "000"
    else:
        return None

# =============================
# INPUT
# =============================
perihal = st.text_area("Masukkan uraian surat")

if st.button("Analisis"):

    if not perihal:
        st.warning("Isi dulu")
        st.stop()

    # =============================
    # FILTER BERDASARKAN KODE
    # =============================
    kode_filter = detect_kode(perihal)

    if kode_filter:
        df_filtered = df[df["kode"].astype(str).str.startswith(kode_filter)]
        st.info(f"🔎 Difilter ke kode {kode_filter}")
    else:
        df_filtered = df
        st.warning("⚠️ Tidak terdeteksi kode, gunakan semua data")

    # =============================
    # EMBEDDING DI DATA FILTERED
    # =============================
    texts = df_filtered["ai_context_final"].astype(str)
    embeddings = encode_data(texts)

    input_vec = model_embed.encode([perihal])
    sim = cosine_similarity(input_vec, embeddings)

    top_idx = sim[0].argsort()[-3:][::-1]
    hasil = df_filtered.iloc[top_idx]

    st.subheader("Hasil Awal")

    kandidat_list = []
    for i, row in hasil.iterrows():
        teks = f"{row['kode']} - {row['uraian']}"
        kandidat_list.append(teks)
        st.write(teks)

    # =============================
    # AI
    # =============================
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    model_ai = genai.GenerativeModel('gemini-1.5-flash')

    prompt = f"""
Anda arsiparis.

Pilih 1 terbaik dari:
{chr(10).join(kandidat_list)}

Perihal: {perihal}

Jelaskan alasan berdasarkan:
- fungsi kegiatan
- objek arsip
"""

    res = model_ai.generate_content(prompt)

    st.subheader("Rekomendasi AI")
    st.write(res.text)
