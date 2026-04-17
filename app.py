import streamlit as st
import pandas as pd
import google.generativeai as genai
import os
import re

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# =============================
# SETUP
# =============================
st.set_page_config(page_title="AI Arsip Muna Barat", layout="centered")

st.title("📂 Penentu Klasifikasi Arsip (Versi Stabil)")
st.caption("Smart Extractor + Embedding + AI")

# =============================
# API KEY
# =============================
if "GEMINI_API_KEY" not in st.secrets:
    st.error("❌ API Key tidak ditemukan!")
    st.stop()

genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

# =============================
# LOAD DATA
# =============================
@st.cache_data
def load_data():
    if os.path.exists("klasifikasi_arsip_optimized.csv"):
        return pd.read_csv("klasifikasi_arsip_optimized.csv")
    elif os.path.exists("klasifikasi_arsip_upgraded.csv"):
        return pd.read_csv("klasifikasi_arsip_upgraded.csv")
    else:
        return pd.read_csv("klasifikasi_arsip.csv")

df = load_data()

# =============================
# CLEAN TEXT
# =============================
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    return text

# =============================
# 🔥 EXTRACTOR BARU (LEBIH TAJAM)
# =============================
def extract_inti(text):
    text = text.lower()

    # hapus kalimat pembuka
    noise = [
        "dengan ini", "saya", "mengajukan", "permohonan",
        "untuk", "melakukan", "dalam rangka",
        "melengkapi", "persyaratan"
    ]

    for n in noise:
        text = text.replace(n, "")

    words = text.split()

    # ambil kata paling akhir (biasanya inti)
    if len(words) > 3:
        words = words[-3:]

    return " ".join(words)

# =============================
# 🔥 BOOST KATA PENTING (GENERIK)
# =============================
def boost_query(text):
    boost = []

    if "pindah" in text:
        boost.append("pegawai mutasi")

    if "cuti" in text:
        boost.append("pegawai cuti")

    if "arsip" in text:
        boost.append("kearsipan arsip")

    return text + " " + " ".join(boost)

# =============================
# PILIH KOLOM
# =============================
def get_search_column(dataframe):
    if "ai_context_final" in dataframe.columns:
        return dataframe["ai_context_final"]
    elif "ai_search_context" in dataframe.columns:
        return dataframe["ai_search_context"]
    else:
        return dataframe["uraian"]

df["search_text"] = get_search_column(df).apply(clean_text)

# =============================
# EMBEDDING
# =============================
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model_embed = load_model()

# =============================
# INPUT
# =============================
perihal = st.text_area("✍️ Masukkan uraian surat")

if st.button("Analisis"):

    if not perihal:
        st.warning("Isi dulu")
        st.stop()

    # =============================
    # EXTRACT INTI
    # =============================
    inti = extract_inti(perihal)
    final_query = boost_query(inti)
    final_query = clean_text(final_query)

    st.info(f"🧠 Inti: {inti}")
    st.info(f"🚀 Query akhir: {final_query}")

    # =============================
    # EMBEDDING
    # =============================
    texts = df["search_text"].astype(str).tolist()
    embeddings = model_embed.encode(texts, show_progress_bar=False)

    input_vec = model_embed.encode([final_query])
    sim = cosine_similarity(input_vec, embeddings)

    top_pos = sim[0].argsort()[-5:][::-1]
    hasil = df.iloc[top_pos]

    # =============================
    # TAMPILKAN
    # =============================
    st.subheader("📊 Rekomendasi Awal")

    kandidat_list = []

    for pos in top_pos:
        row = df.iloc[pos]

        kode = row.get("kode", "-")
        uraian = row.get("uraian", "-")
        skor = sim[0][pos]

        teks = f"{kode} - {uraian}"
        kandidat_list.append(teks)

        st.write(f"**{teks}**")
        st.caption(f"Similarity: {skor:.3f}")

    # =============================
    # AI (FIX TOTAL)
    # =============================
    model_ai = genai.GenerativeModel("gemini-pro")

    with st.spinner("🤖 AI menganalisis..."):
        try:
            prompt = f"""
Anda adalah Arsiparis Ahli.

Inti:
{inti}

Kandidat:
{chr(10).join(kandidat_list)}

Pilih 1 yang paling tepat berdasarkan:
- fungsi kegiatan
- objek arsip

Jawaban:

KODE:
...

ALASAN:
jelaskan logis dan spesifik
"""

            res = model_ai.generate_content(prompt)

            st.subheader("🎯 Hasil AI")
            st.write(res.text)

        except Exception as e:
            st.error(f"AI Error: {e}")

st.divider()
st.caption("Versi Stabil + Extractor Lebih Tajam + Gemini Fix")
