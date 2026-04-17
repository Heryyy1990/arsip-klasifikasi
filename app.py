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

st.title("📂 Penentu Klasifikasi Arsip (Generik + Smart Extractor)")
st.caption("Inti + Keyword + Embedding + AI")

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
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    return text

# =============================
# EKSTRAK INTI (LAMA - TETAP DIPAKAI)
# =============================
def extract_inti(text):
    text = text.lower()

    noise = [
        "berita acara", "surat", "dokumen", "laporan",
        "permohonan", "undangan", "nota dinas",
        "hasil", "tentang", "perihal"
    ]

    for n in noise:
        text = text.replace(n, "")

    return text.strip()

# =============================
# 🔥 EKSTRAK KEYWORD BARU (UNTUK KALIMAT PANJANG)
# =============================
def extract_keyword(text):
    text = text.lower()

    stopwords = [
        "dengan", "ini", "saya", "mengajukan", "untuk",
        "melakukan", "dalam", "rangka", "agar",
        "sebagai", "berikut", "tersebut"
    ]

    words = text.split()
    words = [w for w in words if w not in stopwords]

    # ambil kata penting saja (max 5 kata terakhir biasanya inti)
    return " ".join(words[-5:])

# =============================
# PILIH KOLOM
# =============================
def get_search_column(dataframe):
    if "ai_context_final" in dataframe.columns:
        return dataframe["ai_context_final"]
    elif "ai_search_context" in dataframe.columns:
        return dataframe["ai_search_context"]
    elif "uraian" in dataframe.columns:
        return dataframe["uraian"]
    else:
        return dataframe.iloc[:, 0]

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
    # EKSTRAK INTI + KEYWORD
    # =============================
    inti = extract_inti(perihal)
    keyword = extract_keyword(perihal)

    final_query = clean_text(inti + " " + keyword)

    st.info(f"🧠 Inti: {inti}")
    st.info(f"🔑 Keyword: {keyword}")

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
    # AI (FIX MODEL ERROR)
    # =============================
    try:
        model_ai = genai.GenerativeModel('gemini-1.5-flash-latest')
    except:
        model_ai = genai.GenerativeModel('gemini-pro')

    with st.spinner("🤖 AI menganalisis..."):
        try:
            prompt = f"""
Anda adalah Arsiparis Ahli.

Inti:
{inti}

Keyword:
{keyword}

Kandidat:
{chr(10).join(kandidat_list)}

Tugas:
- Analisis tiap kandidat
- Bandingkan
- Pilih 1 terbaik

Gunakan:
- fungsi kegiatan
- objek arsip

REKOMENDASI:
KODE

ALASAN:
jelaskan logis dan spesifik
"""

            res = model_ai.generate_content(prompt)

            st.subheader("🎯 Hasil AI")
            st.write(res.text)

        except Exception as e:
            st.error(f"AI Error: {e}")

st.divider()
st.caption("Versi Smart Extractor + Stabil AI")
