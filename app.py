import streamlit as st
import pandas as pd
import re

import google.generativeai as genai

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# =============================
# SETUP
# =============================
st.set_page_config(page_title="AI Arsip Muna Barat", layout="centered")
st.title("📂 Penentu Klasifikasi Arsip (Fix 404)")
st.caption("AI stabil + tidak error model")

# =============================
# API CONFIG (RESMI)
# =============================
if "GEMINI_API_KEY" not in st.secrets:
    st.error("API Key tidak ditemukan")
    st.stop()

genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

# 🔥 MODEL PALING AMAN
model = genai.GenerativeModel("gemini-1.5-flash")

# =============================
# AI FUNCTION (ARSIPARIS LOGIC)
# =============================
def analyze_with_ai(perihal):
    prompt = f"""
Anda adalah ARSIPARIS AHLI.

Gunakan langkah berpikir:

1. Tentukan jenis dokumen
2. Tentukan aksi utama
3. Tentukan objek
4. Tentukan inti (maks 5 kata)
5. Tentukan fungsi spesifik (jangan umum)

PERIHAL:
{perihal}

FORMAT:

JENIS:
AKSI:
OBJEK:
INTI:
FUNGSI:
"""

    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"❌ AI Error: {e}"

# =============================
# LOAD DATA
# =============================
@st.cache_data
def load_data():
    return pd.read_csv("klasifikasi_arsip_upgraded.csv")

df = load_data()

def clean(text):
    return re.sub(r'[^a-z0-9\s]', '', str(text).lower())

df["search"] = df["uraian"].apply(clean)

# =============================
# EMBEDDING
# =============================
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

embed_model = load_model()

# =============================
# INPUT
# =============================
perihal = st.text_area("✍️ Masukkan uraian surat")

if st.button("Analisis"):

    if not perihal:
        st.warning("Isi dulu")
        st.stop()

    # =============================
    # AI ANALISIS
    # =============================
    with st.spinner("🧠 AI memahami isi surat..."):
        hasil = analyze_with_ai(perihal)

    if "Error" in hasil:
        st.error(hasil)
        st.stop()

    st.subheader("🧠 Analisis AI")
    st.write(hasil)

    # =============================
    # PARSE
    # =============================
    inti = ""
    fungsi = ""

    for line in hasil.split("\n"):
        if "INTI:" in line:
            inti = line.replace("INTI:", "").strip()
        if "FUNGSI:" in line:
            fungsi = line.replace("FUNGSI:", "").strip()

    query = clean(inti + " " + fungsi)

    # =============================
    # LOCAL MATCHING
    # =============================
    texts = df["search"].tolist()
    emb = embed_model.encode(texts, show_progress_bar=False)

    sim = cosine_similarity(embed_model.encode([query]), emb)[0]

    df["score"] = sim
    top = df.sort_values(by="score", ascending=False).head(5)

    st.subheader("📊 Rekomendasi Kode")

    for _, r in top.iterrows():
        st.write(f"**{r['kode']} - {r['uraian']}**")
        st.caption(f"Score: {r['score']:.3f}")

st.divider()
st.caption("Versi Stabil Tanpa Error 404")
