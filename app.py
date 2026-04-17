import streamlit as st
import pandas as pd
import os
import re
import requests

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# =============================
# SETUP
# =============================
st.set_page_config(page_title="AI Arsip Muna Barat", layout="centered")
st.title("📂 Penentu Klasifikasi Arsip (Hybrid Maksimal)")
st.caption("Local Intelligence + AI Reasoning")

# =============================
# API KEY
# =============================
if "GEMINI_API_KEY" not in st.secrets:
    st.error("❌ API Key tidak ditemukan!")
    st.stop()

API_KEY = st.secrets["GEMINI_API_KEY"]

# =============================
# LOAD DATA
# =============================
@st.cache_data
def load_data():
    if os.path.exists("klasifikasi_arsip_optimized.csv"):
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
# EXTRACT INTI (STABIL)
# =============================
def extract_inti(text):
    text = text.lower()

    if "tentang" in text:
        text = text.split("tentang", 1)[1]

    noise = [
        "dengan ini", "permohonan", "pelaksanaan",
        "kegiatan", "dalam rangka", "sehubungan"
    ]

    for n in noise:
        text = text.replace(n, "")

    text = re.sub(r'\b(19|20)\d{2}\b', '', text)
    text = re.sub(r'\d+', '', text)

    words = text.split()
    important = [w for w in words if len(w) > 3]

    return " ".join(important[:6])

# =============================
# FUNCTION DETECTOR (MULTI)
# =============================
def detect_functions(text):
    fungsi = []

    mapping = {
        "undangan": "rapat",
        "rapat": "rapat koordinasi",
        "cuti": "cuti pegawai",
        "pindah": "mutasi pegawai",
        "mutasi": "mutasi pegawai",
        "pensiun": "pensiun pegawai",
        "arsip": "kearsipan",
        "anggaran": "pengelolaan anggaran",
        "laporan": "pelaporan"
    }

    for k, v in mapping.items():
        if k in text:
            fungsi.append(v)

    return list(set(fungsi))

# =============================
# BUILD QUERY
# =============================
def build_query(inti, fungsi):
    query = inti

    if "berkas" in query:
        query = query.replace("berkas", "")

    for f in fungsi:
        query += " " + f

    return clean_text(query)

# =============================
# SEARCH COLUMN
# =============================
def get_search_column(df):
    if "ai_context_final" in df.columns:
        return df["ai_context_final"]
    else:
        return df["uraian"]

df["search_text"] = get_search_column(df).apply(clean_text)

# =============================
# EMBEDDING
# =============================
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

# =============================
# HYBRID SCORE
# =============================
def calculate_score(query, text, semantic):
    keyword = sum([1 for w in query.split() if w in text])
    keyword = keyword / (len(query.split()) + 1)

    domain = 1 if ("pegawai" in query and "pegawai" in text) else 0

    return (semantic * 0.65) + (keyword * 0.25) + (domain * 0.10)

# =============================
# GEMINI FIX (PALING STABIL)
# =============================
def call_gemini(prompt):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-pro:generateContent?key={API_KEY}"

    data = {
        "contents": [{"parts": [{"text": prompt}]}]
    }

    res = requests.post(url, json=data)

    if res.status_code != 200:
        return f"❌ Error API: {res.text}"

    return res.json()["candidates"][0]["content"]["parts"][0]["text"]

# =============================
# INPUT
# =============================
perihal = st.text_area("✍️ Masukkan uraian surat")

if st.button("Analisis"):

    if not perihal:
        st.warning("Isi dulu")
        st.stop()

    inti = extract_inti(perihal)
    fungsi = detect_functions(inti)
    query = build_query(inti, fungsi)

    st.info(f"🧠 Inti: {inti}")
    st.info(f"🎯 Fungsi: {', '.join(fungsi)}")

    # EMBEDDING
    texts = df["search_text"].tolist()
    embeddings = model.encode(texts, show_progress_bar=False)

    sim = cosine_similarity(model.encode([query]), embeddings)[0]

    scores = [calculate_score(query, texts[i], sim[i]) for i in range(len(df))]
    df["score"] = scores

    top = df.sort_values(by="score", ascending=False).head(5)

    st.subheader("📊 Kandidat Terbaik")

    kandidat = []
    for _, r in top.iterrows():
        teks = f"{r['kode']} - {r['uraian']}"
        kandidat.append(teks)
        st.write(f"**{teks}**")

    # =============================
    # 🔥 AI REASONING (KUNCI)
    # =============================
    with st.spinner("🤖 AI sedang menganalisis mendalam..."):

        prompt = f"""
Anda adalah Arsiparis Ahli.

PERIHAL:
{perihal}

INTI:
{inti}

FUNGSI:
{', '.join(fungsi)}

KANDIDAT:
{chr(10).join(kandidat)}

TUGAS:
1. Analisis fungsi utama arsip
2. Bandingkan setiap kandidat
3. Pilih 1 paling tepat

FORMAT:

ANALISIS:
...

PERBANDINGAN:
...

REKOMENDASI:
KODE

ALASAN:
jelaskan spesifik dan logis
"""

        hasil = call_gemini(prompt)

        st.subheader("🎯 Keputusan Akhir AI")
        st.write(hasil)

st.divider()
st.caption("Hybrid Maksimal - Siap Produksi")
