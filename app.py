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
st.title("📂 Penentu Klasifikasi Arsip (Generik Stabil)")
st.caption("Pattern-Based Extractor + Hybrid Ranking + AI")

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
# 🔥 EXTRACTOR GENERIK (PATTERN BASED)
# =============================
def extract_inti(text):
    text = text.lower()

    # 1. ambil setelah "tentang" jika ada
    if "tentang" in text:
        text = text.split("tentang", 1)[1]

    # 2. hapus kata administratif umum
    noise = [
        "dengan ini", "saya", "kami", "mengajukan",
        "permohonan", "pelaksanaan", "kegiatan",
        "untuk", "dalam rangka", "sehubungan dengan",
        "berdasarkan", "maka", "adalah"
    ]

    for n in noise:
        text = text.replace(n, "")

    # 3. hapus tahun & angka
    text = re.sub(r'\b(19|20)\d{2}\b', '', text)
    text = re.sub(r'\d+', '', text)

    # 4. hapus kata waktu
    waktu = ["tahun", "tanggal", "bulan", "hari"]
    for w in waktu:
        text = text.replace(w, "")

    words = text.split()

    # 5. ambil kata penting (>=4 huruf)
    important = [w for w in words if len(w) > 3]

    # fallback kalau kosong
    if not important:
        important = words

    return " ".join(important[:6])

# =============================
# 🔥 FUNCTION DETECTOR (UMUM)
# =============================
def detect_functions(text):
    fungsi = []

    mapping = {
        "cuti": "cuti pegawai",
        "pindah": "mutasi pegawai",
        "mutasi": "mutasi pegawai",
        "pensiun": "pensiun pegawai",
        "arsip": "kearsipan",
        "libur": "hari libur nasional"
    }

    for k, v in mapping.items():
        if k in text:
            fungsi.append(v)

    return list(set(fungsi))

# =============================
# BUILD QUERY
# =============================
def build_query(inti, fungsi_list):
    query = inti

    # kurangi bias kata umum
    if "berkas" in query:
        query = query.replace("berkas", "")

    for f in fungsi_list:
        query += " " + f

    return clean_text(query)

# =============================
# SEARCH TEXT
# =============================
def get_search_column(df):
    if "ai_context_final" in df.columns:
        return df["ai_context_final"]
    elif "ai_search_context" in df.columns:
        return df["ai_search_context"]
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

    return (semantic * 0.6) + (keyword * 0.25) + (domain * 0.15)

# =============================
# GEMINI FIX
# =============================
def call_gemini(prompt, api_key):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-pro:generateContent?key={api_key}"

    headers = {"Content-Type": "application/json"}
    data = {"contents": [{"parts": [{"text": prompt}]}]}

    res = requests.post(url, headers=headers, json=data)

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
    st.info(f"🚀 Query: {query}")

    texts = df["search_text"].tolist()
    embeddings = model.encode(texts, show_progress_bar=False)

    sim = cosine_similarity(model.encode([query]), embeddings)[0]

    scores = [calculate_score(query, texts[i], sim[i]) for i in range(len(df))]

    df["score"] = scores
    top = df.sort_values(by="score", ascending=False).head(5)

    st.subheader("📊 Rekomendasi Awal")

    kandidat = []
    for _, r in top.iterrows():
        teks = f"{r['kode']} - {r['uraian']}"
        kandidat.append(teks)
        st.write(f"**{teks}**")
        st.caption(f"Score: {r['score']:.3f}")

    # AI
    with st.spinner("🤖 AI menganalisis..."):
        prompt = f"""
Anda arsiparis ahli.

Fungsi:
{', '.join(fungsi)}

Kandidat:
{chr(10).join(kandidat)}

Pilih 1 paling tepat.

KODE:
...

ALASAN:
jelaskan logis
"""
        st.subheader("🎯 Hasil AI")
        st.write(call_gemini(prompt, API_KEY))

st.divider()
st.caption("Versi Generik Stabil - Tidak tergantung contoh")
