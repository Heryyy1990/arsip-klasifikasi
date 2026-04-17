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
st.title("📂 Penentu Klasifikasi Arsip (Next Level)")
st.caption("Function-Based Ranking + AI")

# =============================
# API KEY
# =============================
if "GEMINI_API_KEY" not in st.secrets:
    st.error("❌ API Key tidak ditemukan!")
    st.stop()

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
# CLEAN
# =============================
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    return text

# =============================
# EXTRACT INTI
# =============================
def extract_inti(text):
    text = text.lower()

    noise = [
        "dengan ini", "saya", "mengajukan", "permohonan",
        "untuk", "melakukan", "dalam rangka",
        "melengkapi", "persyaratan"
    ]

    for n in noise:
        text = text.replace(n, "")

    words = text.split()
    if len(words) > 3:
        words = words[-3:]

    return " ".join(words)

# =============================
# 🔥 FUNCTION DETECTOR (KUNCI)
# =============================
def detect_function(text):
    if "pindah" in text or "mutasi" in text:
        return "mutasi pegawai"
    elif "cuti" in text:
        return "cuti pegawai"
    elif "pensiun" in text:
        return "pensiun pegawai"
    elif "arsip" in text:
        return "kearsipan arsip"
    else:
        return ""

# =============================
# BUILD QUERY
# =============================
def build_query(inti):
    fungsi = detect_function(inti)

    query = inti

    if "berkas" in query:
        query = query.replace("berkas", "")

    query = query + " " + fungsi

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
def calculate_score(query, text, semantic_score):
    keyword_score = 0
    domain_score = 0

    for word in query.split():
        if word in text:
            keyword_score += 1

    keyword_score = keyword_score / (len(query.split()) + 1)

    if "pegawai" in query and "pegawai" in text:
        domain_score += 1

    return (semantic_score * 0.6) + (keyword_score * 0.25) + (domain_score * 0.15)

# =============================
# GEMINI API
# =============================
def call_gemini(prompt, api_key):
    url = f"https://generativelanguage.googleapis.com/v1/models/gemini-1.5-flash:generateContent?key={api_key}"

    headers = {"Content-Type": "application/json"}

    data = {
        "contents": [{"parts": [{"text": prompt}]}]
    }

    res = requests.post(url, headers=headers, json=data)

    if res.status_code != 200:
        return f"❌ Error API: {res.text}"

    result = res.json()
    return result["candidates"][0]["content"]["parts"][0]["text"]

# =============================
# INPUT
# =============================
perihal = st.text_area("✍️ Masukkan uraian surat")

if st.button("Analisis"):

    if not perihal:
        st.warning("Isi dulu")
        st.stop()

    inti = extract_inti(perihal)
    fungsi = detect_function(inti)
    query = build_query(inti)

    st.info(f"🧠 Inti: {inti}")
    st.info(f"🎯 Fungsi: {fungsi}")
    st.info(f"🚀 Query: {query}")

    texts = df["search_text"].tolist()
    embeddings = model.encode(texts, show_progress_bar=False)

    input_vec = model.encode([query])
    sim = cosine_similarity(input_vec, embeddings)[0]

    scores = []

    for i, row in df.iterrows():
        text = row["search_text"]
        final_score = calculate_score(query, text, sim[i])
        scores.append(final_score)

    df["final_score"] = scores
    df_sorted = df.sort_values(by="final_score", ascending=False).head(5)

    st.subheader("📊 Rekomendasi Awal")

    kandidat_list = []

    for _, row in df_sorted.iterrows():
        teks = f"{row['kode']} - {row['uraian']}"
        kandidat_list.append(teks)

        st.write(f"**{teks}**")
        st.caption(f"Score: {row['final_score']:.3f}")

    # =============================
    # GEMINI (TETAP ADA)
    # =============================
    with st.spinner("🤖 AI menganalisis..."):

        prompt = f"""
Anda arsiparis ahli.

Fungsi kegiatan:
{fungsi}

Kandidat:
{chr(10).join(kandidat_list)}

Pilih 1 paling tepat.

KODE:
...

ALASAN:
jelaskan berbasis fungsi
"""

        hasil_ai = call_gemini(prompt, st.secrets["GEMINI_API_KEY"])

        st.subheader("🎯 Hasil AI")
        st.write(hasil_ai)

st.divider()
st.caption("Versi Next Level (Function-Based)")
