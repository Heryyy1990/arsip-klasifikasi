import streamlit as st
import pandas as pd
import re
import requests
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# =============================
# SETUP
# =============================
st.set_page_config(page_title="AI Arsip Muna Barat", layout="centered")
st.title("📂 Penentu Klasifikasi Arsip (AI Natural Thinking)")
st.caption("AI berpikir seperti arsiparis (tanpa daftar kaku)")

API_KEY = st.secrets["GEMINI_API_KEY"]

# =============================
# GEMINI (GUIDED THINKING)
# =============================
def call_gemini(perihal):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent?key={API_KEY}"

    prompt = f"""
Anda adalah ARSIPARIS AHLI.

Gunakan cara berpikir profesional:

1. Tentukan jenis dokumen
2. Tentukan kegiatan utama (aksi)
3. Tentukan objek yang diproses
4. Rumuskan INTI (maks 5 kata)
5. Rumuskan FUNGSI secara spesifik (contoh: "administrasi rapat koordinasi", "pengelolaan cuti pegawai")

ATURAN:
- Jangan terlalu umum (hindari: "administrasi")
- Harus spesifik sesuai kegiatan
- Fokus pada fungsi utama, bukan kata terakhir

PERIHAL:
{perihal}

FORMAT:

JENIS:
AKSI:
OBJEK:
INTI:
FUNGSI:
"""

    data = {"contents": [{"parts": [{"text": prompt}]}]}
    res = requests.post(url, json=data)

    if res.status_code != 200:
        return f"❌ Error API: {res.text}"

    return res.json()["candidates"][0]["content"]["parts"][0]["text"]

# =============================
# LOAD DATA
# =============================
@st.cache_data
def load_data():
    return pd.read_csv("klasifikasi_arsip.csv")

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

model = load_model()

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
    with st.spinner("🧠 AI memahami seperti arsiparis..."):
        hasil = call_gemini(perihal)

    if "Error API" in hasil:
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
    emb = model.encode(texts, show_progress_bar=False)

    sim = cosine_similarity(model.encode([query]), emb)[0]

    df["score"] = sim
    top = df.sort_values(by="score", ascending=False).head(5)

    st.subheader("📊 Rekomendasi Kode")

    for _, r in top.iterrows():
        st.write(f"**{r['kode']} - {r['uraian']}**")
        st.caption(f"Score: {r['score']:.3f}")

st.divider()
st.caption("AI Natural Thinking - Tanpa daftar fungsi")
