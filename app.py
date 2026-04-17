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

st.title("📂 Penentu Klasifikasi Arsip (Smart System)")
st.caption("Embedding + Filtering + AI Reasoning")

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
# CLEAN TEXT
# =============================
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    return text

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
# EMBEDDING MODEL
# =============================
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model_embed = load_model()

# =============================
# DETEKSI KODE
# =============================
def detect_kode(perihal):
    p = perihal.lower()

    if any(k in p for k in ["cuti", "pegawai", "asn"]):
        return "800"
    elif any(k in p for k in ["keuangan", "anggaran", "spj"]):
        return "900"
    elif any(k in p for k in ["rapat", "undangan", "surat"]):
        return "000"
    else:
        return None

# =============================
# INPUT
# =============================
perihal = st.text_area("✍️ Masukkan uraian surat")

if st.button("Analisis"):

    if not perihal:
        st.warning("Isi dulu")
        st.stop()

    # =============================
    # FILTER
    # =============================
    kode_filter = detect_kode(perihal)

    if kode_filter:
        df_filtered = df[df["kode"].astype(str).str.startswith(kode_filter)].reset_index(drop=True)
        st.info(f"🔎 Difilter ke kode {kode_filter}")
    else:
        df_filtered = df.reset_index(drop=True)
        st.warning("⚠️ Tidak terdeteksi kode")

    if df_filtered.empty:
        st.error("❌ Data kosong setelah filter")
        st.stop()

    # =============================
    # EMBEDDING
    # =============================
    texts = df_filtered["search_text"].astype(str).tolist()
    embeddings = model_embed.encode(texts, show_progress_bar=False)

    input_vec = model_embed.encode([perihal])
    sim = cosine_similarity(input_vec, embeddings)

    # ambil top 3 posisi (bukan index asli!)
    top_pos = sim[0].argsort()[-3:][::-1]

    hasil = df_filtered.iloc[top_pos]

    # =============================
    # TAMPILKAN
    # =============================
    st.subheader("📊 Rekomendasi Awal")

    kandidat_list = []

    for rank, pos in enumerate(top_pos):
        row = df_filtered.iloc[pos]

        kode = row.get("kode", "-")
        uraian = row.get("uraian", "-")
        skor = sim[0][pos]

        teks = f"{kode} - {uraian}"
        kandidat_list.append(teks)

        st.write(f"**{teks}**")
        st.caption(f"Similarity: {skor:.3f}")

    # =============================
    # AI
    # =============================
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    model_ai = genai.GenerativeModel('gemini-1.5-flash')

    with st.spinner("🤖 AI menganalisis..."):
        try:
            prompt = f"""
Anda adalah Arsiparis Ahli.

Diberikan kandidat:
{chr(10).join(kandidat_list)}

Perihal:
{perihal}

Analisis:
1. Jelaskan relevansi masing-masing
2. Bandingkan
3. Pilih 1 terbaik

Gunakan:
- fungsi kegiatan
- objek arsip
- konteks administrasi

Format:

ANALISIS:
1. ...
2. ...
3. ...

PERBANDINGAN:
...

REKOMENDASI:
KODE

ALASAN:
jelaskan paling logis dan spesifik
"""

            res = model_ai.generate_content(prompt)

            st.subheader("🎯 Hasil AI")
            st.write(res.text)

        except Exception as e:
            st.error(f"AI Error: {e}")

st.divider()
st.caption("Versi Stabil (Index Fix + Filtering + Embedding)")
