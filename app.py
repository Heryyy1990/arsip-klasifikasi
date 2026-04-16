import streamlit as st
import pandas as pd
import numpy as np
import re
import json
import os

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import google.generativeai as genai

# ================= CONFIG =================
st.set_page_config(
    page_title="Klasifikasi Arsip Pemerintah",
    page_icon="🗂️",
    layout="wide"
)

st.title("🗂️ Sistem Klasifikasi Arsip")
st.caption("AI Berbasis Fungsi + Validasi Arsiparis (Gemini)")

# ================= API KEY =================
api_key = st.secrets.get("GEMINI_API_KEY", "") or os.getenv("GEMINI_API_KEY")

if api_key:
    genai.configure(api_key=api_key)

# ================= LOAD DATA =================
@st.cache_resource
def load_data():
    df = pd.read_csv("data/klasifikasi.csv")
    df.columns = df.columns.str.lower().str.strip()

    df = df.rename(columns={
        df.columns[0]: "kode",
        df.columns[1]: "uraian"
    })

    df["kode"] = df["kode"].astype(str).str.strip()
    df["uraian"] = df["uraian"].astype(str)

    df = df[df["uraian"].str.len() > 3].dropna().reset_index(drop=True)

    # hierarchy
    df["level"] = df["kode"].apply(lambda x: x.count("."))
    df["fungsi_kode"] = df["kode"].str.split(".").str[0]

    return df

df = load_data()

# mapping fungsi
fungsi_dict = df[df["level"] == 0].set_index("kode")["uraian"].to_dict()

# ================= MODEL =================
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")  # lebih stabil

model = load_model()

@st.cache_resource
def build_embeddings(data):
    return model.encode(data["uraian"].tolist(), show_progress_bar=False)

embeddings = build_embeddings(df)

# ================= DETEKSI FUNGSI =================
def deteksi_fungsi(teks):

    if not api_key:
        return None

    daftar = "\n".join([f"{k} = {v}" for k, v in fungsi_dict.items()])

    prompt = f"""
Anda arsiparis.

Dokumen:
{teks}

Pilih kode fungsi paling tepat dari daftar:

{daftar}

Jawab HANYA angka.
"""

    try:
        m = genai.GenerativeModel("gemini-1.5-flash-latest")
        res = m.generate_content(prompt)

        raw = res.text.strip()
        kode = re.findall(r"\d+", raw)

        return kode[0] if kode else None

    except:
        return None

# ================= SEARCH =================
def semantic_search(data, query, top_k=3):
    emb = model.encode(data["uraian"].tolist())
    q_emb = model.encode([query])

    skor = cosine_similarity(q_emb, emb)[0]
    idx = np.argsort(skor)[::-1][:top_k]

    return [
        {
            "kode": data.iloc[i]["kode"],
            "uraian": data.iloc[i]["uraian"],
            "skor": skor[i]
        }
        for i in idx
    ]

# ================= UI =================
uraian_input = st.text_area("📝 Uraian Arsip", height=150)

col1, col2 = st.columns(2)
submit = col1.button("🔍 Klasifikasikan", use_container_width=True)
validasi = col2.button("🤖 Validasi AI", use_container_width=True)

# ================= PROSES =================
if submit and uraian_input:

    # ===== Tahap 0: fungsi =====
    fungsi = deteksi_fungsi(uraian_input)

    if fungsi and fungsi in fungsi_dict:
        st.caption(f"🧭 Fungsi: {fungsi} - {fungsi_dict[fungsi]}")
        df_filtered = df[df["fungsi_kode"] == fungsi]
    else:
        st.warning("⚠️ Fungsi tidak terdeteksi → fallback")
        df_filtered = df

    # ===== Tahap 1: kandidat =====
    kandidat = semantic_search(df_filtered, uraian_input, top_k=3)

    st.subheader("📋 Kandidat Klasifikasi")

    for k in kandidat:
        st.write(f"**{k['kode']}** - {k['uraian']} ({k['skor']:.1%})")

    st.session_state["kandidat"] = kandidat
    st.session_state["uraian"] = uraian_input
    st.session_state["fungsi"] = fungsi

# ================= VALIDASI GEMINI =================
if validasi:

    kandidat = st.session_state.get("kandidat", [])
    uraian_input = st.session_state.get("uraian", "")
    fungsi = st.session_state.get("fungsi", "")

    if not kandidat:
        st.warning("⚠️ Jalankan klasifikasi dulu")
    elif not api_key:
        st.error("API key tidak ada")
    else:

        kandidat_text = "\n".join([
            f"{i+1}. {k['kode']} - {k['uraian']}"
            for i, k in enumerate(kandidat)
        ])

        prompt = f"""
Anda arsiparis profesional.

Dokumen:
{uraian_input}

Fungsi:
{fungsi}

Kandidat:
{kandidat_text}

Pilih yang paling tepat.

WAJIB:
- Output JSON
- Tanpa markdown

Format:
{{
"kode_terpilih": "...",
"uraian_terpilih": "...",
"alasan": "..."
}}
"""

        try:
            m = genai.GenerativeModel("gemini-1.5-flash-latest")
            res = m.generate_content(prompt)

            raw = res.text.strip().replace("```json", "").replace("```", "").strip()

            if not raw.startswith("{"):
                raise ValueError("Invalid JSON")

            hasil = json.loads(raw)

        except:
            # fallback aman
            hasil = {
                "kode_terpilih": kandidat[0]["kode"],
                "uraian_terpilih": kandidat[0]["uraian"],
                "alasan": "Fallback: AI gagal parsing"
            }

        st.subheader("✅ Hasil Final")

        col1, col2 = st.columns([1,2])

        with col1:
            st.metric("Kode", hasil["kode_terpilih"])
            st.write(hasil["uraian_terpilih"])

        with col2:
            st.info(hasil["alasan"])

# ================= FOOTER =================
st.markdown("---")
st.caption("EKlasifikasi Arsip © 2026")
