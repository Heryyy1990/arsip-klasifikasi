import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import google.generativeai as genai
import json
import os

st.set_page_config(
    page_title="Klasifikasi Arsip Pemerintah",
    page_icon="🗂️",
    layout="wide"
)

# ── HEADER ─────────────────────────────
st.title("🗂️ Sistem Rekomendasi Klasifikasi Arsip")
st.caption("Model Semantik + Validasi Arsiparis Digital (Gemini)")

# ── LOAD DATA ─────────────────────────
@st.cache_resource
def load_semua():
    # baca CSV (aman dari error)
    try:
        df = pd.read_csv("data/klasifikasi.csv", encoding="utf-8")
    except:
        df = pd.read_csv("data/klasifikasi.csv", encoding="latin-1", on_bad_lines="skip")

    # 🔥 NORMALISASI KOLOM (INI KUNCI)
    df.columns = (
        df.columns
        .str.lower()
        .str.strip()
        .str.replace(";", "", regex=False)
    )

    # ambil kolom secara fleksibel
    kolom_kode = [c for c in df.columns if "kode" in c]
    kolom_uraian = [c for c in df.columns if "uraian" in c]

    if not kolom_kode or not kolom_uraian:
        st.error(f"Kolom tidak ditemukan. Kolom tersedia: {list(df.columns)}")
        st.stop()

    kolom_kode = kolom_kode[0]
    kolom_uraian = kolom_uraian[0]

    # buang data kosong
    df = df[[kolom_kode, kolom_uraian]].dropna()

    # rename standar
    df = df.rename(columns={
        kolom_kode: "kode",
        kolom_uraian: "uraian"
    })

    # model semantic
    model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

    embeddings = model.encode(
        df["uraian"].astype(str).tolist(),
        show_progress_bar=False
    )

    return df, model, embeddings

with st.spinner("⏳ Memuat model AI..."):
    df, model, embeddings = load_semua()

st.success(f"✅ Siap — {len(df)} kode klasifikasi tersedia")
st.divider()

# ── INPUT ─────────────────────────────
uraian_input = st.text_area(
    "📝 Masukkan uraian arsip:",
    height=130
)

proses = st.button("🔍 Klasifikasikan")

# ── PROSES ────────────────────────────
if proses and uraian_input.strip():

    # Tahap 1 — Semantic
    with st.spinner("🔎 Analisis semantik..."):
        q_emb = model.encode([uraian_input])
        skor = cosine_similarity(q_emb, embeddings)[0]

        top3_idx = np.argsort(skor)[::-1][:3]

        kandidat = [
            {
                "kode": str(df.iloc[i]["kode"]),
                "uraian": df.iloc[i]["uraian"],
                "skor": float(skor[i])
            }
            for i in top3_idx
        ]

    st.subheader("📋 3 Kandidat Terbaik")

    for k in kandidat:
        st.write(f"**{k['kode']}** — {k['uraian']} ({k['skor']:.1%})")

    # Tahap 2 — Gemini
    st.subheader("🤖 Validasi AI")

    api_key = os.getenv("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY", "")

    if not api_key:
        st.error("⚠️ API Key Google AI tidak ditemukan.")
    else:
        genai.configure(api_key=api_key)

        kandidat_text = "\n".join([
            f"{i+1}. {k['kode']} - {k['uraian']}"
            for i, k in enumerate(kandidat)
        ])

        prompt = f"""
Anda adalah arsiparis profesional pemerintahan Indonesia.

DOKUMEN:
{uraian_input}

KANDIDAT:
{kandidat_text}

Gunakan langkah:
1. Identifikasi isi utama
2. Tentukan fungsi kegiatan
3. Cocokkan klasifikasi
4. Pilih paling spesifik
5. Perhatikan konteks unit kerja

Jawab hanya JSON:
{{
"kode_terpilih": "...",
"uraian_terpilih": "...",
"alasan": "..."
}}
"""

        try:
            model_gemini = genai.GenerativeModel("gemini-1.5-flash")
            response = model_gemini.generate_content(prompt)

            raw = response.text.strip()
            raw = raw.replace("```json", "").replace("```", "").strip()

            hasil = json.loads(raw)

            st.success("✅ Hasil Final")

            st.metric("📌 Kode", hasil["kode_terpilih"])
            st.write(f"**Uraian:** {hasil['uraian_terpilih']}")
            st.info(f"💡 {hasil['alasan']}")

        except Exception as e:
            st.error(f"Terjadi error: {str(e)}")

elif proses:
    st.warning("⚠️ Masukkan uraian arsip terlebih dahulu.")
