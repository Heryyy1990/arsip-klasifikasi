import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import google.generativeai as genai
import json
import os

st.set_page_config(page_title="EKlasifikasi Arsip", page_icon="📁", layout="centered")

st.title("📁 EKlasifikasi Arsip")
st.caption("Hierarchical Classification (Fungsi → Sub → Detail)")

# ================= CONFIG GEMINI =================
api_key = os.getenv("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY", "")
if api_key:
    genai.configure(api_key=api_key)

# ================= LOAD DATA =================
@st.cache_resource
def load_data():
    try:
        df = pd.read_csv("data/klasifikasi.csv", encoding="utf-8")
    except:
        df = pd.read_csv("data/klasifikasi.csv", encoding="latin-1", on_bad_lines="skip")

    df.columns = df.columns.str.lower().str.strip()

    df = df.rename(columns={
        df.columns[0]: "kode",
        df.columns[1]: "uraian"
    })

    df["kode"] = df["kode"].astype(str)
    df["uraian"] = df["uraian"].astype(str)

    df = df[df["uraian"].str.len() > 3].dropna().reset_index(drop=True)

    # ================= HIERARCHY =================
    df["level"] = df["kode"].apply(lambda x: x.count("."))

    df["fungsi_kode"] = df["kode"].str.split(".").str[0]

    df["subfungsi_kode"] = df["kode"].apply(
        lambda x: ".".join(x.split(".")[:2]) if x.count(".") >= 1 else x
    )

    return df

df = load_data()

# ================= MAPPING FUNGSI =================
fungsi_dict = df[df["level"] == 0].set_index("kode")["uraian"].to_dict()

df["fungsi_nama"] = df["fungsi_kode"].map(fungsi_dict)

# ================= MODEL =================
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

@st.cache_resource
def build_embeddings(data):
    return model.encode(data["uraian"].tolist(), batch_size=32, show_progress_bar=False)

embeddings = build_embeddings(df)

# ================= DETEKSI FUNGSI (OUTPUT KODE) =================
def deteksi_fungsi(teks):

    if not api_key:
        return None

    daftar_fungsi = "\n".join([
        f"{k} = {v}" for k, v in fungsi_dict.items()
    ])

    prompt = f"""
Anda adalah arsiparis.

Dokumen:
{teks}

Pilih kode fungsi PALING TEPAT dari daftar berikut:

{daftar_fungsi}

Jawab hanya kodenya saja.
Contoh: 900
"""

    try:
        m = genai.GenerativeModel("gemini-1.5-flash-latest")
        res = m.generate_content(prompt)
        return res.text.strip()
    except:
        return None

# ================= SEARCH FUNCTION =================
def semantic_search(data, query, top_k=5):
    emb = model.encode(data["uraian"].tolist(), batch_size=32)
    q_emb = model.encode([query])
    skor = cosine_similarity(q_emb, emb)[0]
    idx = np.argsort(skor)[::-1][:top_k]

    hasil = []
    for i in idx:
        hasil.append({
            "kode": data.iloc[i]["kode"],
            "uraian": data.iloc[i]["uraian"],
            "skor": skor[i]
        })
    return hasil

# ================= UI =================
with st.form("form"):
    uraian_input = st.text_input("🔎 Masukkan uraian arsip:")
    col1, col2 = st.columns(2)
    submit = col1.form_submit_button("🚀 Cari")
    validasi = col2.form_submit_button("🤖 Validasi AI")

# ================= PROSES =================
if submit and uraian_input:

    fungsi_kode = deteksi_fungsi(uraian_input)

    if not fungsi_kode or fungsi_kode not in fungsi_dict:
        st.warning("⚠️ Fungsi tidak terdeteksi → fallback ke semua data")
        df_fungsi = df
    else:
        st.caption(f"🧭 Fungsi: {fungsi_kode} - {fungsi_dict[fungsi_kode]}")
        df_fungsi = df[df["fungsi_kode"] == fungsi_kode]

    # ================= LEVEL 2 (SUBFUNGSI) =================
    hasil_sub = semantic_search(df_fungsi, uraian_input, top_k=3)

    sub_kode_terpilih = hasil_sub[0]["kode"]

    df_sub = df[df["subfungsi_kode"] == sub_kode_terpilih]

    # ================= LEVEL 3 (DETAIL) =================
    hasil_detail = semantic_search(df_sub, uraian_input, top_k=3)

    st.write("### 📂 Level Fungsi")
    for k in hasil_sub:
        st.write(f"**{k['kode']}** - {k['uraian']} ({k['skor']:.1%})")

    st.write("### 📄 Level Detail")
    for k in hasil_detail:
        st.write(f"**{k['kode']}** - {k['uraian']} ({k['skor']:.1%})")

    st.session_state["kandidat"] = hasil_detail
    st.session_state["uraian"] = uraian_input
    st.session_state["fungsi"] = fungsi_kode

# ================= VALIDASI AI =================
if validasi:

    kandidat = st.session_state.get("kandidat", [])
    uraian_input = st.session_state.get("uraian", "")
    fungsi = st.session_state.get("fungsi", "")

    if not kandidat:
        st.warning("⚠️ Jalankan pencarian dulu")
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

Jawab JSON:
{{
"kode_terpilih": "...",
"alasan": "..."
}}
"""

        try:
            m = genai.GenerativeModel("gemini-1.5-flash-latest")
            res = m.generate_content(prompt)

            raw = res.text.strip().replace("```json", "").replace("```", "")
            hasil = json.loads(raw)

            st.write("### 🤖 Hasil AI")
            st.success(hasil["kode_terpilih"])
            st.info(hasil["alasan"])

        except Exception as e:
            st.error(f"Error: {str(e)}")

# ================= FOOTER =================
st.markdown("---")
st.caption("EKlasifikasi Arsip © 2026")
