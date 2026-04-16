import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import google.generativeai as genai
import json
import os

# ================= UI CONFIG =================
st.set_page_config(
    page_title="EKlasifikasi Arsip",
    page_icon="📁",
    layout="centered"
)

# ================= STYLE =================
st.markdown("""
<style>
body {
    background-color: #0e1117;
}
.block-container {
    padding-top: 2rem;
}
.title {
    font-size: 32px;
    font-weight: bold;
}
.subtitle {
    color: #9aa0a6;
}
</style>
""", unsafe_allow_html=True)

# ================= HEADER =================
st.markdown('<div class="title">📁 EKlasifikasi Arsip</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">AI Klasifikasi Arsip (Hybrid: Cepat + Pintar)</div>', unsafe_allow_html=True)

st.write("")

# ================= LOAD MODEL =================
@st.cache_resource
def load_data():
    try:
        df = pd.read_csv("data/klasifikasi.csv", encoding="utf-8")
    except:
        df = pd.read_csv("data/klasifikasi.csv", encoding="latin-1", on_bad_lines="skip")

    df.columns = (
        df.columns
        .str.lower()
        .str.strip()
        .str.replace(";", "", regex=False)
    )

    df = df.rename(columns={
        df.columns[0]: "kode",
        df.columns[1]: "uraian"
    })

    model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

    embeddings = model.encode(df["uraian"].astype(str).tolist())

    return df, model, embeddings

df, model, embeddings = load_data()

# ================= FORM (ENTER LANGSUNG SUBMIT) =================
with st.form("form_arsip", clear_on_submit=False):
    uraian_input = st.text_input("🔎 Masukkan uraian arsip:")

    col1, col2 = st.columns(2)

    submit = col1.form_submit_button("🚀 Cari Klasifikasi Cepat")
    validasi = col2.form_submit_button("🤖 Validasi AI (Hemat quota)")

# ================= TAHAP 1 =================
if submit and uraian_input:

    q_emb = model.encode([uraian_input])
    skor = cosine_similarity(q_emb, embeddings)[0]

    top_idx = np.argsort(skor)[::-1][:3]

    kandidat = [
        {
            "kode": str(df.iloc[i]["kode"]),
            "uraian": df.iloc[i]["uraian"],
            "skor": skor[i]
        }
        for i in top_idx
    ]

    st.write("### 📋 Hasil Cepat (Tanpa AI)")
    for k in kandidat:
        st.write(f"**{k['kode']}** - {k['uraian']} ({k['skor']:.1%})")

    st.session_state["kandidat"] = kandidat
    st.session_state["uraian"] = uraian_input

# ================= TAHAP 2 =================
if validasi:

    kandidat = st.session_state.get("kandidat", [])
    uraian_input = st.session_state.get("uraian", "")

    if not kandidat:
        st.warning("⚠️ Jalankan pencarian cepat dulu")
    else:
        api_key = os.getenv("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY", "")

        if not api_key:
            st.error("API Key tidak ditemukan")
        else:
            genai.configure(api_key=api_key)

            kandidat_text = "\n".join([
                f"{i+1}. {k['kode']} - {k['uraian']}"
                for i, k in enumerate(kandidat)
            ])

            prompt = f"""
Anda adalah arsiparis profesional.

Dokumen:
{uraian_input}

Kandidat:
{kandidat_text}

Pilih 1 paling tepat.

Jawab JSON:
{{
"kode_terpilih": "...",
"uraian_terpilih": "...",
"alasan": "..."
}}
"""

            try:
                # 🔥 MODEL YANG PASTI ADA
                model_gemini = genai.GenerativeModel("gemini-1.5-pro")
                response = model_gemini.generate_content(prompt)

                raw = response.text.strip()
                raw = raw.replace("```json", "").replace("```", "")

                hasil = json.loads(raw)

                st.write("### 🤖 Hasil AI")
                st.success(f"{hasil['kode_terpilih']} - {hasil['uraian_terpilih']}")
                st.info(hasil["alasan"])

            except Exception as e:
                st.error(f"Terjadi error: {str(e)}")

# ================= FOOTER =================
st.markdown("---")
st.caption("EKlasifikasi Arsip by Heryanto S.Pd © 2026 | Powered by Gemini AI")
