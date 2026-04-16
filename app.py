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
st.caption("AI Klasifikasi Arsip (Cepat + Validasi AI)")

# ================= LOAD DATA =================
@st.cache_resource
def load_data():
    try:
        df = pd.read_csv("data/klasifikasi.csv", encoding="utf-8")
    except:
        df = pd.read_csv("data/klasifikasi.csv", encoding="latin-1", on_bad_lines="skip")

    df.columns = df.columns.str.lower().str.strip().str.replace(";", "", regex=False)

    df = df.rename(columns={
        df.columns[0]: "kode",
        df.columns[1]: "uraian"
    })

    df["uraian"] = df["uraian"].astype(str)
    df = df[df["uraian"].str.len() > 3].dropna().reset_index(drop=True)

    model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    embeddings = model.encode(df["uraian"].tolist(), batch_size=32, show_progress_bar=False)

    return df, model, embeddings

# ================= FEEDBACK (TIDAK MENGGANGGU RANKING) =================
def load_feedback():
    if os.path.exists("data/feedback.csv"):
        return pd.read_csv("data/feedback.csv")
    return pd.DataFrame(columns=["uraian", "kode"])

def save_feedback(uraian, kode):
    df_fb = load_feedback()
    df_fb = pd.concat([df_fb, pd.DataFrame([{"uraian": uraian, "kode": kode}])])
    df_fb.to_csv("data/feedback.csv", index=False)

# ================= INIT =================
df, model, embeddings = load_data()
feedback_df = load_feedback()

# ================= FORM =================
with st.form("form"):
    uraian_input = st.text_input("🔎 Masukkan uraian arsip:")
    col1, col2 = st.columns(2)
    submit = col1.form_submit_button("🚀 Cari Cepat")
    validasi = col2.form_submit_button("🤖 Validasi AI")

# ================= TAHAP 1 =================
if submit and uraian_input:

    q_emb = model.encode([uraian_input])
    skor = cosine_similarity(q_emb, embeddings)[0]

    idx = np.argsort(skor)[::-1][:3]

    kandidat = [
        {
            "kode": df.iloc[i]["kode"],
            "uraian": df.iloc[i]["uraian"],
            "skor": skor[i]
        }
        for i in idx
    ]

    st.write("### 📋 Rekomendasi")
    for k in kandidat:
        st.write(f"**{k['kode']}** - {k['uraian']} ({k['skor']:.1%})")

    st.session_state["kandidat"] = kandidat
    st.session_state["uraian"] = uraian_input

# ================= PILIH MANUAL =================
if "kandidat" in st.session_state:

    st.write("### ✍️ Pilih Kode yang Benar")

    opsi = [f"{k['kode']} - {k['uraian']}" for k in st.session_state["kandidat"]]

    pilihan = st.selectbox("Pilih hasil yang paling tepat:", opsi)

    if st.button("💾 Simpan"):
        kode_terpilih = pilihan.split(" - ")[0]
        save_feedback(st.session_state["uraian"], kode_terpilih)
        st.success("Tersimpan")

# ================= TAHAP 2 (OPSIONAL AI) =================
if validasi:

    kandidat = st.session_state.get("kandidat", [])
    uraian_input = st.session_state.get("uraian", "")

    if not kandidat:
        st.warning("⚠️ Jalankan pencarian dulu")
    else:
        api_key = os.getenv("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY", "")

        if not api_key:
            st.error("API key tidak ada")
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

Pilih 1 paling tepat berdasarkan fungsi kegiatan.

Jawab JSON:
{{
"kode_terpilih": "...",
"uraian_terpilih": "...",
"alasan": "..."
}}
"""

            try:
                model_gemini = genai.GenerativeModel("gemini-1.5-pro")
                res = model_gemini.generate_content(prompt)

                raw = res.text.strip().replace("```json", "").replace("```", "")
                hasil = json.loads(raw)

                st.write("### 🤖 Hasil AI")
                st.success(f"{hasil['kode_terpilih']} - {hasil['uraian_terpilih']}")
                st.info(hasil["alasan"])

            except Exception as e:
                st.error(f"Error: {str(e)}")

# ================= FOOTER =================
st.markdown("---")
st.caption("EKlasifikasi Arsip © 2026")
