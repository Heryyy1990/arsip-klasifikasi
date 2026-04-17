import streamlit as st
import pandas as pd
import re

from sentence_transformers import SentenceTransformer, CrossEncoder
from sklearn.metrics.pairwise import cosine_similarity

# =============================
# SETUP
# =============================
st.set_page_config(page_title="AI Arsip Muna Barat", layout="centered")
st.title("📂 Penentu Klasifikasi Arsip (Cross-Encoder PRO)")
st.caption("Stage 1: Retrieval | Stage 2: Reranking")

# =============================
# LOAD DATA
# =============================
@st.cache_data
def load_data():
    return pd.read_csv("klasifikasi_arsip_upgraded.csv")

df = load_data()

def clean(text):
    return re.sub(r'[^a-z0-9\s]', '', str(text).lower())

df["search"] = df["uraian"].apply(clean)

# =============================
# PREPROCESS
# =============================
STOPWORDS = ["yang","dan","dengan","tentang","untuk","dari","ini"]

def preprocess(text):
    words = clean(text).split()
    return [w for w in words if w not in STOPWORDS]

SYNONYM = {
    "pindah": "mutasi",
    "permohonan": "pengajuan"
}

def normalize(words):
    return [SYNONYM.get(w, w) for w in words]

def extract_intent(text):
    words = normalize(preprocess(text))
    return " ".join(words)

# =============================
# DOMAIN
# =============================
def predict_domain(q):
    if any(k in q for k in ["pegawai","mutasi","cuti"]):
        return "800"
    if "arsip" in q:
        return "000"
    if "anggaran" in q:
        return "900"
    return None

# =============================
# MODELS
# =============================
@st.cache_resource
def load_models():
    embed = SentenceTransformer('all-MiniLM-L6-v2')
    cross = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    return embed, cross

embed_model, cross_model = load_models()

# =============================
# INPUT
# =============================
perihal = st.text_area("✍️ Masukkan uraian surat")

if st.button("Analisis"):

    if not perihal:
        st.warning("Isi dulu")
        st.stop()

    # =============================
    # INTI
    # =============================
    inti = extract_intent(perihal)

    st.subheader("🧠 Inti")
    st.write(inti)

    # =============================
    # DOMAIN
    # =============================
    domain = predict_domain(inti)

    if domain:
        df_filtered = df[df["kode"].astype(str).str.startswith(domain)].copy()
    else:
        df_filtered = df.copy()

    # =============================
    # STAGE 1 (RETRIEVAL)
    # =============================
    texts = df_filtered["search"].tolist()
    emb = embed_model.encode(texts, show_progress_bar=False)

    sim = cosine_similarity(
        embed_model.encode([inti]),
        emb
    )[0]

    df_filtered["semantic"] = sim

    # ambil top 30 kandidat
    candidates = df_filtered.sort_values(by="semantic", ascending=False).head(30).copy()

    # =============================
    # STAGE 2 (CROSS-ENCODER)
    # =============================
    pairs = [(inti, row["uraian"]) for _, row in candidates.iterrows()]
    scores = cross_model.predict(pairs)

    candidates["cross_score"] = scores

    # =============================
    # FINAL SCORE
    # =============================
    candidates["final_score"] = (
        candidates["semantic"] * 0.4 +
        candidates["cross_score"] * 0.6
    )

    top = candidates.sort_values(by="final_score", ascending=False).head(5)

    # =============================
    # OUTPUT
    # =============================
    st.subheader("📊 Rekomendasi Kode")

    for _, r in top.iterrows():
        st.write(f"**{r['kode']} - {r['uraian']}**")
        st.caption(
            f"Final: {r['final_score']:.3f} | "
            f"S:{r['semantic']:.2f} CE:{r['cross_score']:.2f}"
        )

st.divider()
st.caption("Engine: Two-Stage Retrieval + Cross Encoder")
