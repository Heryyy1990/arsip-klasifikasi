import streamlit as st
import pandas as pd
import re

from sentence_transformers import SentenceTransformer, CrossEncoder
from sklearn.metrics.pairwise import cosine_similarity

# =============================
# SETUP
# =============================
st.set_page_config(page_title="AI Arsip Muna Barat", layout="centered")
st.title("📂 Penentu Klasifikasi Arsip (ULTIMATE ENGINE)")
st.caption("Hybrid AI: NLP + Multi Scoring + Cross Encoder")

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
    "permohonan": "pengajuan",
    "surat": ""
}

def normalize(words):
    return [SYNONYM.get(w, w) for w in words]

# =============================
# INTENT
# =============================
def extract_intent(text):
    words = normalize(preprocess(text))
    return " ".join(words)

# =============================
# DOMAIN
# =============================
def predict_domain(q):
    if any(k in q for k in ["pegawai","mutasi","cuti","pensiun"]):
        return "800"
    if any(k in q for k in ["arsip","pemusnahan","retensi"]):
        return "000"
    if any(k in q for k in ["anggaran","keuangan"]):
        return "900"
    if "rapat" in q:
        return "000"
    return None

# =============================
# QUERY EXPANSION
# =============================
EXPANSION = {
    "mutasi": "mutasi pegawai kepegawaian pindah tugas",
    "cuti": "cuti pegawai kepegawaian izin",
    "arsip": "arsip retensi pemusnahan penyimpanan",
    "rapat": "rapat undangan notulen koordinasi",
    "anggaran": "anggaran keuangan dana kegiatan"
}

def expand_query(q):
    for key in EXPANSION:
        if key in q:
            return q + " " + EXPANSION[key]
    return q

# =============================
# KEYWORD SCORE
# =============================
def keyword_score(q, t):
    q_set = set(q.split())
    t_set = set(t.split())
    if not q_set:
        return 0
    return len(q_set & t_set) / len(q_set)

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
    # INTI + EXPANSION
    # =============================
    inti = extract_intent(perihal)
    query = expand_query(inti)

    st.subheader("🧠 Inti")
    st.write(inti)

    st.subheader("🚀 Query Expansion")
    st.write(query)

    # =============================
    # DOMAIN
    # =============================
    domain = predict_domain(query)

    st.subheader("🎯 Domain")
    st.write(domain if domain else "Semua")

    if domain:
        df_filtered = df[df["kode"].astype(str).str.startswith(domain)].copy()
    else:
        df_filtered = df.copy()

    # =============================
    # STAGE 1 (SEMANTIC)
    # =============================
    texts = df_filtered["search"].tolist()
    emb = embed_model.encode(texts, show_progress_bar=False)

    semantic_scores = cosine_similarity(
        embed_model.encode([query]),
        emb
    )[0]

    df_filtered["semantic"] = semantic_scores

    # ambil kandidat
    candidates = df_filtered.sort_values(by="semantic", ascending=False).head(30).copy()

    # =============================
    # KEYWORD
    # =============================
    candidates["keyword"] = candidates["search"].apply(
        lambda x: keyword_score(query, x)
    )

    # =============================
    # DOMAIN BOOST
    # =============================
    def domain_boost(kode):
        if not domain:
            return 0
        return 1 if str(kode).startswith(domain) else 0

    candidates["domain_boost"] = candidates["kode"].apply(domain_boost)

    # =============================
    # CROSS ENCODER
    # =============================
    pairs = [(query, row["uraian"]) for _, row in candidates.iterrows()]
    cross_scores = cross_model.predict(pairs)

    candidates["cross"] = cross_scores

    # =============================
    # FINAL SCORE
    # =============================
    candidates["final_score"] = (
        candidates["semantic"] * 0.35 +
        candidates["cross"] * 0.40 +
        candidates["keyword"] * 0.15 +
        candidates["domain_boost"] * 0.10
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
            f"S:{r['semantic']:.2f} CE:{r['cross']:.2f} "
            f"K:{r['keyword']:.2f} D:{r['domain_boost']}"
        )

st.divider()
st.caption("ULTIMATE ENGINE (Tanpa Feedback)")
