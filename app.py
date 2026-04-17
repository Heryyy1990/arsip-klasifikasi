import streamlit as st
import pandas as pd
import re
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# =============================
# SETUP
# =============================
st.set_page_config(page_title="AI Arsip Muna Barat", layout="centered")
st.title("📂 Penentu Klasifikasi Arsip (FINAL ENGINE)")
st.caption("Multi Scoring: Semantic + Keyword + Domain")

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
# STOPWORDS
# =============================
STOPWORDS = [
    "yang","dan","dengan","tentang","untuk","dari","ini",
    "tahun","kepada","sehubungan","dalam"
]

def preprocess(text):
    words = clean(text).split()
    return [w for w in words if w not in STOPWORDS]

# =============================
# NORMALISASI
# =============================
SYNONYM = {
    "pindah": "mutasi",
    "berkas": "",
    "permohonan": "pengajuan",
    "surat": "",
}

def normalize(words):
    return [SYNONYM.get(w, w) for w in words]

# =============================
# EKSTRAK INTI
# =============================
def extract_intent(text):
    words = normalize(preprocess(text))
    return " ".join(words)

# =============================
# DOMAIN DETECTION
# =============================
def predict_domain(query):
    if any(k in query for k in ["pegawai","mutasi","cuti","pensiun"]):
        return "800"
    if any(k in query for k in ["arsip","pemusnahan","retensi"]):
        return "000"
    if any(k in query for k in ["anggaran","keuangan","dana"]):
        return "900"
    if "rapat" in query:
        return "000"
    return None

# =============================
# KEYWORD SCORE
# =============================
def keyword_score(query, text):
    q_words = set(query.split())
    t_words = set(text.split())

    if not q_words:
        return 0

    return len(q_words & t_words) / len(q_words)

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
    # INTI
    # =============================
    inti = extract_intent(perihal)

    st.subheader("🧠 Inti")
    st.write(inti)

    # =============================
    # DOMAIN
    # =============================
    domain = predict_domain(inti)

    st.subheader("🎯 Domain")
    st.write(domain if domain else "Semua")

    # =============================
    # FILTER DATA
    # =============================
    if domain:
        df_filtered = df[df["kode"].astype(str).str.startswith(domain)].copy()
    else:
        df_filtered = df.copy()

    # =============================
    # SEMANTIC SCORE
    # =============================
    texts = df_filtered["search"].tolist()
    embeddings = model.encode(texts, show_progress_bar=False)

    semantic_scores = cosine_similarity(
        model.encode([inti]), embeddings
    )[0]

    df_filtered["semantic"] = semantic_scores

    # =============================
    # KEYWORD SCORE
    # =============================
    df_filtered["keyword"] = df_filtered["search"].apply(
        lambda x: keyword_score(inti, x)
    )

    # =============================
    # DOMAIN BOOST
    # =============================
    def domain_boost(kode):
        if not domain:
            return 0
        return 1 if str(kode).startswith(domain) else 0

    df_filtered["domain_boost"] = df_filtered["kode"].apply(domain_boost)

    # =============================
    # FINAL SCORE
    # =============================
    df_filtered["final_score"] = (
        df_filtered["semantic"] * 0.6 +
        df_filtered["keyword"] * 0.25 +
        df_filtered["domain_boost"] * 0.15
    )

    # =============================
    # HASIL
    # =============================
    top = df_filtered.sort_values(by="final_score", ascending=False).head(5)

    st.subheader("📊 Rekomendasi Kode")

    for _, r in top.iterrows():
        st.write(f"**{r['kode']} - {r['uraian']}**")
        st.caption(
            f"Final: {r['final_score']:.3f} | "
            f"S:{r['semantic']:.2f} K:{r['keyword']:.2f} D:{r['domain_boost']}"
        )

st.divider()
st.caption("FINAL ENGINE - Multi Scoring System")
