import streamlit as st
import pandas as pd
import re
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# =============================
# SETUP
# =============================
st.set_page_config(page_title="AI Arsip Muna Barat", layout="centered")
st.title("📂 Penentu Klasifikasi Arsip (Local Intelligence PRO)")
st.caption("Rule + NLP + Semantic")

# =============================
# LOAD DATA
# =============================
@st.cache_data
def load_data():
    return pd.read_csv("klasifikasi_arsip_upgraded.csv")

df = load_data()

# =============================
# TEXT CLEANING
# =============================
def clean(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    return text

STOPWORDS = [
    "yang","dan","dengan","tentang","untuk","dari","ini","tahun",
    "kepada","sehubungan","dalam","berdasarkan"
]

def preprocess(text):
    words = clean(text).split()
    words = [w for w in words if w not in STOPWORDS]
    return words

# =============================
# NORMALISASI DOMAIN
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
# EKSTRAK INTI (SMART)
# =============================
def extract_intent(text):
    words = preprocess(text)
    words = normalize(words)

    teks = " ".join(words)

    aksi = ""
    objek = ""

    # DETEKSI AKSI
    if "pengajuan" in teks:
        aksi = "pengajuan"
    elif "undangan" in teks:
        aksi = "undangan"
    elif "laporan" in teks:
        aksi = "pelaporan"
    elif "pemusnahan" in teks:
        aksi = "pemusnahan"
    elif "rapat" in teks:
        aksi = "rapat"

    # DETEKSI OBJEK
    if "pegawai" in teks or "mutasi" in teks:
        objek = "pegawai"
    elif "arsip" in teks:
        objek = "arsip"
    elif "anggaran" in teks:
        objek = "anggaran"
    elif "cuti" in teks:
        objek = "cuti"

    inti = f"{aksi} {objek}".strip()

    if not inti:
        inti = teks

    return inti

# =============================
# BOOST DOMAIN
# =============================
def boost_score(row, query):
    score = row["score"]

    # kepegawaian
    if "pegawai" in query and str(row["kode"]).startswith("800"):
        score += 0.15

    # arsip
    if "arsip" in query and str(row["kode"]).startswith("000"):
        score += 0.15

    # anggaran
    if "anggaran" in query and str(row["kode"]).startswith("900"):
        score += 0.15

    return score

# =============================
# EMBEDDING
# =============================
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

df["search"] = df["uraian"].apply(clean)

# =============================
# INPUT
# =============================
perihal = st.text_area("✍️ Masukkan uraian surat")

if st.button("Analisis"):

    if not perihal:
        st.warning("Isi dulu")
        st.stop()

    # =============================
    # EKSTRAK INTI
    # =============================
    inti = extract_intent(perihal)

    st.subheader("🧠 Inti Hasil Analisis")
    st.write(inti)

    # =============================
    # EMBEDDING MATCH
    # =============================
    texts = df["search"].tolist()
    emb = model.encode(texts, show_progress_bar=False)

    sim = cosine_similarity(model.encode([inti]), emb)[0]

    df["score"] = sim

    # =============================
    # BOOSTING
    # =============================
    df["final_score"] = df.apply(lambda r: boost_score(r, inti), axis=1)

    top = df.sort_values(by="final_score", ascending=False).head(5)

    st.subheader("📊 Rekomendasi Kode")

    for _, r in top.iterrows():
        st.write(f"**{r['kode']} - {r['uraian']}**")
        st.caption(f"Score: {r['final_score']:.3f}")
