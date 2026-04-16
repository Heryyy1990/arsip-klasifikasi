import streamlit as st
import pandas as pd
import google.generativeai as genai

# --- KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="AI Archiver - Muna Barat",
    page_icon="🧠",
    layout="wide"
)

# --- STYLE CSS (Custom Tampilan) ---
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stButton>button { 
        width: 100%; 
        border-radius: 8px; 
        height: 3.5em; 
        background-color: #1a73e8; 
        color: white; 
        font-weight: bold;
        border: none;
    }
    .stButton>button:hover { background-color: #1557b0; color: white; }
    .result-card { 
        padding: 20px; 
        border-radius: 10px; 
        background-color: white; 
        border-left: 6px solid #1a73e8; 
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 15px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- LOAD DATASET ---
@st.cache_data
def load_data():
    try:
        # Menggunakan file hasil upgrade dari Colab/Script sebelumnya
        df = pd.read_csv('klasifikasi_arsip.csv')
        return df
    except FileNotFoundError:
        return None

df_arsip = load_data()

# --- KONFIGURASI GOOGLE AI (GEMINI) ---
def init_gemini():
    try:
        # Mengambil API Key dari Streamlit Secrets
        api_key = st.secrets["GEMINI_API_KEY"]
        genai.configure(api_key=api_key)
        return True
    except Exception:
        st.error("❌ API Key tidak ditemukan! Masukkan 'GEMINI_API_KEY' di menu Settings > Secrets pada dashboard Streamlit Cloud.")
        return False

# --- LOGIKA ANALISIS ---
def analyze_context(user_text, dataset):
    # Mengambil sampel context untuk membantu AI (30-50 baris pertama sebagai panduan struktur)
    references = dataset['ai_search_context'].head(50).tolist()
    
    # Model name 'gemini-1.5-flash' adalah yang terbaru dan paling stabil
    model = genai.GenerativeModel('gemini-1.5-flash')
    
    prompt = f"""
    Anda adalah pakar Arsiparis Pemerintah Indonesia.
    Gunakan referensi struktur berikut untuk memahami pola pengkodean:
    {references}

    TUGAS:
    Tentukan 3 Kode Klasifikasi yang paling tepat untuk perihal surat berikut ini:
    "{user_text}"

    ATURAN:
    1. Jika input berkaitan dengan SDM/Pegawai/Cuti, WAJIB arahkan ke rumpun kode 800.
    2. Jika berkaitan dengan Anggaran/Keuangan, arahkan ke rumpun 900.
    3. Analisis secara mendalam (kontekstual), jangan hanya terpaku pada satu kata.
    
    FORMAT JAWABAN (Gunakan Markdown):
    ### 🎯 Rekomendasi 1: [KODE] - [URAIAN]
    **Alasan:** [Jelaskan alasan pemilihan secara logis]

    ### 🥈 Rekomendasi 2: [KODE] - [URAIAN]
    **Alasan:** ...
    """
    
    response = model.generate_content(prompt)
    return response.text

# --- TAMPILAN UTAMA ---
def main():
    st.title("🧠 AI Penentu Kode Klasifikasi Arsip")
    st.markdown("##### Dinas Perpustakaan dan Kearsipan Kabupaten Muna Barat")
    st.divider()

    if df_arsip is None:
        st.error("⚠️ File `klasifikasi_arsip_upgraded.csv` tidak ditemukan di repository GitHub kamu.")
        return

    # Layout kolom untuk input
    col1, col2 = st.columns([2, 1])
    
    with col1:
        user_input = st.text_area(
            "✍️ Masukkan Perihal atau Deskripsi Singkat Arsip:",
            placeholder="Contoh: Permohonan cuti tahunan karena ada urusan keluarga di luar kota...",
            height=150
        )
        
        btn_klik = st.button("Mulai Analisis Kontekstual")

    with col2:
        st.info("""
        **Cara Kerja AI:**
        1. Membaca deskripsi surat Anda.
        2. Mencari pola di database klasifikasi.
        3. Menentukan kode yang paling logis berdasarkan aturan kearsipan.
        """)

    if btn_klik:
        if not user_input:
            st.warning("Silakan masukkan deskripsi arsip terlebih dahulu.")
        else:
            if init_gemini():
                with st.spinner('AI sedang berpikir dan menganalisis kode...'):
                    try:
                        hasil = analyze_context(user_input, df_arsip)
                        st.success("✅ Analisis Berhasil!")
                        st.markdown(f'<div class="result-card">{hasil}</div>', unsafe_allow_html=True)
                    except Exception as e:
                        # Fallback jika model name bermasalah di versi tertentu
                        st.error(f"Terjadi kendala: {e}")
                        st.info("Saran: Pastikan file `requirements.txt` sudah berisi `google-generativeai>=0.5.0`.")

if __name__ == "__main__":
    main()
