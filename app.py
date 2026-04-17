# =============================
# 🔥 FUNCTION DETECTOR LEVEL ANRI (UPGRADE)
# =============================
def detect_functions(text):
    fungsi = []

    # PRIORITAS TERTINGGI → jenis dokumen
    if "undangan" in text:
        fungsi.append("rapat")
        fungsi.append("administrasi rapat")

    if "berita acara" in text:
        fungsi.append("dokumentasi kegiatan")

    if "laporan" in text:
        fungsi.append("pelaporan")

    if "nota dinas" in text:
        fungsi.append("administrasi internal")

    # PRIORITAS KEDUA → kegiatan
    if "rapat" in text:
        fungsi.append("rapat koordinasi")

    if "cuti" in text:
        fungsi.append("cuti pegawai")

    if "pindah" in text or "mutasi" in text:
        fungsi.append("mutasi pegawai")

    if "anggaran" in text:
        fungsi.append("pengelolaan anggaran")

    return list(set(fungsi))
