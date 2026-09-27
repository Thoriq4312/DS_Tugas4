"""
app.py — Aplikasi Deployment Segmentasi Siswa (Student Clustering)
====================================================================
Aplikasi ini adalah tahap DEPLOYMENT dari project CRISP-DM:
"Implementasi Clustering untuk Menemukan Pola pada Data Performa Siswa".

Model K-Means (k=2) yang sudah dilatih pada notebook
(student_clustering_crispdm.ipynb) dimuat kembali di sini (tanpa dilatih ulang),
lalu digunakan untuk memprediksi cluster/segmen dari data siswa baru yang
dimasukkan pengguna (guru/staf sekolah) melalui form interaktif.

Sesuai keputusan project, HANYA dua artefak yang disimpan dari notebook:
    - model_kmeans.joblib  (model K-Means final, k=2)
    - scaler.joblib        (StandardScaler yang dipakai saat training)
Urutan fitur dan pemetaan kategori (ParentalSupport) di-hardcode di file ini
agar konsisten persis dengan urutan fitur saat training di notebook.

Cara menjalankan:
    pip install -r requirements.txt
    streamlit run app.py

File yang WAJIB berada di folder yang sama saat menjalankan app ini:
    - model_kmeans.joblib
    - scaler.joblib
    - student_performance_updated_1000.csv (opsional, hanya untuk menampilkan
      ringkasan profil cluster dari data training; app tetap bisa memprediksi
      tanpa file ini)
"""

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Konfigurasi halaman
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Segmentasi Performa Siswa",
    page_icon="🎓",
    layout="centered",
)

# ---------------------------------------------------------------------------
# Konstanta yang HARUS sama persis dengan notebook saat training
# ---------------------------------------------------------------------------
FITUR_CLUSTER = [
    "AttendanceRate",
    "StudyHoursPerWeek",
    "PreviousGrade",
    "FinalGrade",
    "ExtracurricularActivities",
    "ParentalSupport_encoded",
]
PARENTAL_MAPPING = {"Low": 0, "Medium": 1, "High": 2}
DATASET_PATH = "student_performance_updated_1000.csv"


# ---------------------------------------------------------------------------
# Load model & scaler (satu-satunya artefak yang disimpan dari notebook)
# ---------------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load("model_kmeans.joblib")
    scaler = joblib.load("scaler.joblib")
    return model, scaler


try:
    model, scaler = load_artifacts()
    artifacts_ok = True
except FileNotFoundError as e:
    artifacts_ok = False
    missing_file_error = str(e)


@st.cache_data
def hitung_profil_cluster():
    """Membangun ulang profil rata-rata tiap cluster dari data training,
    dengan menerapkan pipeline pembersihan data yang SAMA seperti di notebook,
    lalu memberi label cluster memakai model yang sudah dimuat.
    Fungsi ini bersifat opsional (hanya untuk tampilan ringkasan di UI)."""
    try:
        df_raw = pd.read_csv(DATASET_PATH)
    except FileNotFoundError:
        return None

    df = df_raw.copy()
    df = df.drop(columns=["Attendance (%)", "Study Hours"], errors="ignore")
    df = df.drop(columns=["StudentID", "Name"], errors="ignore")

    kolom_numerik = ["AttendanceRate", "StudyHoursPerWeek", "PreviousGrade",
                      "FinalGrade", "ExtracurricularActivities"]
    kolom_kategorikal = ["Gender", "ParentalSupport", "Online Classes Taken"]

    for col in kolom_numerik:
        df[col] = df[col].fillna(df[col].median())
    for col in kolom_kategorikal:
        df[col] = df[col].fillna(df[col].mode()[0])

    df["ParentalSupport_encoded"] = df["ParentalSupport"].map(PARENTAL_MAPPING)

    X = df[FITUR_CLUSTER]
    X_scaled = scaler.transform(X)
    df["Cluster_KMeans"] = model.predict(X_scaled)

    profil = df.groupby("Cluster_KMeans")[FITUR_CLUSTER].mean().round(2)
    profil["Jumlah Siswa"] = df["Cluster_KMeans"].value_counts().sort_index()
    return profil


def label_rekomendasi(row):
    """Memberi label & rekomendasi sederhana berdasarkan profil rata-rata cluster."""
    final_grade = row["FinalGrade"]
    study_hours = row["StudyHoursPerWeek"]
    attendance = row["AttendanceRate"]

    if final_grade >= 80 and study_hours >= 15:
        return (
            "Siswa Berprestasi & Konsisten",
            "Pertahankan motivasi belajar, berikan program pengayaan (enrichment) "
            "atau tantangan tambahan agar potensi siswa terus berkembang.",
        )
    if final_grade < 75 and (study_hours < 15 or attendance < 80):
        return (
            "Siswa Berisiko / Butuh Perhatian",
            "Disarankan program remedial, pendampingan belajar (mentoring), dan "
            "komunikasi lebih intensif dengan orang tua.",
        )
    if attendance >= 85 and final_grade < 80:
        return (
            "Rajin Hadir, Kurang Efektif Belajar",
            "Fokuskan intervensi pada teknik/strategi belajar (study skills), "
            "bukan sekadar menambah jam belajar.",
        )
    return (
        "Segmen Menengah / Campuran",
        "Pantau perkembangan secara berkala; berikan dukungan sesuai kebutuhan "
        "spesifik siswa pada segmen ini.",
    )


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
st.title("🎓 Segmentasi Performa Siswa (Clustering)")
st.caption(
    "Deployment model K-Means (k=2) dari project CRISP-DM — "
    "masukkan data siswa untuk melihat segmen/cluster hasil prediksi."
)

if not artifacts_ok:
    st.error(
        "File model belum ditemukan di folder ini. Jalankan seluruh sel pada notebook "
        "`student_clustering_crispdm.ipynb` terlebih dahulu (bagian Deployment) untuk "
        "menghasilkan `model_kmeans.joblib` dan `scaler.joblib`, lalu letakkan di folder "
        "yang sama dengan app.py.\n\n"
        f"Detail error: {missing_file_error}"
    )
    st.stop()

st.divider()
st.subheader("Masukkan Data Siswa")

col1, col2 = st.columns(2)

with col1:
    attendance_rate = st.slider("Attendance Rate (%)", 0, 100, 85)
    study_hours = st.slider("Study Hours per Week (jam)", 0, 40, 15)
    previous_grade = st.slider("Previous Grade (nilai sebelumnya)", 0, 100, 75)

with col2:
    final_grade = st.slider("Final Grade (nilai akhir)", 0, 100, 75)
    extracurricular = st.slider("Jumlah Kegiatan Ekstrakurikuler", 0, 5, 1)
    parental_support = st.selectbox(
        "Tingkat Dukungan Orang Tua (Parental Support)",
        options=list(PARENTAL_MAPPING.keys()),
        index=1,
    )

predict_btn = st.button("🔍 Prediksi Segmen Siswa", type="primary", use_container_width=True)

st.divider()

if predict_btn:
    input_dict = {
        "AttendanceRate": attendance_rate,
        "StudyHoursPerWeek": study_hours,
        "PreviousGrade": previous_grade,
        "FinalGrade": final_grade,
        "ExtracurricularActivities": extracurricular,
        "ParentalSupport_encoded": PARENTAL_MAPPING[parental_support],
    }
    input_df = pd.DataFrame([input_dict])[FITUR_CLUSTER]

    input_scaled = scaler.transform(input_df)
    cluster_pred = int(model.predict(input_scaled)[0])

    st.success(f"Siswa ini diprediksi masuk ke **Cluster {cluster_pred}**")

    profil_cluster = hitung_profil_cluster()
    if profil_cluster is not None and cluster_pred in profil_cluster.index:
        profil_terpilih = profil_cluster.loc[cluster_pred]
        label, rekomendasi = label_rekomendasi(profil_terpilih)

        st.markdown(f"**Karakteristik segmen:** {label}")

        st.markdown("#### 📊 Profil Rata-Rata Cluster Ini (dari data training)")
        profil_display = profil_terpilih.drop(labels=["Jumlah Siswa"], errors="ignore")
        st.dataframe(profil_display.rename("Rata-rata").to_frame(), use_container_width=True)

        st.markdown("#### 💡 Rekomendasi Tindak Lanjut")
        st.info(rekomendasi)
    else:
        st.caption(
            f"File `{DATASET_PATH}` tidak ditemukan di folder ini, sehingga profil dan "
            "rekomendasi detail per cluster tidak dapat ditampilkan. Prediksi cluster di "
            "atas tetap valid karena hanya bergantung pada model dan scaler."
        )

    with st.expander("Lihat data input yang dimasukkan"):
        st.dataframe(input_df, use_container_width=True)

st.divider()
profil_semua_cluster = hitung_profil_cluster()
if profil_semua_cluster is not None:
    st.markdown("#### 📋 Ringkasan Seluruh Cluster (Hasil Training)")
    st.caption(
        "Tabel ini menunjukkan karakteristik rata-rata tiap cluster berdasarkan data pelatihan "
        "di notebook — berguna sebagai referensi guru/staf sekolah."
    )
    st.dataframe(profil_semua_cluster, use_container_width=True)

st.markdown(
    "<hr><center><small>Project CRISP-DM — Clustering Performa Siswa · "
    "Dibangun dengan Streamlit</small></center>",
    unsafe_allow_html=True,
)
