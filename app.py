# ======================================================================
# ITEM ANALYSIS - STREAMLIT VERSION (LENGKAP)
# ======================================================================
# Komponen lengkap:
# 1. p (tingkat kesukaran)
# 2. q (1-p)
# 3. pq (varians butir)
# 4. p_high (kelompok atas)
# 5. p_low (kelompok bawah)
# 6. D (daya beda)
# 7. SE (standard error butir) = sqrt(pq/n)
# 8. r_it (validitas corrected)
# 9. KR-20 (reliabilitas)
# 10. Alpha if item deleted
# 11. SEM (standard error measurement)
# 12. Analisis pengecoh
# ======================================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pointbiserialr
import io
import warnings
warnings.filterwarnings('ignore')

# ======================================================================
# KONFIGURASI HALAMAN
# ======================================================================
st.set_page_config(page_title="Item Analysis Tool", page_icon="📊", layout="wide")

st.title("📊 ITEM ANALYSIS TOOL (LENGKAP)")
st.markdown("---")

# ======================================================================
# SIDEBAR - PARAMETER
# ======================================================================
with st.sidebar:
    st.header("⚙️ Parameter Ambang Batas")
    
    st.subheader("Tingkat Kesukaran (p)")
    col1, col2 = st.columns(2)
    with col1:
        batas_sukar = st.number_input("Batas Sukar", value=0.30, step=0.05)
    with col2:
        batas_mudah = st.number_input("Batas Mudah", value=0.80, step=0.05)
    
    st.subheader("Daya Beda (D)")
    col1, col2 = st.columns(2)
    with col1:
        batas_cukup = st.number_input("Batas Cukup", value=0.20, step=0.05)
    with col2:
        batas_baik = st.number_input("Batas Sangat Baik", value=0.40, step=0.05)
    
    st.subheader("Validitas (r_it)")
    batas_valid = st.number_input("Batas Valid", value=0.20, step=0.05)
    
    st.subheader("Kelompok Daya Beda")
    persen_kelompok = st.slider("Persentase Kelompok Atas/Bawah", min_value=10, max_value=50, value=27, step=1)
    
    st.markdown("---")
    st.caption("Scripted by Muhaimin Abdullah")

# ======================================================================
# FUNGSI INTERPRETASI
# ======================================================================
def interpretasi_p(p, batas_sukar, batas_mudah):
    if p < batas_sukar:
        return "Sukar", "Soal terlalu sulit, hanya sedikit siswa yang menjawab benar."
    elif p <= batas_mudah:
        return "Sedang", "Soal memiliki tingkat kesulitan yang baik."
    else:
        return "Mudah", "Soal terlalu mudah, hampir semua siswa menjawab benar."

def interpretasi_d(d, batas_cukup, batas_baik):
    if d < batas_cukup:
        return "Jelek", "Soal tidak bisa membedakan siswa pandai dan kurang pandai."
    elif d < batas_baik:
        return "Cukup", "Soal cukup mampu membedakan siswa."
    else:
        return "Sangat Baik", "Soal sangat baik dalam membedakan siswa."

# ======================================================================
# INITIALIZE SESSION STATE
# ======================================================================
if 'df' not in st.session_state:
    st.session_state.df = None
if 'kunci_df' not in st.session_state:
    st.session_state.kunci_df = None
if 'file_loaded' not in st.session_state:
    st.session_state.file_loaded = False

# ======================================================================
# INPUT DATA
# ======================================================================
tab1, tab2 = st.tabs(["📁 Upload Data", "📊 Hasil Analisis"])

with tab1:
    st.subheader("Upload File Jawaban Siswa")
    
    file_siswa = st.file_uploader("Pilih file CSV jawaban siswa", type=['csv'], key="siswa")
    
    if file_siswa is not None:
        if file_siswa.size > 5 * 1024 * 1024:
            st.error("❌ Ukuran file terlalu besar! Maksimal 5MB.")
        else:
            try:
                file_siswa.seek(0)
                df = pd.read_csv(file_siswa, dtype=str)
                
                if df.empty:
                    st.error("❌ Data kosong! Silakan periksa file Anda.")
                elif len(df.columns) < 2:
                    st.error("❌ Data harus memiliki minimal 2 kolom (kolom siswa + minimal 1 kolom soal)")
                else:
                    st.success(f"✅ File terupload: {file_siswa.name}")
                    st.write(f"Dimensi: {df.shape[0]} baris × {df.shape[1]} kolom")
                    st.subheader("Preview Data")
                    st.dataframe(df.head())
                    
                    st.session_state.df = df
                    st.session_state.file_loaded = True
            except Exception as e:
                st.error(f"❌ Error membaca file: {str(e)}")
    else:
        if not st.session_state.file_loaded:
            st.info("Belum ada file diupload")
    
    st.subheader("Upload File Kunci Jawaban (Opsional)")
    st.caption("Kosongkan jika data sudah dalam format 1/0")
    
    file_kunci = st.file_uploader("Pilih file CSV kunci jawaban", type=['csv'], key="kunci")
    
    if file_kunci is not None:
        if file_kunci.size > 5 * 1024 * 1024:
            st.error("❌ Ukuran file kunci terlalu besar! Maksimal 5MB.")
        else:
            try:
                file_kunci.seek(0)
                df_kunci = pd.read_csv(file_kunci, dtype=str)
                
                if not df_kunci.empty:
                    st.success(f"✅ File kunci terupload")
                    st.session_state.kunci_df = df_kunci
                else:
                    st.warning("⚠️ File kunci kosong")
            except Exception as e:
                st.warning(f"⚠️ Error membaca file kunci: {str(e)}")
                st.session_state.kunci_df = None

# ======================================================================
# PROSES ANALISIS
# ======================================================================
if st.session_state.file_loaded and st.session_state.df is not None:
    
    df = st.session_state.df.copy()
    df_kunci = st.session_state.kunci_df
    
    kolom_soal = df.columns[1:].tolist()
    
    if len(kolom_soal) == 0:
        with tab2:
            st.error("❌ Tidak ada kolom soal! Pastikan file memiliki minimal 2 kolom.")
    else:
        sample = df[kolom_soal[0]].dropna().astype(str).str.strip().values
        sample_clean = [s for s in sample if s not in ['', 'nan', 'NaN', 'None']]
        
        if len(sample_clean) > 0:
            is_biner = all(v in ['0', '1'] for v in sample_clean[:50])
        else:
            is_biner = False
        
        mode = "biner" if is_biner else "pilihan_ganda"
        
        kunci = None
        if mode == "pilihan_ganda" and df_kunci is not None and not df_kunci.empty:
            try:
                if df_kunci.shape[1] > 1:
                    kunci = [str(x).strip().upper() for x in df_kunci.iloc[0, 1:].values]
                else:
                    kunci = [str(df_kunci.iloc[0, 0]).strip().upper()]
            except Exception as e:
                st.warning(f"⚠️ Gagal membaca kunci: {str(e)}")
                kunci = None
        
        df_skor = pd.DataFrame()
        if mode == "pilihan_ganda" and kunci and len(kunci) == len(kolom_soal):
            for i, soal in enumerate(kolom_soal):
                kunci_soal = kunci[i] if i < len(kunci) else None
                if kunci_soal:
                    df_skor[soal] = (df[soal].astype(str).str.strip().str.upper() == kunci_soal).astype(int)
                else:
                    df_skor[soal] = 0
        else:
            for soal in kolom_soal:
                df_skor[soal] = pd.to_numeric(df[soal], errors='coerce').fillna(0).astype(int)
        
        df['skor_total'] = df_skor.sum(axis=1)
        n_siswa = len(df)
        n_soal = len(kolom_soal)
        
        n_kelompok = max(1, int(np.ceil(n_siswa * persen_kelompok / 100)))
        df_sorted = df.sort_values('skor_total', ascending=False).reset_index(drop=True)
        kel_atas = df_sorted.head(n_kelompok)
        kel_bawah = df_sorted.tail(n_kelompok)
        
        # ======================================================================
        # PERHITUNGAN LENGKAP
        # ======================================================================
        stats_hasil = []
        p_vals, q_vals, pq_vals, p_high_vals, p_low_vals, d_vals, se_vals, r_vals, alpha_if_deleted = [], [], [], [], [], [], [], [], []
        
        for i, soal in enumerate(kolom_soal):
            # 1. p (tingkat kesukaran)
            p_val = df_skor[soal].mean()
            p_vals.append(p_val)
            
            # 2. q = 1 - p
            q_val = 1 - p_val
            q_vals.append(q_val)
            
            # 3. pq = p * q (varians butir)
            pq_val = p_val * q_val
            pq_vals.append(pq_val)
            
            # 4. p_high (proporsi benar kelompok atas)
            if mode == "pilihan_ganda" and kunci and i < len(kunci):
                p_high = (kel_atas[soal].astype(str).str.strip().str.upper() == kunci[i]).sum() / n_kelompok
                p_low = (kel_bawah[soal].astype(str).str.strip().str.upper() == kunci[i]).sum() / n_kelompok
            else:
                p_high = pd.to_numeric(kel_atas[soal], errors='coerce').sum() / n_kelompok
                p_low = pd.to_numeric(kel_bawah[soal], errors='coerce').sum() / n_kelompok
            p_high_vals.append(p_high)
            p_low_vals.append(p_low)
            
            # 5. D = p_high - p_low
            d_val = p_high - p_low
            d_vals.append(d_val)
            
            # 6. SE = sqrt(pq / n_siswa)
            se_val = np.sqrt(pq_val / n_siswa)
            se_vals.append(se_val)
            
            # 7. Validitas corrected
            skor_total_minus_item = df['skor_total'] - df_skor[soal]
            if df_skor[soal].var() == 0 or skor_total_minus_item.var() == 0:
                r_it = 0.0
            else:
                r_it, _ = pointbiserialr(df_skor[soal], skor_total_minus_item)
            r_vals.append(r_it)
            
            # 8. Alpha if item deleted
            skor_tanpa = df['skor_total'] - df_skor[soal]
            var_tanpa = skor_tanpa.var(ddof=1)
            pq_tanpa = 0
            for j, soal2 in enumerate(kolom_soal):
                if j != i:
                    p2 = df_skor[soal2].mean()
                    pq_tanpa += p2 * (1 - p2)
            if var_tanpa > 0 and (n_soal - 1) > 1:
                alpha = ((n_soal - 1) / (n_soal - 2)) * (1 - (pq_tanpa / var_tanpa))
            else:
                alpha = 0
            alpha_if_deleted.append(alpha)
            
            # Interpretasi
            kat_p, _ = interpretasi_p(p_val, batas_sukar, batas_mudah)
            kat_d, _ = interpretasi_d(d_val, batas_cukup, batas_baik)
            kat_v = "Valid" if r_it >= batas_valid else "Tidak Valid"
            
            # Rekomendasi
            if r_it >= batas_valid and d_val >= batas_cukup and batas_sukar <= p_val <= batas_mudah:
                rek = "DIGUNAKAN"
            elif r_it < 0.10 or d_val < 0.10:
                rek = "DROP"
            else:
                rek = "REVISI"
            
            stats_hasil.append([
                soal, 
                round(p_val, 4), 
                round(q_val, 4), 
                round(pq_val, 4),
                round(p_high, 4), 
                round(p_low, 4), 
                round(d_val, 4), 
                kat_d,
                round(se_val, 6),
                round(r_it, 4), 
                kat_v,
                round(alpha, 4),
                rek,
                kat_p
            ])
        
        # Reliabilitas KR-20
        var_total = df['skor_total'].var(ddof=1)
        sum_pq = sum(pq_vals)
        
        if var_total > 0 and n_soal > 1:
            kr20 = (n_soal/(n_soal-1)) * (1 - (sum_pq / var_total))
        else:
            kr20 = 0
        
        sem = df['skor_total'].std(ddof=1) * np.sqrt(max(0, 1 - kr20))
        
        # Analisis Pengecoh
        hasil_pengecoh = []
        if mode == "pilihan_ganda" and kunci:
            semua_opsi = set()
            for soal in kolom_soal:
                nilai = df[soal].astype(str).str.strip().str.upper().dropna()
                for n in nilai:
                    if n.isalpha() and len(n) == 1:
                        semua_opsi.add(n)
            opsi_list = sorted(semua_opsi)
            
            for i, soal in enumerate(kolom_soal):
                kunci_soal = kunci[i] if i < len(kunci) else None
                if kunci_soal is None:
                    continue
                data_soal = df[soal].astype(str).str.strip().str.upper()
                for opsi in opsi_list:
                    if opsi == kunci_soal:
                        continue
                    total = (data_soal == opsi).sum()
                    persen = (total / n_siswa) * 100
                    pemilih_atas = (kel_atas[soal].astype(str).str.strip().str.upper() == opsi).sum()
                    pemilih_bawah = (kel_bawah[soal].astype(str).str.strip().str.upper() == opsi).sum()
                    status = "BERFUNGSI" if (persen >= 5 and pemilih_bawah > pemilih_atas) else "TIDAK BERFUNGSI"
                    hasil_pengecoh.append([soal, kunci_soal, opsi, total, round(persen, 1), pemilih_atas, pemilih_bawah, status])
        
        # DataFrame hasil LENGKAP
        df_final = pd.DataFrame(stats_hasil, columns=[
            'Soal', 'p', 'q', 'pq', 'p_high', 'p_low', 'D', 'Interp_D', 
            'SE', 'r_it', 'Interp_V', 'Alpha_if_deleted', 'Rekomendasi', 'Interp_p'
        ])
        
        # ======================================================================
        # TAMPILKAN HASIL
        # ======================================================================
        with tab2:
            st.markdown("## 📋 REKAP ITEM ANALYSIS (LENGKAP)")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Jumlah Siswa", n_siswa)
                st.metric("Jumlah Soal", n_soal)
                st.metric("Mode", mode.upper())
            with col2:
                st.metric("KR-20", f"{kr20:.4f}")
                if kr20 >= 0.80:
                    st.caption("✅ Sangat Baik (Tes sangat konsisten)")
                elif kr20 >= 0.70:
                    st.caption("✅ Baik (Tes dapat digunakan untuk ujian kelas)")
                elif kr20 >= 0.60:
                    st.caption("⚠️ Cukup (Masih bisa digunakan untuk penelitian sederhana)")
                else:
                    st.caption("❌ Kurang (Tes tidak konsisten, perlu perbaikan)")
                st.metric("SEM", f"{sem:.4f}")
                st.caption(f"Σpq = {sum_pq:.4f} | Varians Total = {var_total:.4f}")
            
            st.dataframe(df_final, use_container_width=True)
            
            # Export
            st.markdown("---")
            st.markdown("## 📥 DOWNLOAD HASIL")
            
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                df_final.to_excel(writer, sheet_name='Rekap_Item_Lengkap', index=False)
                pd.DataFrame({
                    'KR-20': [kr20],
                    'SEM': [sem],
                    'Σpq': [sum_pq],
                    'Varians_Total': [var_total],
                    'Jumlah_Siswa': [n_siswa],
                    'Jumlah_Soal': [n_soal]
                }).to_excel(writer, sheet_name='Reliabilitas', index=False)
                if hasil_pengecoh:
                    pd.DataFrame(hasil_pengecoh, columns=['Soal', 'Kunci', 'Opsi', 'Jumlah', 'Persen', 'Atas', 'Bawah', 'Status']).to_excel(writer, sheet_name='Analisis_Pengecoh', index=False)
                
                # Sheet parameter
                pd.DataFrame([
                    {'Komponen': 'p (Tingkat Kesukaran)', 'Rumus': 'p = Σ benar / N', 'Makna': 'Proporsi jawaban benar'},
                    {'Komponen': 'q', 'Rumus': 'q = 1 - p', 'Makna': 'Proporsi jawaban salah'},
                    {'Komponen': 'pq', 'Rumus': 'pq = p × q', 'Makna': 'Varians butir'},
                    {'Komponen': 'p_high', 'Rumus': 'p_high = Σ benar kelompok atas / n_kelompok', 'Makna': 'Proporsi benar kelompok atas (27%)'},
                    {'Komponen': 'p_low', 'Rumus': 'p_low = Σ benar kelompok bawah / n_kelompok', 'Makna': 'Proporsi benar kelompok bawah (27%)'},
                    {'Komponen': 'D (Daya Beda)', 'Rumus': 'D = p_high - p_low', 'Makna': 'Kemampuan butir membedakan siswa'},
                    {'Komponen': 'SE (Standard Error)', 'Rumus': 'SE = √(pq/n)', 'Makna': 'Kesalahan standar estimasi butir'},
                    {'Komponen': 'r_it (Validitas)', 'Rumus': 'Korelasi point-biserial corrected', 'Makna': 'Konsistensi butir dengan skor total'},
                    {'Komponen': 'Alpha_if_deleted', 'Rumus': 'KR-20 tanpa butir i', 'Makna': 'Reliabilitas jika soal dihapus'},
                    {'Komponen': 'KR-20', 'Rumus': '(k/(k-1)) × (1 - Σpq/σ²)', 'Makna': 'Reliabilitas tes (konsistensi internal)'},
                    {'Komponen': 'SEM', 'Rumus': 'SEM = SD × √(1 - KR-20)', 'Makna': 'Standard Error of Measurement'},
                ]).to_excel(writer, sheet_name='Rumus', index=False)
            
            output.seek(0)
            st.download_button(
                label="📥 Download Excel (LENGKAP)",
                data=output,
                file_name="hasil_item_analysis_lengkap.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
            
            st.success("✅ Analisis selesai!")

else:
    with tab2:
        st.info("👈 Silakan upload file CSV terlebih dahulu di tab 'Upload Data'")
