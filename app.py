# ======================================================================
# ITEM ANALYSIS - STREAMLIT VERSION (FINAL FIXED)
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

st.title("📊 ITEM ANALYSIS TOOL")
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
        # Cek ukuran file (max 5MB = 5 * 1024 * 1024 bytes)
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
        # Cek ukuran file (max 5MB)
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
    
    # Deteksi kolom
    kolom_soal = df.columns[1:].tolist()
    
    if len(kolom_soal) == 0:
        with tab2:
            st.error("❌ Tidak ada kolom soal! Pastikan file memiliki minimal 2 kolom.")
    else:
        # Deteksi mode
        sample = df[kolom_soal[0]].dropna().astype(str).str.strip().values
        sample_clean = [s for s in sample if s not in ['', 'nan', 'NaN', 'None']]
        
        if len(sample_clean) > 0:
            is_biner = all(v in ['0', '1'] for v in sample_clean[:50])
        else:
            is_biner = False
        
        mode = "biner" if is_biner else "pilihan_ganda"
        
        # Baca kunci
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
        
        # Konversi ke skor
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
        
        # Kelompok atas/bawah
        n_kelompok = max(1, int(np.ceil(n_siswa * persen_kelompok / 100)))
        df_sorted = df.sort_values('skor_total', ascending=False).reset_index(drop=True)
        kel_atas = df_sorted.head(n_kelompok)
        kel_bawah = df_sorted.tail(n_kelompok)
        
        # Hitung statistik
        stats_hasil = []
        p_vals, q_vals, d_vals, r_vals = [], [], [], []
        
        for i, soal in enumerate(kolom_soal):
            # A. Tingkat Kesukaran (p)
            p_val = df_skor[soal].mean()
            p_vals.append(p_val)
            
            # B. q = 1 - p (proporsi jawaban salah)
            q_val = 1 - p_val
            q_vals.append(q_val)
            
            kat_p, makna_p = interpretasi_p(p_val, batas_sukar, batas_mudah)
            
            # C. Daya Beda (D)
            if mode == "pilihan_ganda" and kunci and i < len(kunci):
                ba = (kel_atas[soal].astype(str).str.strip().str.upper() == kunci[i]).sum()
                bb = (kel_bawah[soal].astype(str).str.strip().str.upper() == kunci[i]).sum()
            else:
                ba = pd.to_numeric(kel_atas[soal], errors='coerce').sum()
                bb = pd.to_numeric(kel_bawah[soal], errors='coerce').sum()
            d_val = (ba - bb) / n_kelompok if n_kelompok > 0 else 0
            d_vals.append(d_val)
            kat_d, makna_d = interpretasi_d(d_val, batas_cukup, batas_baik)
            
            # D. Validitas (Corrected Item-Total Correlation)
            skor_total_minus_item = df['skor_total'] - df_skor[soal]
            if df_skor[soal].var() == 0 or skor_total_minus_item.var() == 0:
                r_it = 0.0
            else:
                r_it, _ = pointbiserialr(df_skor[soal], skor_total_minus_item)
            r_vals.append(r_it)
            kat_v = "Valid" if r_it >= batas_valid else "Tidak Valid"
            
            # E. Rekomendasi Akhir
            if r_it >= batas_valid and d_val >= batas_cukup and batas_sukar <= p_val <= batas_mudah:
                rek = "DIGUNAKAN"
            elif r_it < 0.10 or d_val < 0.10:
                rek = "DROP"
            else:
                rek = "REVISI"
            
            stats_hasil.append([soal, round(p_val, 4), round(q_val, 4), kat_p, round(d_val, 4), kat_d, round(r_it, 4), kat_v, rek])
        
        # Reliabilitas KR-20
        var_total = df['skor_total'].var(ddof=1)
        sum_pq = (df_skor.mean() * (1 - df_skor.mean())).sum()
        
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
        
        # DataFrame hasil dengan q
        df_final = pd.DataFrame(stats_hasil, columns=['Soal', 'p', 'q', 'Interp_p', 'D', 'Interp_D', 'r_it', 'Interp_V', 'Rekomendasi'])
        
        # ======================================================================
        # TAMPILKAN HASIL DI TAB2
        # ======================================================================
        with tab2:
            st.markdown("## 📋 REKAP ITEM ANALYSIS")
            
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
            
            st.markdown("---")
            st.markdown("## 📊 VISUALISASI")
            
            col1, col2 = st.columns(2)
            
            # Grafik 1: Tingkat Kesukaran (p)
            with col1:
                fig1, ax1 = plt.subplots(figsize=(8, 5))
                warna_p = ['red' if x < batas_sukar else ('green' if x <= batas_mudah else 'orange') for x in p_vals]
                ax1.bar(range(1, n_soal+1), p_vals, color=warna_p)
                ax1.axhline(batas_sukar, color='red', linestyle='--', label=f'Batas Sukar ({batas_sukar})')
                ax1.axhline(batas_mudah, color='orange', linestyle='--', label=f'Batas Mudah ({batas_mudah})')
                ax1.set_xlabel('Nomor Soal')
                ax1.set_ylabel('Tingkat Kesukaran (p)')
                ax1.set_title('1. Tingkat Kesukaran (p)')
                ax1.set_xticks(range(1, n_soal+1))
                ax1.set_ylim(0, 1)
                ax1.legend()
                ax1.grid(axis='y', alpha=0.3)
                st.pyplot(fig1)
                plt.close()
            
            # Grafik 2: q = 1-p
            with col2:
                fig2, ax2 = plt.subplots(figsize=(8, 5))
                warna_q = ['blue' for x in q_vals]
                ax2.bar(range(1, n_soal+1), q_vals, color=warna_q, alpha=0.7)
                ax2.set_xlabel('Nomor Soal')
                ax2.set_ylabel('q = 1 - p')
                ax2.set_title('2. Proporsi Jawaban Salah (q)')
                ax2.set_xticks(range(1, n_soal+1))
                ax2.set_ylim(0, 1)
                ax2.grid(axis='y', alpha=0.3)
                st.pyplot(fig2)
                plt.close()
            
            # Grafik 3: Daya Beda
            with col1:
                fig3, ax3 = plt.subplots(figsize=(8, 5))
                warna_d = ['green' if x >= batas_baik else ('orange' if x >= batas_cukup else 'red') for x in d_vals]
                ax3.bar(range(1, n_soal+1), d_vals, color=warna_d)
                ax3.axhline(batas_baik, color='green', linestyle='--', label=f'Sangat Baik ({batas_baik})')
                ax3.axhline(batas_cukup, color='orange', linestyle='--', label=f'Cukup ({batas_cukup})')
                ax3.set_xlabel('Nomor Soal')
                ax3.set_ylabel('Daya Beda (D)')
                ax3.set_title('3. Daya Beda (D)')
                ax3.set_xticks(range(1, n_soal+1))
                ax3.set_ylim(-1, 1)
                ax3.legend()
                ax3.grid(axis='y', alpha=0.3)
                st.pyplot(fig3)
                plt.close()
            
            # Grafik 4: Validitas
            with col2:
                fig4, ax4 = plt.subplots(figsize=(8, 5))
                warna_r = ['green' if x >= batas_valid else 'red' for x in r_vals]
                ax4.bar(range(1, n_soal+1), r_vals, color=warna_r)
                ax4.axhline(batas_valid, color='green', linestyle='--', label=f'Batas Valid ({batas_valid})')
                ax4.axhline(0, color='black', linestyle='-', linewidth=0.5)
                ax4.set_xlabel('Nomor Soal')
                ax4.set_ylabel('Validitas (r_it)')
                ax4.set_title('4. Validitas Butir (Corrected)')
                ax4.set_xticks(range(1, n_soal+1))
                ax4.set_ylim(-1, 1)
                ax4.legend()
                ax4.grid(axis='y', alpha=0.3)
                st.pyplot(fig4)
                plt.close()
            
            # Grafik 5: Rekomendasi
            with col1:
                fig5, ax5 = plt.subplots(figsize=(8, 5))
                warna_rek = ['green' if r == 'DIGUNAKAN' else ('orange' if r == 'REVISI' else 'red') for r in df_final['Rekomendasi']]
                ax5.bar(range(1, n_soal+1), [1]*n_soal, color=warna_rek)
                ax5.set_xlabel('Nomor Soal')
                ax5.set_title('5. Rekomendasi Akhir')
                ax5.set_xticks(range(1, n_soal+1))
                ax5.set_yticks([])
                st.pyplot(fig5)
                plt.close()
            
            # Grafik 6: Distribusi Skor
            with col2:
                fig6, ax6 = plt.subplots(figsize=(8, 5))
                min_skor = int(df['skor_total'].min())
                max_skor = int(df['skor_total'].max())
                bins = range(min_skor, max_skor+2)
                ax6.hist(df['skor_total'], bins=bins, edgecolor='black', alpha=0.7, color='skyblue')
                ax6.axvline(df['skor_total'].mean(), color='red', linestyle='--', label=f"Mean={df['skor_total'].mean():.1f}")
                ax6.axvline(df['skor_total'].median(), color='green', linestyle='--', label=f"Median={df['skor_total'].median():.1f}")
                ax6.set_xlabel('Skor Total')
                ax6.set_ylabel('Frekuensi')
                ax6.set_title('6. Distribusi Skor Total')
                ax6.legend()
                ax6.grid(axis='y', alpha=0.3)
                st.pyplot(fig6)
                plt.close()
            
            # Grafik 7: Pie Chart Rekomendasi
            st.markdown("---")
            col1, col2 = st.columns(2)
            
            with col1:
                fig7, ax7 = plt.subplots(figsize=(6, 5))
                soal_digunakan = sum(1 for r in df_final['Rekomendasi'] if r == 'DIGUNAKAN')
                soal_revisi = sum(1 for r in df_final['Rekomendasi'] if r == 'REVISI')
                soal_drop = sum(1 for r in df_final['Rekomendasi'] if r == 'DROP')
                labels = [f'Digunakan ({soal_digunakan})', f'Revisi ({soal_revisi})', f'Drop ({soal_drop})']
                colors = ['green', 'orange', 'red']
                if soal_digunakan + soal_revisi + soal_drop > 0:
                    ax7.pie([soal_digunakan, soal_revisi, soal_drop], labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
                    ax7.set_title('Ringkasan Rekomendasi Soal')
                st.pyplot(fig7)
                plt.close()
            
            # Heatmap
            if n_soal > 1:
                with col2:
                    st.markdown("### 🔥 Korelasi Antar Butir")
                    fig8, ax8 = plt.subplots(figsize=(max(6, n_soal*0.4), max(5, n_soal*0.35)))
                    korelasi = df_skor.corr()
                    mask = np.triu(np.ones_like(korelasi, dtype=bool))
                    sns.heatmap(korelasi, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r', center=0, square=True, linewidths=0.5, ax=ax8)
                    ax8.set_title('Korelasi Antar Butir (Nilai >0.30 Indikasi Redundansi)')
                    st.pyplot(fig8)
                    plt.close()
            
            # Analisis Pengecoh
            if hasil_pengecoh:
                st.markdown("---")
                st.markdown("## 🎯 ANALISIS PENGECOH")
                df_pengecoh = pd.DataFrame(hasil_pengecoh, columns=['Soal', 'Kunci', 'Opsi', 'Jumlah', 'Persen', 'Atas', 'Bawah', 'Status'])
                st.dataframe(df_pengecoh, use_container_width=True)
            
            # Export
            st.markdown("---")
            st.markdown("## 📥 DOWNLOAD HASIL")
            
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                df_final.to_excel(writer, sheet_name='Rekap_Item', index=False)
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
                    {'Aspek': 'Tingkat Kesukaran (p)', 'Kategori': 'Sukar', 'Rentang': f'< {batas_sukar}', 'Tindakan': 'Perbaiki redaksi, sederhanakan bahasa'},
                    {'Aspek': 'Tingkat Kesukaran (p)', 'Kategori': 'Sedang', 'Rentang': f'{batas_sukar} - {batas_mudah}', 'Tindakan': 'Pertahankan'},
                    {'Aspek': 'Tingkat Kesukaran (p)', 'Kategori': 'Mudah', 'Rentang': f'> {batas_mudah}', 'Tindakan': 'Tingkatkan kesukaran'},
                    {'Aspek': 'Daya Beda (D)', 'Kategori': 'Jelek', 'Rentang': f'< {batas_cukup}', 'Tindakan': 'Drop atau revisi total'},
                    {'Aspek': 'Daya Beda (D)', 'Kategori': 'Cukup', 'Rentang': f'{batas_cukup} - {batas_baik}', 'Tindakan': 'Perbaikan minor'},
                    {'Aspek': 'Daya Beda (D)', 'Kategori': 'Sangat Baik', 'Rentang': f'≥ {batas_baik}', 'Tindakan': 'Pertahankan'},
                    {'Aspek': 'Validitas (r_it)', 'Kategori': 'Tidak Valid', 'Rentang': f'< {batas_valid}', 'Tindakan': 'Perbaiki atau drop'},
                    {'Aspek': 'Validitas (r_it)', 'Kategori': 'Valid', 'Rentang': f'≥ {batas_valid}', 'Tindakan': 'Pertahankan'},
                ]).to_excel(writer, sheet_name='Parameter', index=False)
            
            output.seek(0)
            st.download_button(
                label="📥 Download Excel",
                data=output,
                file_name="hasil_item_analysis.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
            
            st.success("✅ Analisis selesai!")

else:
    with tab2:
        st.info("👈 Silakan upload file CSV terlebih dahulu di tab 'Upload Data'")
