import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import pointbiserialr
import io
import matplotlib.pyplot as plt

# ======================================================================
# ACADEMIC ITEM ANALYSIS - RIGOROUS ENGLISH VERSION (2026)
# ======================================================================

st.set_page_config(page_title="Item Analysis Pro", page_icon="📈", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #f4f4f9; }
    [data-testid="stMetricValue"] { font-family: 'Courier New', Courier, monospace; color: #1a1a1a; }
    .stAlert { border: 2px solid #000; border-radius: 0px; }
    </style>
    """, unsafe_allow_html=True)

st.title("ITEM ANALYSIS TOOL (CTT)")
st.write("Full Classical Test Theory Suite: Methodologically validated metrics for educational research.")

with st.sidebar:
    st.header("📊 Methodological Legend")
    
    with st.expander("1. Difficulty Index (p)", expanded=True):
        st.write("""
        - **Easy (p > 0.70):** 🟢
        - **Moderate (0.30 - 0.70):** 🟡
        - **Difficult (p < 0.30):** 🔴
        """)

    with st.expander("2. Discrimination (d/DDI)", expanded=True):
        st.write("""
        - **d (Discrimination Index):** Focus on the Answer Key.
        - **DDI (Distractor Discrimination):** Focus on the worst distractor.
        - **Excellent (≥ 0.40):** 🟢
        - **Good (0.30 - 0.39):** 🔵
        - **Fair (0.20 - 0.29):** 🟡
        - **Poor (< 0.20):** 🔴
        """)

    st.header("⚙️ Settings")
    group_percent = st.slider("Kelley's Grouping (%)", 10, 50, 27)
    validity_limit = st.number_input("r_pbis Threshold", 0.0, 1.0, 0.25)

u1, u2 = st.columns(2)
with u1:
    student_file = st.file_uploader("Upload Student Data (CSV)", type=['csv'])
with u2:
    key_file = st.file_uploader("Upload Key Data (CSV)", type=['csv'])

if student_file and key_file:
    df = pd.read_csv(student_file).fillna("N/A")
    df_key = pd.read_csv(key_file)
    item_cols = df.columns[1:] 
    id_col_name = df.columns[0]
    answer_key = df_key.iloc[0, 1:].astype(str).str.upper().str.strip().tolist()
    
    df_scores = pd.DataFrame()
    for i, col in enumerate(item_cols):
        df_scores[col] = (df[col].astype(str).str.upper().str.strip() == answer_key[i]).astype(int)
    
    total_scores = df_scores.sum(axis=1)
    df['Total_Score'] = total_scores
    n_students, n_items = len(df), len(item_cols)

    # GATHER GROUP DATA
    n_group = max(1, int(np.ceil(n_students * group_percent / 100)))
    df_sorted = df.sort_values('Total_Score', ascending=False).copy()
    
    # Create Rank and Group Marker
    df_sorted['Rank'] = range(1, n_students + 1)
    df_sorted['Group'] = 'Middle'
    df_sorted.iloc[:n_group, df_sorted.columns.get_loc('Group')] = 'Upper'
    df_sorted.iloc[-n_group:, df_sorted.columns.get_loc('Group')] = 'Lower'
    
    # Ranking Table Summary
    df_ranking = df_sorted[[id_col_name, 'Total_Score', 'Rank', 'Group']].copy()

    up_idx, lo_idx = df_sorted.head(n_group).index, df_sorted.tail(n_group).index

    results = []
    for i, item in enumerate(item_cols):
        p = df_scores[item].mean()
        q = 1 - p
        pq = p * q
        item_var = df_scores[item].var(ddof=0)

        p_up = df_scores.loc[up_idx, item].mean()
        p_lo = df_scores.loc[lo_idx, item].mean()
        d_val = p_up - p_lo

        distractors = []
        for opt in df[item].dropna().astype(str).str.upper().unique():
            opt_clean = opt.strip()
            if opt_clean not in ["", "N/A", answer_key[i]]:
                distractors.append(opt_clean)
                
        ddi_vals = []
        for opt in distractors:
            u_opt = (df.loc[up_idx, item].astype(str).str.upper().str.strip() == opt).mean()
            l_opt = (df.loc[lo_idx, item].astype(str).str.upper().str.strip() == opt).mean()
            ddi_vals.append(l_opt - u_opt)
        
        ddi_final = max(ddi_vals) if ddi_vals else 0
        worst_ddi = min(ddi_vals) if ddi_vals else 0

        corrected_total = total_scores - df_scores[item]
        if df_scores[item].var() != 0:
            r_pb, _ = pointbiserialr(df_scores[item], corrected_total)
            if np.isnan(r_pb):
                r_pb = 0
        else:
            r_pb = 0

        p_desc = "Easy" if p > 0.7 else "Difficult" if p < 0.3 else "Moderate"
        d_desc = "Excellent" if d_val >= 0.4 else "Good" if d_val >= 0.3 else "Fair" if d_val >= 0.2 else "Poor"
        r_desc = "Valid" if r_pb >= validity_limit else "Invalid"
        
        # --- COMBINED FLAGGING SYSTEM ---
        reasons = []
        
        # cek validitas
        if r_pb < validity_limit:
            reasons.append("Low validity")
        
        # cek daya beda
        if d_val < 0.20:
            reasons.append("Poor discrimination")
        # cek difficulty ekstrem
        if p > 0.90:
            reasons.append("Too easy")
        
        if p < 0.20:
            reasons.append("Too difficult")
        
        # cek distractor
        if worst_ddi < -0.10:
            reasons.append("Severely flawed distractor")
        elif worst_ddi < 0:
            reasons.append("Malfunctioning distractor")
        
        # keputusan akhir
        if (r_pb >= validity_limit) and (d_val >= 0.30) and (worst_ddi >= 0):
            decision = "RETAIN"
        
        elif (r_pb < validity_limit and d_val < 0.20) or (worst_ddi < -0.10):
            decision = "REJECT"
        
        else:
            decision = "REVISE"
        
        reason_text = ", ".join(reasons) if reasons else "All criteria satisfied"

        results.append({
            "Item": item, "p": p, "p_Eval": p_desc, "q": q, "pq": pq, "Var": item_var,
            "d": d_val, "d_Eval": d_desc, "Best_DDI": ddi_final, "Worst_DDI": worst_ddi,
            "r_pbis": r_pb, "r_Eval": r_desc,
            "DECISION": decision,
            "REASON": reason_text
        })

    df_res = pd.DataFrame(results)

    # METRICS
    mean_score = total_scores.mean()
    var_total = total_scores.var(ddof=0)
    std_score = np.sqrt(var_total)
    kr20 = (n_items/(n_items-1)) * (1 - (df_res["pq"].sum()/var_total)) if var_total > 0 else 0
    alpha = (n_items/(n_items-1)) * (1 - (df_res["Var"].sum()/var_total)) if var_total > 0 else 0
    sem = std_score * np.sqrt(1 - kr20)

    st.divider()
    m1, m2, m3, m4, m5, m6, m7 = st.columns(7)
    m1.metric("Students (N)", n_students)
    m2.metric("Items (k)", n_items)
    m3.metric("Mean Score", f"{mean_score:.2f}")
    m4.metric("Std. Deviation", f"{std_score:.2f}")
    m5.metric("KR-20", f"{kr20:.4f}")
    m6.metric("Alpha", f"{alpha:.4f}")
    m7.metric("SEM (Error)", f"{sem:.4f}")
    
    st.subheader("🎯 Distractor Functionality Audit")
    min_ddi_global = df_res["Worst_DDI"].min()
    
    if min_ddi_global < 0:
        st.error(f"**Critical Alert:** A malfunctioning distractor was detected (Minimum DDI: {min_ddi_global:.4f}). Negative values indicate that the distractor is more attractive to high-performing students, suggesting a potential flaw in item construction.")
    else:
        st.success("**Perfect Functionality:** No negative DDI detected. All distractors are successfully drawing more students from the lower group than the upper group.")

    # --- DESCRIPTIVE INTERPRETATION (WEB) ---
    st.subheader("📝 Descriptive Interpretation")
    rel_eval = "Excellent" if kr20 >= 0.9 else "High" if kr20 >= 0.8 else "Acceptable" if kr20 >= 0.7 else "Low"
    
    col_int1, col_int2 = st.columns(2)
    with col_int1:
        st.info(f"**Test Reliability:** The KR-20 coefficient of {kr20:.4f} indicates **{rel_eval}** internal consistency. "
                f"This suggests that the instrument is {'highly stable' if kr20 >= 0.8 else 'sufficient'} for measuring the intended academic constructs.")
    with col_int2:
        st.warning(f"**Measurement Error:** The SEM of {sem:.4f} indicates that a student's observed score may fluctuate within this range "
                   f"relative to their theoretical true score. A lower SEM reflects higher precision in score estimation.")

    # --- ITEM MATRIX ---
    st.subheader("📋 Comprehensive Item Statistics Matrix")
    def apply_item_styling(row):
        styles = [''] * len(row)
        dif_color = '#ccffcc' if row['p'] > 0.7 else '#ffcccc' if row['p'] < 0.3 else '#fff2cc'
        styles[1] = styles[2] = f'background-color: {dif_color}; color: black'
        if row['d'] >= 0.4: dis_color, txt = '#2ecc71', 'white'
        elif row['d'] >= 0.3: dis_color, txt = '#3498db', 'white'
        elif row['d'] >= 0.2: dis_color, txt = '#f1c40f', 'black'
        else: dis_color, txt = '#e74c3c', 'white'
        styles[6] = styles[7] = styles[8] = styles[9] = f'background-color: {dis_color}; color: {txt}'
        val_bg = '#ccffcc' if row['r_pbis'] >= validity_limit else '#ffcccc'
        styles[10] = f'background-color: {val_bg}; color: black; font-weight: bold'
        styles[11] = f'background-color: {val_bg}; color: black'
        if row['DECISION'] == "RETAIN": styles[12] = 'background-color: #27ae60; color: white; font-weight: bold'
        elif row['DECISION'] == "REVISE": styles[12] = 'background-color: #f39c12; color: white'
        else: styles[12] = 'background-color: #c0392b; color: white'
        return styles

    st.dataframe(df_res.style.apply(apply_item_styling, axis=1).format("{:.4f}", subset=["p", "q", "pq", "Var", "d", "Best_DDI", "Worst_DDI", "r_pbis"]), use_container_width=True)

        # ======================================================================
    # VISUALISASI BERDASARKAN OUTPUT (DITAMBAHKAN)
    # ======================================================================
    st.markdown("---")
    st.markdown("<h2 style='text-align: center;'>📊 VISUALISASI HASIL ANALISIS ITEM</h2>", unsafe_allow_html=True)
    st.markdown("---")
    
    # --- VISUALISASI 1: STACKED BAR CHART p DAN q ---
    st.subheader("📊 Figure 1. Distribusi Kesulitan Item (p vs q)")
    
    fig1, ax1 = plt.subplots(figsize=(12, 6))
    items = df_res['Item'].tolist()
    p_vals = df_res['p'].tolist()
    q_vals = df_res['q'].tolist()
    
    ax1.bar(items, p_vals, color='#2ecc71', label='p (Benar)', edgecolor='white')
    ax1.bar(items, q_vals, bottom=p_vals, color='#e74c3c', label='q (Salah)', edgecolor='white')
    
    for i, (p, q) in enumerate(zip(p_vals, q_vals)):
        if p > 0.05:
            ax1.text(i, p/2, f'{p:.3f}', ha='center', va='center', fontsize=9, color='white', fontweight='bold')
        if q > 0.05:
            ax1.text(i, p + q/2, f'{q:.3f}', ha='center', va='center', fontsize=9, color='white', fontweight='bold')
    
    ax1.axhline(y=0.70, color='green', linestyle='--', label='Mudah (0.70)')
    ax1.axhline(y=0.30, color='orange', linestyle='--', label='Sulit (0.30)')
    ax1.set_xlabel('Item', fontsize=12)
    ax1.set_ylabel('Proporsi', fontsize=12)
    ax1.set_ylim(0, 1)
    ax1.set_xticklabels(items, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig1)
    plt.close(fig1)
    
    # --- VISUALISASI 2: DAYA BEDA (d) ---
    st.subheader("🎯 Figure 2. Indeks Daya Beda Item (d)")
    
    def get_d_color(d):
        if d >= 0.4: return '#2ecc71'
        elif d >= 0.3: return '#3498db'
        elif d >= 0.2: return '#f1c40f'
        else: return '#e74c3c'
    
    colors_d = [get_d_color(d) for d in df_res['d']]
    
    fig2, ax2 = plt.subplots(figsize=(12, 5))
    bars = ax2.bar(items, df_res['d'], color=colors_d, edgecolor='black')
    
    for bar, d in zip(bars, df_res['d']):
        ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01, f'{d:.3f}', ha='center', fontsize=9)
    
    ax2.axhline(y=0.40, color='#2ecc71', linestyle='--', label='Excellent (0.40)')
    ax2.axhline(y=0.30, color='#3498db', linestyle='--', label='Good (0.30)')
    ax2.axhline(y=0.20, color='#f1c40f', linestyle='--', label='Fair (0.20)')
    ax2.set_xlabel('Item', fontsize=12)
    ax2.set_ylabel('Daya Beda (d)', fontsize=12)
    ax2.set_ylim(-0.05, 1.0)
    ax2.set_xticklabels(items, rotation=45, ha='right')
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close(fig2)
    
    # --- VISUALISASI 3: SCATTER PLOT (LANGSUNG DARI df_res) ---
    st.subheader("🔄 Figure 3. Peta Diagnostik Item (Difficulty vs Discrimination)")
    
    # Warna berdasarkan DECISION dari df_res
    decision_colors = {'RETAIN': '#27ae60', 'REVISE': '#f39c12', 'REJECT': '#e74c3c'}
    
    fig3, ax3 = plt.subplots(figsize=(10, 8))
    
    # Plot setiap item SATU PER SATU dari df_res
    for _, row in df_res.iterrows():
        color = decision_colors[row['DECISION']]
        ax3.scatter(row['p'], row['d'], s=200, c=color, edgecolors='black', linewidth=1.5, zorder=3)
        ax3.annotate(row['Item'], (row['p'], row['d']), 
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=10, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    # Garis threshold
    ax3.axhline(y=0.40, color='#2ecc71', linestyle='--', alpha=0.7, label='d = 0.40 (Excellent)')
    ax3.axhline(y=0.30, color='#3498db', linestyle='--', alpha=0.7, label='d = 0.30 (Good)')
    ax3.axhline(y=0.20, color='#f1c40f', linestyle='--', alpha=0.7, label='d = 0.20 (Fair)')
    ax3.axvline(x=0.70, color='green', linestyle=':', alpha=0.7, label='p = 0.70 (Mudah)')
    ax3.axvline(x=0.30, color='red', linestyle=':', alpha=0.7, label='p = 0.30 (Sulit)')
    
    ax3.set_xlabel('Difficulty (p) → Mudah', fontsize=12)
    ax3.set_ylabel('Discrimination (d) → Baik', fontsize=12)
    ax3.set_title('Peta Diagnostik Item', fontsize=14, fontweight='bold')
    ax3.set_xlim(-0.05, 1.05)
    ax3.set_ylim(-0.05, 1.05)
    ax3.grid(alpha=0.3)
    ax3.legend(loc='upper left', fontsize=9)
    
    # Tambahkan zona interpretasi
    ax3.text(0.85, 0.92, 'IDEAL', fontsize=10, ha='center', bbox=dict(boxstyle='round', facecolor='#27ae60', alpha=0.3))
    ax3.text(0.10, 0.92, 'SULIT\nTAPI BAGUS', fontsize=9, ha='center', bbox=dict(boxstyle='round', facecolor='#3498db', alpha=0.3))
    ax3.text(0.85, 0.10, 'MUDAH\nDAYA BEDA RENDAH', fontsize=9, ha='center', bbox=dict(boxstyle='round', facecolor='#f1c40f', alpha=0.3))
    ax3.text(0.10, 0.10, 'REJECT', fontsize=10, ha='center', fontweight='bold', bbox=dict(boxstyle='round', facecolor='#e74c3c', alpha=0.3))
    
    plt.tight_layout()
    st.pyplot(fig3)
    plt.close(fig3)
    
    # Verifikasi jumlah data
    st.caption(f"✅ Figure 3 menampilkan {len(df_res)} item sesuai dengan data di tabel.")
    
    st.markdown("---")

    # --- RANKING TABLE ---
    st.subheader("🏆 Student Score Ranking & Grouping")
    def apply_rank_styling(row):
        style = [''] * len(row)
        if row['Group'] == 'Upper': bg_color = '#d4edda'
        elif row['Group'] == 'Lower': bg_color = '#f8d7da'
        else: bg_color = 'white'
        return [f'background-color: {bg_color}; color: black'] * len(row)

    st.dataframe(df_ranking.style.apply(apply_rank_styling, axis=1), use_container_width=True)

        # --- DISTRACTORS ---
    st.divider()
    st.subheader("🎯 Distractor Effectiveness")
    dist_data = [df[item].astype(str).str.upper().str.strip().value_counts(normalize=True).to_dict() | {"Item": item} for item in item_cols]
    df_dist = pd.DataFrame(dist_data).set_index('Item').fillna(0)
    cols = sorted([c for c in df_dist.columns if len(str(c)) == 1]) + sorted([c for c in df_dist.columns if len(str(c)) > 1])
    df_dist_final = df_dist[cols].copy()
    
    # Format angka ke persen
    for col in cols: 
        df_dist_final[col] = df_dist_final[col].apply(lambda x: f"{x:.4f} ({x:.2%})")
    
    # Pewarnaan berdasarkan nilai proporsi (menggunakan nilai asli, bukan string)
    def color_distractor(val):
        # val adalah string seperti "0.1234 (12.34%)", kita ekstrak nilai angkanya
        try:
            numeric_val = float(val.split(' ')[0])
        except:
            numeric_val = 0
        
        if numeric_val > 0.50:
            return 'background-color: #2ecc71; color: white;'  # Hijau (dominant)
        elif numeric_val > 0.25:
            return 'background-color: #f1c40f; color: black;'  # Kuning (cukup)
        elif numeric_val < 0.05 and numeric_val > 0:
            return 'background-color: #e74c3c; color: white;'  # Merah (sangat rendah)
        else:
            return ''
    
    # Terapkan styling hanya pada kolom opsi (bukan kolom Item atau Interpretation)
    style_dict = {col: color_distractor for col in cols}
    
    df_dist_styled = df_dist_final.style.applymap(color_distractor, subset=cols)
    
    # Tambah kolom interpretasi
    df_dist_final['Interpretation'] = df_dist[cols].apply(
        lambda row: f"⚠️ Perhatikan: {', '.join([str(opt) for opt, val in row.items() if val < 0.05 and val > 0 and opt != 'N/A'])}" if any(val < 0.05 and val > 0 for val in row.values()) 
        else "✓ Semua opsi berfungsi", axis=1
    )
    
    # Terapkan styling ke kolom interpretasi juga
    df_dist_styled = df_dist_final.style.applymap(color_distractor, subset=cols)
    
    st.dataframe(df_dist_styled, use_container_width=True)

    # --- EXCEL DOWNLOAD ---
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine='xlsxwriter') as writer:
        df_res.to_excel(writer, index=False, sheet_name='Item_Analysis')
        df_ranking.to_excel(writer, index=False, sheet_name='Student_Ranking')
        df_dist_final.to_excel(writer, index=True, sheet_name='Distractor_Analysis')
        
        # GABUNGAN Reliability_Summary + Interpretive_Report
        reliability_interpretation = pd.DataFrame({
            "Metric": [
                "Students (N)", 
                "Items (k)", 
                "Mean Score", 
                "Std. Deviation", 
                "KR-20", 
                "Alpha", 
                "SEM",
                "KR-20 Interpretation",
                "SEM Interpretation"
            ],
            "Value": [
                n_students, 
                n_items, 
                f"{mean_score:.2f}", 
                f"{std_score:.2f}", 
                f"{kr20:.4f}", 
                f"{alpha:.4f}", 
                f"{sem:.4f}",
                f"{rel_eval} reliability",
                f"±{sem:.4f} margin of error"
            ],
            "Interpretation": [
                "Total number of test takers",
                "Total number of items in the test",
                "Average raw score across all students",
                "Spread of scores around the mean",
                f"Internal consistency: {rel_eval}. {'Excellent (>0.90)' if kr20 >= 0.9 else 'High (>0.80)' if kr20 >= 0.8 else 'Acceptable (>0.70)' if kr20 >= 0.7 else 'Low (<0.70)'}",
                "Alternative reliability coefficient (parallel to KR-20)",
                "Standard error of measurement. Lower values = higher precision",
                f"The instrument shows {rel_eval} reliability. High values (>0.70) indicate consistency in measurement.",
                f"The SEM of {sem:.4f} indicates that a student's observed score may fluctuate within this range relative to their theoretical true score."
            ]
        })
        
        reliability_interpretation.to_excel(writer, index=False, sheet_name='Reliability_Interpretation')
            
    st.download_button(label="📥 Download Full Report", data=buf.getvalue(), file_name="Complete_Item_Analysis_Report.xlsx")
