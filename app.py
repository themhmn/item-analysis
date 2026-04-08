import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import pointbiserialr
import io

# ======================================================================
# ACADEMIC ITEM ANALYSIS - RIGOROUS ENGLISH VERSION (2026)
# ======================================================================

st.set_page_config(page_title="Item Analysis Pro", page_icon="📈", layout="wide")

# Enhanced Brutalist Visuals
st.markdown("""
    <style>
    .main { background-color: #f4f4f9; }
    [data-testid="stMetricValue"] { font-family: 'Courier New', Courier, monospace; color: #1a1a1a; }
    .stAlert { border: 2px solid #000; border-radius: 0px; }
    </style>
    """, unsafe_allow_html=True)

st.title("🛡️ RIGOROUS ITEM ANALYSIS TOOL (CTT)")
st.write("Full Classical Test Theory Suite: Methodologically validated metrics for educational research.")

# SIDEBAR: DIKEMBALIKAN KE VERSI ASLI LENGKAP
with st.sidebar:
    st.header("📊 Methodological Legend")
    
    with st.expander("1. Difficulty Index (p/d)", expanded=True):
        st.write("""
        Shows the difficulty level of the test item.
        - **Easy (p > 0.70):** 🟢
        - **Moderate (0.30 - 0.70):** 🟡 (Ideal/Accepted)
        - **Difficult (p < 0.30):** 🔴
        """)

    with st.expander("2. Discrimination (ddi)", expanded=True):
        st.write("""
        The ability to distinguish between high and low-performing students.
        - **Excellent (ddi ≥ 0.40):** 🟢
        - **Good (0.30 - 0.39):** 🔵
        - **Fair (0.20 - 0.29):** 🟡 (Revision Required)
        - **Poor (ddi < 0.20):** 🔴 (Reject/Discard)
        """)

    # INFORMASI YANG SEMPAT HILANG SAYA KEMBALIKAN:
    with st.expander("3. Validity (r_pbis)", expanded=False):
        st.write("""
        Point-Biserial Correlation. Measures how well an item correlates with the total score.
        - **Valid:** ≥ Threshold
        - **Invalid:** < Threshold
        """)

    with st.expander("4. Reliability (KR-20)", expanded=False):
        st.write("""
        Kuder-Richardson Formula 20 measures internal consistency.
        - **Reliable:** ≥ 0.70
        """)

    st.header("⚙️ Settings")
    group_percent = st.slider("Kelley's Grouping (%)", 10, 50, 27)
    validity_limit = st.number_input("r_pbis Threshold", 0.0, 1.0, 0.25)

# FILE UPLOADER
u1, u2 = st.columns(2)
with u1:
    student_file = st.file_uploader("Upload Student Data (CSV)", type=['csv'])
with u2:
    key_file = st.file_uploader("Upload Key Data (CSV)", type=['csv'])

if student_file and key_file:
    # 1. DATA PREP
    df = pd.read_csv(student_file).fillna("N/A")
    df_key = pd.read_csv(key_file)
    item_cols = df.columns[1:] 
    answer_key = df_key.iloc[0, 1:].astype(str).str.upper().str.strip().tolist()
    
    # 2. VECTORIZED SCORING
    df_scores = pd.DataFrame()
    for i, col in enumerate(item_cols):
        df_scores[col] = (df[col].astype(str).str.upper().str.strip() == answer_key[i]).astype(int)
    
    total_scores = df_scores.sum(axis=1)
    df['Total_Score'] = total_scores
    n_students, n_items = len(df), len(item_cols)

    # 3. GROUPING
    n_group = max(1, int(np.ceil(n_students * group_percent / 100)))
    df_sorted = df.sort_values('Total_Score', ascending=False)
    up_idx, lo_idx = df_sorted.head(n_group).index, df_sorted.tail(n_group).index

    # 4. RIGOROUS CALCULATION
    results = []
    for i, item in enumerate(item_cols):
        p = df_scores[item].mean()
        q = 1 - p
        p_up, p_lo = df_scores.loc[up_idx, item].mean(), df_scores.loc[lo_idx, item].mean()
        d_val = p_up - p_lo 
        r_pb, _ = pointbiserialr(df_scores[item], total_scores) if df_scores[item].var() != 0 else (0,0)

        p_desc = "Easy" if p > 0.7 else "Difficult" if p < 0.3 else "Moderate"
        d_desc = "Excellent" if d_val >= 0.4 else "Good" if d_val >= 0.3 else "Fair" if d_val >= 0.2 else "Poor"
        r_desc = "Valid" if r_pb >= validity_limit else "Invalid"
        
        if r_pb >= validity_limit and d_val >= 0.3: decision = "RETAIN"
        elif r_pb >= 0.2 and d_val >= 0.2: decision = "REVISE"
        else: decision = "REJECT"

        results.append({
            "Item": item, "p": p, "p_Eval": p_desc, "q": q, "pq": p*q,
            "ddi": d_val, "d": p, "d_Eval": d_desc, "r_pbis": r_pb, 
            "r_Eval": r_desc, "DECISION": decision
        })

    df_res = pd.DataFrame(results)

    # 5. TEST-LEVEL STATISTICS
    mean_score = total_scores.mean()
    std_score = total_scores.std(ddof=1)
    var_total = total_scores.var(ddof=1)
    kr20 = (n_items/(n_items-1)) * (1 - (df_res["pq"].sum()/var_total)) if var_total > 0 else 0
    sem = std_score * np.sqrt(1 - kr20)

    # DASHBOARD
    st.divider()
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Students (N)", n_students)
    m2.metric("Items (k)", n_items)
    m3.metric("Mean Score", f"{mean_score:.2f}")
    m4.metric("Std. Deviation", f"{std_score:.2f}")
    m5.metric("KR-20 Reliability", f"{kr20:.3f}")
    m6.metric("SEM (Error)", f"{sem:.3f}")

    # 6. FULL STATISTICAL STYLING
    def apply_full_styling(row):
        styles = [''] * len(row)
        dif_color = '#ccffcc' if row['p'] > 0.7 else '#ffcccc' if row['p'] < 0.3 else '#fff2cc'
        styles[1] = styles[2] = styles[6] = f'background-color: {dif_color}; color: black'
        dis_color = '#2ecc71' if row['ddi'] >= 0.4 else '#3498db' if row['ddi'] >= 0.3 else '#f1c40f' if row['ddi'] >= 0.2 else '#e74c3c'
        styles[5] = styles[7] = f'background-color: {dis_color}; color: white'
        val_bg = '#ccffcc' if row['r_pbis'] >= validity_limit else '#ffcccc'
        styles[8] = styles[9] = f'background-color: {val_bg}; color: black; font-weight: bold'
        if row['DECISION'] == "RETAIN": styles[10] = 'background-color: #27ae60; color: white; font-weight: bold'
        elif row['DECISION'] == "REVISE": styles[10] = 'background-color: #f39c12; color: white'
        else: styles[10] = 'background-color: #c0392b; color: white'
        return styles

    st.subheader("📋 Comprehensive Item Statistics Matrix & Validity Report")
    st.dataframe(df_res.style.apply(apply_full_styling, axis=1).format("{:.3f}", subset=["p", "q", "pq", "ddi", "d", "r_pbis"]), use_container_width=True)

    # 7. AUTOMATIC REPORT
    st.divider()
    st.header("📝 Methodological Interpretation")
    c1, c2 = st.columns(2)
    with c1:
        st.write("**Reliability Analysis:**")
        if kr20 >= 0.7: st.success(f"High Reliability ({kr20:.3f}). Instrument is consistent.")
        else: st.error(f"Low Reliability ({kr20:.3f}). Caution: Scores may be unstable.")
    with c2:
        st.write("**Standard Error of Measurement (SEM):**")
        st.info(f"SEM is {sem:.3f}. This figure indicates the range of fluctuation.")

    # 8. DISTRACTOR ANALYSIS (LENGKAP: FORMAT 1 (10%) + WARNA)
    st.subheader("🎯 Distractor Effectiveness (Option Frequency)")
    dist_pct = df[item_cols].apply(lambda x: x.astype(str).str.upper().str.strip().value_counts(normalize=True)).fillna(0).T
    dist_freq = df[item_cols].apply(lambda x: x.astype(str).str.upper().str.strip().value_counts()).fillna(0).T
    
    cols = sorted([c for c in dist_pct.columns if len(str(c)) == 1]) + sorted([c for c in dist_pct.columns if len(str(c)) > 1])
    df_dist_display = pd.DataFrame(index=item_cols)
    for col in cols:
        df_dist_display[col] = [f"{int(f)} ({p:.0%})" for f, p in zip(dist_freq[col], dist_pct[col])]

    def interpret_distractor_v3(row_idx):
        effective = [opt for opt in cols if dist_pct.loc[row_idx, opt] >= 0.05 and opt != "N/A"]
        return f"Effective Options: {', '.join(effective)}" if effective else "No effective distractors"
    
    df_dist_display['Interpretation'] = [interpret_distractor_v3(idx) for idx in df_dist_display.index]

    # Mewarnai berdasarkan angka persentase asli
    def color_cells(val, pct):
        return f'background-color: rgba(46, 204, 113, {pct})'

    st.dataframe(df_dist_display.style.apply(lambda x: [color_cells(v, dist_pct.loc[x.name, col]) if col in dist_pct.columns else '' for col, v in x.items()], axis=1), use_container_width=True)

    # 9. GUIDE DATA
    guide_data = {
        "Metric": ["Difficulty (d)", "Discrimination (ddi)", "r_pbis", "KR-20", "SEM"],
        "Ideal Range": ["0.30 - 0.70", "≥ 0.30", "≥ Threshold", "≥ 0.70", "Lower is Better"],
        "Description": ["Moderate level (d) is best.", "Distinguishes high/low achievers.", "Item-total correlation.", "Internal consistency.", "Score precision."]
    }
    df_guide = pd.DataFrame(guide_data)

    # EXPORT
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine='xlsxwriter') as writer:
        df_res.to_excel(writer, index=False, sheet_name='Item_Analysis')
        df_dist_display.to_excel(writer, index=True, sheet_name='Distractor_Analysis')
        df_guide.to_excel(writer, index=False, sheet_name='Reading_Guide')
        workbook = writer.book
        for sheet in writer.sheets.values():
            sheet.set_column('A:Z', 18)
            
    st.download_button(label="📥 Download Full Report", data=buf.getvalue(), file_name="Complete_Item_Analysis_Report.xlsx")
