import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import pointbiserialr
import io

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

st.title("🛡️ RIGOROUS ITEM ANALYSIS TOOL (CTT)")

with st.sidebar:
    st.header("📊 Methodological Legend")
    with st.expander("1. Difficulty Index (p/d)", expanded=True):
        st.write("- **Easy (p > 0.70):** 🟢\n- **Moderate (0.30 - 0.70):** 🟡\n- **Difficult (p < 0.30):** 🔴")
    with st.expander("2. Discrimination (ddi)", expanded=True):
        st.write("- **Excellent (ddi ≥ 0.40):** 🟢\n- **Good (0.30 - 0.39):** 🔵\n- **Fair (0.20 - 0.29):** 🟡\n- **Poor (ddi < 0.20):** 🔴")
    with st.expander("3. Validity (r_pbis)", expanded=False):
        st.write("- **Valid:** ≥ Threshold\n- **Invalid:** < Threshold")
    with st.expander("4. Reliability (KR-20)", expanded=False):
        st.write("- **Reliable:** ≥ 0.70")
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
    answer_key = df_key.iloc[0, 1:].astype(str).str.upper().str.strip().tolist()
    
    df_scores = pd.DataFrame()
    for i, col in enumerate(item_cols):
        df_scores[col] = (df[col].astype(str).str.upper().str.strip() == answer_key[i]).astype(int)
    
    total_scores = df_scores.sum(axis=1)
    df['Total_Score'] = total_scores
    n_students, n_items = len(df), len(item_cols)

    n_group = max(1, int(np.ceil(n_students * group_percent / 100)))
    df_sorted = df.sort_values('Total_Score', ascending=False)
    up_idx, lo_idx = df_sorted.head(n_group).index, df_sorted.tail(n_group).index

    results = []
    for i, item in enumerate(item_cols):
        p = df_scores[item].mean()
        q = 1 - p
        p_up, p_lo = df_scores.loc[up_idx, item].mean(), df_scores.loc[lo_idx, item].mean()
        d_val = p_up - p_lo 
        r_pb, _ = pointbiserialr(df_scores[item], total_scores) if df_scores[item].var() != 0 else (0,0)
        
        results.append({
            "Item": item, "p": p, "p_Eval": "Easy" if p > 0.7 else "Difficult" if p < 0.3 else "Moderate",
            "q": q, "pq": p*q, "ddi": d_val, "d": p, 
            "d_Eval": "Excellent" if d_val >= 0.4 else "Good" if d_val >= 0.3 else "Fair" if d_val >= 0.2 else "Poor",
            "r_pbis": r_pb, "r_Eval": "Valid" if r_pb >= validity_limit else "Invalid",
            "DECISION": "RETAIN" if r_pb >= validity_limit and d_val >= 0.3 else "REVISE" if r_pb >= 0.2 and d_val >= 0.2 else "REJECT"
        })

    df_res = pd.DataFrame(results)
    kr20 = (n_items/(n_items-1)) * (1 - (df_res["pq"].sum()/total_scores.var())) if total_scores.var() > 0 else 0
    sem = total_scores.std() * np.sqrt(1 - kr20)

    st.divider()
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Students (N)", n_students)
    m2.metric("Items (k)", n_items)
    m3.metric("Mean Score", f"{total_scores.mean():.2f}")
    m4.metric("Std. Deviation", f"{total_scores.std():.2f}")
    m5.metric("KR-20 Reliability", f"{kr20:.3f}")
    m6.metric("SEM (Error)", f"{sem:.3f}")

    def apply_full_styling(row):
        styles = [''] * len(row)
        dif_color = '#ccffcc' if row['p'] > 0.7 else '#ffcccc' if row['p'] < 0.3 else '#fff2cc'
        styles[1] = styles[2] = styles[6] = f'background-color: {dif_color}; color: black'
        dis_color = '#2ecc71' if row['ddi'] >= 0.4 else '#3498db' if row['ddi'] >= 0.3 else '#f1c40f' if row['ddi'] >= 0.2 else '#e74c3c'
        styles[5] = styles[7] = f'background-color: {dis_color}; color: white'
        val_bg = '#ccffcc' if row['r_pbis'] >= validity_limit else '#ffcccc'
        styles[8] = styles[9] = f'background-color: {val_bg}; color: black; font-weight: bold'
        styles[10] = 'background-color: #27ae60; color: white' if row['DECISION'] == "RETAIN" else 'background-color: #f39c12; color: white' if row['DECISION'] == "REVISE" else 'background-color: #c0392b; color: white'
        return styles

    st.subheader("📋 Comprehensive Item Statistics Matrix")
    st.dataframe(df_res.style.apply(apply_full_styling, axis=1).format("{:.3f}", subset=["p", "q", "pq", "ddi", "d", "r_pbis"]), use_container_width=True)

    # DISTRACTOR: SEMUA ANGKA PAKAI DESIMAL (FREKUENSI & PERSENTASE)
    st.subheader("🎯 Distractor Effectiveness")
    dist_pct = df[item_cols].apply(lambda x: x.astype(str).str.upper().str.strip().value_counts(normalize=True)).fillna(0).T
    dist_freq = df[item_cols].apply(lambda x: x.astype(str).str.upper().str.strip().value_counts()).fillna(0).T
    
    df_dist_display = pd.DataFrame(index=item_cols)
    for col in sorted(dist_pct.columns):
        # MODIFIKASI: Angka frekuensi pakai .2f dan persentase pakai .2%
        df_dist_display[col] = [f"{f:.2f} ({p:.2%})" for f, p in zip(dist_freq[col], dist_pct[col])]

    df_dist_display['Interpretation'] = [f"Effective: {', '.join([o for o in dist_pct.columns if dist_pct.loc[i, o] >= 0.05 and o != 'N/A'])}" for i in df_dist_display.index]

    st.dataframe(df_dist_display.style.apply(lambda x: [f'background-color: rgba(46, 204, 113, {dist_pct.loc[x.name, c]})' if c in dist_pct.columns else '' for c in x.index], axis=1), use_container_width=True)

    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine='xlsxwriter') as writer:
        df_res.to_excel(writer, index=False, sheet_name='Item_Analysis')
        df_dist_display.to_excel(writer, index=True, sheet_name='Distractor_Analysis')
    st.download_button(label="📥 Download Full Report", data=buf.getvalue(), file_name="Item_Analysis_Report.xlsx")
