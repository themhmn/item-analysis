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

        distractors = [opt for opt in df[item].unique() if opt != answer_key[i] and opt != "N/A"]
        ddi_vals = []
        for opt in distractors:
            u_opt = (df.loc[up_idx, item].astype(str).str.upper().str.strip() == opt).mean()
            l_opt = (df.loc[lo_idx, item].astype(str).str.upper().str.strip() == opt).mean()
            ddi_vals.append(u_opt - l_opt)
        
        ddi_final = max(ddi_vals) if ddi_vals else 0

        corrected_total = total_scores - df_scores[item]
        r_pb, _ = pointbiserialr(df_scores[item], corrected_total) if df_scores[item].var() != 0 else (0,0)

        p_desc = "Easy" if p > 0.7 else "Difficult" if p < 0.3 else "Moderate"
        d_desc = "Excellent" if d_val >= 0.4 else "Good" if d_val >= 0.3 else "Fair" if d_val >= 0.2 else "Poor"
        r_desc = "Valid" if r_pb >= validity_limit else "Invalid"
        
        if r_pb >= validity_limit and d_val >= 0.3: decision = "RETAIN"
        elif r_pb >= 0.2 and d_val >= 0.2: decision = "REVISE"
        else: decision = "REJECT"

        results.append({
            "Item": item, "p": p, "p_Eval": p_desc, "q": q, "pq": pq, "Var": item_var,
            "d": d_val, "d_Eval": d_desc, "DDI": ddi_final, 
            "r_pbis": r_pb, "r_Eval": r_desc, "DECISION": decision
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
        styles[6] = styles[7] = styles[8] = f'background-color: {dis_color}; color: {txt}'
        val_bg = '#ccffcc' if row['r_pbis'] >= validity_limit else '#ffcccc'
        styles[9] = f'background-color: {val_bg}; color: black; font-weight: bold'
        styles[10] = f'background-color: {val_bg}; color: black'
        if row['DECISION'] == "RETAIN": styles[11] = 'background-color: #27ae60; color: white; font-weight: bold'
        elif row['DECISION'] == "REVISE": styles[11] = 'background-color: #f39c12; color: white'
        else: styles[11] = 'background-color: #c0392b; color: white'
        return styles

    st.dataframe(df_res.style.apply(apply_item_styling, axis=1).format("{:.4f}", subset=["p", "q", "pq", "Var", "d", "DDI", "r_pbis"]), use_container_width=True)

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
    for col in cols: df_dist_final[col] = df_dist_final[col].apply(lambda x: f"{x:.4f} ({x:.2%})")
    
    df_dist_final['Interpretation'] = df_dist[cols].apply(lambda row: f"Effective: {', '.join([str(opt) for opt, val in row.items() if val >= 0.05 and opt != 'N/A'])}", axis=1)
    
    st.dataframe(df_dist[cols].style.background_gradient(cmap='YlGn'), use_container_width=True)

    # --- EXCEL DOWNLOAD (4 SHEETS - FULL ENGLISH) ---
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine='xlsxwriter') as writer:
        df_res.to_excel(writer, index=False, sheet_name='Item_Analysis')
        df_ranking.to_excel(writer, index=False, sheet_name='Student_Ranking')
        df_dist_final.to_excel(writer, index=True, sheet_name='Distractor_Analysis')
        summary_data = {
            "Metric": ["Students (N)", "Items (k)", "Mean Score", "Std. Deviation", "KR-20", "Alpha", "SEM"],
            "Value": [n_students, n_items, mean_score, std_score, kr20, alpha, sem]
        }
        pd.DataFrame(summary_data).to_excel(writer, index=False, sheet_name='Reliability_Summary')
            
    st.download_button(label="📥 Download Full Report (4 Sheets)", data=buf.getvalue(), file_name="Complete_Item_Analysis_Report.xlsx")
