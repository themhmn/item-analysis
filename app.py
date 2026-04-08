import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import pointbiserialr
import io

# ======================================================================
# ACADEMIC ITEM ANALYSIS - COMPREHENSIVE VERSION (2026)
# ======================================================================

st.set_page_config(page_title="Item Analysis Pro", page_icon="📈", layout="wide")

# Visual Setup
st.markdown("""
    <style>
    .main { background-color: #f4f4f9; }
    .stAlert { border: 2px solid #000; border-radius: 0px; }
    </style>
    """, unsafe_allow_html=True)

st.title("🛡️ RIGOROUS ITEM ANALYSIS TOOL (CTT)")
st.write("Full Classical Test Theory Suite: Methodologically validated metrics for educational research.")

# SIDEBAR: SETTINGS & LEGEND
with st.sidebar:
    st.header("📊 Settings & Legend")
    group_percent = st.slider("Kelley's Grouping (%)", 10, 50, 27)
    validity_limit = st.number_input("r_pbis Threshold", 0.0, 1.0, 0.25)
    
    st.divider()
    st.write("**Quick Ref:**")
    st.write("- **p:** Difficulty Index")
    st.write("- **ddi/d:** Discrimination Index")
    st.write("- **r_pbis:** Item Validity")

# FILE UPLOADER
u1, u2 = st.columns(2)
with u1:
    student_file = st.file_uploader("Upload Student Data (CSV)", type=['csv'])
with u2:
    key_file = st.file_uploader("Upload Key Data (CSV)", type=['csv'])

if student_file and key_file:
    # 1. DATA PREPARATION
    df = pd.read_csv(student_file)
    df_key = pd.read_csv(key_file)
    item_cols = df.columns[1:] 
    answer_key = df_key.iloc[0, 1:].astype(str).str.upper().str.strip().tolist()
    
    # 2. SCORING LOGIC
    df_scores = pd.DataFrame()
    for i, col in enumerate(item_cols):
        df_scores[col] = (df[col].astype(str).str.upper().str.strip() == answer_key[i]).astype(int)
    
    total_scores = df_scores.sum(axis=1)
    df['Total_Score'] = total_scores
    n_students, n_items = len(df), len(item_cols)

    # 3. GROUPING (UPPER & LOWER)
    n_group = max(1, int(np.ceil(n_students * group_percent / 100)))
    df_sorted = df.sort_values('Total_Score', ascending=False)
    up_idx, lo_idx = df_sorted.head(n_group).index, df_sorted.tail(n_group).index

    # 4. ITEM ANALYSIS CALCULATION
    results = []
    for i, item in enumerate(item_cols):
        # Difficulty (p) & (q)
        p = df_scores[item].mean()
        q = 1 - p
        pq = p * q
        
        # Discrimination (ddi / d)
        p_up = df_scores.loc[up_idx, item].mean()
        p_lo = df_scores.loc[lo_idx, item].mean()
        d_val = p_up - p_lo
        
        # Validity (r_pbis)
        if df_scores[item].var() != 0:
            r_pb, _ = pointbiserialr(df_scores[item], total_scores)
        else:
            r_pb = 0.0

        # Interpretasi Deskriptif
        p_eval = "Easy" if p > 0.7 else "Difficult" if p < 0.3 else "Moderate"
        d_eval = "Excellent" if d_val >= 0.4 else "Good" if d_val >= 0.3 else "Fair" if d_val >= 0.2 else "Poor"
        v_eval = "Valid" if r_pb >= validity_limit else "Invalid"
        
        # Comprehensive Decision Mapping
        if r_pb >= validity_limit and d_val >= 0.3:
            decision = "RETAIN"
        elif r_pb >= 0.2 and d_val >= 0.2:
            decision = "REVISE"
        else:
            decision = "REJECT"

        results.append({
            "Item": item,
            "p (Difficulty)": p,
            "p_Evaluation": p_eval,
            "q (1-p)": q,
            "pq (Variance)": pq,
            "ddi": d_val,
            "d (Discrimination)": d_val,
            "d_Evaluation": d_eval,
            "r_pbis (Validity)": r_pb,
            "Validity_Evaluation": v_eval,
            "DECISION": decision
        })

    df_res = pd.DataFrame(results)

    # 5. TEST RELIABILITY (KR-20)
    var_total = total_scores.var(ddof=1)
    sum_pq = df_res["pq (Variance)"].sum()
    kr20 = (n_items/(n_items-1)) * (1 - (sum_pq/var_total)) if var_total > 0 else 0
    std_dev = total_scores.std(ddof=1)
    sem = std_dev * np.sqrt(1 - kr20)

    # DASHBOARD METRICS
    st.divider()
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Students (N)", n_students)
    m2.metric("Items (k)", n_items)
    m3.metric("Mean Score", f"{total_scores.mean():.2f}")
    m4.metric("KR-20 Reliability", f"{kr20:.3f}")
    m5.metric("SEM", f"{sem:.3f}")

    # 6. DISTRACTOR ANALYSIS
    dist_data = []
    for item in item_cols:
        counts = df[item].astype(str).str.upper().str.strip().value_counts(normalize=True).to_dict()
        counts['Item'] = item
        dist_data.append(counts)
    df_dist = pd.DataFrame(dist_data).set_index('Item').fillna(0)

    # DISPLAY TABLE
    st.subheader("📋 Item Analysis Matrix")
    st.dataframe(df_res, use_container_width=True)

    st.subheader("🎯 Distractor Effectiveness (%)")
    st.dataframe(df_dist.style.format("{:.2%}"), use_container_width=True)

    # 7. EXCEL EXPORT (2 SHEETS + INTERPRETATION)
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine='xlsxwriter') as writer:
        # Sheet 1: Main Analysis
        df_res.to_excel(writer, index=False, sheet_name='Item_Analysis')
        
        # Sheet 2: Distractor Analysis
        df_dist.to_excel(writer, index=True, sheet_name='Distractor_Analysis')
        
        # Adding Interpretation Summary in Sheet 1
        workbook = writer.book
        worksheet = writer.sheets['Item_Analysis']
        header_format = workbook.add_format({'bold': True, 'bg_color': '#D3D3D3', 'border': 1})
        
        # Append Reliability info at the bottom of Sheet 1
        summary_row = len(df_res) + 3
        worksheet.write(summary_row, 0, "TEST RELIABILITY SUMMARY", header_format)
        worksheet.write(summary_row + 1, 0, f"KR-20 Reliability: {kr20:.4f}")
        worksheet.write(summary_row + 2, 0, f"Standard Error of Measurement (SEM): {sem:.4f}")
        worksheet.write(summary_row + 3, 0, f"Mean Score: {total_scores.mean():.4f}")
        
    st.download_button(
        label="📥 Download Full Academic Report (2 Sheets)",
        data=buf.getvalue(),
        file_name="Complete_Item_Analysis_Report.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
