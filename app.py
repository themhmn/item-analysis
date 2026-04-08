import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pointbiserialr
import io

# ======================================================================
# ACADEMIC ITEM ANALYSIS TOOL (FULL METRICS: P, Q, PQ, D, DDI, RPBS)
# ======================================================================

st.set_page_config(page_title="Pro Item Analysis", page_icon="🧪", layout="wide")
st.title("🧪 FULL-METRIC ITEM ANALYSIS (CTT)")

with st.sidebar:
    st.header("⚙️ Parameters")
    group_percent = st.slider("Grouping % (Kelley)", 10, 50, 27)
    st.info("P = Difficulty Index\nQ = 1 - P\nPQ = Item Variance\nD = Discrimination Index\nDDI = Discrimination Difficulty Index")

# FILE UPLOADER
student_file = st.file_uploader("Upload Student Data (CSV)", type=['csv'])
key_file = st.file_uploader("Upload Answer Key (CSV)", type=['csv'])

if student_file and key_file:
    df = pd.read_csv(student_file).fillna("MISSING")
    df_key = pd.read_csv(key_file)
    
    item_cols = df.columns[1:] 
    answer_key = df_key.iloc[0, 1:].str.upper().str.strip().tolist()
    
    # 1. SCORING & FREQUENCY (FOR DISTRACTOR ANALYSIS)
    df_scores = pd.DataFrame()
    for i, col in enumerate(item_cols):
        df_scores[col] = (df[col].astype(str).str.upper().str.strip() == answer_key[i]).astype(int)
    
    total_scores = df_scores.sum(axis=1)
    df['Total_Score'] = total_scores
    n_students = len(df)
    n_items = len(item_cols)

    # 2. GROUPING
    n_group = max(1, int(np.ceil(n_students * group_percent / 100)))
    df_sorted = df.sort_values('Total_Score', ascending=False)
    upper_idx = df_sorted.head(n_group).index
    lower_idx = df_sorted.tail(n_group).index

    # 3. COMPREHENSIVE CALCULATION
    item_stats = []
    for i, item in enumerate(item_cols):
        # Dasar P, Q, PQ
        p = df_scores[item].mean()
        q = 1 - p
        pq = p * q
        
        # Upper & Lower Performance
        p_upper = df_scores.loc[upper_idx, item].mean()
        p_lower = df_scores.loc[lower_idx, item].mean()
        
        # D (Discrimination Index)
        d_index = p_upper - p_lower
        
        # DDI (Discrimination Difficulty Index)
        # Rumus: (P_upper - P_lower) / (P_upper + P_lower) jika dibutuhkan perbandingan proporsi
        ddi = (p_upper - p_lower) / (p_upper + p_lower) if (p_upper + p_lower) > 0 else 0
        
        # Point Biserial (r_pbis)
        if df_scores[item].var() == 0:
            r_pbis = 0
        else:
            r_pbis, _ = pointbiserialr(df_scores[item], total_scores)

        item_stats.append({
            "Item": item,
            "P (Diff)": round(p, 3),
            "Q (1-P)": round(q, 3),
            "PQ (Var)": round(pq, 3),
            "P_Upper": round(p_upper, 3),
            "P_Lower": round(p_lower, 3),
            "D (Disc)": round(d_index, 3),
            "DDI": round(ddi, 3),
            "r_pbis": round(r_pbis, 3)
        })

    df_results = pd.DataFrame(item_stats)

    # 4. RELIABILITY (KR-20)
    sum_pq = df_results["PQ (Var)"].sum()
    var_total = total_scores.var(ddof=1)
    kr20 = (n_items/(n_items-1)) * (1 - (sum_pq/var_total)) if var_total > 0 else 0

    # DISPLAY METRICS
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Samples (N)", n_students)
    m2.metric("Items (k)", n_items)
    m3.metric("KR-20 Reliability", f"{kr20:.3f}")
    m4.metric("Sum PQ", f"{sum_pq:.3f}")

    st.subheader("📊 Full Item Analysis Table")
    st.dataframe(df_results, use_container_width=True)

    # 5. DISTRACTOR ANALYSIS (Bonus untuk Akademik)
    st.subheader("🎯 Distractor Analysis (Option Frequency)")
    distractor_data = []
    for item in item_cols:
        counts = df[item].value_counts(normalize=True).to_dict()
        counts['Item'] = item
        distractor_data.append(counts)
    
    df_distractor = pd.DataFrame(distractor_data).set_index('Item').fillna(0)
    st.dataframe(df_distractor.style.format("{:.2%}"), use_container_width=True)

    # 6. EXPORT
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df_results.to_excel(writer, index=False, sheet_name='Item_Stats')
        df_distractor.to_excel(writer, sheet_name='Distractor_Analysis')
    
    st.download_button("📥 Download Full Analysis", data=output.getvalue(), file_name="Item_Analysis_Complete.xlsx")
