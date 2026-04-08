import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import pointbiserialr
import io

# ======================================================================
# ITEM ANALYSIS (CTT) - STATISTICAL CORRECT VERSION
# ======================================================================

st.set_page_config(page_title="Item Analysis Pro", page_icon="📈", layout="wide")

st.title("🛡️ RIGOROUS ITEM ANALYSIS TOOL (CTT)")

# SIDEBAR SETTINGS
with st.sidebar:
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
    df = pd.read_csv(student_file).fillna("N/A")
    df_key = pd.read_csv(key_file)
    item_cols = df.columns[1:]
    answer_key = df_key.iloc[0, 1:].astype(str).str.upper().str.strip().tolist()

    # VECTOR SCORING
    df_scores = pd.DataFrame()
    for i, col in enumerate(item_cols):
        df_scores[col] = (df[col].astype(str).str.upper().str.strip() == answer_key[i]).astype(int)

    total_scores = df_scores.sum(axis=1)
    df['Total_Score'] = total_scores
    n_students, n_items = len(df), len(item_cols)

    # GROUPING FOR KELLEY D
    n_group = max(1, int(np.ceil(n_students * group_percent / 100)))
    df_sorted = df.sort_values('Total_Score', ascending=False)
    up_idx = df_sorted.head(n_group).index
    lo_idx = df_sorted.tail(n_group).index

    # CALCULATION
    results = []
    for item in item_cols:
        p = df_scores[item].mean()  # difficulty index
        q = 1 - p

        # Kelley's Discrimination Index (d)
        p_upper = df_scores.loc[up_idx, item].mean()
        p_lower = df_scores.loc[lo_idx, item].mean()
        d = p_upper - p_lower  # correct discrimination index

        # Point-Biserial Correlation (ddi)
        corrected_total = total_scores - df_scores[item]
        if df_scores[item].var() != 0:
            r_pb, _ = pointbiserialr(df_scores[item], corrected_total)
        else:
            r_pb = 0

        # Descriptive Labels
        p_desc = "Easy" if p > 0.7 else "Difficult" if p < 0.3 else "Moderate"
        d_desc = "Excellent" if d >= 0.4 else "Good" if d >= 0.3 else "Fair" if d >= 0.2 else "Poor"
        r_desc = "Valid" if r_pb >= validity_limit else "Invalid"

        # Decision
        if r_pb >= validity_limit and d >= 0.3:
            decision = "RETAIN"
        elif r_pb >= 0.2 and d >= 0.2:
            decision = "REVISE"
        else:
            decision = "REJECT"

        results.append({
            "Item": item,
            "p": p,
            "p_Eval": p_desc,
            "q": q,
            "pq": p*q,
            "d": d,        # KELLEY DISCRIMINATION
            "ddi": r_pb,   # POINT-BISERIAL
            "d_Eval": d_desc,
            "r_pbis": r_pb,
            "r_Eval": r_desc,
            "DECISION": decision
        })

    df_res = pd.DataFrame(results)

    # TEST-LEVEL STATS
    mean_score = total_scores.mean()
    std_score = total_scores.std(ddof=1)
    var_total = total_scores.var(ddof=1)
    kr20 = (n_items / (n_items - 1)) * (1 - (df_res["pq"].sum() / var_total)) if var_total > 0 else 0
    sem = std_score * np.sqrt(1 - kr20)

    # DASHBOARD
    st.divider()
    cols = st.columns(6)
    cols[0].metric("Students (N)", n_students)
    cols[1].metric("Items (k)", n_items)
    cols[2].metric("Mean Score", f"{mean_score:.2f}")
    cols[3].metric("Std. Deviation", f"{std_score:.2f}")
    cols[4].metric("KR-20 Reliability", f"{kr20:.3f}")
    cols[5].metric("SEM (Error)", f"{sem:.3f}")

    # DISPLAY RESULTS
    st.subheader("📋 Item Statistics")
    st.dataframe(df_res.style.format("{:.3f}", subset=["p", "q", "pq", "d", "ddi", "r_pbis"]), use_container_width=True)
