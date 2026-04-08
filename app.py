import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import pointbiserialr
import io

# ======================================================================
# PROFESSIONAL ITEM ANALYSIS TOOL - FULL DESCRIPTIVE EDITION
# ======================================================================

st.set_page_config(page_title="Item Analysis Pro", page_icon="📈", layout="wide")

st.title("📈 ADVANCED ITEM ANALYSIS (CTT STANDARDS)")
st.write("Full statistical metrics with per-item descriptive interpretations and traffic light markers.")

# SIDEBAR PARAMETERS
with st.sidebar:
    st.header("⚙️ Threshold Configuration")
    group_percent = st.slider("Kelley's Grouping (%)", 10, 50, 27)
    validity_limit = st.number_input("Minimum r_pbis", 0.0, 1.0, 0.25)
    st.markdown("---")
    st.write("**Color Guide:**")
    st.write("🟢 Green: Ideal | 🟡 Yellow: Review | 🔴 Red: Reject")

# FILE UPLOADER
u1, u2 = st.columns(2)
with u1:
    student_file = st.file_uploader("Upload Student Responses (CSV)", type=['csv'])
with u2:
    key_file = st.file_uploader("Upload Answer Key (CSV)", type=['csv'])

if student_file and key_file:
    # 1. DATA LOADING & CLEANING
    df = pd.read_csv(student_file).fillna("N/A")
    df_key = pd.read_csv(key_file)
    item_cols = df.columns[1:] 
    answer_key = df_key.iloc[0, 1:].astype(str).str.upper().str.strip().tolist()
    
    # 2. SCORING ENGINE
    df_scores = pd.DataFrame()
    for i, col in enumerate(item_cols):
        df_scores[col] = (df[col].astype(str).str.upper().str.strip() == answer_key[i]).astype(int)
    
    total_scores = df_scores.sum(axis=1)
    df['Total_Score'] = total_scores
    n_students = len(df)
    n_items = len(item_cols)

    # 3. GROUPING
    n_group = max(1, int(np.ceil(n_students * group_percent / 100)))
    df_sorted = df.sort_values('Total_Score', ascending=False)
    up_idx, lo_idx = df_sorted.head(n_group).index, df_sorted.tail(n_group).index

    # 4. FULL STATISTICAL CALCULATION WITH PER-ITEM INTERPRETATION
    results = []
    for i, item in enumerate(item_cols):
        p = df_scores[item].mean()
        q = 1 - p
        pq = p * q
        p_up, p_lo = df_scores.loc[up_idx, item].mean(), df_scores.loc[lo_idx, item].mean()
        d = p_up - p_lo
        ddi = (p_up - p_lo) / p_up if p_up > 0 else 0
        r_pb, _ = pointbiserialr(df_scores[item], total_scores) if df_scores[item].var() != 0 else (0,0)

        # --- INDIVIDUAL DESCRIPTIVE INTERPRETATIONS ---
        # 1. Difficulty Interpretation
        if p > 0.7: diff_desc = "Easy"
        elif p < 0.3: diff_desc = "Difficult"
        else: diff_desc = "Moderate"

        # 2. Discrimination Interpretation
        if d >= 0.4: disc_desc = "Excellent"
        elif d >= 0.3: disc_desc = "Good"
        elif d >= 0.2: disc_desc = "Fair"
        else: disc_desc = "Poor"

        # 3. Validity Interpretation
        val_desc = "Valid" if r_pb >= validity_limit else "Invalid"

        # 4. Final Decision
        decision = "RETAIN" if (r_pb >= validity_limit and d >= 0.2) else "REJECT/REVISE"

        results.append({
            "Item": item,
            "P (Diff)": p,
            "P_Desc": diff_desc,
            "Q": q,
            "PQ": pq,
            "D (Disc)": d,
            "D_Desc": disc_desc,
            "DDI": ddi,
            "r_pbis": r_pb,
            "r_Desc": val_desc,
            "Decision": decision
        })

    df_res = pd.DataFrame(results)

    # 5. GLOBAL RELIABILITY (KR-20)
    sum_pq = df_res["PQ"].sum()
    var_total = total_scores.var(ddof=1)
    kr20 = (n_items/(n_items-1)) * (1 - (sum_pq/var_total)) if var_total > 0 else 0

    # DASHBOARD
    st.divider()
    col_m1, col_m2, col_m3 = st.columns(3)
    col_m1.metric("Samples (N)", n_students)
    col_m2.metric("Items (k)", n_items)
    col_m3.metric("KR-20 Reliability", f"{kr20:.3f}")

    # 6. TRAFFIC LIGHT STYLING
    def apply_traffic_light(row):
        styles = [''] * len(row)
        # Style P
        p_val = row['P (Diff)']
        if p_val < 0.3: styles[1] = 'background-color: #ffcccc' # Red
        elif p_val > 0.7: styles[1] = 'background-color: #ccffcc' # Green
        else: styles[1] = 'background-color: #fff2cc' # Yellow

        # Style D
        d_val = row['D (Disc)']
        if d_val >= 0.4: styles[5] = 'background-color: #2ecc71; color: white'
        elif d_val >= 0.2: styles[5] = 'background-color: #f1c40f'
        else: styles[5] = 'background-color: #e74c3c; color: white'

        # Style r_pbis
        r_val = row['r_pbis']
        if r_val >= validity_limit: styles[8] = 'background-color: #2ecc71; color: white'
        else: styles[8] = 'background-color: #e74c3c; color: white'

        return styles

    st.subheader("📋 Complete Item Analysis Matrix (With Detailed Interpretations)")
    styled_df = df_res.style.apply(apply_traffic_light, axis=1)\
                            .format("{:.3f}", subset=["P (Diff)", "Q", "PQ", "D (Disc)", "DDI", "r_pbis"])
    
    st.dataframe(styled_df, use_container_width=True)

    # 7. AUTOMATIC TEST REPORT
    st.divider()
    st.header("📝 Executive Test Interpretation")
    if kr20 >= 0.8:
        st.success(f"**Reliability ({kr20:.3f}): VERY HIGH.** The instrument is highly consistent and trustworthy.")
    elif kr20 >= 0.6:
        st.warning(f"**Reliability ({kr20:.3f}): MODERATE.** The instrument is acceptable, but check items labeled 'POOR' or 'REVISE'.")
    else:
        st.error(f"**Reliability ({kr20:.3f}): LOW.** Major inconsistencies detected. The test scores are not stable.")

    # 8. DISTRACTOR ANALYSIS
    st.subheader("🎯 Distractor Efficiency (Alphabetical Order)")
    dist_data = [df[item].astype(str).str.upper().str.strip().value_counts(normalize=True).to_dict() | {"Item": item} for item in item_cols]
    df_dist = pd.DataFrame(dist_data).set_index('Item').fillna(0)
    # Order A-E
    cols = sorted([c for c in df_dist.columns if len(str(c)) == 1]) + sorted([c for c in df_dist.columns if len(str(c)) > 1])
    st.dataframe(df_dist[cols].style.background_gradient(cmap='YlGn').format("{:.2%}"), use_container_width=True)

    # EXPORT
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine='xlsxwriter') as writer:
        df_res.to_excel(writer, index=False, sheet_name='Item_Analysis')
        df_dist.to_excel(writer, sheet_name='Distractors')
    st.download_button("📥 Download Full Analysis Report", data=buf.getvalue(), file_name="Full_Item_Analysis.xlsx")
