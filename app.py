import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import pointbiserialr
import io

# ======================================================================
# ACADEMIC ITEM ANALYSIS - STATISTICALLY RIGOROUS VERSION (2026)
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

# SIDEBAR: THE COMPREHENSIVE LEGEND
with st.sidebar:
    st.header("📊 Methodological Legend")
    
    with st.expander("1. Difficulty Index (P)", expanded=True):
        st.write("""
        - **Easy (P > 0.70):** 🟢
        - **Moderate (0.30 - 0.70):** 🟡 (Ideal)
        - **Difficult (P < 0.30):** 🔴
        """)

    with st.expander("2. Discrimination (D)"):
        st.write("""
        - **Excellent (D ≥ 0.40):** 🟢
        - **Good (0.30 - 0.39):** 🔵
        - **Marginal (0.20 - 0.29):** 🟡
        - **Poor (D < 0.20):** 🔴 (Reject)
        """)

    with st.expander("3. Point Biserial (r_pbis)"):
        st.write("""
        Measures item-total correlation. 
        - **Valid:** ≥ Threshold 🟢
        - **Invalid:** < Threshold 🔴
        - *Negative values indicate a flawed item.*
        """)

    with st.expander("4. DDI & PQ"):
        st.write("""
        - **DDI:** Discrimination Difficulty Index.
        - **PQ:** Item Variance ($P \\times Q$). Max variance at $P=0.5$.
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
        d = p_up - p_lo
        r_pb, _ = pointbiserialr(df_scores[item], total_scores) if df_scores[item].var() != 0 else (0,0)

        # Descriptive Logic
        p_desc = "Easy" if p > 0.7 else "Difficult" if p < 0.3 else "Moderate"
        d_desc = "Excellent" if d >= 0.4 else "Good" if d >= 0.3 else "Fair" if d >= 0.2 else "Poor"
        
        # Decision (Statistically Sound)
        if r_pb >= validity_limit and d >= 0.3:
            decision = "RETAIN"
        elif r_pb >= 0.2 and d >= 0.2:
            decision = "REVISE"
        else:
            decision = "REJECT"

        results.append({
            "Item": item, "P": p, "P_Eval": p_desc, "Q": q, "PQ": p*q,
            "D": d, "D_Eval": d_desc, "r_pbis": r_pb, "DECISION": decision
        })

    df_res = pd.DataFrame(results)

    # 5. TEST-LEVEL STATISTICS (KR-20 & SEM)
    var_total = total_scores.var(ddof=1)
    kr20 = (n_items/(n_items-1)) * (1 - (df_res["PQ"].sum()/var_total)) if var_total > 0 else 0
    sem = total_scores.std(ddof=1) * np.sqrt(1 - kr20)

    # METRIC DASHBOARD
    st.divider()
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Sample Size (N)", n_students)
    m2.metric("Items (k)", n_items)
    m3.metric("KR-20 Reliability", f"{kr20:.3f}")
    m4.metric("SEM (Error)", f"{sem:.3f}")

    # 6. STYLING (TRAFFIC LIGHT)
    def apply_academic_style(row):
        styles = [''] * len(row)
        # P Styling
        if row['P'] < 0.3: styles[1] = 'background-color: #ffcccc'
        elif row['P'] > 0.7: styles[1] = 'background-color: #ccffcc'
        else: styles[1] = 'background-color: #fff2cc'
        # D Styling
        if row['D'] >= 0.4: styles[5] = 'background-color: #2ecc71; color: white'
        elif row['D'] < 0.2: styles[5] = 'background-color: #e74c3c; color: white'
        else: styles[5] = 'background-color: #f1c40f'
        # r_pbis
        if row['r_pbis'] < validity_limit: styles[7] = 'color: #e74c3c; font-weight: bold'
        return styles

    st.subheader("📋 Comprehensive Item Statistics Matrix")
    st.dataframe(df_res.style.apply(apply_academic_style, axis=1).format("{:.3f}", subset=["P", "Q", "PQ", "D", "r_pbis"]), use_container_width=True)

    # 7. AUTOMATIC REPORT
    st.divider()
    st.header("📝 Methodological Interpretation")
    c1, c2 = st.columns(2)
    with c1:
        st.write("**Reliability Analysis:**")
        if kr20 >= 0.7: st.success(f"High Reliability ({kr20:.3f}). Instrument is consistent.")
        else: st.error(f"Low Reliability ({kr20:.3f}). Caution: Scores may be unstable.")
    with c2:
        st.write("**Standard Error (SEM):**")
        st.info(f"SEM is {sem:.3f}. This score represents the standard deviation of errors of measurement.")

    # 8. DISTRACTOR ANALYSIS
    st.subheader("🎯 Distractor Effectiveness (Option Frequency)")
    dist_data = [df[item].astype(str).str.upper().str.strip().value_counts(normalize=True).to_dict() | {"Item": item} for item in item_cols]
    df_dist = pd.DataFrame(dist_data).set_index('Item').fillna(0)
    cols = sorted([c for c in df_dist.columns if len(str(c)) == 1]) + sorted([c for c in df_dist.columns if len(str(c)) > 1])
    st.dataframe(df_dist[cols].style.background_gradient(cmap='YlGn').format("{:.2%}"), use_container_width=True)

    # EXPORT
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine='xlsxwriter') as writer:
        df_res.to_excel(writer, index=False, sheet_name='ItemAnalysis')
    st.download_button("📥 Download Academic Report", data=buf.getvalue(), file_name="Item_Analysis_Report.xlsx")
