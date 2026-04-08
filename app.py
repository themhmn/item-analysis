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

# SIDEBAR: THE COMPREHENSIVE LEGEND
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

        # Descriptive Logic
        p_desc = "Easy" if p > 0.7 else "Difficult" if p < 0.3 else "Moderate"
        d_desc = "Excellent" if d_val >= 0.4 else "Good" if d_val >= 0.3 else "Fair" if d_val >= 0.2 else "Poor"
        r_desc = "Valid" if r_pb >= validity_limit else "Invalid"
        
        # Decision Logic
        if r_pb >= validity_limit and d_val >= 0.3:
            decision = "RETAIN"
        elif r_pb >= 0.2 and d_val >= 0.2:
            decision = "REVISE"
        else:
            decision = "REJECT"

        results.append({
            "Item": item, 
            "p": p, "p_Eval": p_desc, 
            "q": q, "pq": p*q,
            "ddi": d_val, 
            "d": p, 
            "d_Eval": d_desc, 
            "r_pbis": r_pb, "r_Eval": r_desc, 
            "DECISION": decision
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

    # 6. FULL STATISTICAL STYLING (TRAFFIC LIGHT SYSTEM)
    def apply_full_styling(row):
        styles = [''] * len(row)
        
        # Difficulty Styling (p & d)
        dif_color = '#ccffcc' if row['p'] > 0.7 else '#ffcccc' if row['p'] < 0.3 else '#fff2cc'
        styles[1] = f'background-color: {dif_color}; color: black' # p
        styles[2] = f'background-color: {dif_color}; color: black' # p_Eval
        styles[6] = f'background-color: {dif_color}; color: black' # d (Difficulty)

        # Discrimination Styling (ddi)
        if row['ddi'] >= 0.4: dis_color, txt = '#2ecc71', 'white'
        elif row['ddi'] >= 0.3: dis_color, txt = '#3498db', 'white'
        elif row['ddi'] >= 0.2: dis_color, txt = '#f1c40f', 'black'
        else: dis_color, txt = '#e74c3c', 'white'
        
        styles[5] = f'background-color: {dis_color}; color: {txt}' # ddi
        styles[7] = f'background-color: {dis_color}; color: {txt}' # d_Eval (Discrimination Desc)

        # Validity Styling (r_pbis)
        val_bg = '#ccffcc' if row['r_pbis'] >= validity_limit else '#ffcccc'
        styles[8] = f'background-color: {val_bg}; color: black; font-weight: bold' # r_pbis
        styles[9] = f'background-color: {val_bg}; color: black' # r_Eval

        # Decision Styling
        if row['DECISION'] == "RETAIN": styles[10] = 'background-color: #27ae60; color: white; font-weight: bold'
        elif row['DECISION'] == "REVISE": styles[10] = 'background-color: #f39c12; color: white'
        else: styles[10] = 'background-color: #c0392b; color: white'

        return styles

    st.subheader("📋 Comprehensive Item Statistics Matrix & Validity Report")
    st.dataframe(
        df_res.style.apply(apply_full_styling, axis=1)
        .format("{:.3f}", subset=["p", "q", "pq", "ddi", "d", "r_pbis"]), 
        use_container_width=True
    )

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
        st.info(f"SEM is {sem:.3f}. This figure indicates the range of fluctuation in students' true scores.")

    # 8. DISTRACTOR ANALYSIS
    st.subheader("🎯 Distractor Effectiveness (Option Frequency)")
    dist_data = [df[item].astype(str).str.upper().str.strip().value_counts(normalize=True).to_dict() | {"Item": item} for item in item_cols]
    df_dist = pd.DataFrame(dist_data).set_index('Item').fillna(0)
    cols = sorted([c for c in df_dist.columns if len(str(c)) == 1]) + sorted([c for c in df_dist.columns if len(str(c)) > 1])
    
    def interpret_distractor(row):
        effective = [opt for opt, val in row.items() if val >= 0.05 and opt != "N/A"]
        return f"Effective Options: {', '.join(effective)}" if effective else "No effective distractors"
    
    df_dist_styled = df_dist[cols].copy()
    df_dist_styled['Interpretation'] = df_dist[cols].apply(interpret_distractor, axis=1)
    st.dataframe(df_dist_styled.style.background_gradient(cmap='YlGn', subset=cols).format("{:.2%}", subset=cols), use_container_width=True)

    # 9. PANDUAN MEMBACA DATA (GUIDE)
    guide_data = {
        "Metric": ["Difficulty (d)", "Discrimination (ddi)", "r_pbis", "KR-20", "SEM"],
        "Ideal Range": ["0.30 - 0.70", "≥ 0.30", "≥ Threshold", "≥ 0.70", "Lower is Better"],
        "Description": [
            "Moderate level (d) is best for norm-referenced tests.",
            "Discrimination (ddi) distinguishes between high and low achievers.",
            "Correlation between item and total test score.",
            "Internal consistency of the entire test.",
            "Precision of the scores obtained."
        ]
    }
    df_guide = pd.DataFrame(guide_data)

    # EXPORT
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine='xlsxwriter') as writer:
        df_res.to_excel(writer, index=False, sheet_name='Item_Analysis')
        df_dist_styled.to_excel(writer, index=True, sheet_name='Distractor_Analysis')
        df_guide.to_excel(writer, index=False, sheet_name='Reading_Guide')
        
        workbook = writer.book
        for sheet in writer.sheets.values():
            sheet.set_column('A:Z', 18)
            
    st.download_button(
        label="📥 Download Full Report",
        data=buf.getvalue(),
        file_name="Complete_Item_Analysis_Report.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
