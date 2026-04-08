# ======================================================================
# ITEM ANALYSIS TOOL - PROFESSIONALLY VALIDATED VERSION
# ======================================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pointbiserialr
import io

# CONFIG
st.set_page_config(page_title="Academic Item Analysis", page_icon="📈", layout="wide")
st.title("📈 ADVANCED ITEM ANALYSIS (CTT VALIDATED)")

# SIDEBAR PARAMETERS
with st.sidebar:
    st.header("⚙️ Analysis Settings")
    group_percent = st.slider("Kelley's Group Grouping (%)", 10, 50, 27)
    r_threshold = st.number_input("Validity Threshold (r_pbis)", 0.0, 1.0, 0.25)
    st.info("Standar umum: D > 0.30 & r_pbis > 0.25")

# FUNCTIONS
def calculate_cronbach_alpha(df_scores):
    k = df_scores.shape[1]
    if k <= 1: return np.nan
    item_vars = df_scores.var(ddof=1).sum()
    total_var = df_scores.sum(axis=1).var(ddof=1)
    if total_var == 0: return 0
    return (k / (k - 1)) * (1 - (item_vars / total_var))

# FILE UPLOADER
student_file = st.file_uploader("Upload Student Responses (CSV)", type=['csv'])
key_file = st.file_uploader("Upload Answer Key (CSV)", type=['csv'])

if student_file and key_file:
    # Load Data
    df = pd.read_csv(student_file).fillna("N/A")
    df_key = pd.read_csv(key_file)
    
    item_cols = df.columns[1:] # Asumsi kolom 0 adalah ID/Nama
    answer_key = df_key.iloc[0, 1:].str.upper().str.strip().tolist()
    
    # 1. SCORING ENGINE
    df_scores = pd.DataFrame()
    for i, col in enumerate(item_cols):
        df_scores[col] = (df[col].astype(str).str.upper().str.strip() == answer_key[i]).astype(int)
    
    total_scores = df_scores.sum(axis=1)
    df['Total_Score'] = total_scores
    n_students = len(df)
    n_items = len(item_cols)

    # 2. GROUPING (UPPER/LOWER)
    n_group = max(1, int(np.ceil(n_students * group_percent / 100)))
    df_sorted = df.sort_values('Total_Score', ascending=False)
    
    upper_indices = df_sorted.head(n_group).index
    lower_indices = df_sorted.tail(n_group).index
    
    # 3. ITEM STATISTICS CALCULATIONS
    item_stats = []
    for i, item in enumerate(item_cols):
        # Difficulty Index (p)
        p = df_scores[item].mean()
        
        # Discrimination Index (D)
        p_upper = df_scores.loc[upper_indices, item].mean()
        p_lower = df_scores.loc[lower_indices, item].mean()
        d_index = p_upper - p_lower
        
        # Point-Biserial Correlation (Validity)
        # Menggunakan skor total (r_pbis)
        if df_scores[item].var() == 0:
            r_pbis = 0
        else:
            r_pbis, _ = pointbiserialr(df_scores[item], total_scores)
            
        # Standard Error of Measurement for Item
        se_item = np.sqrt((p * (1 - p)) / n_students)
        
        # Alpha if Deleted
        remaining_df = df_scores.drop(columns=[item])
        alpha_deleted = calculate_cronbach_alpha(remaining_df)
        
        # Interpretation
        diff_label = "Easy" if p > 0.7 else ("Difficult" if p < 0.3 else "Moderate")
        disc_label = "Good" if d_index >= 0.4 else ("Fair" if d_index >= 0.2 else "Poor")
        status = "RETAIN" if (r_pbis >= r_threshold and d_index >= 0.2) else "REVISE/REJECT"

        item_stats.append({
            "Item": item,
            "Diff_Index (p)": round(p, 3),
            "Difficulty": diff_label,
            "Disc_Index (D)": round(d_index, 3),
            "Discrimination": disc_label,
            "r_pbis (Validity)": round(r_pbis, 3),
            "Alpha_If_Deleted": round(alpha_deleted, 3),
            "SE_Item": round(se_item, 3),
            "Decision": status
        })

    df_results = pd.DataFrame(item_stats)

    # 4. RELIABILITY & AGGREGATE STATS
    # KR-20 formula: (k/(k-1)) * (1 - sum(pq)/var_total)
    sum_pq = (df_scores.mean() * (1 - df_scores.mean())).sum()
    var_total = total_scores.var(ddof=1)
    kr20 = (n_items/(n_items-1)) * (1 - (sum_pq/var_total)) if var_total > 0 else 0
    sem_total = total_scores.std(ddof=1) * np.sqrt(1 - kr20)

    # OUTPUT DISPLAY
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Items", n_items)
    col2.metric("Samples", n_students)
    col3.metric("KR-20 Reliability", f"{kr20:.3f}")
    col4.metric("Mean Score", f"{total_scores.mean():.2f}")

    st.divider()
    
    st.subheader("📋 Item Analysis Summary Table")
    st.dataframe(df_results, use_container_width=True)

    # 5. VISUALIZATIONS
    st.subheader("📊 Visual Distribution")
    v_col1, v_col2 = st.columns(2)
    
    with v_col1:
        fig1, ax1 = plt.subplots()
        sns.histplot(total_scores, kde=True, color='skyblue', ax=ax1)
        ax1.set_title("Score Distribution (Normality)")
        st.pyplot(fig1)
        
    with v_col2:
        fig2, ax2 = plt.subplots()
        sns.scatterplot(data=df_results, x="Diff_Index (p)", y="Disc_Index (D)", hue="Decision", ax=ax2)
        ax2.axhline(0.2, ls='--', color='red', alpha=0.5)
        ax2.set_title("Difficulty vs Discrimination")
        st.pyplot(fig2)

    # 6. EXPORT
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df_results.to_excel(writer, index=False, sheet_name='Analysis_Results')
        # Add summary sheet
        summary_df = pd.DataFrame({
            "Metric": ["KR-20", "SEM", "N Items", "N Students", "Mean"],
            "Value": [kr20, sem_total, n_items, n_students, total_scores.mean()]
        })
        summary_df.to_excel(writer, index=False, sheet_name='Reliability_Summary')
    
    st.download_button(
        label="📥 Download Full Report (Excel)",
        data=output.getvalue(),
        file_name="Item_Analysis_Report.xlsx",
        mime="application/vnd.ms-excel"
    )
