# ======================================================================
# ITEM ANALYSIS TOOL - ADVANCED ACADEMIC VERSION (CTT + VALIDATION)
# ======================================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pointbiserialr
import io
import warnings
warnings.filterwarnings('ignore')

# ======================================================================
# CONFIG
# ======================================================================
st.set_page_config(page_title="Item Analysis Tool", page_icon="📊", layout="wide")
st.title("📊 ITEM ANALYSIS TOOL (ADVANCED ACADEMIC VERSION)")
st.markdown("---")

# ======================================================================
# SIDEBAR
# ======================================================================
with st.sidebar:
    st.header("⚙️ Threshold Parameters")

    difficult_threshold = st.number_input("Difficult (<)", 0.0, 1.0, 0.30, 0.05)
    easy_threshold = st.number_input("Easy (>)", 0.0, 1.0, 0.80, 0.05)

    poor_threshold = st.number_input("D Poor (<)", 0.0, 1.0, 0.20, 0.05)
    good_threshold = st.number_input("D Good (>=)", 0.0, 1.0, 0.40, 0.05)

    valid_threshold = st.number_input("Validity r_it (>=)", 0.0, 1.0, 0.20, 0.05)

    group_percent = st.slider("Upper/Lower Group (%)", 10, 50, 27)

# ======================================================================
# FUNCTIONS
# ======================================================================
def cronbach_alpha(df_scores):
    k = df_scores.shape[1]
    if k <= 1:
        return np.nan
    item_var = df_scores.var(ddof=1)
    total_var = df_scores.sum(axis=1).var(ddof=1)
    if total_var == 0:
        return np.nan
    return (k / (k-1)) * (1 - (item_var.sum() / total_var))

def interpret_reliability(val):
    if np.isnan(val):
        return "Undefined"
    elif val >= 0.80:
        return "Very Good"
    elif val >= 0.70:
        return "Good"
    elif val >= 0.60:
        return "Fair"
    else:
        return "Poor"

def check_dimensionality(df_scores):
    corr = df_scores.corr()
    eigenvalues = np.linalg.eigvals(corr)
    eigenvalues = np.sort(eigenvalues)[::-1]
    return eigenvalues

def standard_error(p, q, n):
    return np.sqrt((p*q)/n) if n > 0 else np.nan

# ======================================================================
# FILE INPUT
# ======================================================================
student_file = st.file_uploader("Upload Student Data (CSV)", type=['csv'])
key_file = st.file_uploader("Upload Answer Key (CSV)", type=['csv'])

if student_file and key_file:

    df = pd.read_csv(student_file, dtype=str)
    df_key = pd.read_csv(key_file, dtype=str)

    item_cols = df.columns[1:]
    n_students = len(df)
    n_items = len(item_cols)

    answer_key = df_key.iloc[0, 1:].str.upper().str.strip().tolist()

    # ==================================================================
    # SCORING
    # ==================================================================
    df_scores = pd.DataFrame()

    for i, col in enumerate(item_cols):
        df_scores[col] = (df[col].str.upper().str.strip() == answer_key[i]).astype(int)

    df['total_score'] = df_scores.sum(axis=1)

    # ==================================================================
    # GROUPING (KELLEY)
    # ==================================================================
    df_idx = df.reset_index().rename(columns={'index': 'idx'})
    df_sorted = df_idx.sort_values('total_score', ascending=False)

    n_group = max(1, int(np.ceil(n_students * group_percent / 100)))

    upper_idx = df_sorted.head(n_group)['idx']
    lower_idx = df_sorted.tail(n_group)['idx']

    upper = df.loc[upper_idx]
    lower = df.loc[lower_idx]

    # ==================================================================
    # ITEM ANALYSIS
    # ==================================================================
    results = []
    pq_values = []

    for i, item in enumerate(item_cols):

        p = df_scores[item].mean()
        q = 1 - p
        pq = p * q
        pq_values.append(pq)

        key = answer_key[i]

        p_upper = (upper[item].str.upper().str.strip() == key).mean()
        p_lower = (lower[item].str.upper().str.strip() == key).mean()

        d = p_upper - p_lower

        se = standard_error(p, q, n_students)

        total_minus = df['total_score'] - df_scores[item]

        if df_scores[item].var(ddof=1) == 0 or total_minus.var(ddof=1) == 0:
            r = 0
        else:
            r, _ = pointbiserialr(df_scores[item], total_minus)

        alpha_del = cronbach_alpha(df_scores.drop(columns=[item]))

        results.append([
            item, p, q, pq, p_upper, p_lower, d, se, r, alpha_del
        ])

    df_results = pd.DataFrame(results, columns=[
        'Item','p','q','pq','p_upper','p_lower','D','SE','r_it','Alpha_if_deleted'
    ])

    # ==================================================================
    # RELIABILITY
    # ==================================================================
    total_variance = df['total_score'].var(ddof=1)
    sum_pq = sum(pq_values)

    kr20 = (n_items/(n_items-1)) * (1 - (sum_pq / total_variance)) if total_variance > 0 and n_items > 1 else np.nan
    alpha = cronbach_alpha(df_scores)
    sem = df['total_score'].std(ddof=1) * np.sqrt(1 - kr20) if not np.isnan(kr20) else np.nan

    # ==================================================================
    # DIMENSIONALITY CHECK
    # ==================================================================
    eigenvalues = check_dimensionality(df_scores)

    # ==================================================================
    # REDUNDANCY CHECK
    # ==================================================================
    corr_matrix = df_scores.corr()
    high_corr_pairs = [(corr_matrix.columns[i], corr_matrix.columns[j])
                       for i in range(len(corr_matrix.columns))
                       for j in range(i+1, len(corr_matrix.columns))
                       if abs(corr_matrix.iloc[i,j]) > 0.70]

    # ==================================================================
    # OUTPUT
    # ==================================================================
    st.subheader("📊 Item Statistics")
    st.dataframe(df_results, use_container_width=True)

    st.subheader("📈 Reliability")
    col1, col2, col3 = st.columns(3)
    col1.metric("KR-20", f"{kr20:.4f}" if not np.isnan(kr20) else "Undefined")
    col2.metric("Cronbach Alpha", f"{alpha:.4f}" if not np.isnan(alpha) else "Undefined")
    col3.metric("SEM", f"{sem:.4f}" if not np.isnan(sem) else "Undefined")
    st.write("Interpretation:", interpret_reliability(alpha))

    st.subheader("🧠 Dimensionality Check (Eigenvalues)")
    st.write("Eigenvalues:", np.round(eigenvalues, 3))
    if eigenvalues[1] > 0 and eigenvalues[0]/eigenvalues[1] > 3:
        st.success("Likely Unidimensional")
    else:
        st.warning("Possible Multidimensional Structure")

    st.subheader("🔁 Item Redundancy (r > 0.70)")
    if high_corr_pairs:
        for pair in high_corr_pairs:
            st.write(pair)
    else:
        st.write("No high redundancy detected")

    st.subheader("📊 Inter-Item Correlation")
    fig, ax = plt.subplots(figsize=(8,6))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
    st.pyplot(fig)

    # ==================================================================
    # DOWNLOAD
    # ==================================================================
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df_results.to_excel(writer, sheet_name='Item Stats', index=False)
        pd.DataFrame({'KR20':[kr20], 'Alpha':[alpha], 'SEM':[sem]}).to_excel(writer, sheet_name='Reliability', index=False)

    st.download_button("📥 Download Excel", data=output.getvalue(), file_name="analysis.xlsx")
