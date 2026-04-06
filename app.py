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
# PAGE CONFIGURATION
# ======================================================================
st.set_page_config(page_title="Item Analysis Tool", page_icon="📊", layout="wide")

st.title("📊 ITEM ANALYSIS TOOL")
st.markdown("---")

# ======================================================================
# SIDEBAR - THRESHOLD PARAMETERS
# ======================================================================
with st.sidebar:
    st.header("⚙️ Threshold Parameters")
    st.caption("Adjust these values based on your assessment standards")
    
    st.subheader("Difficulty Index (p)")
    col1, col2 = st.columns(2)
    with col1:
        difficult_threshold = st.number_input("Difficult (<)", value=0.30, step=0.05)
    with col2:
        easy_threshold = st.number_input("Easy (>)", value=0.80, step=0.05)
    
    st.subheader("Discrimination Index (D)")
    col1, col2 = st.columns(2)
    with col1:
        poor_threshold = st.number_input("Poor (<)", value=0.20, step=0.05)
    with col2:
        good_threshold = st.number_input("Good (≥)", value=0.40, step=0.05)
    
    st.subheader("Validity (r_it)")
    valid_threshold = st.number_input("Valid (≥)", value=0.20, step=0.05)
    
    st.subheader("Group Classification")
    group_percent = st.slider("Upper/Lower Group Percentage", min_value=10, max_value=50, value=27, step=1)
    
    st.markdown("---")
    st.caption("Scripted by Muhaimin Abdullah")
    st.caption("Based on Classical Test Theory (CTT)")

# ======================================================================
# INTERPRETATION FUNCTIONS
# ======================================================================
def interpret_p(p, difficult_threshold, easy_threshold):
    if p < difficult_threshold:
        return "Difficult", "Item is too difficult"
    elif p <= easy_threshold:
        return "Moderate", "Item has optimal difficulty"
    else:
        return "Easy", "Item is too easy"

def interpret_d(d, poor_threshold, good_threshold):
    if d < poor_threshold:
        return "Poor", "Low discrimination"
    elif d < good_threshold:
        return "Fair", "Moderate discrimination"
    else:
        return "Very Good", "Excellent discrimination"

def interpret_ddi(ddi):
    if ddi > 0:
        return "Functional", "Effective distractor"
    elif ddi == 0:
        return "Neutral", "Equal selection"
    else:
        return "Non-Functional", "Ineffective distractor"

# ======================================================================
# INITIALIZE SESSION STATE
# ======================================================================
if 'df' not in st.session_state:
    st.session_state.df = None
if 'answer_key_df' not in st.session_state:
    st.session_state.answer_key_df = None
if 'file_loaded' not in st.session_state:
    st.session_state.file_loaded = False

# ======================================================================
# DATA INPUT
# ======================================================================
tab1, tab2 = st.tabs(["📁 Upload Data", "📊 Analysis Results"])

with tab1:
    st.subheader("Upload Student Response File")
    student_file = st.file_uploader("Choose CSV file", type=['csv'], key="student")
    
    if student_file is not None:
        if student_file.size > 5 * 1024 * 1024:
            st.error("❌ File size exceeds 5MB limit!")
        else:
            try:
                student_file.seek(0)
                df = pd.read_csv(student_file, dtype=str)
                if not df.empty and len(df.columns) >= 2:
                    st.success(f"✅ File uploaded: {student_file.name}")
                    st.session_state.df = df
                    st.session_state.file_loaded = True
                else:
                    st.error("❌ Invalid data format.")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

    st.subheader("Upload Answer Key File (Optional)")
    key_file = st.file_uploader("Choose CSV key file", type=['csv'], key="answer_key")
    if key_file is not None:
        try:
            key_file.seek(0)
            st.session_state.answer_key_df = pd.read_csv(key_file, dtype=str)
        except:
            st.session_state.answer_key_df = None

# ======================================================================
# ANALYSIS PROCESS
# ======================================================================
if st.session_state.file_loaded and st.session_state.df is not None:
    df = st.session_state.df.copy()
    df_key = st.session_state.answer_key_df
    item_columns = df.columns[1:].tolist()
    
    # Mode Detection
    sample = df[item_columns[0]].dropna().astype(str).str.strip().values
    is_binary = all(v in ['0', '1'] for v in sample[:50]) if len(sample) > 0 else False
    mode = "binary" if is_binary else "multiple_choice"

    # Scoring logic
    df_scores = pd.DataFrame()
    if mode == "multiple_choice" and df_key is not None and not df_key.empty:
        answer_key = [str(x).strip().upper() for x in df_key.iloc[0, 1:].values]
        for i, item in enumerate(item_columns):
            key_val = answer_key[i] if i < len(answer_key) else None
            df_scores[item] = (df[item].astype(str).str.strip().str.upper() == key_val).astype(int)
    else:
        for item in item_columns:
            df_scores[item] = pd.to_numeric(df[item], errors='coerce').fillna(0).astype(int)

    df['total_score'] = df_scores.sum(axis=1)
    n_students, n_items = len(df), len(item_columns)
    
    # Upper/Lower Group
    n_group = max(1, int(np.ceil(n_students * group_percent / 100)))
    df_sorted = df.sort_values('total_score', ascending=False).reset_index(drop=True)
    upper_group, lower_group = df_sorted.head(n_group), df_sorted.tail(n_group)

    # Statistics Calculation
    results = []
    p_values, d_values, r_values, se_values, alpha_if_deleted_values = [], [], [], [], []
    pq_all = []

    for i, item in enumerate(item_columns):
        p_val = df_scores[item].mean()
        q_val = 1 - p_val
        pq_val = p_val * q_val
        p_values.append(p_val)
        pq_all.append(pq_val)
        
        # Upper/Lower Prop
        if mode == "multiple_choice" and df_key is not None:
            ans_key = [str(x).strip().upper() for x in df_key.iloc[0, 1:].values]
            p_up = (upper_group[item].astype(str).str.strip().str.upper() == ans_key[i]).sum() / n_group
            p_lo = (lower_group[item].astype(str).str.strip().str.upper() == ans_key[i]).sum() / n_group
        else:
            p_up = pd.to_numeric(upper_group[item], errors='coerce').sum() / n_group
            p_lo = pd.to_numeric(lower_group[item], errors='coerce').sum() / n_group
        
        d_val = p_up - p_lo
        d_values.append(d_val)
        se_val = np.sqrt(pq_val / n_students) if n_students > 0 else 0
        se_values.append(se_val)

        # Validity (Point Biserial Corrected)
        total_minus_item = df['total_score'] - df_scores[item]
        if df_scores[item].var() == 0 or total_minus_item.var() == 0:
            r_it = 0.0
        else:
            r_it, _ = pointbiserialr(df_scores[item], total_minus_item)
        r_values.append(r_it)

        # RUMUS DIPERBAIKI: Alpha if Item Deleted (KR-20 logic)
        var_without = total_minus_item.var(ddof=1)
        # Hitung sum_pq untuk item yang tersisa
        sum_pq_without = sum(df_scores[c].mean() * (1 - df_scores[c].mean()) for c in item_columns if c != item)
        
        # Constraint: k_new = n_items - 1. Rumus KR-20: (k_new/(k_new-1))
        # Pembagi nol terjadi jika n_items <= 2.
        if var_without > 0 and n_items > 2:
            k_new = n_items - 1
            alpha = (k_new / (k_new - 1)) * (1 - (sum_pq_without / var_without))
        else:
            alpha = 0.0
        alpha_if_deleted_values.append(alpha)

        # Interpretations & Recommendation
        p_int, _ = interpret_p(p_val, difficult_threshold, easy_threshold)
        d_int, _ = interpret_d(d_val, poor_threshold, good_threshold)
        val_int = "Valid" if r_it >= valid_threshold else "Invalid"
        
        if r_it >= valid_threshold and d_val >= poor_threshold and difficult_threshold <= p_val <= easy_threshold:
            rec = "RETAIN"
        elif r_it < 0.10 or d_val < 0.10:
            rec = "DROP"
        else:
            rec = "REVISE"

        results.append([item, round(p_val,4), round(q_val,4), round(pq_val,4), round(p_up,4), round(p_lo,4), 
                        round(d_val,4), d_int, round(se_val,6), round(r_it,4), val_int, round(alpha,4), rec, p_int])

    # KR-20 & SEM
    total_var = df['total_score'].var(ddof=1)
    sum_pq = sum(pq_all)
    kr20 = (n_items/(n_items-1)) * (1 - (sum_pq / total_var)) if total_var > 0 and n_items > 1 else 0
    sem = df['total_score'].std(ddof=1) * np.sqrt(max(0, 1 - kr20))

    # Distractor Analysis (DDI)
    distractor_results = []
    if mode == "multiple_choice" and df_key is not None:
        ans_key_list = [str(x).strip().upper() for x in df_key.iloc[0, 1:].values]
        all_opts = sorted(list(set(df[item_columns].values.flatten().astype(str))))
        option_list = [o for o in all_opts if o.isalpha() and len(o) == 1]
        
        for i, item in enumerate(item_columns):
            key_val = ans_key_list[i] if i < len(ans_key_list) else None
            if not key_val: continue
            for opt in option_list:
                if opt == key_val: continue
                n_sel = (df[item].astype(str).str.strip().str.upper() == opt).sum()
                perc = (n_sel / n_students) * 100
                up_sel = (upper_group[item].astype(str).str.strip().str.upper() == opt).sum()
                lo_sel = (lower_group[item].astype(str).str.strip().str.upper() == opt).sum()
                p_up, p_lo = up_sel/n_group, lo_sel/n_group
                ddi = p_lo - p_up
                ddi_int, _ = interpret_ddi(ddi)
                distractor_results.append([item, key_val, opt, n_sel, round(perc,1), up_sel, lo_sel, 
                                           round(p_up,4), round(p_lo,4), round(ddi,4), ddi_int,
                                           "Yes" if perc >= 5.0 else "No", "Yes" if lo_sel > up_sel else "No"])

    df_results = pd.DataFrame(results, columns=['Item','p','q','pq','p_upper','p_lower','D','D_Interpretation',
                                                'SE','r_it','Validity','Alpha_if_deleted','Recommendation','p_Interpretation'])

    with tab2:
        st.markdown("## 📋 ITEM ANALYSIS SUMMARY")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Students", n_students); c2.metric("Items", n_items)
        c3.metric("Mode", mode.upper()); c4.metric("KR-20", f"{kr20:.4f}")
        
        st.info(f"📏 SEM: {sem:.4f} | Σpq: {sum_pq:.4f} | Total Var: {total_var:.4f}")
        st.dataframe(df_results, use_container_width=True)

        if distractor_results:
            st.markdown("---")
            st.markdown("## 🎯 DISTRACTOR ANALYSIS (DDI)")
            df_dist = pd.DataFrame(distractor_results, columns=['Item','Key','Option','N_Select','Percent','Upper_N','Lower_N',
                                                               'Prop_Upper','Prop_Lower','DDI','DDI_Interpretation','>=5%','Lower > Upper'])
            st.dataframe(df_dist, use_container_width=True)

        # VISUALIZATIONS
        st.markdown("---")
        st.markdown("## 📊 VISUALIZATIONS")
        
        # Row 1 & 2 (Difficulty, Discrimination, Validity, Proportion Incorrect)
        col1, col2 = st.columns(2)
        with col1:
            fig1, ax1 = plt.subplots(); ax1.bar(range(1, n_items+1), p_values, color='green'); ax1.set_title("Difficulty (p)"); st.pyplot(fig1)
            fig3, ax3 = plt.subplots(); ax3.bar(range(1, n_items+1), r_values, color='blue'); ax3.set_title("Validity (r_it)"); st.pyplot(fig3)
        with col2:
            fig2, ax2 = plt.subplots(); ax2.bar(range(1, n_items+1), d_values, color='orange'); ax2.set_title("Discrimination (D)"); st.pyplot(fig2)
            fig4, ax4 = plt.subplots(); ax4.bar(range(1, n_items+1), [1-p for p in p_values], color='navy'); ax4.set_title("Prop. Incorrect (q)"); st.pyplot(fig4)

        # Row 3 & 4 (Upper/Lower, Rec, Distribution, Pie)
        col1, col2 = st.columns(2)
        with col1:
            fig5, ax5 = plt.subplots(); ax5.plot(range(1,n_items+1), [r[4] for r in results], 'g-', label='Upper'); ax5.plot(range(1,n_items+1), [r[5] for r in results], 'r-', label='Lower'); ax5.legend(); st.pyplot(fig5)
            fig7, ax7 = plt.subplots(); ax7.hist(df['total_score'], bins=10); ax7.set_title("Score Distribution"); st.pyplot(fig7)
        with col2:
            fig6, ax6 = plt.subplots(); ax6.bar(range(1, n_items+1), [1]*n_items, color='gray'); ax6.set_title("Recommendation Overview"); st.pyplot(fig6)
            fig8, ax8 = plt.subplots(); counts = df_results['Recommendation'].value_counts(); ax8.pie(counts, labels=counts.index, autopct='%1.1f%%'); st.pyplot(fig8)

        # Row 5 (SE & Alpha if Deleted)
        col1, col2 = st.columns(2)
        with col1:
            fig9, ax9 = plt.subplots(); ax9.bar(range(1, n_items+1), se_values, color='purple'); ax9.set_title("Standard Error (SE)"); st.pyplot(fig9)
        with col2:
            fig10, ax10 = plt.subplots(); ax10.bar(range(1, n_items+1), alpha_if_deleted_values, color='teal'); ax10.axhline(kr20, color='red', ls='--'); ax10.set_title("Alpha if Deleted"); st.pyplot(fig10)

        # Row 6: Heatmap (DIPERBAIKI: Ukuran Dinamis)
        if n_items > 1:
            st.markdown("### 11. Inter-Item Correlation Heatmap")
            h_size = min(20, max(8, n_items * 0.4))
            fig11, ax11 = plt.subplots(figsize=(h_size, h_size*0.8))
            sns.heatmap(df_scores.corr(), annot=n_items < 20, fmt='.2f', cmap='RdBu_r', ax=ax11)
            st.pyplot(fig11)

        # DOWNLOAD (COMPLETE)
        st.markdown("---")
        st.markdown("## 📥 DOWNLOAD RESULTS")
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df_results.to_excel(writer, sheet_name='Item_Statistics', index=False)
            pd.DataFrame({'KR-20': [kr20], 'SEM': [sem], 'Sum_pq': [sum_pq], 'Total_Var': [total_var]}).to_excel(writer, sheet_name='Reliability', index=False)
            if distractor_results: df_dist.to_excel(writer, sheet_name='Distractor_Analysis', index=False)
        
        output.seek(0)
        st.download_button(label="📥 Download Excel Report", data=output, file_name="item_analysis_report.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        st.success("✅ Perbaikan rumus selesai & fitur lengkap!")

else:
    with tab1: st.info("👈 Silakan unggah file CSV Anda.")
