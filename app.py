# ======================================================================
# ITEM ANALYSIS - STREAMLIT VERSION (COMPLETE WITH ALL FEATURES)
# ======================================================================
# COMPLETE FEATURES:
# 1. p (difficulty index)
# 2. q (1-p)
# 3. pq (item variance)
# 4. p_upper (upper group proportion correct)
# 5. p_lower (lower group proportion correct)
# 6. D (discrimination index)
# 7. SE (standard error of item)
# 8. r_it (corrected item-total correlation)
# 9. KR-20 (reliability)
# 10. Alpha if item deleted
# 11. SEM (standard error of measurement)
# 12. DDI (Distractor Discrimination Index)
# 13. Explicit criteria flags (>=5%, Lower > Upper)
# 14. Visualizations (10 comprehensive charts)
# 15. Adjustable threshold parameters (sliders)
# 16. Multi-sheet Excel export
# 17. Max file size 5MB
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
        difficult_threshold = st.number_input("Difficult (<)", value=0.30, step=0.05, help="p < this value = Difficult")
    with col2:
        easy_threshold = st.number_input("Easy (>)", value=0.80, step=0.05, help="p > this value = Easy")
    
    st.subheader("Discrimination Index (D)")
    col1, col2 = st.columns(2)
    with col1:
        poor_threshold = st.number_input("Poor (<)", value=0.20, step=0.05, help="D < this value = Poor")
    with col2:
        good_threshold = st.number_input("Good (≥)", value=0.40, step=0.05, help="D ≥ this value = Very Good")
    
    st.subheader("Validity (r_it)")
    valid_threshold = st.number_input("Valid (≥)", value=0.20, step=0.05, help="r_it ≥ this value = Valid")
    
    st.subheader("Distractor Analysis")
    distractor_percent_threshold = st.number_input("Min Selection (%)", value=5.0, step=1.0, help="Distractor must be selected by at least this percentage")
    
    st.subheader("Group Classification")
    group_percent = st.slider("Upper/Lower Group Percentage", min_value=10, max_value=50, value=27, step=1, 
                               help="Kelley (1939) recommends 27% for optimal discrimination")
    
    st.markdown("---")
    st.caption("Scripted by Muhaimin Abdullah")
    st.caption("Based on Classical Test Theory (CTT)")

# ======================================================================
# INTERPRETATION FUNCTIONS
# ======================================================================
def interpret_p(p, difficult_threshold, easy_threshold):
    if p < difficult_threshold:
        return "Difficult", "Item is too difficult; only few students answered correctly"
    elif p <= easy_threshold:
        return "Moderate", "Item has optimal difficulty level"
    else:
        return "Easy", "Item is too easy; most students answered correctly"

def interpret_d(d, poor_threshold, good_threshold):
    if d < poor_threshold:
        return "Poor", "Item cannot distinguish between high and low ability students"
    elif d < good_threshold:
        return "Fair", "Item moderately distinguishes students"
    else:
        return "Very Good", "Item excellently distinguishes students"

def interpret_ddi(ddi):
    if ddi > 0:
        return "Functional", "More low-ability students selected this distractor (effective)"
    elif ddi == 0:
        return "Neutral", "Equal selection by both groups"
    else:
        return "Non-Functional", "More high-ability students selected this distractor (ineffective)"

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
    
    student_file = st.file_uploader("Choose CSV file with student responses", type=['csv'], key="student")
    
    if student_file is not None:
        if student_file.size > 5 * 1024 * 1024:
            st.error("❌ File size exceeds 5MB limit!")
        else:
            try:
                student_file.seek(0)
                df = pd.read_csv(student_file, dtype=str)
                
                if df.empty:
                    st.error("❌ Empty data! Please check your file.")
                elif len(df.columns) < 2:
                    st.error("❌ Data must have at least 2 columns (student ID + at least 1 item)")
                else:
                    st.success(f"✅ File uploaded: {student_file.name}")
                    st.write(f"Dimensions: {df.shape[0]} rows × {df.shape[1]} columns")
                    st.subheader("Data Preview (First 5 rows)")
                    st.dataframe(df.head())
                    
                    st.session_state.df = df
                    st.session_state.file_loaded = True
            except Exception as e:
                st.error(f"❌ Error reading file: {str(e)}")
    else:
        if not st.session_state.file_loaded:
            st.info("No file uploaded yet")
    
    st.subheader("Upload Answer Key File (Optional)")
    st.caption("Leave empty if data is already in binary format (0/1)")
    
    key_file = st.file_uploader("Choose CSV file with answer keys", type=['csv'], key="answer_key")
    
    if key_file is not None:
        if key_file.size > 5 * 1024 * 1024:
            st.error("❌ Answer key file exceeds 5MB limit!")
        else:
            try:
                key_file.seek(0)
                df_key = pd.read_csv(key_file, dtype=str)
                
                if not df_key.empty:
                    st.success(f"✅ Answer key uploaded")
                    st.session_state.answer_key_df = df_key
                else:
                    st.warning("⚠️ Answer key file is empty")
            except Exception as e:
                st.warning(f"⚠️ Error reading answer key: {str(e)}")
                st.session_state.answer_key_df = None

# ======================================================================
# ANALYSIS PROCESS
# ======================================================================
if st.session_state.file_loaded and st.session_state.df is not None:
    
    df = st.session_state.df.copy()
    df_key = st.session_state.answer_key_df
    
    item_columns = df.columns[1:].tolist()
    
    if len(item_columns) == 0:
        with tab2:
            st.error("❌ No item columns found! File must have at least 2 columns.")
    else:
        # Detect data mode (binary or letter-based)
        sample = df[item_columns[0]].dropna().astype(str).str.strip().values
        sample_clean = [s for s in sample if s not in ['', 'nan', 'NaN', 'None']]
        
        if len(sample_clean) > 0:
            is_binary = all(v in ['0', '1'] for v in sample_clean[:50])
        else:
            is_binary = False
        
        mode = "binary" if is_binary else "multiple_choice"
        
        # Read answer key
        answer_key = None
        if mode == "multiple_choice" and df_key is not None and not df_key.empty:
            try:
                if df_key.shape[1] > 1:
                    answer_key = [str(x).strip().upper() for x in df_key.iloc[0, 1:].values]
                else:
                    answer_key = [str(df_key.iloc[0, 0]).strip().upper()]
            except Exception as e:
                st.warning(f"⚠️ Failed to read answer key: {str(e)}")
                answer_key = None
        
        # Convert to binary scores (1/0)
        df_scores = pd.DataFrame()
        if mode == "multiple_choice" and answer_key and len(answer_key) == len(item_columns):
            for i, item in enumerate(item_columns):
                key_value = answer_key[i] if i < len(answer_key) else None
                if key_value:
                    df_scores[item] = (df[item].astype(str).str.strip().str.upper() == key_value).astype(int)
                else:
                    df_scores[item] = 0
        else:
            for item in item_columns:
                df_scores[item] = pd.to_numeric(df[item], errors='coerce').fillna(0).astype(int)
        
        df['total_score'] = df_scores.sum(axis=1)
        n_students = len(df)
        n_items = len(item_columns)
        
        # Upper and lower groups (Kelley's 27% method)
        n_group = max(1, int(np.ceil(n_students * group_percent / 100)))
        df_sorted = df.sort_values('total_score', ascending=False).reset_index(drop=True)
        upper_group = df_sorted.head(n_group)
        lower_group = df_sorted.tail(n_group)
        
        # ======================================================================
        # ITEM STATISTICS CALCULATION
        # ======================================================================
        results = []
        p_values, q_values, pq_values, p_upper_values, p_lower_values = [], [], [], [], []
        d_values, se_values, r_values, alpha_if_deleted_values = [], [], [], []
        
        for i, item in enumerate(item_columns):
            # 1. p (difficulty index)
            p_val = df_scores[item].mean()
            p_values.append(p_val)
            
            # 2. q = 1 - p
            q_val = 1 - p_val
            q_values.append(q_val)
            
            # 3. pq = p * q (item variance)
            pq_val = p_val * q_val
            pq_values.append(pq_val)
            
            # 4. p_upper (proportion correct in upper group)
            if mode == "multiple_choice" and answer_key and i < len(answer_key):
                p_upper = (upper_group[item].astype(str).str.strip().str.upper() == answer_key[i]).sum() / n_group
                p_lower = (lower_group[item].astype(str).str.strip().str.upper() == answer_key[i]).sum() / n_group
            else:
                p_upper = pd.to_numeric(upper_group[item], errors='coerce').sum() / n_group
                p_lower = pd.to_numeric(lower_group[item], errors='coerce').sum() / n_group
            p_upper_values.append(p_upper)
            p_lower_values.append(p_lower)
            
            # 5. D = p_upper - p_lower (discrimination index)
            d_val = p_upper - p_lower
            d_values.append(d_val)
            
            # 6. SE = sqrt(pq / n_students)
            se_val = np.sqrt(pq_val / n_students) if n_students > 0 else 0
            se_values.append(se_val)
            
            # 7. Corrected item-total correlation (validity)
            total_minus_item = df['total_score'] - df_scores[item]
            if df_scores[item].var() == 0 or total_minus_item.var() == 0:
                r_it = 0.0
            else:
                r_it, _ = pointbiserialr(df_scores[item], total_minus_item)
            r_values.append(r_it)
            
            # 8. Alpha if item deleted
            total_without_item = df['total_score'] - df_scores[item]
            var_without = total_without_item.var(ddof=1)
            pq_without = 0
            for j, item2 in enumerate(item_columns):
                if j != i:
                    p2 = df_scores[item2].mean()
                    pq_without += p2 * (1 - p2)
            if var_without > 0 and (n_items - 1) > 1:
                alpha = ((n_items - 1) / (n_items - 2)) * (1 - (pq_without / var_without))
            else:
                alpha = 0
            alpha_if_deleted_values.append(alpha)
            
            # Interpretations
            p_interpretation, _ = interpret_p(p_val, difficult_threshold, easy_threshold)
            d_interpretation, _ = interpret_d(d_val, poor_threshold, good_threshold)
            validity_interpretation = "Valid" if r_it >= valid_threshold else "Invalid"
            
            # Final recommendation
            if r_it >= valid_threshold and d_val >= poor_threshold and difficult_threshold <= p_val <= easy_threshold:
                recommendation = "RETAIN"
            elif r_it < 0.10 or d_val < 0.10:
                recommendation = "DROP"
            else:
                recommendation = "REVISE"
            
            results.append([
                item, 
                round(p_val, 4), 
                round(q_val, 4), 
                round(pq_val, 4),
                round(p_upper, 4), 
                round(p_lower, 4), 
                round(d_val, 4), 
                d_interpretation,
                round(se_val, 6),
                round(r_it, 4), 
                validity_interpretation,
                round(alpha, 4),
                recommendation,
                p_interpretation
            ])
        
        # KR-20 reliability
        total_variance = df['total_score'].var(ddof=1)
        sum_pq = sum(pq_values)
        
        if total_variance > 0 and n_items > 1:
            kr20 = (n_items/(n_items-1)) * (1 - (sum_pq / total_variance))
        else:
            kr20 = 0
        
        sem = df['total_score'].std(ddof=1) * np.sqrt(max(0, 1 - kr20))
        
        # ======================================================================
        # DISTRACTOR ANALYSIS WITH COMPLETE CRITERIA FLAGS
        # ======================================================================
        distractor_results = []
        if mode == "multiple_choice" and answer_key:
            all_options = set()
            for item in item_columns:
                values = df[item].astype(str).str.strip().str.upper().dropna()
                for v in values:
                    if v.isalpha() and len(v) == 1:
                        all_options.add(v)
            option_list = sorted(all_options)
            
            for i, item in enumerate(item_columns):
                key_value = answer_key[i] if i < len(answer_key) else None
                if key_value is None:
                    continue
                item_data = df[item].astype(str).str.strip().str.upper()
                
                for option in option_list:
                    if option == key_value:
                        continue
                    
                    total_select = (item_data == option).sum()
                    percent = (total_select / n_students) * 100
                    
                    upper_select = (upper_group[item].astype(str).str.strip().str.upper() == option).sum()
                    lower_select = (lower_group[item].astype(str).str.strip().str.upper() == option).sum()
                    
                    prop_upper = upper_select / n_group if n_group > 0 else 0
                    prop_lower = lower_select / n_group if n_group > 0 else 0
                    
                    # DDI = proportion lower - proportion upper
                    ddi = prop_lower - prop_upper
                    
                    ddi_interpretation, _ = interpret_ddi(ddi)
                    
                    # EXPLICIT CRITERIA FLAGS
                    meets_percent_criteria = percent >= distractor_percent_threshold
                    meets_lower_upper_criteria = lower_select > upper_select
                    
                    # Final distractor status based on both criteria
                    if meets_percent_criteria and meets_lower_upper_criteria:
                        final_status = "✅ FUNCTIONAL"
                        final_status_color = "green"
                    elif meets_percent_criteria and not meets_lower_upper_criteria:
                        final_status = "⚠️ PARTIALLY FUNCTIONAL (Low > High criteria not met)"
                        final_status_color = "orange"
                    elif not meets_percent_criteria and meets_lower_upper_criteria:
                        final_status = "⚠️ PARTIALLY FUNCTIONAL (≥5% criteria not met)"
                        final_status_color = "orange"
                    else:
                        final_status = "❌ NON-FUNCTIONAL"
                        final_status_color = "red"
                    
                    distractor_results.append([
                        item, key_value, option, 
                        total_select, round(percent, 1), 
                        upper_select, lower_select, 
                        round(prop_upper, 4), round(prop_lower, 4),
                        round(ddi, 4), ddi_interpretation,
                        "✅ Yes" if meets_percent_criteria else "❌ No",
                        "✅ Yes" if meets_lower_upper_criteria else "❌ No",
                        final_status
                    ])
        
        # Final results dataframe
        df_results = pd.DataFrame(results, columns=[
            'Item', 'p', 'q', 'pq', 'p_upper', 'p_lower', 'D', 'D_Interpretation', 
            'SE', 'r_it', 'Validity', 'Alpha_if_deleted', 'Recommendation', 'p_Interpretation'
        ])
        
        # ======================================================================
        # DISPLAY RESULTS IN TAB2
        # ======================================================================
        with tab2:
            # ==================================================================
            # SECTION 1: SUMMARY METRICS
            # ==================================================================
            st.markdown("## 📋 ITEM ANALYSIS SUMMARY")
            st.markdown("Below are the key statistics for your test and items.")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("📊 Number of Students", n_students)
            with col2:
                st.metric("📝 Number of Items", n_items)
            with col3:
                st.metric("⚙️ Response Mode", mode.upper())
            with col4:
                st.metric("📈 KR-20 Reliability", f"{kr20:.4f}")
            
            # Reliability interpretation
            col1, col2 = st.columns(2)
            with col1:
                if kr20 >= 0.80:
                    st.success(f"✅ **Reliability:** {kr20:.4f} (Very Good - suitable for high-stakes testing)")
                    st.caption("Interpretation: The test has excellent internal consistency. Results are highly reliable.")
                elif kr20 >= 0.70:
                    st.info(f"📘 **Reliability:** {kr20:.4f} (Good - suitable for classroom exams)")
                    st.caption("Interpretation: The test has acceptable internal consistency for most educational purposes.")
                elif kr20 >= 0.60:
                    st.warning(f"⚠️ **Reliability:** {kr20:.4f} (Fair - acceptable for exploratory research)")
                    st.caption("Interpretation: The test has marginal reliability. Consider improving weak items.")
                else:
                    st.error(f"❌ **Reliability:** {kr20:.4f} (Poor - needs significant improvement)")
                    st.caption("Interpretation: The test lacks internal consistency. Major revision or replacement recommended.")
            
            with col2:
                st.info(f"📏 **Standard Error of Measurement (SEM):** {sem:.4f}")
                st.caption(f"95% Confidence Interval for true scores: ±{sem*1.96:.2f} points")
                st.caption(f"Σpq = {sum_pq:.4f} | Total Variance = {total_variance:.4f}")
                st.caption("Interpretation: SEM indicates the precision of individual student scores.")
            
            # ==================================================================
            # SECTION 2: COMPLETE ITEM STATISTICS TABLE
            # ==================================================================
            st.markdown("---")
            st.markdown("## 📊 COMPLETE ITEM STATISTICS")
            st.caption("This table shows all psychometric properties for each test item.")
            st.caption("**Recommendation Guide:** RETAIN = meets all criteria | REVISE = partially meets criteria | DROP = fails critical criteria")
            
            st.dataframe(df_results, use_container_width=True)
            
            # ==================================================================
            # SECTION 3: DISTRACTOR ANALYSIS WITH COMPLETE CRITERIA
            # ==================================================================
            if distractor_results:
                st.markdown("---")
                st.markdown("## 🎯 DISTRACTOR ANALYSIS WITH COMPLETE CRITERIA")
                
                st.markdown("""
                ### 📖 How to Interpret Distractor Analysis:
                
                A **functional distractor** must meet **BOTH** criteria:
                
                | Criterion | Description | Formula |
                |-----------|-------------|---------|
                | **1. Selection Rate** | At least **≥5%** of all students select this distractor | (N_Select / Total Students) × 100% ≥ 5% |
                | **2. Lower > Upper** | More **low-ability students** (lower group) select this distractor than high-ability students | Lower_N > Upper_N |
                
                **Status Legend:**
                - ✅ **FUNCTIONAL**: Both criteria met → Effective distractor, keep it
                - ⚠️ **PARTIALLY FUNCTIONAL**: One criterion met → Needs revision
                - ❌ **NON-FUNCTIONAL**: Neither criterion met → Replace this distractor
                """)
                
                df_distractor = pd.DataFrame(distractor_results, columns=[
                    'Item', 'Key', 'Option', 
                    'N_Select', 'Percent', 
                    'Upper_N', 'Lower_N',
                    'Prop_Upper', 'Prop_Lower',
                    'DDI', 'DDI_Interpretation',
                    '≥5%?', 'Lower > Upper?',
                    'Final Status'
                ])
                
                st.dataframe(df_distractor, use_container_width=True)
                
                # DDI Summary by item
                st.markdown("### 📊 DDI Summary by Item")
                st.caption("Mean DDI > 0 indicates distractors generally function well for this item")
                
                ddi_summary = []
                for item in item_columns:
                    item_distractors = df_distractor[df_distractor['Item'] == item]['DDI'].values
                    if len(item_distractors) > 0:
                        mean_ddi = np.mean(item_distractors)
                        functional_count = sum(1 for idx, row in df_distractor.iterrows() if row['Item'] == item and '✅ FUNCTIONAL' in row['Final Status'])
                        ddi_summary.append([item, len(item_distractors), round(mean_ddi, 4), functional_count])
                
                df_ddi_summary = pd.DataFrame(ddi_summary, columns=['Item', '# Distractors', 'Mean DDI', '# Functional'])
                st.dataframe(df_ddi_summary, use_container_width=True)
            
            # ==================================================================
            # SECTION 4: VISUALIZATIONS (ALL CHARTS BELOW TABLES)
            # ==================================================================
            st.markdown("---")
            st.markdown("## 📊 VISUALIZATIONS")
            st.caption("The following charts provide visual interpretation of item statistics.")
            
            # Row 1: Difficulty and Discrimination
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 1. Item Difficulty (p)")
                fig1, ax1 = plt.subplots(figsize=(8, 5))
                colors_p = ['#e74c3c' if x < difficult_threshold else ('#2ecc71' if x <= easy_threshold else '#f39c12') for x in p_values]
                bars1 = ax1.bar(range(1, n_items+1), p_values, color=colors_p, edgecolor='black', linewidth=0.5)
                ax1.axhline(difficult_threshold, color='#e74c3c', linestyle='--', linewidth=1.5, label=f'Difficult Threshold (<{difficult_threshold})')
                ax1.axhline(easy_threshold, color='#f39c12', linestyle='--', linewidth=1.5, label=f'Easy Threshold (>{easy_threshold})')
                ax1.set_xlabel('Item Number', fontsize=11)
                ax1.set_ylabel('Difficulty Index (p)', fontsize=11)
                ax1.set_title('Item Difficulty: Red=Difficult, Green=Moderate, Orange=Easy', fontsize=12)
                ax1.set_xticks(range(1, n_items+1))
                ax1.set_ylim(0, 1)
                ax1.legend(loc='lower right', fontsize=9)
                ax1.grid(axis='y', alpha=0.3)
                
                for bar, p_val in zip(bars1, p_values):
                    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, f'{p_val:.2f}', ha='center', va='bottom', fontsize=8)
                st.pyplot(fig1)
                plt.close()
            
            with col2:
                st.markdown("### 2. Item Discrimination (D)")
                fig2, ax2 = plt.subplots(figsize=(8, 5))
                colors_d = ['#2ecc71' if x >= good_threshold else ('#f39c12' if x >= poor_threshold else '#e74c3c') for x in d_values]
                bars2 = ax2.bar(range(1, n_items+1), d_values, color=colors_d, edgecolor='black', linewidth=0.5)
                ax2.axhline(good_threshold, color='#2ecc71', linestyle='--', linewidth=1.5, label=f'Very Good (≥{good_threshold})')
                ax2.axhline(poor_threshold, color='#f39c12', linestyle='--', linewidth=1.5, label=f'Fair (≥{poor_threshold})')
                ax2.set_xlabel('Item Number', fontsize=11)
                ax2.set_ylabel('Discrimination Index (D)', fontsize=11)
                ax2.set_title('Item Discrimination: Green=Very Good, Orange=Fair, Red=Poor', fontsize=12)
                ax2.set_xticks(range(1, n_items+1))
                ax2.set_ylim(-1, 1)
                ax2.legend(loc='lower right', fontsize=9)
                ax2.grid(axis='y', alpha=0.3)
                
                for bar, d_val in zip(bars2, d_values):
                    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.03, f'{d_val:.2f}', ha='center', va='bottom', fontsize=8)
                st.pyplot(fig2)
                plt.close()
            
            # Row 2: Validity and Proportion Incorrect
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 3. Item Validity (r_it)")
                fig3, ax3 = plt.subplots(figsize=(8, 5))
                colors_r = ['#2ecc71' if x >= valid_threshold else '#e74c3c' for x in r_values]
                bars3 = ax3.bar(range(1, n_items+1), r_values, color=colors_r, edgecolor='black', linewidth=0.5)
                ax3.axhline(valid_threshold, color='#2ecc71', linestyle='--', linewidth=1.5, label=f'Valid Threshold (≥{valid_threshold})')
                ax3.axhline(0, color='black', linestyle='-', linewidth=0.8)
                ax3.set_xlabel('Item Number', fontsize=11)
                ax3.set_ylabel('Corrected Item-Total Correlation (r_it)', fontsize=11)
                ax3.set_title('Item Validity: Green=Valid, Red=Invalid', fontsize=12)
                ax3.set_xticks(range(1, n_items+1))
                ax3.set_ylim(-1, 1)
                ax3.legend(loc='lower right', fontsize=9)
                ax3.grid(axis='y', alpha=0.3)
                
                for bar, r_val in zip(bars3, r_values):
                    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.03, f'{r_val:.2f}', ha='center', va='bottom', fontsize=8)
                st.pyplot(fig3)
                plt.close()
            
            with col2:
                st.markdown("### 4. Proportion Incorrect (q = 1-p)")
                fig4, ax4 = plt.subplots(figsize=(8, 5))
                bars4 = ax4.bar(range(1, n_items+1), q_values, color='#3498db', alpha=0.7, edgecolor='black', linewidth=0.5)
                ax4.set_xlabel('Item Number', fontsize=11)
                ax4.set_ylabel('q = 1 - p', fontsize=11)
                ax4.set_title('Proportion of Students Answering Incorrectly', fontsize=12)
                ax4.set_xticks(range(1, n_items+1))
                ax4.set_ylim(0, 1)
                ax4.grid(axis='y', alpha=0.3)
                
                for bar, q_val in zip(bars4, q_values):
                    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, f'{q_val:.2f}', ha='center', va='bottom', fontsize=8)
                st.pyplot(fig4)
                plt.close()
            
            # Row 3: Upper vs Lower Group Comparison
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 5. Upper vs Lower Group Performance")
                fig5, ax5 = plt.subplots(figsize=(8, 5))
                x = range(1, n_items+1)
                ax5.plot(x, p_upper_values, 'o-', color='#2ecc71', label='p_upper (Upper 27%)', linewidth=2, markersize=8)
                ax5.plot(x, p_lower_values, 's-', color='#e74c3c', label='p_lower (Lower 27%)', linewidth=2, markersize=8)
                ax5.fill_between(x, p_lower_values, p_upper_values, alpha=0.2, color='gray')
                ax5.set_xlabel('Item Number', fontsize=11)
                ax5.set_ylabel('Proportion Correct', fontsize=11)
                ax5.set_title('Comparison of Upper and Lower Group Performance', fontsize=12)
                ax5.set_xticks(range(1, n_items+1))
                ax5.set_ylim(0, 1)
                ax5.legend(loc='lower right', fontsize=9)
                ax5.grid(axis='both', alpha=0.3)
                st.pyplot(fig5)
                plt.close()
            
            with col2:
                st.markdown("### 6. Item Recommendations")
                fig6, ax6 = plt.subplots(figsize=(8, 5))
                colors_rec = ['#2ecc71' if r == 'RETAIN' else ('#f39c12' if r == 'REVISE' else '#e74c3c') for r in df_results['Recommendation']]
                bars6 = ax6.bar(range(1, n_items+1), [1]*n_items, color=colors_rec, edgecolor='black', linewidth=0.5)
                ax6.set_xlabel('Item Number', fontsize=11)
                ax6.set_title('Item Recommendations: Green=RETAIN, Orange=REVISE, Red=DROP', fontsize=12)
                ax6.set_xticks(range(1, n_items+1))
                ax6.set_yticks([])
                
                for bar, rec in zip(bars6, df_results['Recommendation']):
                    ax6.text(bar.get_x() + bar.get_width()/2, 1.05, rec, ha='center', va='bottom', fontsize=8, rotation=45)
                st.pyplot(fig6)
                plt.close()
            
            # Row 4: Score Distribution and Recommendation Pie Chart
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 7. Total Score Distribution")
                fig7, ax7 = plt.subplots(figsize=(8, 5))
                min_score = int(df['total_score'].min())
                max_score = int(df['total_score'].max())
                bins = range(min_score, max_score+2)
                ax7.hist(df['total_score'], bins=bins, edgecolor='black', alpha=0.7, color='#3498db')
                ax7.axvline(df['total_score'].mean(), color='#e74c3c', linestyle='--', linewidth=2, label=f"Mean = {df['total_score'].mean():.2f}")
                ax7.axvline(df['total_score'].median(), color='#2ecc71', linestyle='--', linewidth=2, label=f"Median = {df['total_score'].median():.2f}")
                ax7.set_xlabel('Total Score', fontsize=11)
                ax7.set_ylabel('Frequency', fontsize=11)
                ax7.set_title('Distribution of Student Total Scores', fontsize=12)
                ax7.legend(loc='upper right', fontsize=9)
                ax7.grid(axis='y', alpha=0.3)
                st.pyplot(fig7)
                plt.close()
            
            with col2:
                st.markdown("### 8. Recommendation Summary")
                fig8, ax8 = plt.subplots(figsize=(7, 5))
                retain_count = sum(1 for r in df_results['Recommendation'] if r == 'RETAIN')
                revise_count = sum(1 for r in df_results['Recommendation'] if r == 'REVISE')
                drop_count = sum(1 for r in df_results['Recommendation'] if r == 'DROP')
                
                if retain_count + revise_count + drop_count > 0:
                    sizes = [retain_count, revise_count, drop_count]
                    labels = ['RETAIN', 'REVISE', 'DROP']
                    colors_pie = ['#2ecc71', '#f39c12', '#e74c3c']
                    explode = (0.05, 0.05, 0.05)
                    
                    wedges, texts, autotexts = ax8.pie(
                        sizes, 
                        explode=explode,
                        labels=None,
                        colors=colors_pie, 
                        autopct='%1.1f%%', 
                        startangle=90,
                        textprops={'fontsize': 11, 'fontweight': 'bold'}
                    )
                    ax8.legend(
                        wedges, 
                        [f'{label}: {size} items ({size/n_items*100:.1f}%)' for label, size in zip(labels, sizes)],
                        title="Recommendation",
                        loc="center left",
                        bbox_to_anchor=(1, 0.5),
                        fontsize=10
                    )
                    ax8.set_title('Proportion of Item Recommendations', fontsize=12, fontweight='bold')
                    ax8.axis('equal')
                    
                    for autotext in autotexts:
                        autotext.set_color('white')
                        autotext.set_fontweight('bold')
                        autotext.set_fontsize(10)
                
                st.pyplot(fig8)
                plt.close()
            
            # Row 5: Additional Charts
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 9. Standard Error by Item")
                fig9, ax9 = plt.subplots(figsize=(8, 5))
                bars9 = ax9.bar(range(1, n_items+1), se_values, color='#9b59b6', alpha=0.7, edgecolor='black', linewidth=0.5)
                ax9.set_xlabel('Item Number', fontsize=11)
                ax9.set_ylabel('Standard Error (SE)', fontsize=11)
                ax9.set_title('Standard Error of Item Proportion', fontsize=12)
                ax9.set_xticks(range(1, n_items+1))
                ax9.grid(axis='y', alpha=0.3)
                
                for bar, se_val in zip(bars9, se_values):
                    ax9.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, f'{se_val:.4f}', ha='center', va='bottom', fontsize=7)
                st.pyplot(fig9)
                plt.close()
            
            with col2:
                st.markdown("### 10. Alpha if Item Deleted")
                fig10, ax10 = plt.subplots(figsize=(8, 5))
                colors_alpha = ['#2ecc71' if x <= kr20 else '#e74c3c' for x in alpha_if_deleted_values]
                bars10 = ax10.bar(range(1, n_items+1), alpha_if_deleted_values, color=colors_alpha, edgecolor='black', linewidth=0.5)
                ax10.axhline(kr20, color='blue', linestyle='--', linewidth=1.5, label=f'Overall KR-20 = {kr20:.4f}')
                ax10.set_xlabel('Item Number', fontsize=11)
                ax10.set_ylabel('Alpha if Item Deleted', fontsize=11)
                ax10.set_title('Impact of Removing Each Item on Reliability', fontsize=12)
                ax10.set_xticks(range(1, n_items+1))
                ax10.legend(loc='lower right', fontsize=9)
                ax10.grid(axis='y', alpha=0.3)
                
                for bar, alpha_val in zip(bars10, alpha_if_deleted_values):
                    ax10.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, f'{alpha_val:.3f}', ha='center', va='bottom', fontsize=7)
                st.pyplot(fig10)
                plt.close()
            
            # Row 6: Correlation Heatmap
            if n_items > 1:
                st.markdown("### 11. Inter-Item Correlation Heatmap")
                fig11, ax11 = plt.subplots(figsize=(max(10, n_items*0.6), max(8, n_items*0.5)))
                correlation_matrix = df_scores.corr()
                mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
                sns.heatmap(correlation_matrix, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r', center=0, 
                            square=True, linewidths=0.5, ax=ax11, 
                            annot_kws={'size': 9}, cbar_kws={'shrink': 0.8})
                ax11.set_title('Inter-Item Correlation Matrix (r > 0.30 indicates potential item redundancy)', fontsize=12)
                st.pyplot(fig11)
                plt.close()
            
            # ==================================================================
            # SECTION 5: DOWNLOAD RESULTS
            # ==================================================================
            st.markdown("---")
            st.markdown("## 📥 DOWNLOAD RESULTS")
            st.caption("Download the complete analysis report as an Excel file with multiple sheets.")
            
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                df_results.to_excel(writer, sheet_name='Item_Statistics', index=False)
                pd.DataFrame({
                    'KR-20': [kr20],
                    'SEM': [sem],
                    'Sum_pq': [sum_pq],
                    'Total_Variance': [total_variance],
                    'Number_of_Students': [n_students],
                    'Number_of_Items': [n_items],
                    'Group_Percentage': [group_percent],
                    'Difficulty_Threshold': [f"Difficult: <{difficult_threshold}, Easy: >{easy_threshold}"],
                    'Discrimination_Threshold': [f"Poor: <{poor_threshold}, Very Good: ≥{good_threshold}"],
                    'Validity_Threshold': [f"Valid: ≥{valid_threshold}"],
                    'Distractor_Percent_Threshold': [f"≥{distractor_percent_threshold}%"]
                }).to_excel(writer, sheet_name='Reliability_Summary', index=False)
                
                if distractor_results:
                    df_distractor.to_excel(writer, sheet_name='Distractor_Analysis', index=False)
                
                # Formulas sheet
                pd.DataFrame([
                    {'Component': 'p (Difficulty Index)', 'Formula': 'p = Σ correct / N', 'Meaning': 'Proportion of students answering correctly', 'Range': '0.00 - 1.00', 'Ideal': '0.30 - 0.80'},
                    {'Component': 'q', 'Formula': 'q = 1 - p', 'Meaning': 'Proportion of students answering incorrectly', 'Range': '0.00 - 1.00', 'Ideal': 'Complement of p'},
                    {'Component': 'pq', 'Formula': 'pq = p × q', 'Meaning': 'Item variance', 'Range': '0.00 - 0.25', 'Ideal': 'Maximum at p=0.50'},
                    {'Component': 'p_upper', 'Formula': 'p_upper = Σ correct (upper group) / n_group', 'Meaning': 'Proportion correct in upper group', 'Range': '0.00 - 1.00', 'Ideal': 'High (>0.70)'},
                    {'Component': 'p_lower', 'Formula': 'p_lower = Σ correct (lower group) / n_group', 'Meaning': 'Proportion correct in lower group', 'Range': '0.00 - 1.00', 'Ideal': 'Low (<0.30)'},
                    {'Component': 'D (Discrimination)', 'Formula': 'D = p_upper - p_lower', 'Meaning': 'Ability to distinguish high vs low ability students', 'Range': '-1.00 - 1.00', 'Ideal': '≥0.40'},
                    {'Component': 'SE', 'Formula': 'SE = √(pq/n)', 'Meaning': 'Standard error of item proportion', 'Range': '>0', 'Ideal': 'Small (<0.05)'},
                    {'Component': 'r_it (Validity)', 'Formula': 'Point-biserial correlation (corrected)', 'Meaning': 'Consistency of item with total test score', 'Range': '-1.00 - 1.00', 'Ideal': '≥0.20'},
                    {'Component': 'Alpha_if_deleted', 'Formula': 'KR-20 without item i', 'Meaning': 'Reliability if item is removed', 'Range': '0.00 - 1.00', 'Ideal': 'Less than overall KR-20'},
                    {'Component': 'KR-20', 'Formula': '(k/(k-1)) × (1 - Σpq/σ²)', 'Meaning': 'Internal consistency reliability', 'Range': '0.00 - 1.00', 'Ideal': '≥0.70'},
                    {'Component': 'SEM', 'Formula': 'SEM = SD × √(1 - KR-20)', 'Meaning': 'Standard Error of Measurement', 'Range': '>0', 'Ideal': 'Small relative to score range'},
                    {'Component': 'DDI', 'Formula': 'DDI = Prop_Lower - Prop_Upper', 'Meaning': 'Distractor Discrimination Index', 'Range': '-1.00 - 1.00', 'Ideal': '>0'},
                ]).to_excel(writer, sheet_name='Formulas_Guide', index=False)
                
                # Threshold parameters sheet
                pd.DataFrame([
                    {'Aspect': 'Difficulty (p)', 'Category': 'Difficult', 'Range': f'< {difficult_threshold}', 'Action': 'Revise wording, simplify language', 'Interpretation': 'Too few students answered correctly'},
                    {'Aspect': 'Difficulty (p)', 'Category': 'Moderate', 'Range': f'{difficult_threshold} - {easy_threshold}', 'Action': 'Retain', 'Interpretation': 'Optimal difficulty level'},
                    {'Aspect': 'Difficulty (p)', 'Category': 'Easy', 'Range': f'> {easy_threshold}', 'Action': 'Increase difficulty', 'Interpretation': 'Too many students answered correctly'},
                    {'Aspect': 'Discrimination (D)', 'Category': 'Poor', 'Range': f'< {poor_threshold}', 'Action': 'Drop or major revision', 'Interpretation': 'Cannot distinguish ability levels'},
                    {'Aspect': 'Discrimination (D)', 'Category': 'Fair', 'Range': f'{poor_threshold} - {good_threshold}', 'Action': 'Minor revision', 'Interpretation': 'Moderately distinguishes ability'},
                    {'Aspect': 'Discrimination (D)', 'Category': 'Very Good', 'Range': f'≥ {good_threshold}', 'Action': 'Retain', 'Interpretation': 'Excellent discrimination'},
                    {'Aspect': 'Validity (r_it)', 'Category': 'Invalid', 'Range': f'< {valid_threshold}', 'Action': 'Revise or drop', 'Interpretation': 'Item inconsistent with test'},
                    {'Aspect': 'Validity (r_it)', 'Category': 'Valid', 'Range': f'≥ {valid_threshold}', 'Action': 'Retain', 'Interpretation': 'Item consistent with test'},
                    {'Aspect': 'Distractor', 'Category': 'Functional', 'Range': '≥5% AND Lower_N > Upper_N', 'Action': 'Retain', 'Interpretation': 'Effective distractor'},
                    {'Aspect': 'Distractor', 'Category': 'Partially Functional', 'Range': 'Meets only one criterion', 'Action': 'Revise', 'Interpretation': 'Needs improvement'},
                    {'Aspect': 'Distractor', 'Category': 'Non-Functional', 'Range': 'Meets no criteria', 'Action': 'Replace', 'Interpretation': 'Ineffective distractor'},
                ]).to_excel(writer, sheet_name='Decision_Guide', index=False)
            
            output.seek(0)
            st.download_button(
                label="📥 Download Complete Excel Report",
                data=output,
                file_name="item_analysis_complete_report.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )
            
            st.success("✅ Analysis complete! The report includes item statistics, distractor analysis with explicit criteria flags, and comprehensive visualizations.")

else:
    with tab2:
        st.info("👈 Please upload your CSV file in the 'Upload Data' tab first.")
        st.markdown("""
        ### Expected CSV Format:
        
        | Student_ID | Item_1 | Item_2 | Item_3 | ... |
        |------------|--------|--------|--------|-----|
        | S1 | 1 | 0 | 1 | ... |
        | S2 | 0 | 1 | 1 | ... |
        | S3 | 1 | 1 | 0 | ... |
        
        **For multiple-choice:** Use letters (A, B, C, D, E) and upload answer key file.
        **For binary scores:** Use 1 (correct) and 0 (incorrect).
        """)
