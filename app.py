# ======================================================================
# ITEM ANALYSIS - STREAMLIT VERSION (COMPLETE + FULLY FIXED)
# ======================================================================
# CRITICAL FIXES:
# 1. Upper_N = 0 issue resolved by preserving original indices
# 2. Fixed Alpha if Deleted calculation (now uses proper KR-20 formula)
# 3. Fixed Standard Error formula (uses n-1 for unbiased estimate)
# 4. Enhanced D interpretation with negative discrimination handling
# 5. Improved recommendation logic with more granular categories
# 6. Added minimum student validation (N >= 30 warning)
# 7. Fixed answer key validation for NaN/None values
# 8. Removed debug info from production display
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
        difficult_threshold = st.number_input("Difficult (<)", value=0.30, step=0.05)
    with col2:
        easy_threshold = st.number_input("Easy (>)", value=0.80, step=0.05)
    
    st.subheader("Discrimination Index (D)")
    col1, col2 = st.columns(2)
    with col1:
        poor_threshold = st.number_input("Poor (<)", value=0.20, step=0.05)
    with col2:
        good_threshold = st.number_input("Good (>=)", value=0.40, step=0.05)
    
    st.subheader("Validity (r_it)")
    valid_threshold = st.number_input("Valid (>=)", value=0.20, step=0.05)
    
    st.subheader("Group Classification")
    group_percent = st.slider("Upper/Lower Group Percentage", min_value=10, max_value=50, value=27, step=1)
    
    st.markdown("---")
    st.caption("Scripted by Muhaimin Abdullah | Based on Classical Test Theory (CTT)")

# ======================================================================
# FUNCTIONS (IMPROVED)
# ======================================================================
def interpret_p(p):
    if p < difficult_threshold:
        return "Difficult"
    elif p <= easy_threshold:
        return "Moderate"
    else:
        return "Easy"

def interpret_d(d):
    """Enhanced interpretation with negative discrimination handling"""
    if d < 0:
        return "Very Poor (Negative)"
    elif d < poor_threshold:
        return "Poor"
    elif d < good_threshold:
        return "Fair"
    else:
        return "Very Good"

def interpret_ddi(ddi):
    if ddi > 0:
        return "Functional"
    elif ddi == 0:
        return "Neutral"
    else:
        return "Non-Functional"

def get_recommendation(p, d, r):
    """
    Enhanced recommendation logic with more granular categories
    Based on classical test theory principles
    """
    # Critical case: negative discrimination - must drop
    if d < 0:
        return "DROP (Negative Discrim)"
    
    # Ideal item: valid, good discrimination, moderate difficulty
    if r >= valid_threshold and d >= good_threshold and difficult_threshold <= p <= easy_threshold:
        return "RETAIN"
    
    # Very poor items that should be dropped
    if r < 0.0 or d < 0.10:
        return "DROP"
    
    # Items needing revision with specific reasons
    if r < valid_threshold:
        if r < 0.10:
            return "REVISE (Very Low Validity)"
        else:
            return "REVISE (Low Validity)"
    elif d < poor_threshold:
        if d < 0:
            return "REVISE (Negative Discrim)"
        else:
            return "REVISE (Low Discrimination)"
    elif p < difficult_threshold:
        return "REVISE (Too Difficult)"
    elif p > easy_threshold:
        return "REVISE (Too Easy)"
    else:
        return "REVISE (Minor Issues)"

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
                    st.error("❌ Empty data!")
                elif len(df.columns) < 2:
                    st.error("❌ Need at least 2 columns!")
                else:
                    st.success(f"✅ File uploaded: {student_file.name}")
                    st.write(f"Dimensions: {df.shape[0]} rows x {df.shape[1]} columns")
                    st.subheader("Data Preview (First 5 rows)")
                    st.dataframe(df.head())
                    
                    st.session_state.df = df
                    st.session_state.file_loaded = True
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
    
    st.subheader("Upload Answer Key File (Required for Multiple Choice)")
    key_file = st.file_uploader("Choose CSV file with answer keys", type=['csv'], key="answer_key")
    
    if key_file is not None:
        if key_file.size > 5 * 1024 * 1024:
            st.error("❌ File too large!")
        else:
            try:
                key_file.seek(0)
                df_key = pd.read_csv(key_file, dtype=str)
                if not df_key.empty:
                    # Validate answer key values (no NaN/None)
                    first_row = df_key.iloc[0].values
                    valid_keys = all(pd.notna(first_row))
                    if not valid_keys:
                        st.warning("⚠️ Answer key contains empty values. Please check your file.")
                    else:
                        st.success(f"✅ Answer key uploaded")
                        st.session_state.answer_key_df = df_key
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

# ======================================================================
# ANALYSIS PROCESS
# ======================================================================
if st.session_state.file_loaded and st.session_state.df is not None:
    
    df = st.session_state.df.copy()
    df_key = st.session_state.answer_key_df
    
    item_columns = df.columns[1:].tolist()
    n_items = len(item_columns)
    n_students = len(df)
    
    # Warning for small sample size
    if n_students < 30:
        with tab2:
            st.warning(f"⚠️ **Small Sample Warning:** Only {n_students} students. Item analysis is more reliable with N ≥ 30 students. Results should be interpreted cautiously.")
    
    if n_items == 0:
        with tab2:
            st.error("❌ No item columns found!")
    elif df_key is None or df_key.empty:
        with tab2:
            st.error("❌ ANSWER KEY IS REQUIRED! Please upload answer key file.")
    else:
        # Read answer key from row 0 with improved validation
        try:
            if df_key.shape[1] > 1:
                answer_key_raw = [str(x).strip().upper() if pd.notna(x) else None for x in df_key.iloc[0, 1:].values]
            else:
                answer_key_raw = [str(df_key.iloc[0, 0]).strip().upper() if pd.notna(df_key.iloc[0, 0]) else None]
            
            # Check for None values in answer key
            if any(k is None for k in answer_key_raw):
                with tab2:
                    st.error("❌ Answer key contains empty/missing values! Please ensure all answer keys are provided.")
                    answer_key = None
            else:
                answer_key = answer_key_raw
        except Exception as e:
            with tab2:
                st.error(f"❌ Failed to read answer key: {str(e)}")
                answer_key = None
        
        if answer_key is None:
            with tab2:
                st.error("❌ Could not read answer key!")
        elif len(answer_key) != n_items:
            with tab2:
                st.error(f"❌ Answer key length ({len(answer_key)}) does not match number of items ({n_items})")
        else:
            # Convert to binary scores (1/0)
            df_scores = pd.DataFrame()
            for i, item in enumerate(item_columns):
                key_value = answer_key[i]
                df_scores[item] = (df[item].astype(str).str.strip().str.upper() == key_value).astype(int)
            
            df['total_score'] = df_scores.sum(axis=1)
            
            # ==================================================================
            # CRITICAL BUG FIX: Preserve original indices for correct group formation
            # ==================================================================
            # Step 1: Add original index as a column
            df_with_idx = df.reset_index().rename(columns={'index': 'original_idx'})
            
            # Step 2: Sort by total_score
            df_sorted = df_with_idx.sort_values('total_score', ascending=False).reset_index(drop=True)
            
            # Step 3: Determine group size
            n_group = max(1, int(np.ceil(n_students * group_percent / 100)))
            
            # Step 4: Get original indices of upper and lower groups
            upper_indices = df_sorted.head(n_group)['original_idx'].tolist()
            lower_indices = df_sorted.tail(n_group)['original_idx'].tolist()
            
            # Step 5: Extract groups from original DataFrame using .loc[]
            upper_group = df.loc[upper_indices].copy()
            lower_group = df.loc[lower_indices].copy()
            
            # Also create score-based groups for validation
            upper_scores = df_scores.loc[upper_indices].copy()
            lower_scores = df_scores.loc[lower_indices].copy()
            
            # ==================================================================
            # ITEM STATISTICS (WITH IMPROVED FORMULAS)
            # ==================================================================
            results = []
            p_values = []
            q_values = []
            pq_values = []
            p_upper_values = []
            p_lower_values = []
            d_values = []
            se_values = []
            r_values = []
            alpha_if_deleted_values = []
            
            for i, item in enumerate(item_columns):
                p_val = df_scores[item].mean()
                p_values.append(p_val)
                
                q_val = 1 - p_val
                q_values.append(q_val)
                
                pq_val = p_val * q_val
                pq_values.append(pq_val)
                
                key_value = answer_key[i]
                
                # Calculate using correctly formed groups
                upper_correct = (upper_group[item].astype(str).str.strip().str.upper() == key_value).sum()
                p_upper = upper_correct / n_group if n_group > 0 else 0
                
                lower_correct = (lower_group[item].astype(str).str.strip().str.upper() == key_value).sum()
                p_lower = lower_correct / n_group if n_group > 0 else 0
                
                p_upper_values.append(p_upper)
                p_lower_values.append(p_lower)
                
                d_val = p_upper - p_lower
                d_values.append(d_val)
                
                # FIXED: Standard Error with n-1 for unbiased estimate
                se_val = np.sqrt(pq_val / (n_students - 1)) if n_students > 1 else 0
                se_values.append(se_val)
                
                total_minus_item = df['total_score'] - df_scores[item]
                if df_scores[item].var() == 0 or total_minus_item.var() == 0:
                    r_it = 0.0
                else:
                    r_it, _ = pointbiserialr(df_scores[item], total_minus_item)
                r_values.append(r_it)
                
                # FIXED: Alpha if item deleted using proper KR-20 formula
                # Calculate KR-20 without item i
                sum_pq_without = 0
                for j, item2 in enumerate(item_columns):
                    if j != i:
                        p2 = df_scores[item2].mean()
                        sum_pq_without += p2 * (1 - p2)
                
                total_without = df['total_score'] - df_scores[item]
                var_total_without = total_without.var(ddof=1)
                
                if var_total_without > 0 and (n_items - 1) >= 2:
                    alpha = ((n_items - 1) / (n_items - 2)) * (1 - (sum_pq_without / var_total_without))
                elif var_total_without > 0 and (n_items - 1) == 1:
                    alpha = 1.0  # Only 1 item left, perfect correlation
                else:
                    alpha = np.nan
                alpha_if_deleted_values.append(alpha if not np.isnan(alpha) else 0)
                
                # Interpretations & Recommendation
                p_int = interpret_p(p_val)
                d_int = interpret_d(d_val)
                v_int = "Valid" if r_it >= valid_threshold else "Invalid"
                rec = get_recommendation(p_val, d_val, r_it)
                
                results.append([
                    item, 
                    round(p_val, 4), 
                    round(q_val, 4), 
                    round(pq_val, 4),
                    round(p_upper, 4), 
                    round(p_lower, 4), 
                    round(d_val, 4), 
                    d_int,
                    round(se_val, 6),
                    round(r_it, 4), 
                    v_int, 
                    round(alpha_if_deleted_values[-1], 4) if not np.isnan(alpha_if_deleted_values[-1]) else 0, 
                    rec, 
                    p_int
                ])
            
            # KR-20
            total_variance = df['total_score'].var(ddof=1)
            sum_pq = sum(pq_values)
            if total_variance > 0 and n_items > 1:
                kr20 = (n_items/(n_items-1)) * (1 - (sum_pq / total_variance))
            else:
                kr20 = 0
            sem = df['total_score'].std(ddof=1) * np.sqrt(max(0, 1 - kr20))
            
            # ==================================================================
            # DISTRACTOR ANALYSIS (ENHANCED)
            # ==================================================================
            distractor_results = []
            
            # Auto-detect options from data (including multi-character options)
            all_options = set()
            for item in item_columns:
                values = df[item].astype(str).str.strip().str.upper().dropna()
                for v in values:
                    if v != '' and v != 'NAN':
                        all_options.add(v)
            option_list = sorted(all_options)
            
            for i, item in enumerate(item_columns):
                key_value = answer_key[i]
                item_data = df[item].astype(str).str.strip().str.upper()
                
                for option in option_list:
                    if option == key_value:
                        continue
                    
                    total_select = (item_data == option).sum()
                    percent = (total_select / n_students) * 100 if n_students > 0 else 0
                    
                    # Use correctly formed groups
                    upper_select = (upper_group[item].astype(str).str.strip().str.upper() == option).sum()
                    lower_select = (lower_group[item].astype(str).str.strip().str.upper() == option).sum()
                    
                    prop_upper = upper_select / n_group if n_group > 0 else 0
                    prop_lower = lower_select / n_group if n_group > 0 else 0
                    
                    ddi = prop_lower - prop_upper
                    ddi_int = interpret_ddi(ddi)
                    
                    meets_percent = percent >= 5.0
                    meets_lower_upper = lower_select > upper_select
                    
                    distractor_results.append([
                        item, key_value, option,
                        total_select, round(percent, 1),
                        upper_select, lower_select,
                        round(prop_upper, 4), round(prop_lower, 4),
                        round(ddi, 4), ddi_int,
                        "Yes" if meets_percent else "No",
                        "Yes" if meets_lower_upper else "No"
                    ])
            
            df_results = pd.DataFrame(results, columns=[
                'Item', 'p', 'q', 'pq', 'p_upper', 'p_lower', 'D', 'D_Interpretation',
                'SE', 'r_it', 'Validity', 'Alpha_if_deleted', 'Recommendation', 'p_Interpretation'
            ])
            
            # ==================================================================
            # DISPLAY RESULTS IN TAB2
            # ==================================================================
            with tab2:
                # --------------------------------------------------------------
                # DATA SUMMARY
                # --------------------------------------------------------------
                st.markdown("## 📋 DATA SUMMARY")
                
                mean_score = df['total_score'].mean()
                median_score = df['total_score'].median()
                min_score = df['total_score'].min()
                max_score = df['total_score'].max()
                std_score = df['total_score'].std(ddof=1)
                
                upper_group_scores = df_sorted['total_score'].iloc[:n_group].tolist()
                lower_group_scores = df_sorted['total_score'].iloc[-n_group:].tolist()
                
                summary_data = {
                    'Parameter': [
                        'Number of Students (N)', 
                        'Number of Items (k)', 
                        'Group Percentage', 
                        'Group Size (n_group)',
                        'Mean Total Score (M)', 
                        'Median Total Score', 
                        'Standard Deviation (SD)',
                        'Minimum Score', 
                        'Maximum Score',
                        'Upper Group Score Range',
                        'Lower Group Score Range'
                    ],
                    'Value': [
                        n_students, 
                        n_items, 
                        f"{group_percent}% (Kelley, 1939)", 
                        n_group,
                        f"{mean_score:.2f}", 
                        f"{median_score:.2f}", 
                        f"{std_score:.2f}",
                        min_score, 
                        max_score,
                        f"{min(upper_group_scores)} - {max(upper_group_scores)}",
                        f"{min(lower_group_scores)} - {max(lower_group_scores)}"
                    ]
                }
                
                df_summary = pd.DataFrame(summary_data)
                st.dataframe(df_summary, use_container_width=True)
                
                # --------------------------------------------------------------
                # RELIABILITY SUMMARY WITH INTERPRETATION
                # --------------------------------------------------------------
                st.markdown("## 📊 RELIABILITY SUMMARY")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("KR-20", f"{kr20:.4f}")
                    if kr20 >= 0.80:
                        st.success("✅ Very Good (suitable for high-stakes testing)")
                        st.caption("**Interpretation:** The test has excellent internal consistency. Results are highly reliable.")
                    elif kr20 >= 0.70:
                        st.info("📘 Good (suitable for classroom exams)")
                        st.caption("**Interpretation:** The test has acceptable internal consistency for most educational purposes.")
                    elif kr20 >= 0.60:
                        st.warning("⚠️ Fair (acceptable for exploratory research)")
                        st.caption("**Interpretation:** The test has marginal reliability. Consider improving weak items.")
                    else:
                        st.error("❌ Poor (needs significant improvement)")
                        st.caption("**Interpretation:** The test lacks internal consistency. Major revision recommended.")
                
                with col2:
                    st.metric("SEM", f"{sem:.4f}")
                    st.caption(f"95% CI: ±{sem*1.96:.2f} points")
                    st.caption("**Interpretation:** Individual scores may vary within this range due to measurement error.")
                
                with col3:
                    st.metric("Σpq", f"{sum_pq:.4f}")
                    st.caption(f"Total Variance: {total_variance:.4f}")
                    st.caption("**Interpretation:** Σpq is the sum of item variances used in KR-20 calculation.")
                
                # --------------------------------------------------------------
                # COMPLETE ITEM STATISTICS TABLE WITH COLOR CODING
                # --------------------------------------------------------------
                st.markdown("---")
                st.markdown("## 📊 COMPLETE ITEM STATISTICS")
                st.caption("This table shows all psychometric properties for each test item.")
                st.caption("**Color coding:** p (Green=Moderate, Red=Difficult, Orange=Easy) | D (Green=Very Good, Orange=Fair, Red=Poor/Pink=Negative) | r_it (Green=Valid, Red=Invalid)")
                
                def color_p(val):
                    if val < difficult_threshold:
                        return 'background-color: #ffcccc'
                    elif val <= easy_threshold:
                        return 'background-color: #ccffcc'
                    else:
                        return 'background-color: #ffe6cc'
                
                def color_d(val):
                    if val < 0:
                        return 'background-color: #ff99cc'  # Pink for negative
                    elif val < poor_threshold:
                        return 'background-color: #ffcccc'
                    elif val < good_threshold:
                        return 'background-color: #ffe6cc'
                    else:
                        return 'background-color: #ccffcc'
                
                def color_r(val):
                    if val >= valid_threshold:
                        return 'background-color: #ccffcc'
                    else:
                        return 'background-color: #ffcccc'
                
                styled_df = df_results.style
                styled_df = styled_df.map(color_p, subset=['p'])
                styled_df = styled_df.map(color_d, subset=['D'])
                styled_df = styled_df.map(color_r, subset=['r_it'])
                
                st.dataframe(styled_df, use_container_width=True)
                
                # --------------------------------------------------------------
                # DISTRACTOR ANALYSIS
                # --------------------------------------------------------------
                if distractor_results:
                    st.markdown("---")
                    st.markdown("## 🎯 DISTRACTOR ANALYSIS")
                    st.caption("**DDI (Distractor Discrimination Index)** = Proportion Lower - Proportion Upper | DDI > 0 indicates a functional distractor")
                    st.caption("**Criteria for functional distractor:** (1) Selected by >=5% of students, (2) More low-ability than high-ability students choose them")
                    
                    df_distractor = pd.DataFrame(distractor_results, columns=[
                        'Item', 'Key', 'Option', 
                        'N_Select', 'Percent', 
                        'Upper_N', 'Lower_N',
                        'Prop_Upper', 'Prop_Lower',
                        'DDI', 'DDI_Interpretation',
                        '>=5%', 'Lower > Upper'
                    ])
                    
                    st.dataframe(df_distractor, use_container_width=True)
                    
                    st.markdown("### 📊 DDI Summary by Item")
                    ddi_summary = []
                    for item in item_columns:
                        item_distractors = df_distractor[df_distractor['Item'] == item]['DDI'].values
                        if len(item_distractors) > 0:
                            mean_ddi = np.mean(item_distractors)
                            functional_count = sum(1 for d in item_distractors if d > 0)
                            ddi_summary.append([item, len(item_distractors), round(mean_ddi, 4), functional_count])
                    
                    df_ddi_summary = pd.DataFrame(ddi_summary, columns=['Item', 'Num_Distractors', 'Mean_DDI', 'Functional_Count'])
                    st.dataframe(df_ddi_summary, use_container_width=True)
                
                # --------------------------------------------------------------
                # VISUALIZATIONS (ALL 11 CHARTS)
                # --------------------------------------------------------------
                st.markdown("---")
                st.markdown("## 📊 VISUALIZATIONS")
                st.caption("The following charts provide visual interpretation of item statistics.")
                
                # Row 1: Difficulty and Discrimination
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("### 1. Item Difficulty (p)")
                    fig1, ax1 = plt.subplots(figsize=(8, 5))
                    colors_p = ['red' if x < difficult_threshold else ('green' if x <= easy_threshold else 'orange') for x in p_values]
                    ax1.bar(range(1, n_items+1), p_values, color=colors_p)
                    ax1.axhline(difficult_threshold, color='red', linestyle='--', label=f'Difficult (<{difficult_threshold})')
                    ax1.axhline(easy_threshold, color='orange', linestyle='--', label=f'Easy (>{easy_threshold})')
                    ax1.set_xlabel('Item Number')
                    ax1.set_ylabel('Difficulty Index (p)')
                    ax1.set_title('Item Difficulty: Red=Difficult, Green=Moderate, Orange=Easy')
                    ax1.set_xticks(range(1, n_items+1))
                    ax1.set_ylim(0, 1)
                    ax1.legend(loc='lower right')
                    ax1.grid(axis='y', alpha=0.3)
                    st.pyplot(fig1)
                    plt.close()
                
                with col2:
                    st.markdown("### 2. Item Discrimination (D)")
                    fig2, ax2 = plt.subplots(figsize=(8, 5))
                    colors_d = []
                    for x in d_values:
                        if x < 0:
                            colors_d.append('#ff99cc')  # Pink for negative
                        elif x >= good_threshold:
                            colors_d.append('#2ecc71')
                        elif x >= poor_threshold:
                            colors_d.append('#f39c12')
                        else:
                            colors_d.append('#e74c3c')
                    ax2.bar(range(1, n_items+1), d_values, color=colors_d)
                    ax2.axhline(good_threshold, color='green', linestyle='--', label=f'Very Good (≥{good_threshold})')
                    ax2.axhline(poor_threshold, color='orange', linestyle='--', label=f'Fair (≥{poor_threshold})')
                    ax2.axhline(0, color='red', linestyle=':', linewidth=1, label='Zero Line')
                    ax2.set_xlabel('Item Number')
                    ax2.set_ylabel('Discrimination Index (D)')
                    ax2.set_title('Item Discrimination: Green=Very Good, Orange=Fair, Red=Poor, Pink=Negative')
                    ax2.set_xticks(range(1, n_items+1))
                    ax2.set_ylim(-1, 1)
                    ax2.legend(loc='lower right')
                    ax2.grid(axis='y', alpha=0.3)
                    st.pyplot(fig2)
                    plt.close()
                
                # Row 2: Validity and q
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("### 3. Item Validity (r_it)")
                    fig3, ax3 = plt.subplots(figsize=(8, 5))
                    colors_r = ['green' if x >= valid_threshold else 'red' for x in r_values]
                    ax3.bar(range(1, n_items+1), r_values, color=colors_r)
                    ax3.axhline(valid_threshold, color='green', linestyle='--', label=f'Valid (≥{valid_threshold})')
                    ax3.axhline(0, color='black', linestyle='-', linewidth=0.5)
                    ax3.set_xlabel('Item Number')
                    ax3.set_ylabel('Corrected Item-Total Correlation (r_it)')
                    ax3.set_title('Item Validity: Green=Valid, Red=Invalid')
                    ax3.set_xticks(range(1, n_items+1))
                    ax3.set_ylim(-1, 1)
                    ax3.legend(loc='lower right')
                    ax3.grid(axis='y', alpha=0.3)
                    st.pyplot(fig3)
                    plt.close()
                
                with col2:
                    st.markdown("### 4. Proportion Incorrect (q = 1-p)")
                    fig4, ax4 = plt.subplots(figsize=(8, 5))
                    ax4.bar(range(1, n_items+1), q_values, color='navy', alpha=0.7)
                    ax4.set_xlabel('Item Number')
                    ax4.set_ylabel('q = 1 - p')
                    ax4.set_title('Proportion of Students Answering Incorrectly')
                    ax4.set_xticks(range(1, n_items+1))
                    ax4.set_ylim(0, 1)
                    ax4.grid(axis='y', alpha=0.3)
                    st.pyplot(fig4)
                    plt.close()
                
                # Row 3: Upper vs Lower and Recommendations
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("### 5. Upper vs Lower Group Performance")
                    fig5, ax5 = plt.subplots(figsize=(8, 5))
                    x = range(1, n_items+1)
                    ax5.plot(x, p_upper_values, 'o-', color='green', label=f'p_upper (Upper {group_percent}%)', linewidth=2, markersize=8)
                    ax5.plot(x, p_lower_values, 's-', color='red', label=f'p_lower (Lower {group_percent}%)', linewidth=2, markersize=8)
                    ax5.fill_between(x, p_lower_values, p_upper_values, alpha=0.2, color='gray')
                    ax5.set_xlabel('Item Number')
                    ax5.set_ylabel('Proportion Correct')
                    ax5.set_title('Comparison of Upper and Lower Group Performance')
                    ax5.set_xticks(range(1, n_items+1))
                    ax5.set_ylim(0, 1)
                    ax5.legend(loc='lower right')
                    ax5.grid(axis='both', alpha=0.3)
                    st.pyplot(fig5)
                    plt.close()
                
                with col2:
                    st.markdown("### 6. Item Recommendations")
                    fig6, ax6 = plt.subplots(figsize=(8, 5))
                    
                    def get_rec_color(rec):
                        if 'RETAIN' in rec:
                            return '#2ecc71'
                        elif 'DROP' in rec:
                            return '#e74c3c'
                        else:
                            return '#f39c12'
                    
                    colors_rec = [get_rec_color(r) for r in df_results['Recommendation']]
                    ax6.bar(range(1, n_items+1), [1]*n_items, color=colors_rec)
                    ax6.set_xlabel('Item Number')
                    ax6.set_title('Item Recommendations: Green=RETAIN, Orange=REVISE, Red=DROP')
                    ax6.set_xticks(range(1, n_items+1))
                    ax6.set_yticks([])
                    
                    for bar, rec in zip(ax6.patches, df_results['Recommendation']):
                        rec_text = rec[:12] + "..." if len(rec) > 12 else rec
                        ax6.text(bar.get_x() + bar.get_width()/2, 1.05, rec_text, ha='center', va='bottom', fontsize=7, rotation=45)
                    st.pyplot(fig6)
                    plt.close()
                
                # Row 4: Score Distribution and Pie Chart
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("### 7. Total Score Distribution")
                    fig7, ax7 = plt.subplots(figsize=(8, 5))
                    min_score_val = int(df['total_score'].min())
                    max_score_val = int(df['total_score'].max())
                    bins = range(min_score_val, max_score_val+2)
                    ax7.hist(df['total_score'], bins=bins, edgecolor='black', alpha=0.7, color='skyblue')
                    ax7.axvline(df['total_score'].mean(), color='red', linestyle='--', linewidth=2, label=f"Mean = {df['total_score'].mean():.2f}")
                    ax7.axvline(df['total_score'].median(), color='green', linestyle='--', linewidth=2, label=f"Median = {df['total_score'].median():.2f}")
                    ax7.set_xlabel('Total Score')
                    ax7.set_ylabel('Frequency')
                    ax7.set_title('Distribution of Student Total Scores')
                    ax7.legend(loc='upper right')
                    ax7.grid(axis='y', alpha=0.3)
                    st.pyplot(fig7)
                    plt.close()
                
                with col2:
                    st.markdown("### 8. Recommendation Summary")
                    fig8, ax8 = plt.subplots(figsize=(7, 5))
                    
                    # Count based on main categories (RETAIN, REVISE, DROP)
                    retain_count = sum(1 for r in df_results['Recommendation'] if 'RETAIN' in r)
                    revise_count = sum(1 for r in df_results['Recommendation'] if 'REVISE' in r)
                    drop_count = sum(1 for r in df_results['Recommendation'] if 'DROP' in r)
                    
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
                            [f'{label} ({size} items)' for label, size in zip(labels, sizes)],
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
                    
                    st.pyplot(fig8)
                    plt.close()
                
                # Row 5: SE and Alpha if Deleted
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("### 9. Standard Error by Item")
                    fig9, ax9 = plt.subplots(figsize=(8, 5))
                    ax9.bar(range(1, n_items+1), se_values, color='#9b59b6', alpha=0.7, edgecolor='black')
                    ax9.set_xlabel('Item Number')
                    ax9.set_ylabel('Standard Error (SE)')
                    ax9.set_title('Standard Error of Item Proportion (with n-1 correction)')
                    ax9.set_xticks(range(1, n_items+1))
                    ax9.grid(axis='y', alpha=0.3)
                    
                    for bar, se_val in zip(ax9.patches, se_values):
                        ax9.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, f'{se_val:.4f}', ha='center', va='bottom', fontsize=7)
                    st.pyplot(fig9)
                    plt.close()
                
                with col2:
                    st.markdown("### 10. Alpha if Item Deleted")
                    fig10, ax10 = plt.subplots(figsize=(8, 5))
                    
                    # Filter out NaN values for coloring
                    alpha_clean = [x if not np.isnan(x) else 0 for x in alpha_if_deleted_values]
                    colors_alpha = ['#2ecc71' if x <= kr20 else '#e74c3c' for x in alpha_clean]
                    ax10.bar(range(1, n_items+1), alpha_clean, color=colors_alpha, edgecolor='black')
                    ax10.axhline(kr20, color='blue', linestyle='--', linewidth=2, label=f'Overall KR-20 = {kr20:.4f}')
                    ax10.set_xlabel('Item Number')
                    ax10.set_ylabel('Alpha if Item Deleted')
                    ax10.set_title('Impact on Reliability if Item Removed')
                    ax10.set_xticks(range(1, n_items+1))
                    ax10.legend()
                    ax10.grid(axis='y', alpha=0.3)
                    
                    for bar, alpha_val in zip(ax10.patches, alpha_clean):
                        ax10.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, f'{alpha_val:.3f}', ha='center', va='bottom', fontsize=7)
                    st.pyplot(fig10)
                    plt.close()
                
                # Row 6: Heatmap
                if n_items > 1:
                    st.markdown("### 11. Inter-Item Correlation Heatmap")
                    fig11, ax11 = plt.subplots(figsize=(max(8, n_items*0.5), max(6, n_items*0.4)))
                    correlation_matrix = df_scores.corr()
                    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
                    sns.heatmap(correlation_matrix, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r', center=0, 
                                square=True, linewidths=0.5, ax=ax11, 
                                annot_kws={'size': 8}, cbar_kws={'shrink': 0.8})
                    ax11.set_title('Inter-Item Correlation Matrix (r > 0.30 indicates potential redundancy)', fontsize=12)
                    st.pyplot(fig11)
                    plt.close()
                
                # --------------------------------------------------------------
                # DOWNLOAD RESULTS (ENHANCED)
                # --------------------------------------------------------------
                st.markdown("---")
                st.markdown("## 📥 DOWNLOAD RESULTS")
                
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    df_results.to_excel(writer, sheet_name='Item_Statistics', index=False)
                    pd.DataFrame({
                        'KR-20': [kr20],
                        'SEM': [sem],
                        'Sum_pq': [sum_pq],
                        'Total_Variance': [total_variance],
                        'Number_of_Students': [n_students],
                        'Number_of_Items': [n_items]
                    }).to_excel(writer, sheet_name='Reliability', index=False)
                    
                    if distractor_results:
                        df_distractor.to_excel(writer, sheet_name='Distractor_Analysis_DDI', index=False)
                        df_ddi_summary.to_excel(writer, sheet_name='DDI_Summary', index=False)
                    
                    pd.DataFrame([
                        {'Component': 'p (Difficulty Index)', 'Formula': 'p = Σ correct / N', 'Meaning': 'Proportion of students answering correctly'},
                        {'Component': 'q', 'Formula': 'q = 1 - p', 'Meaning': 'Proportion of students answering incorrectly'},
                        {'Component': 'pq', 'Formula': 'pq = p × q', 'Meaning': 'Item variance'},
                        {'Component': 'p_upper', 'Formula': 'p_upper = Σ correct (upper group) / n_group', 'Meaning': 'Proportion correct in upper group'},
                        {'Component': 'p_lower', 'Formula': 'p_lower = Σ correct (lower group) / n_group', 'Meaning': 'Proportion correct in lower group'},
                        {'Component': 'D (Discrimination)', 'Formula': 'D = p_upper - p_lower', 'Meaning': 'Ability to distinguish high vs low ability students'},
                        {'Component': 'SE', 'Formula': 'SE = √(pq/(n-1))', 'Meaning': 'Standard error of item proportion (unbiased estimate)'},
                        {'Component': 'r_it (Validity)', 'Formula': 'Point-biserial correlation (corrected)', 'Meaning': 'Consistency of item with total test score'},
                        {'Component': 'Alpha_if_deleted', 'Formula': 'KR-20 without item i', 'Meaning': 'Reliability if item is removed'},
                        {'Component': 'KR-20', 'Formula': '(k/(k-1)) × (1 - Σpq/σ²)', 'Meaning': 'Internal consistency reliability'},
                        {'Component': 'SEM', 'Formula': 'SEM = SD × √(1 - KR-20)', 'Meaning': 'Standard Error of Measurement'},
                        {'Component': 'DDI', 'Formula': 'DDI = Prop_Lower - Prop_Upper', 'Meaning': 'Distractor Discrimination Index'},
                    ]).to_excel(writer, sheet_name='Formulas', index=False)
                    
                    pd.DataFrame([
                        {'Aspect': 'Difficulty (p)', 'Category': 'Difficult', 'Range': f'< {difficult_threshold}', 'Action': 'Revise wording, simplify language'},
                        {'Aspect': 'Difficulty (p)', 'Category': 'Moderate', 'Range': f'{difficult_threshold} - {easy_threshold}', 'Action': 'Retain'},
                        {'Aspect': 'Difficulty (p)', 'Category': 'Easy', 'Range': f'> {easy_threshold}', 'Action': 'Increase difficulty'},
                        {'Aspect': 'Discrimination (D)', 'Category': 'Very Poor (Negative)', 'Range': '< 0', 'Action': 'Immediate DROP - Item confusing'},
                        {'Aspect': 'Discrimination (D)', 'Category': 'Poor', 'Range': f'0 - {poor_threshold}', 'Action': 'Drop or major revision'},
                        {'Aspect': 'Discrimination (D)', 'Category': 'Fair', 'Range': f'{poor_threshold} - {good_threshold}', 'Action': 'Minor revision'},
                        {'Aspect': 'Discrimination (D)', 'Category': 'Very Good', 'Range': f'≥ {good_threshold}', 'Action': 'Retain'},
                        {'Aspect': 'Validity (r_it)', 'Category': 'Invalid', 'Range': f'< {valid_threshold}', 'Action': 'Revise or drop'},
                        {'Aspect': 'Validity (r_it)', 'Category': 'Valid', 'Range': f'≥ {valid_threshold}', 'Action': 'Retain'},
                        {'Aspect': 'DDI', 'Category': 'Functional', 'Range': '> 0', 'Action': 'Retain distractor'},
                        {'Aspect': 'DDI', 'Category': 'Neutral', 'Range': '= 0', 'Action': 'Evaluate, consider revision'},
                        {'Aspect': 'DDI', 'Category': 'Non-Functional', 'Range': '< 0', 'Action': 'Replace distractor'},
                    ]).to_excel(writer, sheet_name='Threshold_Parameters', index=False)
                    
                    # Add summary statistics sheet
                    pd.DataFrame({
                        'Statistic': ['Mean', 'Median', 'Min', 'Max', 'Std Dev', 'Skewness', 'Kurtosis'],
                        'Total_Score': [
                            df['total_score'].mean(),
                            df['total_score'].median(),
                            df['total_score'].min(),
                            df['total_score'].max(),
                            df['total_score'].std(),
                            df['total_score'].skew(),
                            df['total_score'].kurtosis()
                        ]
                    }).to_excel(writer, sheet_name='Score_Summary', index=False)
                
                output.seek(0)
                st.download_button(
                    label="📥 Download Excel Report (Complete)",
                    data=output,
                    file_name="item_analysis_report.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
                
                st.success("✅ Analysis complete! The report includes all item statistics, distractor analysis with DDI, and visualizations.")
                
                # Display sample size warning if applicable
                if n_students < 30:
                    st.info("ℹ️ **Note:** Analysis completed with small sample size. Results should be validated with larger sample when possible.")

else:
    with tab2:
        st.info("👈 Please upload CSV and answer key files in the 'Upload Data' tab")
