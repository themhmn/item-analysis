import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import pointbiserialr
import io
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ======================================================================
# ACADEMIC ITEM ANALYSIS - RIGOROUS ENGLISH VERSION (2026)
# ======================================================================

st.set_page_config(page_title="Item Analysis Pro", page_icon="📈", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #f4f4f9; }
    [data-testid="stMetricValue"] { font-family: 'Courier New', Courier, monospace; color: #1a1a1a; }
    .stAlert { border: 2px solid #000; border-radius: 0px; }
    </style>
    """, unsafe_allow_html=True)

st.title("ITEM ANALYSIS TOOL (CTT)")
st.write("Full Classical Test Theory Suite: Methodologically validated metrics for educational research.")

with st.sidebar:
    st.header("📊 Methodological Legend")
    
    with st.expander("1. Difficulty Index (p)", expanded=True):
        st.write("""
        - **Easy (p > 0.70):** 🟢
        - **Moderate (0.30 - 0.70):** 🟡
        - **Difficult (p < 0.30):** 🔴
        """)

    with st.expander("2. Discrimination (d/DDI)", expanded=True):
        st.write("""
        - **d (Discrimination Index):** Focus on the Answer Key.
        - **DDI (Distractor Discrimination):** Focus on the worst distractor.
        - **Excellent (≥ 0.40):** 🟢
        - **Good (0.30 - 0.39):** 🔵
        - **Fair (0.20 - 0.29):** 🟡
        - **Poor (< 0.20):** 🔴
        """)

    st.header("⚙️ Settings")
    group_percent = st.slider("Kelley's Grouping (%)", 10, 50, 27)
    validity_limit = st.number_input("r_pbis Threshold", 0.0, 1.0, 0.25)

u1, u2 = st.columns(2)
with u1:
    student_file = st.file_uploader("Upload Student Data (CSV)", type=['csv'])
with u2:
    key_file = st.file_uploader("Upload Key Data (CSV)", type=['csv'])

if student_file and key_file:
    df = pd.read_csv(student_file).fillna("N/A")
    df_key = pd.read_csv(key_file)
    item_cols = df.columns[1:] 
    id_col_name = df.columns[0]
    answer_key = df_key.iloc[0, 1:].astype(str).str.upper().str.strip().tolist()
    
    df_scores = pd.DataFrame()
    for i, col in enumerate(item_cols):
        df_scores[col] = (df[col].astype(str).str.upper().str.strip() == answer_key[i]).astype(int)
    
    total_scores = df_scores.sum(axis=1)
    df['Total_Score'] = total_scores
    n_students, n_items = len(df), len(item_cols)

    # GATHER GROUP DATA
    n_group = max(1, int(np.ceil(n_students * group_percent / 100)))
    df_sorted = df.sort_values('Total_Score', ascending=False).copy()
    
    # Create Rank and Group Marker
    df_sorted['Rank'] = range(1, n_students + 1)
    df_sorted['Group'] = 'Middle'
    df_sorted.iloc[:n_group, df_sorted.columns.get_loc('Group')] = 'Upper'
    df_sorted.iloc[-n_group:, df_sorted.columns.get_loc('Group')] = 'Lower'
    
    # Ranking Table Summary
    df_ranking = df_sorted[[id_col_name, 'Total_Score', 'Rank', 'Group']].copy()

    up_idx, lo_idx = df_sorted.head(n_group).index, df_sorted.tail(n_group).index

    results = []
    for i, item in enumerate(item_cols):
        p = df_scores[item].mean()
        q = 1 - p
        pq = p * q
        item_var = df_scores[item].var(ddof=0)

        p_up = df_scores.loc[up_idx, item].mean()
        p_lo = df_scores.loc[lo_idx, item].mean()
        d_val = p_up - p_lo

        distractors = []
        for opt in df[item].dropna().astype(str).str.upper().unique():
            opt_clean = opt.strip()
            if opt_clean not in ["", "N/A", answer_key[i]]:
                distractors.append(opt_clean)
                
        ddi_vals = []
        for opt in distractors:
            u_opt = (df.loc[up_idx, item].astype(str).str.upper().str.strip() == opt).mean()
            l_opt = (df.loc[lo_idx, item].astype(str).str.upper().str.strip() == opt).mean()
            ddi_vals.append(l_opt - u_opt)
        
        ddi_final = max(ddi_vals) if ddi_vals else 0
        worst_ddi = min(ddi_vals) if ddi_vals else 0

        corrected_total = total_scores - df_scores[item]
        if df_scores[item].var() != 0:
            r_pb, _ = pointbiserialr(df_scores[item], corrected_total)
            if np.isnan(r_pb):
                r_pb = 0
        else:
            r_pb = 0

        p_desc = "Easy" if p > 0.7 else "Difficult" if p < 0.3 else "Moderate"
        d_desc = "Excellent" if d_val >= 0.4 else "Good" if d_val >= 0.3 else "Fair" if d_val >= 0.2 else "Poor"
        r_desc = "Valid" if r_pb >= validity_limit else "Invalid"
        
        # --- COMBINED FLAGGING SYSTEM ---
        reasons = []
        
        # cek validitas
        if r_pb < validity_limit:
            reasons.append("Low validity")
        
        # cek daya beda
        if d_val < 0.20:
            reasons.append("Poor discrimination")
        # cek difficulty ekstrem
        if p > 0.90:
            reasons.append("Too easy")
        
        if p < 0.20:
            reasons.append("Too difficult")
        
        # cek distractor
        if worst_ddi < -0.10:
            reasons.append("Severely flawed distractor")
        elif worst_ddi < 0:
            reasons.append("Malfunctioning distractor")
        
        # keputusan akhir
        if (r_pb >= validity_limit) and (d_val >= 0.30) and (worst_ddi >= 0):
            decision = "RETAIN"
        
        elif (r_pb < validity_limit and d_val < 0.20) or (worst_ddi < -0.10):
            decision = "REJECT"
        
        else:
            decision = "REVISE"
        
        reason_text = ", ".join(reasons) if reasons else "All criteria satisfied"

        results.append({
            "Item": item, "p": p, "p_Eval": p_desc, "q": q, "pq": pq, "Var": item_var,
            "d": d_val, "d_Eval": d_desc, "Best_DDI": ddi_final, "Worst_DDI": worst_ddi,
            "r_pbis": r_pb, "r_Eval": r_desc,
            "DECISION": decision,
            "REASON": reason_text
        })

    df_res = pd.DataFrame(results)

    # METRICS
    mean_score = total_scores.mean()
    var_total = total_scores.var(ddof=0)
    std_score = np.sqrt(var_total)
    kr20 = (n_items/(n_items-1)) * (1 - (df_res["pq"].sum()/var_total)) if var_total > 0 else 0
    alpha = (n_items/(n_items-1)) * (1 - (df_res["Var"].sum()/var_total)) if var_total > 0 else 0
    sem = std_score * np.sqrt(1 - kr20)

    st.divider()
    m1, m2, m3, m4, m5, m6, m7 = st.columns(7)
    m1.metric("Students (N)", n_students)
    m2.metric("Items (k)", n_items)
    m3.metric("Mean Score", f"{mean_score:.2f}")
    m4.metric("Std. Deviation", f"{std_score:.2f}")
    m5.metric("KR-20", f"{kr20:.4f}")
    m6.metric("Alpha", f"{alpha:.4f}")
    m7.metric("SEM (Error)", f"{sem:.4f}")
    
    # ======================================================================
    # VISUALIZATION SECTION - PUBLICATION QUALITY
    # ======================================================================
    st.subheader("📊 Item Difficulty Visualization (p & q Stacked Bar Chart)")
    
    # --- STACKED BAR CHART FOR p AND q (1 BAR CONTAINING BOTH p AND q) ---
    df_pq = df_res[['Item', 'p', 'q']].copy()
    df_pq_melted = df_pq.melt(id_vars=['Item'], var_name='Component', value_name='Proportion')
    
    # Color mapping for p and q
    color_map = {'p': '#2ecc71', 'q': '#e74c3c'}
    
    fig_pq = go.Figure()
    
    # Add p bars (bottom portion)
    fig_pq.add_trace(go.Bar(
        name='p (Correct / Difficulty)',
        x=df_pq['Item'],
        y=df_pq['p'],
        marker_color='#2ecc71',
        text=df_pq['p'].apply(lambda x: f'{x:.3f}'),
        textposition='inside',
        textfont=dict(color='white', size=11, family='Arial Black'),
        hovertemplate='<b>%{x}</b><br>p (Difficulty): %{y:.3f}<br>q (Incorrect): %{customdata:.3f}<extra></extra>',
        customdata=df_pq['q'],
        width=0.7
    ))
    
    # Add q bars (top portion - stacked)
    fig_pq.add_trace(go.Bar(
        name='q (Incorrect / 1-p)',
        x=df_pq['Item'],
        y=df_pq['q'],
        marker_color='#e74c3c',
        text=df_pq['q'].apply(lambda x: f'{x:.3f}'),
        textposition='inside',
        textfont=dict(color='white', size=11, family='Arial Black'),
        hovertemplate='<b>%{x}</b><br>q (Incorrect): %{y:.3f}<extra></extra>',
        width=0.7
    ))
    
    fig_pq.update_layout(
        barmode='stack',
        title=dict(
            text='<b>Item Difficulty Distribution: p (Correct) vs q (Incorrect)</b><br><sup>Each bar represents 100% of responses per item</sup>',
            font=dict(size=16, family='Arial Black'),
            x=0.5,
            xanchor='center'
        ),
        xaxis=dict(
            title='<b>Item Number</b>',
            tickangle=45,
            tickfont=dict(size=11),
            title_font=dict(size=13, family='Arial Black')
        ),
        yaxis=dict(
            title='<b>Proportion of Responses</b>',
            range=[0, 1],
            tickformat='.0%',
            tickfont=dict(size=11),
            title_font=dict(size=13, family='Arial Black'),
            gridcolor='lightgray',
            gridwidth=0.5
        ),
        legend=dict(
            title='<b>Component</b>',
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='center',
            x=0.5,
            font=dict(size=12)
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        hoverlabel=dict(bgcolor='white', font_size=12, font_family='Arial'),
        height=500,
        margin=dict(t=80, b=80, l=60, r=60)
    )
    
    # Add horizontal reference lines for difficulty thresholds
    fig_pq.add_hline(y=0.70, line_dash="dash", line_color="#27ae60", opacity=0.7,
                     annotation_text="Easy threshold (0.70)", annotation_position="top right")
    fig_pq.add_hline(y=0.30, line_dash="dash", line_color="#e67e22", opacity=0.7,
                     annotation_text="Difficult threshold (0.30)", annotation_position="bottom right")
    
    st.plotly_chart(fig_pq, use_container_width=True)
    
    # --- ADDITIONAL VISUALIZATION 1: Discrimination Index (d) Bar Chart ---
    st.subheader("🎯 Item Discrimination Index (d)")
    
    # Color function for discrimination values
    def get_d_color(d_val):
        if d_val >= 0.4: return '#2ecc71'  # Excellent
        elif d_val >= 0.3: return '#3498db'  # Good
        elif d_val >= 0.2: return '#f1c40f'  # Fair
        else: return '#e74c3c'  # Poor
    
    colors_d = [get_d_color(d) for d in df_res['d']]
    
    fig_d = go.Figure()
    fig_d.add_trace(go.Bar(
        x=df_res['Item'],
        y=df_res['d'],
        marker_color=colors_d,
        text=df_res['d'].apply(lambda x: f'{x:.3f}'),
        textposition='outside',
        textfont=dict(size=11),
        hovertemplate='<b>%{x}</b><br>Discrimination Index (d): %{y:.3f}<extra></extra>',
        width=0.7
    ))
    
    fig_d.update_layout(
        title=dict(
            text='<b>Item Discrimination Index (d)</b><br><sup>Upper-Lower Group Method (Kelley\'s %)</sup>',
            font=dict(size=16, family='Arial Black'),
            x=0.5,
            xanchor='center'
        ),
        xaxis=dict(
            title='<b>Item Number</b>',
            tickangle=45,
            tickfont=dict(size=11),
            title_font=dict(size=13, family='Arial Black')
        ),
        yaxis=dict(
            title='<b>Discrimination Index (d)</b>',
            range=[-0.1, 1.0],
            tickfont=dict(size=11),
            title_font=dict(size=13, family='Arial Black'),
            gridcolor='lightgray',
            gridwidth=0.5
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        hoverlabel=dict(bgcolor='white', font_size=12),
        height=450,
        margin=dict(t=80, b=80)
    )
    
    # Add reference lines
    fig_d.add_hline(y=0.40, line_dash="dash", line_color="#2ecc71", opacity=0.7,
                    annotation_text="Excellent (≥0.40)", annotation_position="top right")
    fig_d.add_hline(y=0.30, line_dash="dash", line_color="#3498db", opacity=0.7,
                    annotation_text="Good (≥0.30)", annotation_position="right")
    fig_d.add_hline(y=0.20, line_dash="dash", line_color="#f1c40f", opacity=0.7,
                    annotation_text="Fair (≥0.20)", annotation_position="bottom right")
    
    st.plotly_chart(fig_d, use_container_width=True)
    
    # --- ADDITIONAL VISUALIZATION 2: Point-Biserial Correlation (r_pbis) ---
    st.subheader("📈 Point-Biserial Correlation (Item-Whole Correlation)")
    
    def get_r_color(r_val, threshold):
        return '#2ecc71' if r_val >= threshold else '#e74c3c'
    
    colors_r = [get_r_color(r, validity_limit) for r in df_res['r_pbis']]
    
    fig_r = go.Figure()
    fig_r.add_trace(go.Bar(
        x=df_res['Item'],
        y=df_res['r_pbis'],
        marker_color=colors_r,
        text=df_res['r_pbis'].apply(lambda x: f'{x:.3f}'),
        textposition='outside',
        textfont=dict(size=11),
        hovertemplate='<b>%{x}</b><br>r_pbis: %{y:.3f}<extra></extra>',
        width=0.7
    ))
    
    fig_r.update_layout(
        title=dict(
            text=f'<b>Point-Biserial Correlation (r_pbis)</b><br><sup>Threshold for validity: {validity_limit}</sup>',
            font=dict(size=16, family='Arial Black'),
            x=0.5,
            xanchor='center'
        ),
        xaxis=dict(
            title='<b>Item Number</b>',
            tickangle=45,
            tickfont=dict(size=11),
            title_font=dict(size=13, family='Arial Black')
        ),
        yaxis=dict(
            title='<b>Point-Biserial Correlation (r_pbis)</b>',
            range=[-0.5, 1.0],
            tickfont=dict(size=11),
            title_font=dict(size=13, family='Arial Black'),
            gridcolor='lightgray',
            gridwidth=0.5
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        hoverlabel=dict(bgcolor='white', font_size=12),
        height=450,
        margin=dict(t=80, b=80)
    )
    
    fig_r.add_hline(y=validity_limit, line_dash="dash", line_color="#e74c3c", opacity=0.7,
                    annotation_text=f"Validity threshold ({validity_limit})", annotation_position="bottom right")
    
    st.plotly_chart(fig_r, use_container_width=True)
    
    # --- ADDITIONAL VISUALIZATION 3: Score Distribution Histogram ---
    st.subheader("📊 Score Distribution Analysis")
    
    fig_hist = go.Figure()
    fig_hist.add_trace(go.Histogram(
        x=total_scores,
        nbinsx=20,
        marker_color='#3498db',
        marker_line_color='white',
        marker_line_width=1,
        opacity=0.85,
        hovertemplate='Score: %{x}<br>Frequency: %{y}<extra></extra>'
    ))
    
    # Add normal distribution curve overlay
    from scipy.stats import norm
    x_norm = np.linspace(total_scores.min(), total_scores.max(), 100)
    y_norm = norm.pdf(x_norm, mean_score, std_score) * (len(total_scores) * (total_scores.max() - total_scores.min()) / 20)
    
    fig_hist.add_trace(go.Scatter(
        x=x_norm,
        y=y_norm,
        mode='lines',
        name='Normal Distribution',
        line=dict(color='#e74c3c', width=2, dash='dash'),
        hovertemplate='Theoretical density<br>Score: %{x:.1f}<extra></extra>'
    ))
    
    fig_hist.update_layout(
        title=dict(
            text='<b>Distribution of Total Test Scores</b><br><sup>With theoretical normal curve overlay</sup>',
            font=dict(size=16, family='Arial Black'),
            x=0.5,
            xanchor='center'
        ),
        xaxis=dict(
            title='<b>Total Score</b>',
            tickfont=dict(size=11),
            title_font=dict(size=13, family='Arial Black'),
            gridcolor='lightgray'
        ),
        yaxis=dict(
            title='<b>Frequency</b>',
            tickfont=dict(size=11),
            title_font=dict(size=13, family='Arial Black'),
            gridcolor='lightgray'
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        hoverlabel=dict(bgcolor='white', font_size=12),
        height=450,
        margin=dict(t=80),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5)
    )
    
    # Add mean and median lines
    fig_hist.add_vline(x=mean_score, line_dash="solid", line_color="#2ecc71", opacity=0.8,
                       annotation_text=f"Mean: {mean_score:.2f}", annotation_position="top")
    fig_hist.add_vline(x=total_scores.median(), line_dash="dot", line_color="#e67e22", opacity=0.8,
                       annotation_text=f"Median: {total_scores.median():.2f}", annotation_position="top")
    
    st.plotly_chart(fig_hist, use_container_width=True)
    
    # --- ADDITIONAL VISUALIZATION 4: Difficulty vs Discrimination Scatter Plot ---
    st.subheader("🔄 Item Diagnostics: Difficulty vs Discrimination")
    
    # Add decision-based coloring
    decision_colors = {'RETAIN': '#27ae60', 'REVISE': '#f39c12', 'REJECT': '#c0392b'}
    colors_decision = [decision_colors[dec] for dec in df_res['DECISION']]
    
    fig_scatter = go.Figure()
    fig_scatter.add_trace(go.Scatter(
        x=df_res['p'],
        y=df_res['d'],
        mode='markers+text',
        marker=dict(
            size=18,
            color=colors_decision,
            line=dict(width=2, color='white'),
            symbol='circle'
        ),
        text=df_res['Item'],
        textposition='middle center',
        textfont=dict(size=10, color='white', family='Arial Black'),
        hovertemplate='<b>Item %{text}</b><br>Difficulty (p): %{x:.3f}<br>Discrimination (d): %{y:.3f}<br>Decision: %{customdata}<extra></extra>',
        customdata=df_res['DECISION']
    ))
    
    # Add quadrant zones
    fig_scatter.add_hrect(y0=0.40, y1=1.0, line_width=0, fillcolor="#2ecc71", opacity=0.08,
                          annotation_text="Excellent Discrimination", annotation_position="top left")
    fig_scatter.add_hrect(y0=0.30, y1=0.40, line_width=0, fillcolor="#3498db", opacity=0.08,
                          annotation_text="Good", annotation_position="top left")
    fig_scatter.add_hrect(y0=0.20, y1=0.30, line_width=0, fillcolor="#f1c40f", opacity=0.08,
                          annotation_text="Fair", annotation_position="top left")
    
    fig_scatter.update_layout(
        title=dict(
            text='<b>Item Diagnostic Map: Difficulty (p) vs Discrimination (d)</b><br><sup>Colored by Retention Decision</sup>',
            font=dict(size=16, family='Arial Black'),
            x=0.5,
            xanchor='center'
        ),
        xaxis=dict(
            title='<b>Difficulty Index (p)</b>',
            range=[0, 1],
            tickformat='.2f',
            tickfont=dict(size=11),
            title_font=dict(size=13, family='Arial Black'),
            gridcolor='lightgray',
            gridwidth=0.5
        ),
        yaxis=dict(
            title='<b>Discrimination Index (d)</b>',
            range=[-0.1, 1.0],
            tickfont=dict(size=11),
            title_font=dict(size=13, family='Arial Black'),
            gridcolor='lightgray',
            gridwidth=0.5
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        hoverlabel=dict(bgcolor='white', font_size=12),
        height=500,
        margin=dict(t=80, l=60, r=60, b=60)
    )
    
    # Add vertical zones for difficulty
    fig_scatter.add_vrect(x0=0.70, x1=1.0, line_width=0, fillcolor="#27ae60", opacity=0.08,
                          annotation_text="Easy Zone", annotation_position="top right")
    fig_scatter.add_vrect(x0=0.30, x1=0.70, line_width=0, fillcolor="#f39c12", opacity=0.08,
                          annotation_text="Moderate", annotation_position="top right")
    fig_scatter.add_vrect(x0=0, x1=0.30, line_width=0, fillcolor="#e74c3c", opacity=0.08,
                          annotation_text="Difficult Zone", annotation_position="top right")
    
    st.plotly_chart(fig_scatter, use_container_width=True)
    
    # ======================================================================
    # END OF VISUALIZATION SECTION
    # ======================================================================
    
    st.subheader("🎯 Distractor Functionality Audit")
    min_ddi_global = df_res["Worst_DDI"].min()
    
    if min_ddi_global < 0:
        st.error(f"**Critical Alert:** A malfunctioning distractor was detected (Minimum DDI: {min_ddi_global:.4f}). Negative values indicate that the distractor is more attractive to high-performing students, suggesting a potential flaw in item construction.")
    else:
        st.success("**Perfect Functionality:** No negative DDI detected. All distractors are successfully drawing more students from the lower group than the upper group.")

    # --- DESCRIPTIVE INTERPRETATION (WEB) ---
    st.subheader("📝 Descriptive Interpretation")
    rel_eval = "Excellent" if kr20 >= 0.9 else "High" if kr20 >= 0.8 else "Acceptable" if kr20 >= 0.7 else "Low"
    
    col_int1, col_int2 = st.columns(2)
    with col_int1:
        st.info(f"**Test Reliability:** The KR-20 coefficient of {kr20:.4f} indicates **{rel_eval}** internal consistency. "
                f"This suggests that the instrument is {'highly stable' if kr20 >= 0.8 else 'sufficient'} for measuring the intended academic constructs.")
    with col_int2:
        st.warning(f"**Measurement Error:** The SEM of {sem:.4f} indicates that a student's observed score may fluctuate within this range "
                   f"relative to their theoretical true score. A lower SEM reflects higher precision in score estimation.")

    # --- ITEM MATRIX ---
    st.subheader("📋 Comprehensive Item Statistics Matrix")
    def apply_item_styling(row):
        styles = [''] * len(row)
        dif_color = '#ccffcc' if row['p'] > 0.7 else '#ffcccc' if row['p'] < 0.3 else '#fff2cc'
        styles[1] = styles[2] = f'background-color: {dif_color}; color: black'
        if row['d'] >= 0.4: dis_color, txt = '#2ecc71', 'white'
        elif row['d'] >= 0.3: dis_color, txt = '#3498db', 'white'
        elif row['d'] >= 0.2: dis_color, txt = '#f1c40f', 'black'
        else: dis_color, txt = '#e74c3c', 'white'
        styles[6] = styles[7] = styles[8] = styles[9] = f'background-color: {dis_color}; color: {txt}'
        val_bg = '#ccffcc' if row['r_pbis'] >= validity_limit else '#ffcccc'
        styles[10] = f'background-color: {val_bg}; color: black; font-weight: bold'
        styles[11] = f'background-color: {val_bg}; color: black'
        if row['DECISION'] == "RETAIN": styles[12] = 'background-color: #27ae60; color: white; font-weight: bold'
        elif row['DECISION'] == "REVISE": styles[12] = 'background-color: #f39c12; color: white'
        else: styles[12] = 'background-color: #c0392b; color: white'
        return styles

    st.dataframe(df_res.style.apply(apply_item_styling, axis=1).format("{:.4f}", subset=["p", "q", "pq", "Var", "d", "Best_DDI", "Worst_DDI", "r_pbis"]), use_container_width=True)

    # --- RANKING TABLE ---
    st.subheader("🏆 Student Score Ranking & Grouping")
    def apply_rank_styling(row):
        style = [''] * len(row)
        if row['Group'] == 'Upper': bg_color = '#d4edda'
        elif row['Group'] == 'Lower': bg_color = '#f8d7da'
        else: bg_color = 'white'
        return [f'background-color: {bg_color}; color: black'] * len(row)

    st.dataframe(df_ranking.style.apply(apply_rank_styling, axis=1), use_container_width=True)

    # --- DISTRACTORS ---
    st.divider()
    st.subheader("🎯 Distractor Effectiveness")
    dist_data = [df[item].astype(str).str.upper().str.strip().value_counts(normalize=True).to_dict() | {"Item": item} for item in item_cols]
    df_dist = pd.DataFrame(dist_data).set_index('Item').fillna(0)
    cols = sorted([c for c in df_dist.columns if len(str(c)) == 1]) + sorted([c for c in df_dist.columns if len(str(c)) > 1])
    df_dist_final = df_dist[cols].copy()
    for col in cols: df_dist_final[col] = df_dist_final[col].apply(lambda x: f"{x:.4f} ({x:.2%})")
    
    df_dist_final['Interpretation'] = df_dist[cols].apply(lambda row: f"Effective: {', '.join([str(opt) for opt, val in row.items() if val >= 0.05 and opt != 'N/A'])}", axis=1)
    
    df_dist_formatted = df_dist[cols].style.format(lambda x: f"{x:.4f} ({x:.2%})").background_gradient(cmap='YlGn')
    st.dataframe(df_dist_formatted, use_container_width=True)

    # --- EXCEL DOWNLOAD ---
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine='xlsxwriter') as writer:
        df_res.to_excel(writer, index=False, sheet_name='Item_Analysis')
        df_ranking.to_excel(writer, index=False, sheet_name='Student_Ranking')
        df_dist_final.to_excel(writer, index=True, sheet_name='Distractor_Analysis')
        
        # GABUNGAN Reliability_Summary + Interpretive_Report
        reliability_interpretation = pd.DataFrame({
            "Metric": [
                "Students (N)", 
                "Items (k)", 
                "Mean Score", 
                "Std. Deviation", 
                "KR-20", 
                "Alpha", 
                "SEM",
                "KR-20 Interpretation",
                "SEM Interpretation"
            ],
            "Value": [
                n_students, 
                n_items, 
                f"{mean_score:.2f}", 
                f"{std_score:.2f}", 
                f"{kr20:.4f}", 
                f"{alpha:.4f}", 
                f"{sem:.4f}",
                f"{rel_eval} reliability",
                f"±{sem:.4f} margin of error"
            ],
            "Interpretation": [
                "Total number of test takers",
                "Total number of items in the test",
                "Average raw score across all students",
                "Spread of scores around the mean",
                f"Internal consistency: {rel_eval}. {'Excellent (>0.90)' if kr20 >= 0.9 else 'High (>0.80)' if kr20 >= 0.8 else 'Acceptable (>0.70)' if kr20 >= 0.7 else 'Low (<0.70)'}",
                "Alternative reliability coefficient (parallel to KR-20)",
                "Standard error of measurement. Lower values = higher precision",
                f"The instrument shows {rel_eval} reliability. High values (>0.70) indicate consistency in measurement.",
                f"The SEM of {sem:.4f} indicates that a student's observed score may fluctuate within this range relative to their theoretical true score."
            ]
        })
        
        reliability_interpretation.to_excel(writer, index=False, sheet_name='Reliability_Interpretation')
            
    st.download_button(label="📥 Download Full Report", data=buf.getvalue(), file_name="Complete_Item_Analysis_Report.xlsx")
