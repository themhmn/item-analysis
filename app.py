import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import pointbiserialr
import io
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import matplotlib.patches as mpatches
import plotly.graph_objects as go
import plotly.express as px

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
        
        if r_pb < validity_limit:
            reasons.append("Low validity")
        
        if d_val < 0.20:
            reasons.append("Poor discrimination")
        
        if p > 0.90:
            reasons.append("Too easy")
        
        if p < 0.20:
            reasons.append("Too difficult")
        
        if worst_ddi < -0.10:
            reasons.append("Severely flawed distractor")
        elif worst_ddi < 0:
            reasons.append("Malfunctioning distractor")
        
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
    # PROFESSIONAL VISUALIZATION SECTION - PUBLICATION QUALITY
    # ======================================================================
    
    st.markdown("---")
    st.markdown("<h2 style='text-align: center; color: #2c3e50;'>📊 ITEM ANALYSIS VISUALIZATION DASHBOARD</h2>", unsafe_allow_html=True)
    st.markdown("---")
    
    # ----------------------------------------------------------------------
    # FIGURE 1: STACKED BAR CHART (p & q)
    # ----------------------------------------------------------------------
    st.markdown("<h3 style='color: #2c3e50; margin-top: 30px;'>📍 Figure 1. Item Difficulty Distribution</h3>", unsafe_allow_html=True)
    st.markdown("<p style='color: #7f8c8d; margin-bottom: 20px;'><i>Stacked bar chart showing proportion of correct (green) vs incorrect (red) responses per item. Horizontal dashed lines indicate difficulty thresholds.</i></p>", unsafe_allow_html=True)
    
    fig1 = plt.figure(figsize=(14, 7), facecolor='white', dpi=100)
    ax1 = fig1.add_subplot(111)
    ax1.set_facecolor('#fafafa')
    
    items = df_res['Item'].tolist()
    p_vals = df_res['p'].tolist()
    q_vals = df_res['q'].tolist()
    
    bars_p = ax1.bar(items, p_vals, width=0.7, color='#2ecc71', 
                     label='p (Proportion Correct)', edgecolor='white', linewidth=1, zorder=3)
    bars_q = ax1.bar(items, q_vals, bottom=p_vals, width=0.7, color='#e74c3c', 
                     label='q (Proportion Incorrect)', edgecolor='white', linewidth=1, zorder=3)
    
    for i, (p, q) in enumerate(zip(p_vals, q_vals)):
        if p > 0.08:
            ax1.text(i, p/2, f'{p:.3f}', ha='center', va='center', fontsize=10, 
                    color='white', fontweight='bold')
        if q > 0.08:
            ax1.text(i, p + q/2, f'{q:.3f}', ha='center', va='center', fontsize=10, 
                    color='white', fontweight='bold')
    
    ax1.axhline(y=0.70, color='#27ae60', linestyle='--', alpha=0.8, linewidth=2, zorder=4)
    ax1.axhline(y=0.30, color='#e67e22', linestyle='--', alpha=0.8, linewidth=2, zorder=4)
    
    ax1.set_xlabel('Item Number', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Proportion of Responses', fontsize=13, fontweight='bold')
    ax1.set_title('Item Difficulty Distribution: p (Correct) vs q (Incorrect)', 
                 fontsize=15, fontweight='bold', pad=15)
    ax1.set_ylim(0, 1.02)
    ax1.set_xticks(range(len(items)))
    ax1.set_xticklabels(items, rotation=45, ha='right', fontsize=10)
    ax1.set_yticks(np.arange(0, 1.1, 0.1))
    ax1.set_yticklabels([f'{x:.0%}' for x in np.arange(0, 1.1, 0.1)], fontsize=10)
    ax1.grid(axis='y', alpha=0.3, linestyle='-', linewidth=0.5, zorder=0)
    ax1.legend(loc='upper right', fontsize=11, framealpha=0.95, fancybox=True, shadow=True)
    
    plt.tight_layout()
    st.pyplot(fig1)
    plt.close(fig1)
    
    # ----------------------------------------------------------------------
    # FIGURE 2: DISCRIMINATION INDEX
    # ----------------------------------------------------------------------
    st.markdown("<h3 style='color: #2c3e50; margin-top: 40px;'>📍 Figure 2. Item Discrimination Index (d)</h3>", unsafe_allow_html=True)
    st.markdown("<p style='color: #7f8c8d; margin-bottom: 20px;'><i>Higher values indicate better item discrimination. Color coding: Green (Excellent ≥0.40), Blue (Good ≥0.30), Yellow (Fair ≥0.20), Red (Poor <0.20).</i></p>", unsafe_allow_html=True)
    
    def get_d_color(d_val):
        if d_val >= 0.4: return '#2ecc71'
        elif d_val >= 0.3: return '#3498db'
        elif d_val >= 0.2: return '#f1c40f'
        else: return '#e74c3c'
    
    colors_d = [get_d_color(d) for d in df_res['d']]
    
    fig2 = plt.figure(figsize=(14, 6), facecolor='white', dpi=100)
    ax2 = fig2.add_subplot(111)
    ax2.set_facecolor('#fafafa')
    
    bars_d = ax2.bar(items, df_res['d'], width=0.7, color=colors_d, 
                     edgecolor='black', linewidth=0.8, zorder=3)
    
    for bar, d_val in zip(bars_d, df_res['d']):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.015, f'{d_val:.3f}', 
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax2.axhline(y=0.40, color='#2ecc71', linestyle='--', alpha=0.8, linewidth=2, label='Excellent (≥0.40)')
    ax2.axhline(y=0.30, color='#3498db', linestyle='--', alpha=0.8, linewidth=2, label='Good (≥0.30)')
    ax2.axhline(y=0.20, color='#f1c40f', linestyle='--', alpha=0.8, linewidth=2, label='Fair (≥0.20)')
    
    ax2.set_xlabel('Item Number', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Discrimination Index (d)', fontsize=13, fontweight='bold')
    ax2.set_title('Item Discrimination Index - Upper-Lower Group Method', 
                 fontsize=15, fontweight='bold', pad=15)
    ax2.set_ylim(-0.05, 1.05)
    ax2.set_xticks(range(len(items)))
    ax2.set_xticklabels(items, rotation=45, ha='right', fontsize=10)
    ax2.grid(axis='y', alpha=0.3, linestyle='-', linewidth=0.5, zorder=0)
    ax2.legend(loc='upper left', fontsize=10, framealpha=0.95, fancybox=True, shadow=True)
    
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close(fig2)
    
    # ----------------------------------------------------------------------
    # FIGURE 3: POINT-BISERIAL CORRELATION
    # ----------------------------------------------------------------------
    st.markdown("<h3 style='color: #2c3e50; margin-top: 40px;'>📍 Figure 3. Point-Biserial Correlation (r_pbis)</h3>", unsafe_allow_html=True)
    st.markdown(f"<p style='color: #7f8c8d; margin-bottom: 20px;'><i>Correlation between item score and total test score. Green bars exceed the validity threshold ({validity_limit}), indicating valid items.</i></p>", unsafe_allow_html=True)
    
    colors_r = ['#2ecc71' if r >= validity_limit else '#e74c3c' for r in df_res['r_pbis']]
    
    fig3 = plt.figure(figsize=(14, 6), facecolor='white', dpi=100)
    ax3 = fig3.add_subplot(111)
    ax3.set_facecolor('#fafafa')
    
    bars_r = ax3.bar(items, df_res['r_pbis'], width=0.7, color=colors_r, 
                     edgecolor='black', linewidth=0.8, zorder=3)
    
    for bar, r_val in zip(bars_r, df_res['r_pbis']):
        height = bar.get_height()
        if r_val >= 0:
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.02, f'{r_val:.3f}', 
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
        else:
            ax3.text(bar.get_x() + bar.get_width()/2., height - 0.03, f'{r_val:.3f}', 
                    ha='center', va='top', fontsize=9, fontweight='bold')
    
    ax3.axhline(y=validity_limit, color='#e74c3c', linestyle='--', alpha=0.8, linewidth=2, 
                label=f'Validity Threshold ({validity_limit})')
    ax3.axhline(y=0, color='#95a5a6', linestyle='-', alpha=0.5, linewidth=1)
    
    ax3.set_xlabel('Item Number', fontsize=13, fontweight='bold')
    ax3.set_ylabel('Point-Biserial Correlation (r_pbis)', fontsize=13, fontweight='bold')
    ax3.set_title('Item-Whole Correlation: Point-Biserial Coefficient', 
                 fontsize=15, fontweight='bold', pad=15)
    ax3.set_ylim(-0.55, 1.05)
    ax3.set_xticks(range(len(items)))
    ax3.set_xticklabels(items, rotation=45, ha='right', fontsize=10)
    ax3.grid(axis='y', alpha=0.3, linestyle='-', linewidth=0.5, zorder=0)
    ax3.legend(loc='upper left', fontsize=10, framealpha=0.95, fancybox=True, shadow=True)
    
    plt.tight_layout()
    st.pyplot(fig3)
    plt.close(fig3)
    
    # ----------------------------------------------------------------------
    # FIGURE 4: SCORE DISTRIBUTION HISTOGRAM
    # ----------------------------------------------------------------------
    st.markdown("<h3 style='color: #2c3e50; margin-top: 40px;'>📍 Figure 4. Total Score Distribution</h3>", unsafe_allow_html=True)
    st.markdown("<p style='color: #7f8c8d; margin-bottom: 20px;'><i>Histogram of total test scores with mean (green) and median (orange) indicators.</i></p>", unsafe_allow_html=True)
    
    fig4 = plt.figure(figsize=(14, 6), facecolor='white', dpi=100)
    ax4 = fig4.add_subplot(111)
    ax4.set_facecolor('#fafafa')
    
    n_bins = min(15, len(np.unique(total_scores)))
    ax4.hist(total_scores, bins=n_bins, color='#3498db', edgecolor='white', 
             linewidth=1.5, alpha=0.7, density=False, zorder=3)
    
    ax4.axvline(x=mean_score, color='#2ecc71', linestyle='-', linewidth=2.5, 
                label=f'Mean = {mean_score:.2f}', zorder=4)
    ax4.axvline(x=total_scores.median(), color='#e67e22', linestyle='--', linewidth=2.5, 
                label=f'Median = {total_scores.median():.2f}', zorder=4)
    
    ax4.set_xlabel('Total Score', fontsize=13, fontweight='bold')
    ax4.set_ylabel('Frequency', fontsize=13, fontweight='bold')
    ax4.set_title('Distribution of Total Test Scores', fontsize=15, fontweight='bold', pad=15)
    ax4.grid(axis='y', alpha=0.3, linestyle='-', linewidth=0.5, zorder=0)
    ax4.legend(loc='upper right', fontsize=11, framealpha=0.95, fancybox=True, shadow=True)
    
    stats_text = f'N = {n_students}\nMin = {total_scores.min():.0f}\nMax = {total_scores.max():.0f}'
    ax4.text(0.02, 0.98, stats_text, transform=ax4.transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    st.pyplot(fig4)
    plt.close(fig4)
    
    # ----------------------------------------------------------------------
    # FIGURE 5: DIFFICULTY VS DISCRIMINATION SCATTER PLOT (USING PLOTLY - MORE RELIABLE)
    # ----------------------------------------------------------------------
    st.markdown("<h3 style='color: #2c3e50; margin-top: 40px;'>📍 Figure 5. Item Diagnostic Map</h3>", unsafe_allow_html=True)
    st.markdown("<p style='color: #7f8c8d; margin-bottom: 20px;'><i>Scatter plot mapping each item's difficulty (p) against discrimination (d). Color indicates recommended action: Green (Retain), Orange (Revise), Red (Reject). Hover over points for details.</i></p>", unsafe_allow_html=True)
    
    # Prepare data for Plotly
    decision_colors_plotly = {'RETAIN': '#27ae60', 'REVISE': '#f39c12', 'REJECT': '#c0392b'}
    decision_labels_plotly = {'RETAIN': 'RETAIN (Pertahankan)', 'REVISE': 'REVISE (Revisi)', 'REJECT': 'REJECT (Tolak)'}
    
    df_plotly = df_res.copy()
    df_plotly['Decision_Label'] = df_plotly['DECISION'].map(decision_labels_plotly)
    df_plotly['Decision_Color'] = df_plotly['DECISION'].map(decision_colors_plotly)
    df_plotly['Tooltip'] = df_plotly.apply(lambda x: f"<b>Item {x['Item']}</b><br>p (Difficulty): {x['p']:.3f}<br>d (Discrimination): {x['d']:.3f}<br>r_pbis: {x['r_pbis']:.3f}<br>Decision: {decision_labels_plotly[x['DECISION']]}", axis=1)
    
    # Create Plotly figure
    fig5 = go.Figure()
    
    # Add scatter points
    fig5.add_trace(go.Scatter(
        x=df_plotly['p'],
        y=df_plotly['d'],
        mode='markers+text',
        marker=dict(
            size=18,
            color=df_plotly['Decision_Color'],
            line=dict(width=2, color='white'),
            symbol='circle'
        ),
        text=df_plotly['Item'],
        textposition='middle center',
        textfont=dict(size=11, color='white', family='Arial Black'),
        hovertemplate='%{customdata}<extra></extra>',
        customdata=df_plotly['Tooltip'],
        name='Items'
    ))
    
    # Add threshold lines
    fig5.add_hline(y=0.40, line_dash="dash", line_color="#2ecc71", opacity=0.8, 
                   annotation_text="Excellent (d ≥ 0.40)", annotation_position="top right")
    fig5.add_hline(y=0.30, line_dash="dash", line_color="#3498db", opacity=0.8,
                   annotation_text="Good (d ≥ 0.30)", annotation_position="right")
    fig5.add_hline(y=0.20, line_dash="dash", line_color="#f1c40f", opacity=0.8,
                   annotation_text="Fair (d ≥ 0.20)", annotation_position="bottom right")
    fig5.add_vline(x=0.70, line_dash="dot", line_color="#27ae60", opacity=0.8,
                   annotation_text="Easy (p > 0.70)", annotation_position="top")
    fig5.add_vline(x=0.30, line_dash="dot", line_color="#e74c3c", opacity=0.8,
                   annotation_text="Difficult (p < 0.30)", annotation_position="bottom")
    
    # Add shaded background zones
    fig5.add_hrect(y0=0.40, y1=1.05, line_width=0, fillcolor="#2ecc71", opacity=0.08)
    fig5.add_hrect(y0=0.30, y1=0.40, line_width=0, fillcolor="#3498db", opacity=0.08)
    fig5.add_hrect(y0=0.20, y1=0.30, line_width=0, fillcolor="#f1c40f", opacity=0.08)
    fig5.add_vrect(x0=0.70, x1=1.05, line_width=0, fillcolor="#27ae60", opacity=0.06)
    fig5.add_vrect(x0=-0.05, x1=0.30, line_width=0, fillcolor="#e74c3c", opacity=0.06)
    
    # Update layout
    fig5.update_layout(
        title=dict(
            text='<b>Item Diagnostic Map: Difficulty (p) vs Discrimination (d)</b>',
            font=dict(size=18, family='Arial Black', color='#2c3e50'),
            x=0.5,
            xanchor='center'
        ),
        xaxis=dict(
            title='<b>Difficulty Index (p)</b> → Easier',
            range=[-0.05, 1.05],
            tickformat='.2f',
            tickfont=dict(size=12),
            title_font=dict(size=14, family='Arial Black'),
            gridcolor='lightgray',
            gridwidth=0.5,
            showgrid=True,
            zeroline=False
        ),
        yaxis=dict(
            title='<b>Discrimination Index (d)</b> → Better',
            range=[-0.05, 1.05],
            tickfont=dict(size=12),
            title_font=dict(size=14, family='Arial Black'),
            gridcolor='lightgray',
            gridwidth=0.5,
            showgrid=True,
            zeroline=False
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        hoverlabel=dict(bgcolor='white', font_size=12, font_family='Arial'),
        height=650,
        width=1000,
        margin=dict(t=80, l=80, r=80, b=80),
        legend=dict(
            title='<b>Recommended Action</b>',
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='center',
            x=0.5,
            font=dict(size=12)
        )
    )
    
    # Add zone annotations
    fig5.add_annotation(
        x=0.88, y=0.92, text="★ IDEAL ZONE ★",
        showarrow=False, font=dict(size=11, color='#27ae60', family='Arial Black'),
        bgcolor='white', bordercolor='#27ae60', borderwidth=1, borderpad=4
    )
    fig5.add_annotation(
        x=0.10, y=0.92, text="Too Hard<br>Good Disc.",
        showarrow=False, font=dict(size=10, color='#3498db'),
        bgcolor='white', bordercolor='#3498db', borderwidth=1, borderpad=4
    )
    fig5.add_annotation(
        x=0.88, y=0.10, text="Too Easy<br>Poor Disc.",
        showarrow=False, font=dict(size=10, color='#f39c12'),
        bgcolor='white', bordercolor='#f39c12', borderwidth=1, borderpad=4
    )
    fig5.add_annotation(
        x=0.10, y=0.10, text="Poor Quality<br>REJECT",
        showarrow=False, font=dict(size=10, color='#c0392b', weight='bold'),
        bgcolor='white', bordercolor='#c0392b', borderwidth=1, borderpad=4
    )
    
    st.plotly_chart(fig5, use_container_width=True)
    
    st.markdown("---")
    st.markdown("<p style='text-align: center; color: #7f8c8d; font-size: 12px;'>Visualization generated using Classical Test Theory (CTT) framework. For publication, figures can be exported as high-resolution images.</p>", unsafe_allow_html=True)
    st.markdown("---")
    
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
