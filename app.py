import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import pointbiserialr, norm
from scipy.optimize import minimize, brentq
from scipy.special import expit  # logistic sigmoid
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import io
import warnings
warnings.filterwarnings('ignore')

# ======================================================================
# ACADEMIC ITEM ANALYSIS — CTT + IRT SUITE (2026)
# CTT + IRT 1PL/2PL/3PL + Auto-Interpretation
# ======================================================================

st.set_page_config(page_title="Item Analysis Pro by Muhaimin Abdullah", page_icon="📊", layout="wide")

# ── Custom CSS ──────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600;700&display=swap');

html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
.main { background-color: #0d1117; }
h1, h2, h3 { font-family: 'IBM Plex Mono', monospace !important; }

[data-testid="stMetricValue"] {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 1.6rem !important;
    font-weight: 600 !important;
    color: #58a6ff !important;
}
[data-testid="stMetricLabel"] { color: #8b949e !important; font-size: 0.75rem !important; }
[data-testid="stMetricDelta"] { font-family: 'IBM Plex Mono', monospace !important; }

.metric-card {
    background: linear-gradient(135deg, #161b22, #1c2128);
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 1rem;
    margin: 0.25rem 0;
}
.flag-retain { background:#1a4731; color:#3fb950; border:1px solid #3fb950; border-radius:4px; padding:2px 8px; font-weight:700; font-family:'IBM Plex Mono',monospace; font-size:0.8rem; }
.flag-revise { background:#3d2b00; color:#d29922; border:1px solid #d29922; border-radius:4px; padding:2px 8px; font-weight:700; font-family:'IBM Plex Mono',monospace; font-size:0.8rem; }
.flag-reject { background:#3d1212; color:#f85149; border:1px solid #f85149; border-radius:4px; padding:2px 8px; font-weight:700; font-family:'IBM Plex Mono',monospace; font-size:0.8rem; }

div[data-testid="stExpander"] { border: 1px solid #30363d; border-radius:6px; }
.stAlert { border-radius:6px !important; }
.stTabs [data-baseweb="tab-list"] { gap: 8px; background: #161b22; border-radius: 8px; padding: 4px; }
.stTabs [data-baseweb="tab"] { border-radius: 6px; color: #8b949e; font-family: 'IBM Plex Mono', monospace; font-size: 0.82rem; }
.stTabs [aria-selected="true"] { background:#21262d !important; color:#58a6ff !important; }
</style>
""", unsafe_allow_html=True)

# ── Title ────────────────────────────────────────────────────────────────
st.markdown("# ITEM ANALYSIS by Muhaimin Abdullah")
st.markdown("**Classical Test Theory (CTT) + Item Response Theory (IRT 1PL / 2PL / 3PL)** · *Methodologically Validated · 2026 Edition*")
st.divider()

# ══════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### ⚙️ Analysis Settings")

    st.markdown("**CTT Parameters**")
    group_percent = st.slider("Kelley's Grouping (%)", 10, 50, 27,
        help="Percentage of top/bottom students used to compute discrimination index d. Kelley (1939) recommends 27%.")
    validity_limit = st.number_input("r_pbis Validity Threshold", 0.0, 1.0, 0.25, step=0.05,
        help="Minimum point-biserial correlation for an item to be considered valid. Common threshold: 0.20–0.30.")

    st.markdown("---")
    st.markdown("**IRT Parameters**")
    irt_model = st.selectbox("IRT Model", ["1PL (Rasch)", "2PL", "3PL"],
        help="1PL: only difficulty. 2PL: difficulty + discrimination. 3PL: difficulty + discrimination + pseudo-guessing.")
    irt_max_iter = st.slider("Max EM Iterations", 50, 500, 200, step=50)

    st.markdown("---")
    st.markdown("### 📖 Legend")
    with st.expander("Difficulty Index (p)", expanded=False):
        st.markdown("""
| Range | Label | Symbol |
|---|---|---|
| > 0.70 | Easy | 🟢 |
| 0.30–0.70 | Moderate | 🟡 |
| < 0.30 | Difficult | 🔴 |

*Optimal range: 0.30–0.70 (maximises score variance and information).*
        """)
    with st.expander("Discrimination (d & DDI)", expanded=False):
        st.markdown("""
| d-value | Label |
|---|---|
| ≥ 0.40 | 🟢 Excellent |
| 0.30–0.39 | 🔵 Good |
| 0.20–0.29 | 🟡 Fair |
| < 0.20 | 🔴 Poor |

**d** = proportion correct (Upper) − proportion correct (Lower)  
**DDI** = Distractor Discrimination Index (for wrong options)
        """)
    with st.expander("IRT Parameters", expanded=False):
        st.markdown("""
**b (difficulty):** Theta (ability) level where P(correct) = 0.50 for 1PL/2PL, or (1+c)/2 for 3PL.  
**a (discrimination):** Slope of ICC at inflection point. Higher = better discriminating item.  
**c (pseudo-guessing):** Probability of correct response by chance for very low-ability students.  
**INFIT/OUTFIT:** Model fit statistics. Ideal range: 0.70–1.30 (Rasch).
        """)
    with st.expander("Reliability Benchmarks", expanded=False):
        st.markdown("""
| KR-20 | Interpretation |
|---|---|
| ≥ 0.90 | Excellent |
| 0.80–0.89 | High |
| 0.70–0.79 | Acceptable |
| < 0.70 | Low |
        """)

# ══════════════════════════════════════════════════════════════════════
# FILE UPLOAD
# ══════════════════════════════════════════════════════════════════════
st.markdown("### 📁 Data Input")
u1, u2 = st.columns(2)
with u1:
    student_file = st.file_uploader("Upload Student Response Data (CSV)", type=['csv'],
        help="First column = Student ID. Subsequent columns = responses to each item (A/B/C/D or numeric).")
with u2:
    key_file = st.file_uploader("Upload Answer Key (CSV)", type=['csv'],
        help="Same column structure as student data. First row = correct answers.")

st.markdown("""
<details><summary>📋 <b>CSV Format Guide</b></summary>

**Student CSV:**
```
StudentID, Q1, Q2, Q3, ...
S001,      A,  C,  B, ...
S002,      B,  C,  A, ...
```
**Key CSV:**
```
Key, Q1, Q2, Q3, ...
ANS, A,  C,  A, ...
```
</details>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════
# IRT CORE FUNCTIONS
# ══════════════════════════════════════════════════════════════════════

def irt_prob(theta, a, b, c):
    """3PL ICC: P(theta) = c + (1-c) / (1 + exp(-a*(theta-b)))"""
    return c + (1 - c) * expit(a * (theta - b))

def estimate_irt_em(X, model='2PL', max_iter=200, tol=1e-5):
    """
    EM Algorithm for IRT parameter estimation.
    X: binary response matrix (n_students x n_items)
    Returns: params dict with a, b, c arrays and theta estimates.
    """
    n, k = X.shape
    # Gauss-Hermite quadrature nodes & weights (21-point approximation)
    n_quad = 21
    theta_q = np.linspace(-4, 4, n_quad)
    # Normal prior weights
    w_q = norm.pdf(theta_q, 0, 1)
    w_q /= w_q.sum()

    # Initialize parameters
    p_mean = X.mean(axis=0).clip(0.05, 0.95)
    b_init = -norm.ppf(p_mean)
    a_init = np.ones(k)
    c_init = np.zeros(k)

    if model == '1PL':
        a_fixed = 1.0
    
    params = {'a': a_init.copy(), 'b': b_init.copy(), 'c': c_init.copy()}

    prev_ll = -np.inf

    for iteration in range(max_iter):
        a_arr, b_arr, c_arr = params['a'], params['b'], params['c']

        # E-step: compute posterior weights r[q, j] (expected count) and
        # f[q] (expected freq of each quadrature point)
        # P_qj: (n_quad, k) — probability correct for each theta_q, item j
        P_qj = np.array([[irt_prob(th, a_arr[j], b_arr[j], c_arr[j])
                          for j in range(k)] for th in theta_q])  # (Q, k)

        # L_qi: likelihood of each student's pattern at each theta_q
        # L_qi: (Q, n)
        L_qi = np.ones((n_quad, n))
        for q, th in enumerate(theta_q):
            p_q = P_qj[q]  # (k,)
            # product over items
            L_qi[q] = np.prod(
                np.where(X == 1, p_q, 1 - p_q), axis=1
            )

        # marginal likelihood per student: (n,)
        f_qi = L_qi * w_q[:, None]  # (Q, n)
        marg = f_qi.sum(axis=0)  # (n,) marginal likelihood
        marg = np.maximum(marg, 1e-300)

        # posterior: (Q, n)
        post = f_qi / marg[None, :]

        # Expected counts per quadrature point
        r_q = post.sum(axis=1)     # (Q,) total weight at each theta_q
        # Expected correct at each quad point for each item: (Q, k)
        rj_q = post @ X            # (Q, k)

        log_lik = np.sum(np.log(marg))

        # M-step: update item parameters
        for j in range(k):
            def neg_loglik_item(pars):
                if model == '1PL':
                    a_j, b_j, c_j = 1.0, pars[0], 0.0
                elif model == '2PL':
                    a_j, b_j, c_j = pars[0], pars[1], 0.0
                else:  # 3PL
                    a_j, b_j, c_j = pars[0], pars[1], pars[2]
                p_j = np.array([irt_prob(th, a_j, b_j, c_j) for th in theta_q])
                p_j = np.clip(p_j, 1e-9, 1 - 1e-9)
                ll = np.sum(rj_q[:, j] * np.log(p_j) +
                            (r_q - rj_q[:, j]) * np.log(1 - p_j))
                return -ll

            if model == '1PL':
                x0 = [b_arr[j]]
                bounds = [(-4, 4)]
            elif model == '2PL':
                x0 = [a_arr[j], b_arr[j]]
                bounds = [(0.3, 3.0), (-4, 4)]
            else:
                x0 = [a_arr[j], b_arr[j], c_arr[j]]
                bounds = [(0.3, 3.0), (-4, 4), (0.0, 0.40)]

            try:
                res = minimize(neg_loglik_item, x0, bounds=bounds,
                               method='L-BFGS-B',
                               options={'maxiter': 50, 'ftol': 1e-8})
                if model == '1PL':
                    params['b'][j] = res.x[0]
                elif model == '2PL':
                    params['a'][j], params['b'][j] = res.x
                else:
                    params['a'][j], params['b'][j], params['c'][j] = res.x
            except Exception:
                pass

        if abs(log_lik - prev_ll) < tol:
            break
        prev_ll = log_lik

    # Estimate theta per student using MLE (given item params)
    thetas = []
    for i in range(n):
        def neg_ll_theta(th):
            p_j = np.array([irt_prob(th[0], params['a'][j], params['b'][j], params['c'][j])
                            for j in range(k)])
            p_j = np.clip(p_j, 1e-9, 1 - 1e-9)
            return -np.sum(X[i] * np.log(p_j) + (1 - X[i]) * np.log(1 - p_j))
        try:
            r = minimize(neg_ll_theta, [0.0], bounds=[(-4, 4)], method='L-BFGS-B')
            thetas.append(float(r.x[0]))
        except Exception:
            thetas.append(0.0)

    return params, np.array(thetas), log_lik

def compute_item_info(theta_range, a, b, c):
    """Item Information Function: I(theta) = (dP/dtheta)^2 / (P*Q)"""
    P = irt_prob(theta_range, a, b, c)
    Q = 1 - P
    dP = a * (P - c) * Q / (1 - c)
    info = np.where(P * Q > 1e-10, dP**2 / (P * Q), 0)
    return info

def rasch_fit_stats(X, b_arr, theta_arr):
    """
    Compute INFIT and OUTFIT mean-square fit statistics (Rasch/1PL only).
    INFIT = variance-weighted residual; OUTFIT = unweighted (outlier-sensitive).
    Ideal range: 0.70–1.30.
    """
    n, k = X.shape
    infit_list, outfit_list = [], []
    for j in range(k):
        P_ij = expit(theta_arr - b_arr[j])  # (n,)
        W_ij = P_ij * (1 - P_ij)
        Z_ij = X[:, j] - P_ij
        Z2_ij = Z_ij ** 2
        W2_ij = W_ij ** 2

        outfit = np.mean(Z2_ij / np.maximum(P_ij * (1 - P_ij), 1e-9))
        infit_num = np.sum(Z2_ij * W_ij)
        infit_den = np.sum(W2_ij)
        infit = infit_num / max(infit_den, 1e-9)

        infit_list.append(infit)
        outfit_list.append(outfit)
    return np.array(infit_list), np.array(outfit_list)

# ══════════════════════════════════════════════════════════════════════
# DESCRIPTIVE AUTO-INTERPRETATION FUNCTIONS
# ══════════════════════════════════════════════════════════════════════

def interpret_reliability(kr20, sem, n_items):
    if kr20 >= 0.90:
        rel_label = "Excellent"
        rel_detail = ("The instrument demonstrates **excellent internal consistency** (KR-20 ≥ 0.90). "
                      "All items appear to measure the same latent construct cohesively. "
                      "This level of reliability is suitable for **high-stakes decisions** such as certification or selection.")
    elif kr20 >= 0.80:
        rel_label = "High"
        rel_detail = ("The instrument shows **high reliability** (KR-20 0.80–0.89), indicating strong internal consistency. "
                      "Suitable for most formal assessments. A small subset of items may still benefit from revision "
                      "to push reliability toward the excellent threshold.")
    elif kr20 >= 0.70:
        rel_label = "Acceptable"
        rel_detail = ("Reliability is **acceptable** (KR-20 0.70–0.79), meeting the minimum standard for classroom research. "
                      "However, for high-stakes contexts, further item refinement is strongly recommended. "
                      "Focus revision efforts on items with low r_pbis and poor discrimination.")
    else:
        rel_label = "Low"
        rel_detail = ("Reliability is **low** (KR-20 < 0.70), suggesting significant item heterogeneity or measurement noise. "
                      "This instrument **should not be used for individual-level decisions** without major revision. "
                      "Consider removing or rewriting flagged items and increasing item count.")

    sem_detail = (f"The Standard Error of Measurement (SEM = {sem:.3f}) implies that a student's true score "
                  f"lies within ±{sem:.2f} score points of their observed score with ~68% confidence, "
                  f"or within ±{2*sem:.2f} with ~95% confidence. "
                  + ("This is an acceptably small margin." if sem < n_items * 0.1 else
                     "This margin is relatively large — score interpretations should be made cautiously."))

    return rel_label, rel_detail, sem_detail

def interpret_item_profile(df_res, n_items, validity_limit):
    n_retain = (df_res['DECISION'] == 'RETAIN').sum()
    n_revise = (df_res['DECISION'] == 'REVISE').sum()
    n_reject = (df_res['DECISION'] == 'REJECT').sum()
    pct_retain = n_retain / n_items * 100

    n_easy = (df_res['p'] > 0.70).sum()
    n_mod  = ((df_res['p'] >= 0.30) & (df_res['p'] <= 0.70)).sum()
    n_hard = (df_res['p'] < 0.30).sum()

    n_valid = (df_res['r_pbis'] >= validity_limit).sum()

    summary = (
        f"Out of **{n_items} items**, **{n_retain} ({pct_retain:.0f}%)** are recommended for RETAIN, "
        f"**{n_revise}** require REVISE, and **{n_reject}** should be REJECTED. "
        f"\n\n**Difficulty profile:** {n_easy} easy · {n_mod} moderate · {n_hard} difficult. "
        f"Optimal test design targets ≥ 50% of items in the moderate range for maximum score variance."
        f"\n\n**Validity:** {n_valid}/{n_items} items meet the r_pbis ≥ {validity_limit} threshold. "
        + ("The test has strong item validity overall." if n_valid/n_items >= 0.75 else
           "A significant portion of items are invalid — this impacts construct validity of the total score.")
    )
    return summary, n_retain, n_revise, n_reject, n_easy, n_mod, n_hard

def interpret_irt_params(a_arr, b_arr, c_arr, model):
    b_mean, b_std = b_arr.mean(), b_arr.std()
    items = []
    if model in ['2PL', '3PL']:
        a_mean = a_arr.mean()
        items.append(f"Mean discrimination (a̅) = {a_mean:.3f} — "
                     + ("items are highly discriminating." if a_mean >= 1.5 else
                        "items have moderate discrimination." if a_mean >= 0.8 else
                        "items have weak discrimination — review item construction."))
    items.append(f"Mean difficulty (b̅) = {b_mean:.3f}, SD = {b_std:.3f}. "
                 + ("Items are well-centered around the mean ability level (b̅ ≈ 0)." if abs(b_mean) < 0.3 else
                    f"Items tend to be {'easier' if b_mean < 0 else 'harder'} than average test-taker ability."))
    if model == '3PL':
        c_mean = c_arr.mean()
        items.append(f"Mean pseudo-guessing (c̅) = {c_mean:.3f}. "
                     + ("Guessing is negligible." if c_mean < 0.10 else
                        "Moderate guessing detected — consider lengthening stems or improving distractors." if c_mean < 0.25 else
                        "High guessing — items may lack sufficient distractors or are poorly written."))
    return "\n\n".join(items)

# ══════════════════════════════════════════════════════════════════════
# MATPLOTLIB FIGURE HELPERS
# ══════════════════════════════════════════════════════════════════════

def dark_fig(figsize=(10, 5)):
    fig = plt.figure(figsize=figsize, facecolor='#0d1117')
    return fig

def style_ax(ax, title='', xlabel='', ylabel=''):
    ax.set_facecolor('#161b22')
    ax.tick_params(colors='#8b949e', labelsize=8)
    ax.xaxis.label.set_color('#c9d1d9')
    ax.yaxis.label.set_color('#c9d1d9')
    ax.title.set_color('#e6edf3')
    for sp in ax.spines.values():
        sp.set_color('#30363d')
    if title: ax.set_title(title, fontsize=10, fontweight='bold', pad=8)
    if xlabel: ax.set_xlabel(xlabel, fontsize=8)
    if ylabel: ax.set_ylabel(ylabel, fontsize=8)
    ax.grid(True, color='#21262d', linewidth=0.5, alpha=0.7)

COLORS = {
    'blue': '#58a6ff', 'green': '#3fb950', 'red': '#f85149',
    'yellow': '#d29922', 'purple': '#bc8cff', 'cyan': '#39d353',
    'orange': '#ffa657', 'grey': '#8b949e'
}

# ══════════════════════════════════════════════════════════════════════
# MAIN ANALYSIS BLOCK
# ══════════════════════════════════════════════════════════════════════

if student_file and key_file:
    # ── Load Data ────────────────────────────────────────────────────
    df = pd.read_csv(student_file).fillna("N/A")
    df_key = pd.read_csv(key_file)

    item_cols  = df.columns[1:]
    id_col_name = df.columns[0]
    answer_key  = df_key.iloc[0, 1:].astype(str).str.upper().str.strip().tolist()

    # ── Score Matrix ─────────────────────────────────────────────────
    df_scores = pd.DataFrame()
    for i, col in enumerate(item_cols):
        df_scores[col] = (df[col].astype(str).str.upper().str.strip() == answer_key[i]).astype(int)

    total_scores = df_scores.sum(axis=1)
    df['Total_Score'] = total_scores
    n_students, n_items = len(df), len(item_cols)

    # ── Grouping (Kelley's Method) ────────────────────────────────────
    n_group = int(n_students * group_percent / 100)
    df_sorted = df.sort_values('Total_Score', ascending=False).copy()
    df_sorted['Rank'] = range(1, n_students + 1)
    df_sorted['Group'] = 'Middle'
    df_sorted.iloc[:n_group, df_sorted.columns.get_loc('Group')] = 'Upper'
    df_sorted.iloc[-n_group:, df_sorted.columns.get_loc('Group')] = 'Lower'
    df_ranking = df_sorted[[id_col_name, 'Total_Score', 'Rank', 'Group']].copy()
    up_idx = df_sorted.head(n_group).index
    lo_idx = df_sorted.tail(n_group).index

    # ══════════════════════════════════════════════════════════════════
    # CTT ITEM ANALYSIS
    # ══════════════════════════════════════════════════════════════════
    with st.spinner("⚙️ Running CTT analysis..."):
        results = []
        for i, item in enumerate(item_cols):
            p   = df_scores[item].mean()
            q   = 1 - p
            pq  = p * q
            item_var = df_scores[item].var(ddof=0)

            p_up = df_scores.loc[up_idx, item].mean()
            p_lo = df_scores.loc[lo_idx, item].mean()
            d_val = p_up - p_lo

            # All options chosen by students
            distractors = [
                opt.strip() for opt in df[item].dropna().astype(str).str.upper().unique()
                if opt.strip() not in ["", "N/A", answer_key[i]]
            ]

            ddi_vals = []
            for opt in distractors:
                u_opt = (df.loc[up_idx, item].astype(str).str.upper().str.strip() == opt).mean()
                l_opt = (df.loc[lo_idx, item].astype(str).str.upper().str.strip() == opt).mean()
                ddi_vals.append(l_opt - u_opt)

            ddi_best  = max(ddi_vals) if ddi_vals else 0
            ddi_worst = min(ddi_vals) if ddi_vals else 0

            # Corrected item-total correlation (r_pbis)
            corrected_total = total_scores - df_scores[item]
            if df_scores[item].var() != 0 and corrected_total.var() != 0:
                r_pb, _ = pointbiserialr(df_scores[item], corrected_total)
                r_pb = 0 if np.isnan(r_pb) else r_pb
            else:
                r_pb = 0

            # ── Descriptive Labels ──
            p_desc = "Easy" if p > 0.70 else "Difficult" if p < 0.30 else "Moderate"
            d_desc = ("Excellent" if d_val >= 0.40 else "Good" if d_val >= 0.30
                      else "Fair" if d_val >= 0.20 else "Poor")
            r_desc = "Valid" if r_pb >= validity_limit else "Invalid"

            # ── Decision Logic ──────────────────────────────────────
            reasons = []
            if r_pb < validity_limit:       reasons.append("Low validity (r_pbis)")
            if d_val < 0.20:                reasons.append("Poor discrimination (d < 0.20)")
            if p > 0.90:                    reasons.append("Too easy (p > 0.90)")
            if p < 0.10:                    reasons.append("Too difficult (p < 0.10)")
            if ddi_worst < -0.10:           reasons.append("Severely flawed distractor (DDI < −0.10)")
            elif ddi_worst < 0:             reasons.append("Malfunctioning distractor (DDI < 0)")

            # Primary decision:
            # RETAIN: meets all criteria
            # REJECT: fails validity AND discrimination, OR has severely flawed distractor
            # REVISE: borderline — at least one issue but not multiple severe failures
            if (r_pb >= validity_limit) and (d_val >= 0.30) and (ddi_worst >= 0):
                decision = "RETAIN"
            elif ((r_pb < validity_limit and d_val < 0.20) or ddi_worst < -0.10
                  or (p > 0.95) or (p < 0.05)):
                decision = "REJECT"
            else:
                decision = "REVISE"

            reason_text = ", ".join(reasons) if reasons else "All criteria satisfied"

            results.append({
                "Item": item,
                "p": p, "p_Eval": p_desc,
                "q": q, "pq": pq, "Var": item_var,
                "p_Upper": p_up, "p_Lower": p_lo,
                "d": d_val, "d_Eval": d_desc,
                "Best_DDI": ddi_best, "Worst_DDI": ddi_worst,
                "r_pbis": r_pb, "r_Eval": r_desc,
                "DECISION": decision,
                "REASON": reason_text
            })

    df_res = pd.DataFrame(results)

    # ── CTT Reliability Statistics ────────────────────────────────────
    mean_score = total_scores.mean()
    var_total  = total_scores.var(ddof=0)
    std_score  = np.sqrt(var_total)
    # KR-20 (Kuder-Richardson Formula 20) — exact formula
    kr20 = ((n_items / (n_items - 1)) * (1 - df_res["pq"].sum() / var_total)
            if var_total > 0 else 0)
    # Cronbach's Alpha (generalisation of KR-20 for polytomous; same here as items binary)
    alpha = ((n_items / (n_items - 1)) * (1 - df_res["Var"].sum() / var_total)
             if var_total > 0 else 0)
    sem = std_score * np.sqrt(1 - kr20)
    # Split-half reliability (corrected with Spearman-Brown)
    odd_scores  = df_scores.iloc[:, 0::2].sum(axis=1)
    even_scores = df_scores.iloc[:, 1::2].sum(axis=1)
    if odd_scores.var() > 0 and even_scores.var() > 0:
        r_half = np.corrcoef(odd_scores, even_scores)[0, 1]
        split_half = 2 * r_half / (1 + r_half)
    else:
        r_half = split_half = 0

    # ══════════════════════════════════════════════════════════════════
    # IRT ESTIMATION
    # ══════════════════════════════════════════════════════════════════
    X_binary = df_scores[list(item_cols)].values.astype(float)

    model_key = irt_model.split()[0]  # '1PL', '2PL', '3PL'

    with st.spinner(f"🔬 Fitting IRT {model_key} model via EM algorithm ({irt_max_iter} max iterations)..."):
        irt_params, theta_hat, log_lik = estimate_irt_em(
            X_binary, model=model_key, max_iter=irt_max_iter)

    a_arr = irt_params['a']
    b_arr = irt_params['b']
    c_arr = irt_params['c']

    # IRT reliability: separation reliability
    theta_var  = theta_hat.var()
    # Item information at each student's theta
    item_info_at_theta = np.array([
        compute_item_info(theta_hat, a_arr[j], b_arr[j], c_arr[j])
        for j in range(n_items)
    ])
    test_info_at_theta = item_info_at_theta.sum(axis=0)
    avg_info   = test_info_at_theta.mean()
    # Marginal reliability (Green, 1984)
    irt_rel    = avg_info * theta_var / (1 + avg_info * theta_var) if avg_info > 0 else 0

    # INFIT/OUTFIT (only meaningful for 1PL/Rasch)
    if model_key == '1PL':
        infit_arr, outfit_arr = rasch_fit_stats(X_binary, b_arr, theta_hat)
    else:
        infit_arr = outfit_arr = np.full(n_items, np.nan)

    # Add IRT columns to df_res
    df_res['IRT_b']    = b_arr
    df_res['IRT_a']    = a_arr
    df_res['IRT_c']    = c_arr
    df_res['IRT_INFIT'] = infit_arr
    df_res['IRT_OUTFIT'] = outfit_arr

    # Item information value at b (peak)
    df_res['Item_Info_Peak'] = [
        compute_item_info(np.array([b_arr[j]]), a_arr[j], b_arr[j], c_arr[j])[0]
        for j in range(n_items)
    ]

    # ══════════════════════════════════════════════════════════════════
    # DISPLAY — DASHBOARD
    # ══════════════════════════════════════════════════════════════════
    st.divider()
    st.markdown("## 📈 Test-Level Summary")

    # Row 1: basic metrics
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Students (N)", f"{n_students:,}")
    m2.metric("Items (k)", f"{n_items:,}")
    m3.metric("Mean Score", f"{mean_score:.2f}", delta=f"/{n_items} ({mean_score/n_items:.0%})")
    m4.metric("Std. Deviation", f"{std_score:.2f}")

    # Row 2: reliability metrics
    rel_label, rel_detail, sem_detail = interpret_reliability(kr20, sem, n_items)
    r1, r2, r3, r4, r5 = st.columns(5)
    r1.metric("KR-20", f"{kr20:.4f}", delta=rel_label)
    r2.metric("Cronbach's α", f"{alpha:.4f}")
    r3.metric("Split-Half (SB)", f"{split_half:.4f}",
              help="Spearman-Brown corrected odd/even split-half reliability.")
    r4.metric("SEM", f"{sem:.3f}",
              help="Standard Error of Measurement = SD × √(1 − KR20)")
    r5.metric(f"IRT Rel. ({model_key})", f"{irt_rel:.4f}",
              help="Marginal IRT reliability (Green 1984). Comparable to KR-20.")

    # ══════════════════════════════════════════════════════════════════
    # TABS
    # ══════════════════════════════════════════════════════════════════
    tab_ctt, tab_irt, tab_dist, tab_rank, tab_interp, tab_report = st.tabs([
        "📋 CTT Item Matrix",
        "🔬 IRT Analysis",
        "🎯 Distractor Analysis",
        "🏆 Student Ranking",
        "📝 Interpretive Report",
        "📥 Download"
    ])

    # ──────────────────────────────────────────────────────────────────
    # TAB 1: CTT ITEM MATRIX
    # ──────────────────────────────────────────────────────────────────
    with tab_ctt:
        st.markdown("### CTT Item Statistics Matrix")
        st.caption("Color coding: **Green** = good · **Yellow** = borderline · **Red** = problematic")

        # Distractor alert
        min_ddi = df_res["Worst_DDI"].min()
        if min_ddi < -0.10:
            st.error(f"⚠️ **Severe Distractor Alert:** Minimum DDI = {min_ddi:.4f}. "
                     "At least one distractor is more attractive to high-performing students — "
                     "indicating a potential keying error or misleading stem.")
        elif min_ddi < 0:
            st.warning(f"⚠️ **Malfunctioning Distractor:** Minimum DDI = {min_ddi:.4f}. "
                       "Some distractors attract more upper-group students than lower-group students.")
        else:
            st.success("✅ **All distractors functional.** No negative DDI detected.")

        display_cols = ["Item", "p", "p_Eval", "q", "pq", "Var",
                        "p_Upper", "p_Lower", "d", "d_Eval",
                        "Best_DDI", "Worst_DDI", "r_pbis", "r_Eval",
                        "DECISION", "REASON"]

        def apply_item_styling(row):
            styles = [''] * len(row)
            base = 'color: black;'
            # p columns (index 1,2)
            dif_color = ('#ccffcc' if row['p'] > 0.70 else
                         '#ffcccc' if row['p'] < 0.30 else '#fff2cc')
            styles[1] = styles[2] = f'background-color: {dif_color}; {base}'

            # d columns (index 8,9,10,11)
            if row['d'] >= 0.40:   dc, tc = '#2ecc71', 'white'
            elif row['d'] >= 0.30: dc, tc = '#3498db', 'white'
            elif row['d'] >= 0.20: dc, tc = '#f1c40f', 'black'
            else:                   dc, tc = '#e74c3c', 'white'
            for idx in [8, 9, 10, 11]:
                styles[idx] = f'background-color: {dc}; color: {tc}'

            # r_pbis (index 12,13)
            val_bg = '#ccffcc' if row['r_pbis'] >= validity_limit else '#ffcccc'
            styles[12] = f'background-color: {val_bg}; {base} font-weight: bold'
            styles[13] = f'background-color: {val_bg}; {base}'

            # DECISION (index 14)
            if row['DECISION'] == "RETAIN":
                styles[14] = 'background-color: #27ae60; color: white; font-weight: bold'
            elif row['DECISION'] == "REVISE":
                styles[14] = 'background-color: #f39c12; color: white; font-weight: bold'
            else:
                styles[14] = 'background-color: #c0392b; color: white; font-weight: bold'
            return styles

        st.dataframe(
            df_res[display_cols].style
                .apply(apply_item_styling, axis=1)
                .format("{:.4f}", subset=["p", "q", "pq", "Var", "p_Upper", "p_Lower",
                                          "d", "Best_DDI", "Worst_DDI", "r_pbis"]),
            use_container_width=True, height=500
        )

        # ── CTT Charts ────────────────────────────────────────────────
        st.markdown("---")
        c1, c2 = st.columns(2)

        with c1:
            fig = dark_fig((7, 4))
            ax = fig.add_subplot(111)
            colors_p = [COLORS['green'] if p > 0.70 else COLORS['red'] if p < 0.30
                        else COLORS['yellow'] for p in df_res['p']]
            bars = ax.bar(df_res['Item'], df_res['p'], color=colors_p, edgecolor='#30363d', linewidth=0.5)
            ax.axhline(0.70, color=COLORS['green'], linestyle='--', linewidth=1, alpha=0.6, label='Easy threshold (0.70)')
            ax.axhline(0.30, color=COLORS['red'],   linestyle='--', linewidth=1, alpha=0.6, label='Difficult threshold (0.30)')
            ax.set_ylim(0, 1.1)
            ax.legend(fontsize=7, facecolor='#21262d', labelcolor='#c9d1d9', framealpha=0.8)
            ax.tick_params(axis='x', rotation=45)
            style_ax(ax, 'Item Difficulty (p)', 'Item', 'Proportion Correct')
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        with c2:
            fig = dark_fig((7, 4))
            ax = fig.add_subplot(111)
            colors_d = [COLORS['green'] if d >= 0.40 else COLORS['blue'] if d >= 0.30
                        else COLORS['yellow'] if d >= 0.20 else COLORS['red']
                        for d in df_res['d']]
            ax.bar(df_res['Item'], df_res['d'], color=colors_d, edgecolor='#30363d', linewidth=0.5)
            ax.axhline(0.40, color=COLORS['green'],  linestyle='--', linewidth=1, alpha=0.6, label='Excellent (0.40)')
            ax.axhline(0.20, color=COLORS['yellow'], linestyle='--', linewidth=1, alpha=0.6, label='Fair/Poor boundary (0.20)')
            ax.axhline(0,    color=COLORS['red'],    linestyle='-',  linewidth=0.8, alpha=0.4)
            ax.legend(fontsize=7, facecolor='#21262d', labelcolor='#c9d1d9', framealpha=0.8)
            ax.tick_params(axis='x', rotation=45)
            style_ax(ax, 'Item Discrimination (d)', 'Item', 'Discrimination Index (d)')
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        c3, c4 = st.columns(2)
        with c3:
            fig = dark_fig((7, 4))
            ax = fig.add_subplot(111)
            colors_r = [COLORS['green'] if r >= validity_limit else COLORS['red'] for r in df_res['r_pbis']]
            ax.bar(df_res['Item'], df_res['r_pbis'], color=colors_r, edgecolor='#30363d', linewidth=0.5)
            ax.axhline(validity_limit, color=COLORS['yellow'], linestyle='--', linewidth=1,
                       label=f'Validity threshold ({validity_limit})')
            ax.axhline(0, color='#555', linewidth=0.8)
            ax.legend(fontsize=7, facecolor='#21262d', labelcolor='#c9d1d9', framealpha=0.8)
            ax.tick_params(axis='x', rotation=45)
            style_ax(ax, 'Point-Biserial Correlation (r_pbis)', 'Item', 'r_pbis (corrected)')
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        with c4:
            # Decision pie chart
            fig = dark_fig((5, 4))
            ax = fig.add_subplot(111)
            dec_counts = df_res['DECISION'].value_counts()
            pie_colors = {'RETAIN': '#27ae60', 'REVISE': '#f39c12', 'REJECT': '#c0392b'}
            clrs = [pie_colors.get(l, '#888') for l in dec_counts.index]
            wedges, texts, autotexts = ax.pie(
                dec_counts.values, labels=dec_counts.index,
                autopct='%1.0f%%', colors=clrs,
                textprops={'color': '#c9d1d9', 'fontsize': 9},
                wedgeprops={'edgecolor': '#0d1117', 'linewidth': 2}
            )
            for at in autotexts: at.set_fontsize(9); at.set_fontweight('bold')
            ax.set_facecolor('#0d1117')
            style_ax(ax, 'Item Decision Distribution')
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        # Score distribution histogram
        st.markdown("#### 📊 Score Distribution")
        fig = dark_fig((10, 3.5))
        ax = fig.add_subplot(111)
        ax.hist(total_scores, bins=min(n_items, 20), color=COLORS['blue'],
                edgecolor='#0d1117', linewidth=0.5, alpha=0.85, density=False)
        ax.axvline(mean_score, color=COLORS['orange'], linestyle='--', linewidth=1.5, label=f'Mean = {mean_score:.2f}')
        ax.axvline(mean_score - sem, color=COLORS['grey'], linestyle=':', linewidth=1, label=f'±SEM = ±{sem:.2f}')
        ax.axvline(mean_score + sem, color=COLORS['grey'], linestyle=':', linewidth=1)
        ax.legend(fontsize=8, facecolor='#21262d', labelcolor='#c9d1d9', framealpha=0.8)
        style_ax(ax, 'Raw Score Distribution', 'Total Score', 'Frequency')
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    # ──────────────────────────────────────────────────────────────────
    # TAB 2: IRT ANALYSIS
    # ──────────────────────────────────────────────────────────────────
    with tab_irt:
        st.markdown(f"### IRT Analysis — {irt_model}")

        i1, i2, i3, i4 = st.columns(4)
        i1.metric("Log-Likelihood", f"{log_lik:.2f}")
        i2.metric("IRT Reliability", f"{irt_rel:.4f}")
        i3.metric("Mean b (Difficulty)", f"{b_arr.mean():.3f}")
        i4.metric("Mean θ (Ability)", f"{theta_hat.mean():.3f}", delta=f"SD={theta_hat.std():.3f}")

        if model_key in ['2PL', '3PL']:
            ia1, ia2, ia3 = st.columns(3)
            ia1.metric("Mean a (Discrimination)", f"{a_arr.mean():.3f}")
            ia2.metric("Mean c (Pseudo-guess)", f"{c_arr.mean():.3f}")
            ia3.metric("AIC (approx)", f"{-2*log_lik + 2*(n_items*(1 if model_key=='1PL' else 2 if model_key=='2PL' else 3)):.0f}")

        st.markdown("---")

        # ── IRT Parameter Table ────────────────────────────────────────
        st.markdown("#### Item Parameter Estimates")
        irt_display = df_res[['Item', 'IRT_b', 'IRT_a', 'IRT_c', 'IRT_INFIT', 'IRT_OUTFIT', 'Item_Info_Peak']].copy()
        irt_display.columns = ['Item', 'b (Difficulty)', 'a (Discrimination)', 'c (Pseudo-guess)',
                                'INFIT MNSQ', 'OUTFIT MNSQ', 'Peak Info']

        def style_irt(row):
            styles = [''] * len(row)
            # b: easy < -1 (green), hard > 1 (red)
            b = row['b (Difficulty)']
            styles[1] = ('background-color:#ccffcc; color:black' if b < -1 else
                         'background-color:#ffcccc; color:black' if b > 1 else
                         'background-color:#fff2cc; color:black')
            # INFIT/OUTFIT: ideal 0.70-1.30
            for idx, col in [(4, 'INFIT MNSQ'), (5, 'OUTFIT MNSQ')]:
                v = row[col]
                if np.isnan(v): continue
                styles[idx] = ('background-color:#ccffcc; color:black' if 0.70 <= v <= 1.30 else
                                'background-color:#ffcccc; color:black')
            return styles

        st.dataframe(
            irt_display.style.apply(style_irt, axis=1)
                .format("{:.4f}", subset=['b (Difficulty)', 'a (Discrimination)', 'c (Pseudo-guess)', 'Peak Info'])
                .format(lambda x: f"{x:.4f}" if not np.isnan(x) else "N/A",
                        subset=['INFIT MNSQ', 'OUTFIT MNSQ']),
            use_container_width=True
        )

        if model_key == '1PL':
            st.caption("**INFIT/OUTFIT ideal range: 0.70–1.30** (Wright & Masters, 1982). "
                       "Values > 1.30 indicate unexpected responses (misfit); < 0.70 indicate overfitting (redundancy).")

        st.markdown("---")

        # ── ICC Curves ─────────────────────────────────────────────────
        st.markdown("#### Item Characteristic Curves (ICC)")
        theta_range = np.linspace(-4, 4, 300)

        n_cols_icc = min(4, n_items)
        n_rows_icc = int(np.ceil(n_items / n_cols_icc))
        fig = dark_fig((n_cols_icc * 3.5, n_rows_icc * 2.8))
        gs = GridSpec(n_rows_icc, n_cols_icc, figure=fig, hspace=0.45, wspace=0.35)

        for j, item in enumerate(item_cols):
            row_i, col_i = divmod(j, n_cols_icc)
            ax = fig.add_subplot(gs[row_i, col_i])
            P = irt_prob(theta_range, a_arr[j], b_arr[j], c_arr[j])
            ax.plot(theta_range, P, color=COLORS['blue'], linewidth=1.8)
            ax.axvline(b_arr[j], color=COLORS['orange'], linestyle='--', linewidth=1, alpha=0.8)
            ax.axhline(0.5, color=COLORS['grey'], linestyle=':', linewidth=0.7, alpha=0.5)
            if model_key == '3PL':
                ax.axhline(c_arr[j], color=COLORS['purple'], linestyle=':', linewidth=1, alpha=0.6)
            ax.set_ylim(-0.05, 1.10)
            ax.set_xlim(-4, 4)
            style_ax(ax, title=f"{item}\nb={b_arr[j]:.2f}, a={a_arr[j]:.2f}",
                     xlabel='θ (ability)', ylabel='P(correct)')
            ax.set_title(f"{item}  b={b_arr[j]:.2f} a={a_arr[j]:.2f}",
                         color='#e6edf3', fontsize=8.5, fontweight='bold', pad=4)

        fig.suptitle(f"Item Characteristic Curves — {irt_model}", color='#e6edf3',
                     fontsize=11, fontweight='bold', y=1.01)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        # ── Item & Test Information Functions ──────────────────────────
        st.markdown("#### Item Information Functions (IIF) & Test Information Function (TIF)")
        fig2 = dark_fig((12, 4.5))
        ax_left  = fig2.add_subplot(121)
        ax_right = fig2.add_subplot(122)

        tif_total = np.zeros(len(theta_range))
        palette = plt.cm.plasma(np.linspace(0.1, 0.9, n_items))
        for j, item in enumerate(item_cols):
            iif = compute_item_info(theta_range, a_arr[j], b_arr[j], c_arr[j])
            ax_left.plot(theta_range, iif, color=palette[j], linewidth=1.2, alpha=0.8, label=item)
            tif_total += iif

        style_ax(ax_left, 'Item Information Functions (IIF)', 'θ (Ability)', 'Information I(θ)')
        if n_items <= 15:
            ax_left.legend(fontsize=6, facecolor='#21262d', labelcolor='#c9d1d9',
                           framealpha=0.7, loc='upper right', ncol=2)

        ax_right.plot(theta_range, tif_total, color=COLORS['cyan'], linewidth=2.2)
        ax_right.fill_between(theta_range, tif_total, alpha=0.15, color=COLORS['cyan'])
        # SEM from IRT: SEM(θ) = 1/√I(θ)
        sem_irt = np.where(tif_total > 0, 1 / np.sqrt(tif_total), np.nan)
        ax_r2 = ax_right.twinx()
        ax_r2.plot(theta_range, sem_irt, color=COLORS['orange'], linewidth=1.5,
                   linestyle='--', alpha=0.8, label='SEM(θ)')
        ax_r2.set_ylabel('SEM(θ) = 1/√I(θ)', color=COLORS['orange'], fontsize=8)
        ax_r2.tick_params(colors=COLORS['orange'], labelsize=7)
        for sp in ax_r2.spines.values(): sp.set_color('#30363d')
        style_ax(ax_right, 'Test Information Function (TIF) + IRT-SEM', 'θ (Ability)', 'Total Information')

        fig2.tight_layout()
        st.pyplot(fig2, use_container_width=True)
        plt.close(fig2)

        # ── Wright Map ─────────────────────────────────────────────────
        st.markdown("#### Wright Map (Person-Item Map)")
        st.caption("Aligns student ability estimates (θ) with item difficulty (b) on the same logit scale.")

        fig3 = dark_fig((8, max(5, n_items * 0.5)))
        ax3 = fig3.add_subplot(111)

        # Person distribution (left)
        bins_theta = np.linspace(-4, 4, 25)
        hist_vals, bin_edges = np.histogram(theta_hat, bins=bins_theta)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        max_hist = max(hist_vals) if hist_vals.max() > 0 else 1
        ax3.barh(bin_centers, -hist_vals / max_hist * 1.5, height=0.25,
                 color=COLORS['blue'], alpha=0.6, label='Students (θ)')

        # Item b positions (right)
        for j, item in enumerate(item_cols):
            ax3.scatter(0.05, b_arr[j], color=COLORS['orange'], s=70, zorder=5)
            ax3.text(0.12, b_arr[j], f"  {item} (b={b_arr[j]:.2f})",
                     va='center', ha='left', fontsize=7.5, color='#c9d1d9')

        ax3.axvline(0, color='#30363d', linewidth=0.8)
        ax3.set_xlim(-2, 2)
        ax3.set_ylim(-4.2, 4.2)
        ax3.set_xlabel('← Persons (normalized count)   |   Item b →', color='#c9d1d9', fontsize=8)
        ax3.set_ylabel('Logit Scale (θ / b)', color='#c9d1d9', fontsize=8)

        p_patch = mpatches.Patch(color=COLORS['blue'], alpha=0.6, label='Students (θ distribution)')
        i_patch = mpatches.Patch(color=COLORS['orange'], label='Items (b parameter)')
        ax3.legend(handles=[p_patch, i_patch], fontsize=8,
                   facecolor='#21262d', labelcolor='#c9d1d9', framealpha=0.8)
        style_ax(ax3, f'Wright Map — {irt_model}')
        fig3.tight_layout()
        st.pyplot(fig3, use_container_width=True)
        plt.close(fig3)

        # ── Ability Estimates Table ────────────────────────────────────
        st.markdown("#### Student Ability Estimates (θ)")
        df_theta = pd.DataFrame({
            id_col_name: df[id_col_name].values,
            'Total_Score': total_scores.values,
            'θ (IRT Ability)': theta_hat.round(4),
            'SEM_θ': [1/np.sqrt(max(ti, 1e-9)) for ti in test_info_at_theta]
        }).sort_values('θ (IRT Ability)', ascending=False).reset_index(drop=True)
        df_theta['Rank'] = range(1, n_students + 1)

        st.dataframe(
            df_theta.style.background_gradient(subset=['θ (IRT Ability)'], cmap='Blues'),
            use_container_width=True, height=300
        )

    # ──────────────────────────────────────────────────────────────────
    # TAB 3: DISTRACTOR ANALYSIS
    # ──────────────────────────────────────────────────────────────────
    with tab_dist:
        st.markdown("### Distractor Effectiveness Analysis")
        st.caption("Proportion of students selecting each option. Effective distractors: chosen by ≥ 5% of students, "
                   "more frequently by lower-group students.")

        dist_data = []
        for item in item_cols:
            counts = df[item].astype(str).str.upper().str.strip().value_counts(normalize=True)
            row = counts.to_dict()
            row['Item'] = item
            dist_data.append(row)

        df_dist = pd.DataFrame(dist_data).set_index('Item').fillna(0)
        options_sorted = (sorted([c for c in df_dist.columns if len(str(c)) == 1]) +
                          sorted([c for c in df_dist.columns if len(str(c)) > 1]))
        df_dist = df_dist[[c for c in options_sorted if c in df_dist.columns]]

        # Flag key vs distractor
        header_labels = {}
        for i, item in enumerate(item_cols):
            for opt in df_dist.columns:
                header_labels[(item, opt)] = f"★ {opt}" if opt == answer_key[i] else opt

        df_dist_pct = df_dist.copy()
        for col in df_dist_pct.columns:
            df_dist_pct[col] = df_dist_pct[col].apply(lambda x: f"{x:.3f} ({x:.1%})")

        # Effectiveness tag
        def tag_effectiveness(row, item_idx):
            ak = answer_key[item_idx]
            tags = []
            for opt, val in row.items():
                if opt == ak: continue
                if val < 0.05: tags.append(f"⚠️ {opt} nonfunctional")
                elif val >= 0.05: tags.append(f"✅ {opt} effective")
            return " · ".join(tags) if tags else "—"

        df_dist_pct['Distractor Effectiveness'] = [
            tag_effectiveness(df_dist.iloc[i], i) for i in range(len(df_dist))
        ]

        st.dataframe(
            df_dist[df_dist.columns].style
                .format(lambda x: f"{x:.3f} ({x:.1%})")
                .background_gradient(cmap='YlGn', axis=1),
            use_container_width=True
        )

        st.markdown("---")
        st.markdown("#### Option Response Heatmap")
        fig = dark_fig((max(8, len(df_dist.columns) * 1.5), max(5, n_items * 0.55)))
        ax = fig.add_subplot(111)
        data_arr = df_dist.values
        im = ax.imshow(data_arr, cmap='YlGn', aspect='auto', vmin=0, vmax=1)
        ax.set_xticks(range(len(df_dist.columns)))
        ax.set_xticklabels(df_dist.columns, color='#c9d1d9', fontsize=9)
        ax.set_yticks(range(len(item_cols)))
        ax.set_yticklabels(list(item_cols), color='#c9d1d9', fontsize=8)
        for i in range(n_items):
            for j2, col in enumerate(df_dist.columns):
                val = data_arr[i, j2]
                ax.text(j2, i, f"{val:.2f}", ha='center', va='center',
                        fontsize=7, color='black' if val > 0.4 else '#c9d1d9')
        plt.colorbar(im, ax=ax, label='Proportion Selected')
        style_ax(ax, 'Option Selection Heatmap (proportion per item)')
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    # ──────────────────────────────────────────────────────────────────
    # TAB 4: STUDENT RANKING
    # ──────────────────────────────────────────────────────────────────
    with tab_rank:
        st.markdown("### Student Score Ranking & Group Assignment")
        st.caption(f"Grouping: Top {group_percent}% = **Upper Group** · Bottom {group_percent}% = **Lower Group** "
                   f"(Kelley's {group_percent}% criterion, n={n_group} per group)")

        col_r1, col_r2, col_r3 = st.columns(3)
        col_r1.metric("Upper Group (n)", n_group)
        col_r2.metric("Middle Group (n)", n_students - 2 * n_group)
        col_r3.metric("Lower Group (n)", n_group)

        def apply_rank_styling(row):
            if row['Group'] == 'Upper': bg = '#1a4731'
            elif row['Group'] == 'Lower': bg = '#3d1212'
            else: bg = '#161b22'
            return [f'background-color: {bg}; color: #e6edf3'] * len(row)

        # Add theta to ranking
        theta_dict = dict(zip(df[id_col_name].values, theta_hat))
        df_ranking_display = df_ranking.copy()
        df_ranking_display['θ (IRT)'] = df_ranking_display[id_col_name].map(theta_dict).round(4)

        st.dataframe(
            df_ranking_display.style.apply(apply_rank_styling, axis=1),
            use_container_width=True, height=500
        )

    # ──────────────────────────────────────────────────────────────────
    # TAB 5: INTERPRETIVE REPORT
    # ──────────────────────────────────────────────────────────────────
    with tab_interp:
        st.markdown("### 📝 Automated Interpretive Report")
        st.caption("Narrative interpretations generated from actual data — for use in research reports and theses.")

        # ── Section 1: Overview ──────────────────────────────────────
        st.markdown("#### 1. Test Administration Overview")
        st.info(
            f"This item analysis covers a test administered to **{n_students} students** across **{n_items} items**. "
            f"The mean raw score was **{mean_score:.2f}** (SD = {std_score:.2f}), representing "
            f"**{mean_score/n_items:.1%}** of the maximum possible score. "
            f"Score distribution ranged from {total_scores.min()} to {total_scores.max()} points."
        )

        # ── Section 2: Reliability ───────────────────────────────────
        st.markdown("#### 2. Reliability Analysis")
        st.markdown(rel_detail)
        st.markdown(sem_detail)
        st.markdown(
            f"**Cross-validation:** Cronbach's α = {alpha:.4f} (consistent with KR-20 = {kr20:.4f}, "
            f"as expected for dichotomous items). "
            f"The Spearman-Brown corrected split-half coefficient = {split_half:.4f}, "
            + ("further confirming internal consistency." if abs(split_half - kr20) < 0.10 else
               "showing some discrepancy — consider reviewing item placement order.")
        )

        # ── Section 3: Item Profile ──────────────────────────────────
        st.markdown("#### 3. Item Difficulty & Discrimination Profile")
        item_summary, n_retain, n_revise, n_reject, n_easy, n_mod, n_hard = interpret_item_profile(
            df_res, n_items, validity_limit)
        st.markdown(item_summary)

        # Specific flagged items
        reject_items = df_res[df_res['DECISION'] == 'REJECT']['Item'].tolist()
        revise_items = df_res[df_res['DECISION'] == 'REVISE']['Item'].tolist()
        if reject_items:
            st.error(f"**Items recommended for REJECTION ({len(reject_items)}):** {', '.join(reject_items)}  \n"
                     "These items should be removed from the test or completely rewritten before future use.")
        if revise_items:
            st.warning(f"**Items recommended for REVISION ({len(revise_items)}):** {', '.join(revise_items)}  \n"
                       "Review stems, distractors, or answer keys. Moderate issues detected.")

        # ── Section 4: IRT ───────────────────────────────────────────
        st.markdown(f"#### 4. IRT Analysis ({irt_model})")
        irt_interp = interpret_irt_params(a_arr, b_arr, c_arr, model_key)
        st.markdown(irt_interp)
        st.markdown(
            f"The IRT marginal reliability coefficient = **{irt_rel:.4f}**, "
            + ("consistent with the KR-20, validating internal consistency from both frameworks." 
               if abs(irt_rel - kr20) < 0.10 else
               f"which differs from KR-20 ({kr20:.4f}) — this may reflect non-normality of the latent trait distribution.")
        )

        # Theta vs raw score correlation
        r_theta_raw = np.corrcoef(theta_hat, total_scores)[0, 1]
        st.markdown(
            f"Person ability estimates (θ) correlate with raw scores at r = **{r_theta_raw:.4f}**, "
            + ("confirming strong monotonic alignment between CTT and IRT orderings." if r_theta_raw > 0.95 else
               "indicating some re-ranking of students by IRT — this is common when items vary substantially in discrimination.")
        )

        if model_key == '1PL':
            poor_fit = df_res[(~df_res['IRT_INFIT'].isna()) & ((df_res['IRT_INFIT'] > 1.30) | (df_res['IRT_INFIT'] < 0.70))]
            if len(poor_fit) > 0:
                st.warning(f"**Rasch Fit:** {len(poor_fit)} item(s) showed INFIT MNSQ outside the 0.70–1.30 ideal range: "
                           f"{', '.join(poor_fit['Item'].tolist())}. These items violate the Rasch model assumption of equal discrimination.")

        # ── Section 5: Distractor ────────────────────────────────────
        st.markdown("#### 5. Distractor Functionality")
        n_neg_ddi = (df_res['Worst_DDI'] < 0).sum()
        n_severe  = (df_res['Worst_DDI'] < -0.10).sum()
        if n_severe > 0:
            st.error(f"{n_severe} item(s) have severely malfunctioning distractors (DDI < −0.10). "
                     "These distractors attract more high-ability than low-ability students, "
                     "suggesting possible keying errors or ambiguous options.")
        elif n_neg_ddi > 0:
            st.warning(f"{n_neg_ddi} item(s) have at least one negative DDI distractor. "
                       "While not severe, these distractors should be reviewed for clarity.")
        else:
            st.success("All distractors are functioning as intended — "
                       "more lower-group students selected each distractor than upper-group students.")

        # ── Section 6: Recommendations ───────────────────────────────
        st.markdown("#### 6. Recommendations")
        recommendations = []
        if kr20 < 0.70:
            recommendations.append("🔴 **Add more items** to increase reliability. Reliability below 0.70 is insufficient for individual assessment.")
        if n_reject > 0:
            recommendations.append(f"🔴 **Remove or rewrite {n_reject} rejected item(s)** before reuse.")
        if n_revise > 0:
            recommendations.append(f"🟡 **Revise {n_revise} item(s)** — check stems, distractors, and keying.")
        if n_hard > n_items * 0.3:
            recommendations.append("🟡 **Too many difficult items** — consider item scaffolding or adjusting cognitive level.")
        if n_easy > n_items * 0.4:
            recommendations.append("🟡 **Too many easy items** — score ceiling effect may reduce discrimination.")
        if model_key == '1PL' and len(poor_fit) > 0:
            recommendations.append("🟡 **Rasch model fit issues detected** — consider 2PL model or revise misfitting items.")
        if model_key == '3PL' and c_arr.mean() > 0.20:
            recommendations.append("🟡 **High average pseudo-guessing (c > 0.20)** — improve distractor quality or add more options.")
        if not recommendations:
            recommendations.append("🟢 **No critical issues detected.** The test meets psychometric standards for its intended purpose.")

        for rec in recommendations:
            st.markdown(rec)

    # ──────────────────────────────────────────────────────────────────
    # TAB 6: DOWNLOAD
    # ──────────────────────────────────────────────────────────────────
    with tab_report:
        st.markdown("### 📥 Export Full Report")

        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine='xlsxwriter') as writer:
            wb = writer.book

            # ── Formats ───────────────────────────────────────────────
            hdr_fmt = wb.add_format({'bold': True, 'bg_color': '#1f3864', 'font_color': 'white',
                                     'border': 1, 'align': 'center', 'valign': 'vcenter'})
            retain_fmt = wb.add_format({'bg_color': '#c6efce', 'font_color': '#276221', 'bold': True, 'border': 1})
            revise_fmt = wb.add_format({'bg_color': '#ffeb9c', 'font_color': '#9c6500', 'bold': True, 'border': 1})
            reject_fmt = wb.add_format({'bg_color': '#ffc7ce', 'font_color': '#9c0006', 'bold': True, 'border': 1})
            num_fmt    = wb.add_format({'num_format': '0.0000', 'border': 1, 'align': 'center'})
            txt_fmt    = wb.add_format({'border': 1})
            upper_fmt  = wb.add_format({'bg_color': '#c6efce', 'border': 1})
            lower_fmt  = wb.add_format({'bg_color': '#ffc7ce', 'border': 1})
            mid_fmt    = wb.add_format({'border': 1})
            title_fmt  = wb.add_format({'bold': True, 'font_size': 14, 'font_color': '#1f3864'})
            sub_fmt    = wb.add_format({'bold': True, 'font_size': 11})
            body_fmt   = wb.add_format({'text_wrap': True, 'valign': 'top'})

            # ── Sheet 1: CTT Item Analysis ─────────────────────────────
            ctt_cols = ["Item", "p", "p_Eval", "q", "pq", "Var",
                        "p_Upper", "p_Lower", "d", "d_Eval",
                        "Best_DDI", "Worst_DDI", "r_pbis", "r_Eval",
                        "DECISION", "REASON"]
            df_res[ctt_cols].to_excel(writer, index=False, sheet_name='CTT_Item_Analysis', startrow=1)
            ws1 = writer.sheets['CTT_Item_Analysis']
            ws1.write(0, 0, 'CTT Item Analysis Report', title_fmt)
            for col_i, col_name in enumerate(ctt_cols):
                ws1.write(1, col_i, col_name, hdr_fmt)
            for row_i, row in df_res[ctt_cols].iterrows():
                for col_i, col_name in enumerate(ctt_cols):
                    val = row[col_name]
                    if col_name == 'DECISION':
                        fmt = retain_fmt if val == 'RETAIN' else revise_fmt if val == 'REVISE' else reject_fmt
                        ws1.write(row_i + 2, col_i, val, fmt)
                    elif isinstance(val, float):
                        ws1.write(row_i + 2, col_i, val, num_fmt)
                    else:
                        ws1.write(row_i + 2, col_i, val, txt_fmt)
            ws1.set_column('A:A', 12)
            ws1.set_column('B:N', 12)
            ws1.set_column('O:O', 10)
            ws1.set_column('P:P', 40)

            # ── Sheet 2: IRT Parameters ────────────────────────────────
            irt_export = df_res[['Item', 'IRT_b', 'IRT_a', 'IRT_c',
                                  'IRT_INFIT', 'IRT_OUTFIT', 'Item_Info_Peak']].copy()
            irt_export.columns = ['Item', 'b (Difficulty)', 'a (Discrimination)', 'c (Pseudo-guess)',
                                   'INFIT MNSQ', 'OUTFIT MNSQ', 'Peak Information']
            irt_export.to_excel(writer, index=False, sheet_name='IRT_Parameters', startrow=1)
            ws2 = writer.sheets['IRT_Parameters']
            ws2.write(0, 0, f'IRT Parameter Estimates — {irt_model}', title_fmt)
            for col_i, cn in enumerate(irt_export.columns):
                ws2.write(1, col_i, cn, hdr_fmt)
            ws2.set_column('A:G', 18)

            # ── Sheet 3: Student Ranking + Theta ──────────────────────
            df_theta_export = df_ranking.copy()
            df_theta_export['θ (IRT Ability)'] = df_ranking[id_col_name].map(
                dict(zip(df[id_col_name].values, theta_hat))).round(4)
            df_theta_export.to_excel(writer, index=False, sheet_name='Student_Ranking', startrow=1)
            ws3 = writer.sheets['Student_Ranking']
            ws3.write(0, 0, 'Student Score Ranking & IRT Ability', title_fmt)
            for col_i, cn in enumerate(df_theta_export.columns):
                ws3.write(1, col_i, cn, hdr_fmt)
            for row_i, row in df_theta_export.iterrows():
                grp = row.get('Group', '')
                fmt = upper_fmt if grp == 'Upper' else lower_fmt if grp == 'Lower' else mid_fmt
                for col_i, val in enumerate(row):
                    ws3.write(row_i + 2, col_i, val, fmt)
            ws3.set_column('A:E', 18)

            # ── Sheet 4: Distractor Analysis ──────────────────────────
            df_dist_pct.to_excel(writer, index=True, sheet_name='Distractor_Analysis', startrow=1)
            ws4 = writer.sheets['Distractor_Analysis']
            ws4.write(0, 0, 'Distractor Effectiveness (Proportion Selected)', title_fmt)
            ws4.set_column('A:Z', 16)

            # ── Sheet 5: Reliability & Interpretation ─────────────────
            rel_label_val, _, _ = interpret_reliability(kr20, sem, n_items)
            reliability_interpretation = pd.DataFrame({
                "Metric": ["N (Students)", "k (Items)", "Mean", "SD", "KR-20", "Alpha",
                           "Split-Half (SB)", "SEM", "IRT Reliability", "Log-Likelihood",
                           "KR-20 Interpretation", "SEM Interpretation"],
                "Value": [n_students, n_items, f"{mean_score:.2f}", f"{std_score:.2f}",
                          f"{kr20:.4f}", f"{alpha:.4f}", f"{split_half:.4f}",
                          f"{sem:.4f}", f"{irt_rel:.4f}", f"{log_lik:.2f}",
                          f"{rel_label_val} reliability",
                          f"±{sem:.4f} (±{2*sem:.4f} at 95% CI)"],
                "Interpretation": [
                    "Total number of test takers",
                    "Total number of items in the test",
                    "Average raw score across all students",
                    "Spread of scores around the mean",
                    f"Internal consistency: {rel_label_val}. Benchmark: Excellent ≥0.90, High 0.80–0.89, Acceptable 0.70–0.79",
                    "Cronbach's generalization of KR-20 (identical for binary items)",
                    "Spearman-Brown corrected odd/even split. Cross-validates KR-20.",
                    "Standard error of measurement. Lower = more precise score estimates.",
                    f"Marginal IRT reliability ({irt_model}). Comparable to KR-20.",
                    "Marginal log-likelihood from EM estimation. Higher (less negative) = better fit.",
                    rel_detail.replace("**", "").replace("\n\n", " "),
                    sem_detail.replace("**", "")
                ]
            })
            reliability_interpretation.to_excel(writer, index=False,
                                                 sheet_name='Reliability_Interpretation', startrow=1)
            ws5 = writer.sheets['Reliability_Interpretation']
            ws5.write(0, 0, 'Reliability Summary & Interpretive Report', title_fmt)
            for col_i, cn in enumerate(reliability_interpretation.columns):
                ws5.write(1, col_i, cn, hdr_fmt)
            ws5.set_column('A:A', 22)
            ws5.set_column('B:B', 20)
            ws5.set_column('C:C', 80)

        st.download_button(
            label="📥 Download Complete Item Analysis Report (Excel)",
            data=buf.getvalue(),
            file_name="Item_Analysis_Pro_Report.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
        st.success("✅ Report includes: CTT Matrix · IRT Parameters · Wright Map Data · Student Rankings + θ · Distractor Analysis · Reliability Interpretation")

else:
    # ── Welcome / Instructions ──────────────────────────────────────────
    st.markdown("### Getting Started")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
**Step 1: Prepare Student CSV**, use the following format:
```
StudentID, Q1, Q2, Q3
S001,       A,  C,  B
S002,       B,  C,  A
S003,       A,  A,  B
```
*First column = ID. Rest = responses.*
        """)
    with col2:
        st.markdown("""
**Step 2: Prepare Key CSV**
```
Key, Q1, Q2, Q3
ANS, A,  C,  A
```
*First column = label. Rest = correct answers.*
        """)
    with col3:
        st.markdown("""
**Step 3: Configure & Analyze**
- Set Kelley's % (default 27%)
- Set r_pbis threshold (default 0.25)
- Choose IRT model (1PL/2PL/3PL)
- Upload files → results appear instantly
        """)

    st.info("📌 **Note:** IRT estimation uses an EM algorithm with Gauss-Hermite quadrature. "
            "For small samples (N < 100), use 1PL (Rasch) for stability. "
            "2PL/3PL recommended for N ≥ 200.")
