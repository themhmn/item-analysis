import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import pointbiserialr
import io

# =========================================================
# ITEM ANALYSIS PRO (CTT - IMPROVED VERSION)
# =========================================================

st.set_page_config(page_title="Item Analysis Pro", page_icon="📈", layout="wide")

st.markdown("""
<style>
.main { background-color: #f4f4f9; }
[data-testid="stMetricValue"] { font-family: monospace; }
</style>
""", unsafe_allow_html=True)

st.title("ITEM ANALYSIS TOOL (CTT - IMPROVED)")
st.write("Classical Test Theory with improved statistical rigor.")

with st.sidebar:
    st.header("Settings")
    group_percent = st.slider("Kelley Group (%)", 10, 50, 27)
    validity_limit = st.number_input("r_pb threshold", 0.0, 1.0, 0.25)

col1, col2 = st.columns(2)
student_file = col1.file_uploader("Student CSV", type="csv")
key_file = col2.file_uploader("Answer Key CSV", type="csv")


# =========================
# MAIN PROCESS
# =========================
if student_file and key_file:

    df = pd.read_csv(student_file).fillna("")
    df_key = pd.read_csv(key_file)

    id_col = df.columns[0]
    item_cols = df.columns[1:]

    # CLEAN ANSWERS
    df_clean = df.copy()
    for c in item_cols:
        df_clean[c] = df_clean[c].astype(str).str.upper().str.strip()

    answer_key = (
        df_key.iloc[0, 1:]
        .astype(str).str.upper().str.strip()
        .tolist()
    )

    # =========================
    # SCORING (0/1)
    # =========================
    df_scores = pd.DataFrame()

    for i, col in enumerate(item_cols):
        df_scores[col] = (df_clean[col] == answer_key[i]).astype(int)

    total_scores = df_scores.sum(axis=1)
    df["Total"] = total_scores

    n_students = len(df)
    n_items = len(item_cols)

    # =========================
    # GROUPING (KELLEY METHOD)
    # =========================
    n_group = max(1, int(np.ceil(n_students * group_percent / 100)))
    df_sorted = df.copy().sort_values("Total", ascending=False)

    df_sorted["Group"] = "Middle"
    df_sorted.iloc[:n_group, df_sorted.columns.get_loc("Group")] = "Upper"
    df_sorted.iloc[-n_group:, df_sorted.columns.get_loc("Group")] = "Lower"

    up_idx = df_sorted.index[:n_group]
    lo_idx = df_sorted.index[-n_group:]


    results = []

    # =========================
    # ITEM ANALYSIS CORE
    # =========================
    for i, item in enumerate(item_cols):

        item_score = df_scores[item]

        p = item_score.mean()
        q = 1 - p
        pq = p * q
        var_item = item_score.var(ddof=1)

        # difficulty group comparison
        p_up = df_scores.loc[up_idx, item].mean()
        p_lo = df_scores.loc[lo_idx, item].mean()
        d_index = p_up - p_lo

        # =========================
        # DISTRACTOR ANALYSIS (FIXED)
        # =========================
        all_options = df_clean[item].unique()
        correct_answer = answer_key[i]

        distractors = [
            opt for opt in all_options
            if opt not in ["", "N/A", correct_answer]
        ]

        ddi_vals = []

        for opt in distractors:

            up_prop = (df_clean.loc[up_idx, item] == opt).mean()
            lo_prop = (df_clean.loc[lo_idx, item] == opt).mean()

            # IMPORTANT: Lower - Upper (true distractor power)
            ddi_vals.append(lo_prop - up_prop)

        worst_ddi = min(ddi_vals) if ddi_vals else 0
        best_ddi = max(ddi_vals) if ddi_vals else 0

        # =========================
        # POINT BISERIAL (ROBUST)
        # =========================
        try:
            corrected_total = total_scores - item_score
            if item_score.std() == 0:
                r_pb = 0
            else:
                r_pb, _ = pointbiserialr(item_score, corrected_total)
                if np.isnan(r_pb):
                    r_pb = 0
        except:
            r_pb = 0

        # =========================
        # INTERPRETATION
        # =========================
        if p > 0.7:
            p_desc = "Easy"
        elif p < 0.3:
            p_desc = "Difficult"
        else:
            p_desc = "Moderate"

        if d_index >= 0.4:
            d_desc = "Excellent"
        elif d_index >= 0.3:
            d_desc = "Good"
        elif d_index >= 0.2:
            d_desc = "Fair"
        else:
            d_desc = "Poor"

        r_desc = "Valid" if r_pb >= validity_limit else "Invalid"

        # =========================
        # DECISION SYSTEM (FIXED LOGIC)
        # =========================
        reasons = []

        if r_pb < validity_limit:
            reasons.append("Low validity")

        if d_index < 0.20:
            reasons.append("Low discrimination")

        if p > 0.90:
            reasons.append("Too easy")

        if p < 0.20:
            reasons.append("Too difficult")

        if worst_ddi < -0.10:
            reasons.append("Broken distractor")

        if worst_ddi < 0:
            reasons.append("Weak distractor")

        if (r_pb >= validity_limit) and (d_index >= 0.30) and (worst_ddi >= 0):
            decision = "RETAIN"
        elif (r_pb < validity_limit and d_index < 0.20) or (worst_ddi < -0.10):
            decision = "REJECT"
        else:
            decision = "REVISE"

        results.append({
            "Item": item,
            "p": p,
            "q": q,
            "p*q": pq,
            "Var": var_item,
            "Discrimination": d_index,
            "Distractor_Best": best_ddi,
            "Distractor_Worst": worst_ddi,
            "r_pb": r_pb,
            "Decision": decision,
            "Reason": ", ".join(reasons) if reasons else "OK"
        })

    df_res = pd.DataFrame(results)

    # =========================
    # RELIABILITY
    # =========================
    total_var = total_scores.var(ddof=1)

    kr20 = (
        (n_items / (n_items - 1)) *
        (1 - df_res["p*q"].sum() / total_var)
        if total_var > 0 else 0
    )

    alpha = (
        (n_items / (n_items - 1)) *
        (1 - df_res["Var"].sum() / total_var)
        if total_var > 0 else 0
    )

    sem = np.sqrt(total_var) * np.sqrt(1 - kr20)

    # =========================
    # OUTPUT
    # =========================
    st.subheader("Summary Metrics")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("N Students", n_students)
    c2.metric("Items", n_items)
    c3.metric("KR-20", round(kr20, 4))
    c4.metric("Alpha", round(alpha, 4))

    st.subheader("Item Analysis Table")
    st.dataframe(df_res)

    st.subheader("Ranking")
    st.dataframe(df_sorted[[id_col, "Total", "Group"]])

    # =========================
    # EXPORT
    # =========================
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
        df_res.to_excel(writer, index=False, sheet_name="Item_Analysis")
        df_sorted.to_excel(writer, index=False, sheet_name="Ranking")

    st.download_button(
        "Download Report",
        buffer.getvalue(),
        file_name="item_analysis.xlsx"
    )
