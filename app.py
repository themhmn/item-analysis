import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import pointbiserialr
import io

# ===============================
# ITEM ANALYSIS PRO ULTRA (CTT + IRT LIGHT)
# ===============================

st.set_page_config(page_title="Item Analysis Pro ULTRA", page_icon="📈", layout="wide")

st.title("ITEM ANALYSIS TOOL (CTT + IRT ENHANCED)")
st.write("Advanced psychometric analysis: Classical Test Theory + IRT approximation")

with st.sidebar:
    st.header("Settings")
    group_percent = st.slider("Kelley Group (%)", 10, 50, 27)
    validity_limit = st.number_input("r_pb Threshold", 0.0, 1.0, 0.25)

u1, u2 = st.columns(2)
with u1:
    student_file = st.file_uploader("Student CSV", type=["csv"])
with u2:
    key_file = st.file_uploader("Answer Key CSV", type=["csv"])

if student_file and key_file:

    df = pd.read_csv(student_file).fillna("N/A")
    df_key = pd.read_csv(key_file)

    item_cols = df.columns[1:]
    id_col = df.columns[0]

    key = df_key.iloc[0, 1:].astype(str).str.upper().str.strip().tolist()

    # ===============================
    # SCORE MATRIX
    # ===============================
    df_scores = pd.DataFrame()
    for i, col in enumerate(item_cols):
        df_scores[col] = (df[col].astype(str).str.upper().str.strip() == key[i]).astype(int)

    total = df_scores.sum(axis=1)
    df["Total"] = total

    n, k = len(df), len(item_cols)

    # ===============================
    # GROUPING (UPPER-LOWER)
    # ===============================
    g = max(1, int(np.ceil(n * group_percent / 100)))
    df_sorted = df.sort_values("Total", ascending=False).copy()

    df_sorted["Rank"] = range(1, n + 1)
    df_sorted["Group"] = "Middle"
    df_sorted.iloc[:g, df_sorted.columns.get_loc("Group")] = "Upper"
    df_sorted.iloc[-g:, df_sorted.columns.get_loc("Group")] = "Lower"

    upper = df_sorted.head(g).index
    lower = df_sorted.tail(g).index

    results = []

    # ===============================
    # ITEM ANALYSIS CORE LOOP
    # ===============================
    for i, item in enumerate(item_cols):

        score = df_scores[item]

        p = score.mean()
        q = 1 - p
        pq = p * q
        var = score.var(ddof=0)

        # discrimination
        p_u = df_scores.loc[upper, item].mean()
        p_l = df_scores.loc[lower, item].mean()
        d = p_u - p_l

        # distractor analysis
        distractors = []
        for opt in df[item].astype(str).str.upper().unique():
            if opt not in ["", "N/A", key[i]]:
                distractors.append(opt.strip())

        ddi = []
        for opt in distractors:
            u = (df.loc[upper, item].astype(str).str.upper().str.strip() == opt).mean()
            l = (df.loc[lower, item].astype(str).str.upper().str.strip() == opt).mean()
            ddi.append(l - u)

        best_ddi = max(ddi) if ddi else 0
        worst_ddi = min(ddi) if ddi else 0

        # point biserial
        corrected = total - score
        if score.var() != 0:
            r_pb, _ = pointbiserialr(score, corrected)
            r_pb = 0 if np.isnan(r_pb) else r_pb
        else:
            r_pb = 0

        # ===============================
        # 🔥 IRT APPROXIMATION (NEW)
        # ===============================
        theta = (total - total.mean()) / (total.std() + 1e-9)

        a = max(0.5, min(2.5, abs(d) * 3))   # discrimination proxy
        b = np.log((1 - p + 1e-9) / (p + 1e-9))  # difficulty proxy

        p_irt = 1 / (1 + np.exp(-a * (theta.mean() - b)))

        item_info = a**2 * p * (1 - p)
        sem_item = np.sqrt(1 / (item_info + 1e-9))

        # ===============================
        # CLASSICAL LABELS
        # ===============================
        p_lab = "Easy" if p > 0.7 else "Difficult" if p < 0.3 else "Moderate"
        d_lab = "Excellent" if d >= 0.4 else "Good" if d >= 0.3 else "Fair" if d >= 0.2 else "Poor"

        # ===============================
        # FINAL DECISION ENGINE (ENHANCED)
        # ===============================
        issues = []

        if r_pb < validity_limit:
            issues.append("Low validity")

        if d < 0.2:
            issues.append("Low discrimination")

        if p > 0.9:
            issues.append("Too easy")

        if p < 0.2:
            issues.append("Too difficult")

        if worst_ddi < 0:
            issues.append("Bad distractor")

        if item_info < 0.15:
            issues.append("Low information (IRT)")

        if sem_item > 1.0:
            issues.append("High measurement error")

        if (r_pb >= validity_limit) and (d >= 0.3) and (worst_ddi >= 0) and (item_info >= 0.2):
            decision = "RETAIN"
        elif len(issues) >= 3:
            decision = "REJECT"
        else:
            decision = "REVISE"

        # ===============================
        # OUTPUT
        # ===============================
        results.append({
            "Item": item,
            "p": p,
            "p_label": p_lab,
            "q": q,
            "pq": pq,
            "d": d,
            "DDI_Worst": worst_ddi,
            "r_pb": r_pb,

            # IRT ENHANCED
            "IRT_a": a,
            "IRT_b": b,
            "IRT_p": p_irt,
            "Item_Info": item_info,
            "SEM_Item": sem_item,

            "Decision": decision,
            "Issues": ", ".join(issues) if issues else "OK"
        })

    df_res = pd.DataFrame(results)

    # ===============================
    # METRICS (ENHANCED)
    # ===============================
    mean = total.mean()
    var_t = total.var(ddof=0)
    sd = np.sqrt(var_t)

    kr20 = (k/(k-1)) * (1 - df_res["pq"].sum()/var_t) if var_t > 0 else 0
    alpha = (k/(k-1)) * (1 - df_res["pq"].sum()/var_t) if var_t > 0 else 0

    sem = sd * np.sqrt(1 - kr20)

    # ===============================
    # OUTPUT UI
    # ===============================
    st.metric("KR-20", round(kr20, 4))
    st.metric("Alpha", round(alpha, 4))
    st.metric("SEM", round(sem, 4))

    st.subheader("Item Table (CTT + IRT)")
    st.dataframe(df_res)

    # ===============================
    # DOWNLOAD
    # ===============================
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter") as w:
        df_res.to_excel(w, index=False)

    st.download_button(
        "Download Report",
        buf.getvalue(),
        file_name="ITEM_ANALYSIS_ULTRA.xlsx"
    )
