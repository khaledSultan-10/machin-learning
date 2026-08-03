import pandas as pd
import numpy as np
import streamlit as st
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, precision_recall_curve
)

# ==========================================
# 1. إعدادات الصفحة
# ==========================================
st.set_page_config(
    page_title="Bank Marketing Dashboard",
    page_icon="🏦",
    layout="wide"
)

# ==========================================
# 2. دوال تحميل البيانات والتجهيز (Cached)
# ==========================================
@st.cache_data
def load_data():
    # يبحث عن الملف بجانب app.py مهما كان مجلد التشغيل
    base_dir = Path(__file__).parent
    csv_path = base_dir / "bank-additional-full.csv"
    df = pd.read_csv(csv_path, sep=";")
    return df


@st.cache_data
def get_preprocessed_data():
    df = load_data().copy()

    # 1. إزالة العمود أحادي القيمة (default) — تباين شبه معدوم وضجيج فقط
    if "default" in df.columns:
        df = df.drop(columns=["default"])

    # 2. هندسة pdays: 999 = لم يُتّصل به من قبل ➜ متغير ثنائي
    df["contacted_before"] = df["pdays"].apply(lambda x: 0 if x == 999 else 1)
    df = df.drop(columns=["pdays"])

    # 3. ترميز المتغير الهدف
    df["y"] = df["y"].map({"no": 0, "yes": 1})

    # 4. One-Hot Encoding للأعمدة النصية
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
    df_encoded = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    # نسختان: كاملة (بـ duration = مسرّبة) وصادقة (بدون duration)
    df_full = df_encoded.copy()
    df_honest = df_encoded.drop(columns=["duration"])

    return df_full, df_honest


@st.cache_data
def get_train_test_splits():
    df_full, df_honest = get_preprocessed_data()

    # التقسيم الصادق (Stratified 75/25)
    X_h = df_honest.drop(columns=["y"])
    y_h = df_honest["y"]
    X_train_h, X_test_h, y_train_h, y_test_h = train_test_split(
        X_h, y_h, test_size=0.25, random_state=42, stratify=y_h
    )

    # التقسيم المسرّب (مع duration)
    X_f = df_full.drop(columns=["y"])
    y_f = df_full["y"]
    X_train_f, X_test_f, y_train_f, y_test_f = train_test_split(
        X_f, y_f, test_size=0.25, random_state=42, stratify=y_f
    )

    return (X_train_h, X_test_h, y_train_h, y_test_h), (X_train_f, X_test_f, y_train_f, y_test_f)


# ==========================================
# 3. تدريب النماذج (Cached) لتفادي إعادة التدريب مع الـ Slider
# ==========================================
@st.cache_resource
def train_all_models():
    (X_train_h, X_test_h, y_train_h, y_test_h), (X_train_f, X_test_f, y_train_f, y_test_f) = get_train_test_splits()

    # 1. النموذج الصادق: Random Forest
    rf_honest = RandomForestClassifier(random_state=42, class_weight="balanced")
    rf_honest.fit(X_train_h, y_train_h)

    # 2. النموذج الصادق: Logistic Regression (عائلة ثانية) — مع قياس عادل للميزات
    lr_honest = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)),
    ])
    lr_honest.fit(X_train_h, y_train_h)

    # 3. النموذج المسرّب: Random Forest مع duration
    rf_leaked = RandomForestClassifier(random_state=42, class_weight="balanced")
    rf_leaked.fit(X_train_f, y_train_f)

    return rf_honest, lr_honest, rf_leaked


# تحميل عالمي
df = load_data()
(X_train_h, X_test_h, y_train_h, y_test_h), (X_train_f, X_test_f, y_train_f, y_test_f) = get_train_test_splits()
rf_honest, lr_honest, rf_leaked = train_all_models()

# قيمة خط الأساس (على مجموعة الاختبار) — نحسبها مرة ونستخدمها في كل مكان
BASELINE_ACC = (y_test_h == 0).mean() * 100
TEST_SIZE = len(y_test_h)

# ==========================================
# 4. القائمة الجانبية + العناصر دائمة الظهور
# ==========================================
st.sidebar.title("🏦 Marketing Campaign Dashboard")
section = st.sidebar.radio(
    "Select a section to navigate:",
    [
        "1. Exploratory Data Analysis",
        "2. Data Preprocessing",
        "3. Model Training & Baseline",
        "4. Model Evaluation & Comparison",
        "5. Prediction Dashboard",
    ],
)

# --- خط الأساس دائم الظهور (الشرط 2: "leave it there permanently") ---
st.sidebar.markdown("---")
st.sidebar.subheader("🎯 The Bar (Baseline)")
st.sidebar.metric(
    label=f"Always-Guess-NO Accuracy (Test, n={TEST_SIZE:,})",
    value=f"{BASELINE_ACC:.2f}%",
    delta="Recall for YES = 0%",
    delta_color="inverse",
)
st.sidebar.caption("Any model must clear this bar by a meaningful margin, and must never be judged on accuracy alone.")

# --- القيود دائمة الظهور (الشرط 7: "always visible") ---
st.sidebar.markdown("---")
with st.sidebar.expander("⚠️ What This Model Cannot Do (always visible)", expanded=False):
    st.markdown(
        """
        **1. Macroeconomic drift.** The model leans heavily on `euribor3m` and
        `nr.employed`. If rates or employment move sharply next month, its
        predictions decay — it learned 2008–2010 conditions, not universal laws.

        **2. Historical & geographic bias.** Data is from one Portuguese bank,
        2008–2010. It may not transfer to other countries, demographics, or
        today's digital-banking behaviour.

        **3. Cost of what we removed.** We dropped `duration` to kill data
        leakage and `default` for zero variance. That trade lost real predictive
        signal in exchange for an honest, deployable model.

        **4. Correlation, not causation.** A high score flags a *likely* yes, not
        a guaranteed one. Service quality, timing, and offer terms still decide
        the outcome.
        """
    )

# ==========================================
# 5. محتوى الأقسام
# ==========================================

# --- Section 1: EDA ---
if section == "1. Exploratory Data Analysis":
    st.title("📊 Section 1: Exploratory Data Analysis (EDA)")

    target_counts = df["y"].value_counts()
    target_percentages = df["y"].value_counts(normalize=True) * 100

    st.subheader("1️⃣ Target Balance: How Many People Actually Said Yes?")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Clients", f"{len(df):,}")
    col2.metric("Subscribed (YES)", f"{target_counts.get('yes', 0):,} ({target_percentages.get('yes', 0):.2f}%)")
    col3.metric("Not Subscribed (NO)", f"{target_counts.get('no', 0):,} ({target_percentages.get('no', 0):.2f}%)")

    st.warning(
        "⚠️ **Class Imbalance:** 'YES' is only ~11.3%. This alone tells us the problem is imbalanced — "
        "a lazy model that always guesses 'NO' scores ~88.7% accuracy while catching **zero** subscribers. "
        "That is why accuracy alone is meaningless here."
    )

    st.markdown("---")
    st.subheader("2️⃣ Where the Missing Data Is Actually Hiding")
    st.caption("There are no blank cells, but missing data is hidden as the literal text 'unknown'.")
    unknown_counts = (df == "unknown").sum()
    unknown_cols = unknown_counts[unknown_counts > 0].sort_values(ascending=False)
    if not unknown_cols.empty:
        st.write("**Count of hidden 'unknown' values per column:**")
        st.bar_chart(unknown_cols)

    st.markdown("---")
    st.subheader("3️⃣ Two Deceptive Columns")
    col_a, col_b = st.columns(2)
    with col_a:
        st.info(
            "📌 **`default` — looks useful, isn't.** Almost entirely 'no', with a "
            "handful of 'unknown' and virtually no 'yes'. Near-zero variance ➜ dropped."
        )
    with col_b:
        st.info(
            "📌 **`pdays` — looks numeric, isn't.** The value `999` is a *code* for "
            "'never contacted before', not a real distance ➜ converted to the binary "
            "flag `contacted_before`."
        )

# --- Section 2: Preprocessing ---
elif section == "2. Data Preprocessing":
    st.title("⚙️ Section 2: Data Preprocessing")
    df_full, df_honest = get_preprocessed_data()
    st.success(
        "✅ Dropped `default` (low variance) · engineered `contacted_before` from `pdays` · "
        "mapped target `y` (no➜0, yes➜1) · One-Hot encoded categoricals."
    )
    c1, c2 = st.columns(2)
    c1.metric("Honest dataset shape (no `duration`)", f"{df_honest.shape[0]:,} × {df_honest.shape[1]}")
    c2.metric("Full dataset shape (with `duration`)", f"{df_full.shape[0]:,} × {df_full.shape[1]}")
    st.write("**Preview of the honest, model-ready dataset:**")
    st.dataframe(df_honest.head())

# --- Section 3: Baseline & Splitting ---
elif section == "3. Model Training & Baseline":
    st.title("🤖 Section 3: The Bar & An Honest Split")

    st.subheader("📌 1. The Dumbest Possible Model (The Bar)")
    col_b1, col_b2 = st.columns(2)
    col_b1.metric("🎯 Baseline Test Accuracy (Always Guess NO)", f"{BASELINE_ACC:.2f}%")
    col_b2.metric("📊 Baseline Recall for Class 'YES'", "0.00%")
    st.caption(f"🔒 Evaluated strictly on the test set ({TEST_SIZE:,} samples). This number is pinned in the sidebar too.")

    st.markdown("---")
    st.subheader("📌 2. Is a Random Split Honest Here?")
    st.info(
        """
        **The honest answer: a random split is optimistic, and we say so.**

        Per `bank-additional-names.txt`, the rows are ordered **chronologically**
        (May 2008 → November 2010), and the last five columns are macroeconomic
        indicators (`emp.var.rate`, `cons.price.idx`, `cons.conf.idx`,
        `euribor3m`, `nr.employed`) that move with time. A random split therefore
        lets the model *see future economic conditions during training* — a mild
        temporal leak that flatters the numbers.

        A strict **time-based split** (train on the oldest months, test on the
        newest) would better mimic real deployment. We deliberately chose a
        **Stratified 75/25 random split** instead, for one reason only: to keep
        the rare ~11% positive class in identical proportion across train and
        test so the metrics stay stable and comparable. We accept the trade-off
        and read our results as an **optimistic upper bound**, not a deployment
        guarantee. Every metric in this dashboard is computed on the held-out
        test set only.
        """
    )

# --- Section 4: Model Evaluation ---
elif section == "4. Model Evaluation & Comparison":
    st.title("📊 Section 4: Model Evaluation & Comparison")
    st.caption("🔒 All metrics below are computed STRICTLY on the test set.")

    # التنبؤات
    y_pred_leaked = rf_leaked.predict(X_test_f)
    y_prob_leaked = rf_leaked.predict_proba(X_test_f)[:, 1]

    y_pred_rf = rf_honest.predict(X_test_h)
    y_prob_rf = rf_honest.predict_proba(X_test_h)[:, 1]

    y_pred_lr = lr_honest.predict(X_test_h)
    y_prob_lr = lr_honest.predict_proba(X_test_h)[:, 1]

    def metric_block(y_true, y_pred, y_prob):
        st.write(f"- Accuracy: `{accuracy_score(y_true, y_pred):.4f}`")
        st.write(f"- Precision: `{precision_score(y_true, y_pred, zero_division=0):.4f}`")
        st.write(f"- Recall: `{recall_score(y_true, y_pred, zero_division=0):.4f}`")
        st.write(f"- ROC AUC: `{roc_auc_score(y_true, y_prob):.4f}`")
        cm = confusion_matrix(y_true, y_pred)
        cm_df = pd.DataFrame(
            cm,
            index=["Actual NO", "Actual YES"],
            columns=["Pred NO", "Pred YES"],
        )
        st.write("**Confusion Matrix:**")
        st.dataframe(cm_df)

    st.subheader("⚔️ Leaked Model vs. Two Honest Model Families")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.error("❌ **Leaked — RF with `duration`**")
        metric_block(y_test_f, y_pred_leaked, y_prob_leaked)
    with c2:
        st.success("✅ **Honest — Random Forest**")
        metric_block(y_test_h, y_pred_rf, y_prob_rf)
    with c3:
        st.info("✅ **Honest — Logistic Regression**")
        metric_block(y_test_h, y_pred_lr, y_prob_lr)

    st.warning(
        "🧠 **Why the 'worse' model is the better one.** The leaked model's strength comes almost "
        "entirely from `duration` (call length) — but you only know a call's length *after* you've made it. "
        "Using it to decide **who to call** is impossible in practice (data leakage). Removing `duration` drops "
        "every metric, yet it's the only version that can actually run *before* the call — which is the whole task."
    )

    st.markdown("---")

    # --- الشرط 6: منحنى الدقة/الاستدعاء عبر العتبات + Slider ---
    st.subheader("🎚️ Where Do You Draw the Line? (Threshold Trade-off)")

    prec, rec, thr = precision_recall_curve(y_test_h, y_prob_rf)
    curve_df = pd.DataFrame({
        "Threshold": thr,
        "Precision": prec[:-1],
        "Recall": rec[:-1],
    }).set_index("Threshold")
    st.write("**Precision and recall as the decision threshold moves (Honest RF):**")
    st.line_chart(curve_df)

    threshold = st.slider("Select decision threshold:", 0.0, 1.0, 0.35, 0.05)
    y_pred_custom = (y_prob_rf >= threshold).astype(int)
    col_t1, col_t2, col_t3 = st.columns(3)
    col_t1.metric("Precision", f"{precision_score(y_test_h, y_pred_custom, zero_division=0):.4f}")
    col_t2.metric("Recall", f"{recall_score(y_test_h, y_pred_custom, zero_division=0):.4f}")
    col_t3.metric("F1-Score", f"{f1_score(y_test_h, y_pred_custom, zero_division=0):.4f}")

    st.info(
        "📝 **Chosen threshold ≈ 0.35.** We lower it below the default 0.5 to buy **recall**: missing a "
        "customer who would have subscribed loses a whole sale, while calling a wrong prospect costs only a "
        "short, cheap phone call. When a miss is more expensive than a false alarm, you move the line down."
    )

    st.markdown("---")
    st.subheader("⚠️ Reminder: What This Model Cannot Do")
    st.caption("Full text is always available in the sidebar expander. Short version:")
    st.error(
        "Depends on 2008–2010 Portuguese macro conditions · won't generalize across time/geography without "
        "retraining · dropping `duration` and `default` cost real signal · finds correlation, not causation."
    )

# --- Section 5: Prediction Dashboard ---
elif section == "5. Prediction Dashboard":
    st.title("🔮 Section 5: Single Customer Prediction (Before the Call)")
    st.write("Estimate subscription likelihood **before placing the call**, using the honest Random Forest model.")

    col_in1, col_in2, col_in3 = st.columns(3)
    with col_in1:
        age = st.number_input("Age", 17, 100, 35)
        job = st.selectbox("Job", ["admin.", "blue-collar", "technician", "services", "management",
                                    "retired", "entrepreneur", "self-employed", "housemaid",
                                    "unemployed", "student", "unknown"])
        marital = st.selectbox("Marital Status", ["married", "single", "divorced", "unknown"])
        education = st.selectbox("Education", ["university.degree", "high.school", "basic.9y",
                                               "professional.course", "basic.4y", "basic.6y",
                                               "unknown", "illiterate"])
    with col_in2:
        housing = st.selectbox("Has Housing Loan?", ["no", "yes", "unknown"])
        loan = st.selectbox("Has Personal Loan?", ["no", "yes", "unknown"])
        contact = st.selectbox("Contact Communication Type", ["cellular", "telephone"])
        month = st.selectbox("Last Contact Month", ["may", "jul", "aug", "jun", "nov",
                                                    "apr", "oct", "sep", "mar", "dec"])
        day_of_week = st.selectbox("Last Contact Day", ["mon", "thu", "tue", "wed", "fri"])
    with col_in3:
        campaign = st.number_input("Campaign Contacts", 1, 50, 1)
        previous = st.number_input("Previous Contacts", 0, 10, 0)
        poutcome = st.selectbox("Previous Outcome", ["nonexistent", "failure", "success"])
        contacted_before = 1 if previous > 0 else 0

    st.subheader("📈 Macroeconomic Context")
    col_eco1, col_eco2, col_eco3 = st.columns(3)
    with col_eco1:
        emp_var_rate = st.number_input("emp.var.rate", value=-1.8)
        cons_price_idx = st.number_input("cons.price.idx", value=92.893)
    with col_eco2:
        cons_conf_idx = st.number_input("cons.conf.idx", value=-46.2)
        euribor3m = st.number_input("euribor3m", value=1.313)
    with col_eco3:
        nr_employed = st.number_input("nr.employed", value=5099.1)

    input_data = pd.DataFrame([{
        'age': age, 'job': job, 'marital': marital, 'education': education,
        'housing': housing, 'loan': loan, 'contact': contact, 'month': month,
        'day_of_week': day_of_week, 'campaign': campaign, 'previous': previous,
        'poutcome': poutcome, 'emp.var.rate': emp_var_rate, 'cons.price.idx': cons_price_idx,
        'cons.conf.idx': cons_conf_idx, 'euribor3m': euribor3m, 'nr.employed': nr_employed,
        'contacted_before': contacted_before
    }])

    # مطابقة الأعمدة مع بيانات التدريب (الأعمدة المفقودة ➜ 0)
    input_encoded = pd.get_dummies(input_data).reindex(columns=X_train_h.columns, fill_value=0)

    decision_threshold = 0.35
    if st.button("🚀 Predict Subscription Probability", type="primary", use_container_width=True):
        pred_proba = rf_honest.predict_proba(input_encoded)[0][1]

        col_res1, col_res2 = st.columns(2)
        col_res1.metric("Subscription Probability (YES)", f"{pred_proba * 100:.2f}%")
        with col_res2:
            if pred_proba >= decision_threshold:
                st.success("🎯 **RECOMMENDATION: CALL THIS CUSTOMER**")
                st.write(f"Above the business threshold ({decision_threshold}).")
            else:
                st.error("🚫 **RECOMMENDATION: DO NOT CALL**")
                st.write(f"Below the business threshold ({decision_threshold}). Save the call effort.")

        st.caption(
            "🔒 Note: `call duration` was **not** required for this prediction — the model decides "
            "*before* the call, exactly as the task demands."
        )