import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.dummy import DummyClassifier

# -----------------------------------------------------------------------------
# 1. Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="뇌졸중 예측 실습실 - 채점표",
    page_icon="🧠",
    layout="wide"
)

# -----------------------------------------------------------------------------
# 2. Data Loading & Split Strategy (Same as Classification Model Page)
# -----------------------------------------------------------------------------
DATA_URL = "https://raw.githubusercontent.com/parksso0613/data/refs/heads/main/stroke.csv"

@st.cache_data
def load_data():
    """Load dataset from Github repository."""
    df = pd.read_csv(DATA_URL, encoding="utf-8")
    return df

try:
    df = load_data()
    data_load_success = True
except Exception as e:
    data_load_success = False
    st.error(f"데이터를 불러오는 중 오류가 발생했습니다: {e}")

st.title("📋 모델 평가 및 채점표")
st.markdown("테스트 데이터를 바탕으로 모델의 예측 성능을 다양한 평가 지표와 상세 오차 목록을 통해 점검합니다.")
st.divider()

if data_load_success:
    # -------------------------------------------------------------------------
    # 3. Data Processing & Custom Train / Test Split
    # -------------------------------------------------------------------------
    # 기본 입력 속성 설정: age, avg_glucose_level, hypertension, heart_disease
    selected_features = ["age", "avg_glucose_level", "hypertension", "heart_disease"]

    # 번호(id 또는 기존 순서) 순 정렬 후 10명씩 묶어 앞 3명을 테스트용으로 고정 (1,533명)
    df_sorted = df.copy()
    if "id" in df_sorted.columns:
        df_sorted = df_sorted.sort_values("id").reset_index(drop=True)
    else:
        df_sorted = df_sorted.reset_index(drop=True)

    test_indices = []
    train_indices = []

    for i in range(len(df_sorted)):
        if (i % 10) < 3:
            test_indices.append(i)
        else:
            train_indices.append(i)

    train_df = df_sorted.iloc[train_indices].copy()
    test_df = df_sorted.iloc[test_indices].copy()

    X_train = train_df[selected_features].values
    y_train = train_df["stroke"].values
    X_test = test_df[selected_features].values
    y_test = test_df["stroke"].values

    # -------------------------------------------------------------------------
    # 4. Model Training
    # -------------------------------------------------------------------------
    # 1) Dummy Classifier (훈련용에서 많은 쪽으로만 답하는 모델)
    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(X_train, y_train)

    # 2) Logistic Regression (로지스틱 회귀)
    log_reg = LogisticRegression(random_state=42, max_iter=1000)
    log_reg.fit(X_train, y_train)

    # 3) Decision Tree (의사결정트리)
    dt_clf = DecisionTreeClassifier(max_depth=3, min_samples_split=5, random_state=42)
    dt_clf.fit(X_train, y_train)

    # Predict on test set
    dummy_pred = dummy.predict(X_test)
    log_pred = log_reg.predict(X_test)
    dt_pred = dt_clf.predict(X_test)

    # Helper function to compute Confusion Matrix components & Metrics
    def compute_metrics(y_true, y_pred):
        # 1: Positive (뇌졸중), 0: Negative (아님)
        tp = int(np.sum((y_true == 1) & (y_pred == 1)))
        fp = int(np.sum((y_true == 0) & (y_pred == 1)))
        fn = int(np.sum((y_true == 1) & (y_pred == 0)))
        tn = int(np.sum((y_true == 0) & (y_pred == 0)))

        total = tp + fp + fn + tn
        acc = (tp + tn) / total if total > 0 else 0.0

        # Recall = TP / (TP + FN) [실제 뇌졸중 중 맞춘 비율]
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        # Precision = TP / (TP + FP) [뇌졸중이라 예측한 것 중 진짜인 비율]
        pred_pos_count = tp + fp
        if pred_pos_count == 0:
            precision_str = "계산할 수 없음"
            f1_str = "계산할 수 없음"
            precision_val = None
            f1_val = None
        else:
            precision_val = tp / pred_pos_count
            precision_str = f"{precision_val * 100:.2f}%"
            if precision_val + recall == 0:
                f1_str = "계산할 수 없음"
                f1_val = None
            else:
                f1_val = (2 * precision_val * recall) / (precision_val + recall)
                f1_str = f"{f1_val * 100:.2f}%"

        acc_str = f"{acc * 100:.2f}%"
        recall_str = f"{recall * 100:.2f}%"

        return {
            "TP": tp, "FP": fp, "FN": fn, "TN": tn,
            "Accuracy_str": acc_str,
            "Recall_str": recall_str,
            "Precision_str": precision_str,
            "F1_str": f1_str
        }

    dummy_m = compute_metrics(y_test, dummy_pred)
    log_m = compute_metrics(y_test, log_pred)
    dt_m = compute_metrics(y_test, dt_pred)

    # -------------------------------------------------------------------------
    # 5. Summary Info Banner
    # -------------------------------------------------------------------------
    test_total_count = len(test_df)
    test_stroke_count = int(test_df["stroke"].sum())
    st.info(f"📊 **테스트 데이터 세트 현황**: 전체 **{test_total_count:,}명** 중 실제 뇌졸중 환자는 **{test_stroke_count:,}명**입니다.")

    st.markdown("---")

    # -------------------------------------------------------------------------
    # 6. Evaluation Metrics Table
    # -------------------------------------------------------------------------
    st.subheader("🎯 모델별 평가 지표 비교")

    metrics_summary_df = pd.DataFrame([
        {
            "모델 이름": "훈련용에서 많은 쪽으로만 답하는 모델",
            "정확도 (Accuracy)": dummy_m["Accuracy_str"],
            "재현율 (Recall)": dummy_m["Recall_str"],
            "정밀도 (Precision)": dummy_m["Precision_str"],
            "F1 점수 (F1-Score)": dummy_m["F1_str"]
        },
        {
            "모델 이름": "로지스틱 회귀(확률로 답하는 모델)",
            "정확도 (Accuracy)": log_m["Accuracy_str"],
            "재현율 (Recall)": log_m["Recall_str"],
            "정밀도 (Precision)": log_m["Precision_str"],
            "F1 점수 (F1-Score)": log_m["F1_str"]
        },
        {
            "모델 이름": "의사결정트리(질문으로 답하는 모델)",
            "정확도 (Accuracy)": dt_m["Accuracy_str"],
            "재현율 (Recall)": dt_m["Recall_str"],
            "정밀도 (Precision)": dt_m["Precision_str"],
            "F1 점수 (F1-Score)": dt_m["F1_str"]
        }
    ])

    st.dataframe(metrics_summary_df, use_container_width=True, hide_index=True)

    st.markdown("---")

    # -------------------------------------------------------------------------
    # 7. Confusion Matrices (2x2 Tables with Row/Col Totals)
    # -------------------------------------------------------------------------
    st.subheader("🧩 모델별 혼동행렬 (Confusion Matrix)")

    col_cm1, col_cm2 = st.columns(2)

    def display_confusion_matrix_table(m_dict, title_name):
        tp, fp, fn, tn = m_dict["TP"], m_dict["FP"], m_dict["FN"], m_dict["TN"]
        
        cm_df = pd.DataFrame({
            "예측: 아님 (0)": [
                f"{tn:,}명 (TN)",
                f"{fn:,}명 (FN)",
                f"{tn + fn:,}명"
            ],
            "예측: 뇌졸중 (1)": [
                f"{fp:,}명 (FP)",
                f"{tp:,}명 (TP)",
                f"{fp + tp:,}명"
            ],
            "합계": [
                f"{tn + fp:,}명",
                f"{fn + tp:,}명",
                f"{tn + fp + fn + tp:,}명"
            ]
        }, index=["실제: 아님 (0)", "실제: 뇌졸중 (1)", "합계"])

        st.markdown(f"##### {title_name}")
        st.table(cm_df)

    with col_cm1:
        display_confusion_matrix_table(log_m, "📈 로지스틱 회귀(확률로 답하는 모델)")

    with col_cm2:
        display_confusion_matrix_table(dt_m, "🌳 의사결정트리(질문으로 답하는 모델)")

    st.markdown("---")

    # -------------------------------------------------------------------------
    # 8. Detailed Error Lists ( 놓친 환자 vs 헛짚은 사람 )
    # -------------------------------------------------------------------------
    st.subheader("🔍 세부 예측 오류 명단 조회")

    selected_model_name = st.radio(
        "조회할 모델을 선택하세요:",
        options=["로지스틱 회귀(확률로 답하는 모델)", "의사결정트리(질문으로 답하는 모델)"],
        horizontal=True
    )

    if selected_model_name == "로지스틱 회귀(확률로 답하는 모델)":
        active_preds = log_pred
        active_m = log_m
    else:
        active_preds = dt_pred
        active_m = dt_m

    # Create temporary view dataframe for analysis
    test_analysis_df = test_df.copy()
    test_analysis_df["pred_stroke"] = active_preds

    # 1) 놓친 환자 (FN): 실제 1 (뇌졸중) 인데 예측 0 (아님)
    fn_df = test_analysis_df[(test_analysis_df["stroke"] == 1) & (test_analysis_df["pred_stroke"] == 0)].copy()
    fn_df = fn_df.sort_values(by="age", ascending=False)

    # 2) 헛짚은 사람 (FP): 실제 0 (아님) 인데 예측 1 (뇌졸중)
    fp_df = test_analysis_df[(test_analysis_df["stroke"] == 0) & (test_analysis_df["pred_stroke"] == 1)].copy()
    fp_df = fp_df.sort_values(by="age", ascending=False)

    # Column formatting for Korean display
    display_cols = ["id", "age", "avg_glucose_level", "hypertension", "heart_disease"]
    col_korean_names = {
        "id": "번호(id)",
        "age": "나이",
        "avg_glucose_level": "평균 혈당",
        "hypertension": "고혈압 여부",
        "heart_disease": "심장병 여부"
    }

    def format_list_df(df_input):
        if df_input.empty:
            return pd.DataFrame(columns=["번호(id)", "나이", "평균 혈당", "고혈압 여부", "심장병 여부"])
        
        df_out = df_input[display_cols].copy()
        df_out["hypertension"] = df_out["hypertension"].map({0: "없음(0)", 1: "있음(1)"})
        df_out["heart_disease"] = df_out["heart_disease"].map({0: "없음(0)", 1: "있음(1)"})
        df_out = df_out.rename(columns=col_korean_names)
        return df_out

    fn_display_df = format_list_df(fn_df)
    fp_display_df = format_list_df(fp_df)

    col_list1, col_list2 = st.columns(2)

    with col_list1:
        fn_count = len(fn_df)
        st.markdown(f"##### ⚠️ 실제 뇌졸중인데 아니라고 예측한 사람 (놓친 환자) [총 {fn_count:,}명]")
        if fn_count > 20:
            st.caption("※ 나이가 많은 순서로 상위 20명만 표시합니다.")
            st.dataframe(fn_display_df.head(20), use_container_width=True, hide_index=True)
        else:
            st.dataframe(fn_display_df, use_container_width=True, hide_index=True)

    with col_list2:
        fp_count = len(fp_df)
        st.markdown(f"##### 🚨 아닌데 뇌졸중이라고 예측한 사람 (헛짚은 사람) [총 {fp_count:,}명]")
        if fp_count > 20:
            st.caption("※ 나이가 많은 순서로 상위 20명만 표시합니다.")
            st.dataframe(fp_display_df.head(20), use_container_width=True, hide_index=True)
        else:
            st.dataframe(fp_display_df, use_container_width=True, hide_index=True)

    st.markdown(
        f"💡 **참고**: 위 두 목록의 사람 수(**놓친 환자: {fn_count}명**, **헛짚은 사람: {fp_count}명**)는 "
        f"상단 혼동행렬 표의 **FN({active_m['FN']}명)** 및 **FP({active_m['FP']}명)** 칸의 수치와 정확히 일치합니다."
    )
