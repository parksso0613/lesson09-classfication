import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

# -----------------------------------------------------------------------------
# 1. Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="뇌졸중 예측 실습실 - 성능 높이기",
    page_icon="🧠",
    layout="wide"
)

# -----------------------------------------------------------------------------
# 2. Data Loading & Caching
# -----------------------------------------------------------------------------
DATA_URL = "https://raw.githubusercontent.com/parksso0613/data/refs/heads/main/stroke.csv"

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_URL, encoding="utf-8")
    return df

try:
    df = load_data()
    data_load_success = True
except Exception as e:
    data_load_success = False
    st.error(f"데이터를 불러오는 중 오류가 발생했습니다: {e}")

st.title("⚡ 클래스 가중치 조절로 모델 성능 높이기")
st.markdown(
    "뇌졸중 환자(양성) 비율이 매우 적은 불균형 데이터셋에서 **클래스 가중치(Class Weight)**를 맞추기 전과 후의 "
    "모델 예측 성능 및 결정 경계(Decision Boundary) 변화를 비교 분석합니다."
)
st.divider()

if data_load_success:
    # -------------------------------------------------------------------------
    # 3. Feature Selection with Korean Names
    # -------------------------------------------------------------------------
    feature_map = {
        "age": "나이",
        "avg_glucose_level": "평균 혈당",
        "bmi": "체질량지수",
        "hypertension": "고혈압",
        "heart_disease": "심장병"
    }
    inv_feature_map = {v: k for k, v in feature_map.items()}

    default_selected_korean = ["나이", "평균 혈당", "고혈압", "심장병"]
    all_korean_options = list(feature_map.values())

    selected_korean = st.multiselect(
        "분류 모델 입력으로 사용할 속성을 선택하세요 (2개 이상 선택):",
        options=all_korean_options,
        default=default_selected_korean
    )

    if len(selected_korean) < 2:
        st.warning("⚠️ 최소 2개 이상의 속성을 선택해야 모델 학습 및 시각화가 가능합니다.")
        st.stop()

    selected_features = [inv_feature_map[k] for k in selected_korean]

    # -------------------------------------------------------------------------
    # 4. Custom Train / Test Split Strategy
    # -------------------------------------------------------------------------
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

    # Missing value imputation (BMI) using training set median
    if "bmi" in selected_features:
        bmi_median_train = train_df["bmi"].median()
        train_df["bmi"] = train_df["bmi"].fillna(bmi_median_train)
        test_df["bmi"] = test_df["bmi"].fillna(bmi_median_train)

    X_train = train_df[selected_features].values
    y_train = train_df["stroke"].values
    X_test = test_df[selected_features].values
    y_test = test_df["stroke"].values

    # -------------------------------------------------------------------------
    # 5. Model Training (Before & After Class Weight Balance)
    # -------------------------------------------------------------------------
    # Logistic Regression
    log_unbalanced = LogisticRegression(random_state=42, max_iter=1000)
    log_unbalanced.fit(X_train, y_train)

    log_balanced = LogisticRegression(random_state=42, max_iter=1000, class_weight="balanced")
    log_balanced.fit(X_train, y_train)

    # Decision Tree
    dt_unbalanced = DecisionTreeClassifier(max_depth=3, min_samples_split=5, random_state=42)
    dt_unbalanced.fit(X_train, y_train)

    dt_balanced = DecisionTreeClassifier(max_depth=3, min_samples_split=5, random_state=42, class_weight="balanced")
    dt_balanced.fit(X_train, y_train)

    # Helper function to compute Confusion Matrix components & Metrics
    def compute_metrics(y_true, y_pred):
        tp = int(np.sum((y_true == 1) & (y_pred == 1)))
        fp = int(np.sum((y_true == 0) & (y_pred == 1)))
        fn = int(np.sum((y_true == 1) & (y_pred == 0)))
        tn = int(np.sum((y_true == 0) & (y_pred == 0)))

        total = tp + fp + fn + tn
        acc = (tp + tn) / total if total > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        pred_pos_count = tp + fp
        if pred_pos_count == 0:
            precision_str = "계산할 수 없음"
            f1_str = "계산할 수 없음"
        else:
            precision_val = tp / pred_pos_count
            precision_str = f"{precision_val * 100:.2f}%"
            if precision_val + recall == 0:
                f1_str = "계산할 수 없음"
            else:
                f1_val = (2 * precision_val * recall) / (precision_val + recall)
                f1_str = f"{f1_val * 100:.2f}%"

        return {
            "TP": tp, "FP": fp, "FN": fn, "TN": tn,
            "Accuracy": f"{acc * 100:.2f}%",
            "Recall": f"{recall * 100:.2f}%",
            "Precision": precision_str,
            "F1": f1_str
        }

    log_unbal_m = compute_metrics(y_test, log_unbalanced.predict(X_test))
    log_bal_m = compute_metrics(y_test, log_balanced.predict(X_test))

    dt_unbal_m = compute_metrics(y_test, dt_unbalanced.predict(X_test))
    dt_bal_m = compute_metrics(y_test, dt_balanced.predict(X_test))

    # -------------------------------------------------------------------------
    # 6. Before & After Comparison Table
    # -------------------------------------------------------------------------
    st.subheader("📊 가중치 조절 전·후 성능 비교 표")

    def render_model_comparison(unbal_m, bal_m, model_title):
        st.markdown(f"#### 📌 {model_title}")
        comp_df = pd.DataFrame([
            {
                "구분": "가중치 조절 전 (None)",
                "TN (실제0→예측0)": f"{unbal_m['TN']:,}명",
                "FP (실제0→예측1)": f"{unbal_m['FP']:,}명",
                "FN (실제1→예측0)": f"{unbal_m['FN']:,}명",
                "TP (실제1→예측1)": f"{unbal_m['TP']:,}명",
                "정확도": unbal_m["Accuracy"],
                "재현율 (Recall)": unbal_m["Recall"],
                "정밀도 (Precision)": unbal_m["Precision"],
                "F1 점수": unbal_m["F1"]
            },
            {
                "구분": "가중치 조절 후 (Balanced)",
                "TN (실제0→예측0)": f"{bal_m['TN']:,}명",
                "FP (실제0→예측1)": f"{bal_m['FP']:,}명",
                "FN (실제1→예측0)": f"{bal_m['FN']:,}명",
                "TP (실제1→예측1)": f"{bal_m['TP']:,}명",
                "정확도": bal_m["Accuracy"],
                "재현율 (Recall)": bal_m["Recall"],
                "정밀도 (Precision)": bal_m["Precision"],
                "F1 점수": bal_m["F1"]
            }
        ])
        st.dataframe(comp_df, use_container_width=True, hide_index=True)

        st.info(
            f"💡 **변화 요약 ({model_title})**: "
            f"찾아낸 환자 수(TP)는 **{unbal_m['TP']}명 ➔ {bal_m['TP']}명**, "
            f"놓친 환자 수(FN)는 **{unbal_m['FN']}명 ➔ {bal_m['FN']}명**, "
            f"헛짚은 사람 수(FP)는 **{unbal_m['FP']}명 ➔ {bal_m['FP']}명**으로 바뀌었습니다."
        )

    render_model_comparison(log_unbal_m, log_bal_m, "로지스틱 회귀(확률로 답하는 모델)")
    st.markdown("---")
    render_model_comparison(dt_unbal_m, dt_bal_m, "의사결정트리(질문으로 답하는 모델)")

    st.markdown("---")

    # -------------------------------------------------------------------------
    # 7. 2D Scatter Plot & Decision Boundaries Overlap
    # -------------------------------------------------------------------------
    st.subheader("🎨 2D 산점도 및 가중치 조절 전·후 0.5 경계선 비교")

    col_x, col_y = st.columns(2)
    with col_x:
        x_korean = st.selectbox("가로축(X축) 속성 선택:", selected_korean, index=0)
    with col_y:
        y_korean_options = [k for k in selected_korean if k != x_korean]
        y_korean = st.selectbox("세로축(Y축) 속성 선택:", y_korean_options, index=0)

    x_feat = inv_feature_map[x_korean]
    y_feat = inv_feature_map[y_korean]

    x_idx = selected_features.index(x_feat)
    y_idx = selected_features.index(y_feat)

    other_feats = [f for f in selected_features if f not in [x_feat, y_feat]]
    other_medians = {}
    fixed_values_text = []

    for f in other_feats:
        med_val = float(np.median(test_df[f]))
        other_medians[f] = med_val
        fixed_values_text.append(f"**{feature_map[f]}**: {med_val:.2f}")

    if fixed_values_text:
        st.info("💡 **선택되지 않은 축 속성의 고정값 (테스트 데이터 중앙값)**: " + ", ".join(fixed_values_text))

    x_min, x_max = test_df[x_feat].min(), test_df[x_feat].max()
    y_min, y_max = test_df[y_feat].min(), test_df[y_feat].max()

    x_margin = (x_max - x_min) * 0.05 if x_max != x_min else 1.0
    y_margin = (y_max - y_min) * 0.05 if y_max != y_min else 1.0

    x_grid = np.linspace(x_min - x_margin, x_max + x_margin, 200)

    fig2d = go.Figure()

    # Scatter points
    test_df_copy = test_df.copy()
    test_df_copy["stroke_label"] = test_df_copy["stroke"].map({0: "아님 (0)", 1: "뇌졸중 (1)"})

    for label, color, symbol in [("아님 (0)", "#1f77b4", "circle"), ("뇌졸중 (1)", "#d62728", "x")]:
        sub = test_df_copy[test_df_copy["stroke_label"] == label]
        fig2d.add_trace(go.Scatter(
            x=sub[x_feat],
            y=sub[y_feat],
            mode="markers",
            name=f"실제 {label}",
            marker=dict(color=color, symbol=symbol, size=7, opacity=0.7)
        ))

    # Calculate 0.5 decision lines for Logistic Regression
    def add_logistic_line(model, name_str, line_color, dash_type):
        coef = model.coef_[0]
        intercept = model.intercept_[0]

        w_x = coef[x_idx]
        w_y = coef[y_idx]

        const_term = intercept
        for f in other_feats:
            f_idx = selected_features.index(f)
            const_term += coef[f_idx] * other_medians[f]

        if abs(w_y) > 1e-6:
            y_line = -(w_x * x_grid + const_term) / w_y
            valid_mask = (y_line >= y_min - y_margin) & (y_line <= y_max + y_margin)
            if np.any(valid_mask):
                fig2d.add_trace(go.Scatter(
                    x=x_grid[valid_mask],
                    y=y_line[valid_mask],
                    mode="lines",
                    name=name_str,
                    line=dict(color=line_color, width=3, dash=dash_type)
                ))
                return True
        return False

    line1_ok = add_logistic_line(log_unbalanced, "조절 전 0.5 경계선", "#2ca02c", "dash")
    line2_ok = add_logistic_line(log_balanced, "조절 후 0.5 경계선", "#ff7f0e", "solid")

    fig2d.update_layout(
        title=f"로지스틱 회귀 0.5 결정 경계선 비교 ({x_korean} vs {y_korean})",
        xaxis_title=x_korean,
        yaxis_title=y_korean,
        legend_title="범례"
    )

    st.plotly_chart(fig2d, use_container_width=True)

    if not line1_ok:
        st.warning("⚠️ 가중치 조절 전 0.5 경계선이 현재 표시된 그래프 영역 밖에 존재합니다.")
    if not line2_ok:
        st.warning("⚠️ 가중치 조절 후 0.5 경계선이 현재 표시된 그래프 영역 밖에 존재합니다.")

    # -------------------------------------------------------------------------
    # 8. 3D Scatter Plot & Planes (If >= 3 features selected)
    # -------------------------------------------------------------------------
    if len(selected_korean) >= 3:
        st.markdown("---")
        st.subheader("🧊 3차원 산점도 및 결정 평면 (3D Plane) 비교")
        st.caption("※ 마우스를 클릭하고 드래그하여 3차원 그래프를 자유롭게 회전시켜 볼 수 있습니다.")

        col3_1, col3_2, col3_3 = st.columns(3)
        with col3_1:
            x3_kor = st.selectbox("3D X축 속성:", selected_korean, index=0, key="3d_x")
        with col3_2:
            y3_opts = [k for k in selected_korean if k != x3_kor]
            y3_kor = st.selectbox("3D Y축 속성:", y3_opts, index=0, key="3d_y")
        with col3_3:
            z3_opts = [k for k in selected_korean if k not in [x3_kor, y3_kor]]
            z3_kor = st.selectbox("3D Z축 속성:", z3_opts, index=0, key="3d_z")

        x3_feat, y3_feat, z3_feat = inv_feature_map[x3_kor], inv_feature_map[y3_kor], inv_feature_map[z3_kor]
        x3_idx, y3_idx, z3_idx = selected_features.index(x3_feat), selected_features.index(y3_feat), selected_features.index(z3_feat)

        other3_feats = [f for f in selected_features if f not in [x3_feat, y3_feat, z3_feat]]
        other3_medians = {f: float(np.median(test_df[f])) for f in other3_feats}

        fig3d = go.Figure()

        # 3D Scatter points
        for label, color, symbol in [("아님 (0)", "#1f77b4", "circle"), ("뇌졸중 (1)", "#d62728", "x")]:
            sub = test_df_copy[test_df_copy["stroke_label"] == label]
            fig3d.add_trace(go.Scatter3d(
                x=sub[x3_feat],
                y=sub[y3_feat],
                z=sub[z3_feat],
                mode="markers",
                name=f"실제 {label}",
                marker=dict(color=color, symbol=symbol, size=4, opacity=0.7)
            ))

        # Build 3D Plane Mesh for Logistic Regression
        x3_min, x3_max = test_df[x3_feat].min(), test_df[x3_feat].max()
        y3_min, y3_max = test_df[y3_feat].min(), test_df[y3_feat].max()

        x3_g = np.linspace(x3_min, x3_max, 20)
        y3_g = np.linspace(y3_min, y3_max, 20)
        xx3, yy3 = np.meshgrid(x3_g, y3_g)

        def add_3d_plane(model, plane_name, color_str):
            coef = model.coef_[0]
            intercept = model.intercept_[0]

            w_x, w_y, w_z = coef[x3_idx], coef[y3_idx], coef[z3_idx]
            const_term = intercept
            for f in other3_feats:
                f_idx = selected_features.index(f)
                const_term += coef[f_idx] * other3_medians[f]

            if abs(w_z) > 1e-6:
                zz3 = -(w_x * xx3 + w_y * yy3 + const_term) / w_z
                fig3d.add_trace(go.Surface(
                    x=xx3,
                    y=yy3,
                    z=zz3,
                    name=plane_name,
                    showscale=False,
                    opacity=0.5,
                    colorscale=[[0, color_str], [1, color_str]]
                ))

        add_3d_plane(log_unbalanced, "조절 전 평면", "green")
        add_3d_plane(log_balanced, "조절 후 평면", "orange")

        fig3d.update_layout(
            title=f"3차원 공간 결정 평면 시각화 ({x3_kor}, {y3_kor}, {z3_kor})",
            scene=dict(
                xaxis_title=x3_kor,
                yaxis_title=y3_kor,
                zaxis_title=z3_kor
            ),
            height=650
        )

        st.plotly_chart(fig3d, use_container_width=True)
