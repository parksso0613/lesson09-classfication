import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.dummy import DummyClassifier

# -----------------------------------------------------------------------------
# 1. Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="뇌졸중 예측 실습실 - 분류 모델",
    page_icon="🧠",
    layout="wide"
)

# -----------------------------------------------------------------------------
# 2. Data Loading
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

st.title("🤖 분류 모델 실습")
st.markdown("머신러닝 분류 모델을 학습시키고, 결정 경계 및 의사결정나무 구조를 시각화하여 비교합니다.")
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
    # 번호(id 또는 기존 순서) 순 정렬 후 10명씩 묶어 앞 3명을 테스트용으로 고정
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

    # -------------------------------------------------------------------------
    # 5. Missing Value Imputation (BMI)
    # -------------------------------------------------------------------------
    if "bmi" in selected_features:
        bmi_median_train = train_df["bmi"].median()
        train_df["bmi"] = train_df["bmi"].fillna(bmi_median_train)
        test_df["bmi"] = test_df["bmi"].fillna(bmi_median_train)

    X_train = train_df[selected_features].values
    y_train = train_df["stroke"].values
    X_test = test_df[selected_features].values
    y_test = test_df["stroke"].values

    # -------------------------------------------------------------------------
    # 6. Model Training
    # -------------------------------------------------------------------------
    # 1) Dummy Classifier (기준 모델)
    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(X_train, y_train)

    # 2) Logistic Regression (로지스틱 회귀)
    log_reg = LogisticRegression(random_state=42, max_iter=1000)
    log_reg.fit(X_train, y_train)

    # 3) Decision Tree (의사결정트리: 질문 3번까지, 최소 분할 샘플 5명, 난수 고정)
    dt_clf = DecisionTreeClassifier(max_depth=3, min_samples_split=5, random_state=42)
    dt_clf.fit(X_train, y_train)

    # Accuracies
    dummy_train_acc = dummy.score(X_train, y_train)
    dummy_test_acc = dummy.score(X_test, y_test)

    log_train_acc = log_reg.score(X_train, y_train)
    log_test_acc = log_reg.score(X_test, y_test)

    dt_train_acc = dt_clf.score(X_train, y_train)
    dt_test_acc = dt_clf.score(X_test, y_test)

    # -------------------------------------------------------------------------
    # 7. Model Accuracy Display Cards
    # -------------------------------------------------------------------------
    st.subheader("📊 모델 평가 및 정확도 비교")
    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric(
            label="📌 기준 모델 (가장 많은 쪽으로만 답하는 모델)",
            value=f"{dummy_test_acc * 100:.2f}%"
        )
        st.caption(f"훈련 정확도: **{dummy_train_acc * 100:.2f}%** | 테스트 정확도: **{dummy_test_acc * 100:.2f}%**")

    with c2:
        st.metric(
            label="📈 로지스틱 회귀 (확률로 답하는 모델)",
            value=f"{log_test_acc * 100:.2f}%"
        )
        st.caption(f"훈련 정확도: **{log_train_acc * 100:.2f}%** | 테스트 정확도: **{log_test_acc * 100:.2f}%**")

    with c3:
        st.metric(
            label="🌳 의사결정트리 (질문으로 답하는 모델)",
            value=f"{dt_test_acc * 100:.2f}%"
        )
        st.caption(f"훈련 정확도: **{dt_train_acc * 100:.2f}%** | 테스트 정확도: **{dt_test_acc * 100:.2f}%**")

    st.markdown("---")

    # -------------------------------------------------------------------------
    # 8. Scatter Plot & Decision Boundary Visualization
    # -------------------------------------------------------------------------
    st.subheader("🎨 산점도 및 결정 경계 시각화")

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

    # 테스트 데이터 중 선택되지 않은 속성의 중앙값 계산
    other_feats = [f for f in selected_features if f not in [x_feat, y_feat]]
    other_medians = {}
    fixed_values_text = []

    for f in other_feats:
        med_val = float(np.median(test_df[f]))
        other_medians[f] = med_val
        fixed_values_text.append(f"**{feature_map[f]}**: {med_val:.2f}")

    if fixed_values_text:
        st.info("💡 **선택되지 않은 축 속성의 고정값 (테스트 데이터 중앙값)**: " + ", ".join(fixed_values_text))

    # Grid generation for decision regions
    x_min, x_max = test_df[x_feat].min(), test_df[x_feat].max()
    y_min, y_max = test_df[y_feat].min(), test_df[y_feat].max()

    x_margin = (x_max - x_min) * 0.05 if x_max != x_min else 1.0
    y_margin = (y_max - y_min) * 0.05 if y_max != y_min else 1.0

    x_grid = np.linspace(x_min - x_margin, x_max + x_margin, 100)
    y_grid = np.linspace(y_min - y_margin, y_max + y_margin, 100)
    xx, yy = np.meshgrid(x_grid, y_grid)

    grid_points = np.zeros((xx.size, len(selected_features)))
    grid_points[:, x_idx] = xx.ravel()
    grid_points[:, y_idx] = yy.ravel()

    for f in other_feats:
        idx = selected_features.index(f)
        grid_points[:, idx] = other_medians[f]

    # Predict Decision Tree regions on grid
    dt_grid_preds = dt_clf.predict(grid_points).reshape(xx.shape)

    fig = go.Figure()

    # Contour for Decision Tree regions
    fig.add_trace(go.Contour(
        x=x_grid,
        y=y_grid,
        z=dt_grid_preds,
        showscale=False,
        colorscale=[[0, 'rgba(144, 238, 144, 0.25)'], [1, 'rgba(255, 182, 193, 0.25)']],
        name="의사결정트리 영역",
        hoverinfo="skip"
    ))

    # Scatter points for test set
    test_df_copy = test_df.copy()
    test_df_copy["stroke_label"] = test_df_copy["stroke"].map({0: "아님 (0)", 1: "뇌졸중 (1)"})

    for label, color, symbol in [("아님 (0)", "#1f77b4", "circle"), ("뇌졸중 (1)", "#d62728", "x")]:
        sub = test_df_copy[test_df_copy["stroke_label"] == label]
        fig.add_trace(go.Scatter(
            x=sub[x_feat],
            y=sub[y_feat],
            mode="markers",
            name=f"실제 {label}",
            marker=dict(color=color, symbol=symbol, size=7, opacity=0.8)
        ))

    # Logistic Regression 0.5 decision line calculation
    # w_x * x + w_y * y + sum(w_other * val_other) + intercept = 0
    coef = log_reg.coef_[0]
    intercept = log_reg.intercept_[0]

    w_x = coef[x_idx]
    w_y = coef[y_idx]

    const_term = intercept
    for f in other_feats:
        f_idx = selected_features.index(f)
        const_term += coef[f_idx] * other_medians[f]

    is_line_outside = False

    if abs(w_y) > 1e-6:
        y_line = -(w_x * x_grid + const_term) / w_y
        valid_mask = (y_line >= y_min - y_margin) & (y_line <= y_max + y_margin)
        if np.any(valid_mask):
            fig.add_trace(go.Scatter(
                x=x_grid[valid_mask],
                y=y_line[valid_mask],
                mode="lines",
                name="로지스틱 회귀 경계선 (0.5)",
                line=dict(color="black", width=2, dash="dash")
            ))
        else:
            is_line_outside = True
    else:
        is_line_outside = True

    fig.update_layout(
        title=f"테스트 데이터 산점도 및 결정 경계 ({x_korean} vs {y_korean})",
        xaxis_title=x_korean,
        yaxis_title=y_korean,
        legend_title="범례"
    )

    st.plotly_chart(fig, use_container_width=True)

    if is_line_outside:
        st.warning("⚠️ 로지스틱 회귀 모델의 0.5 결정 경계선이 현재 표시된 그래프 영역 밖에 존재합니다.")

    st.markdown("---")

    # -------------------------------------------------------------------------
    # 9. Decision Tree Graphviz Visualization
    # -------------------------------------------------------------------------
    st.subheader("🌳 의사결정트리 구조 (질문 가지)")

    tree_ = dt_clf.tree_

    def build_dot(tree, feature_names):
        dot_lines = [
            'digraph Tree {',
            'node [shape=box, style="filled, rounded", color="black", fontname="helvetica"];',
            'edge [fontname="helvetica"];'
        ]

        asked_features = set()
        leaf_count = 0
        normal_leaf_count = 0

        def traverse(node_id):
            nonlocal leaf_count, normal_leaf_count
            n_samples = tree.n_node_samples[node_id]
            value = tree.value[node_id][0]
            stroke_cnt = int(value[1]) if len(value) > 1 else 0
            ratio = (stroke_cnt / n_samples * 100) if n_samples > 0 else 0.0

            is_leaf = (tree.children_left[node_id] == -1) and (tree.children_right[node_id] == -1)

            if is_leaf:
                leaf_count += 1
                pred_cls = np.argmax(value)
                if pred_cls == 0:
                    normal_leaf_count += 1
                    fillcolor = "#e5f5e0"  # 연한 녹색 (아님)
                    ans_str = "답: 아님 (0)"
                else:
                    fillcolor = "#fee0d2"  # 연한 빨간색 (뇌졸중)
                    ans_str = "답: 뇌졸중 (1)"

                label = f"{ans_str}\\n훈련 사람 수: {n_samples:,}명\\n실제 뇌졸중: {stroke_cnt:,}명 ({ratio:.1f}%)"
                dot_lines.append(f'{node_id} [label="{label}", fillcolor="{fillcolor}"];')
            else:
                f_idx = tree.feature[node_id]
                f_name_eng = feature_names[f_idx]
                f_name_kor = feature_map[f_name_eng]
                asked_features.add(f_name_kor)
                threshold = tree.threshold[node_id]

                label = f"{f_name_kor} <= {threshold:.2f}?\\n훈련 사람 수: {n_samples:,}명\\n실제 뇌졸중: {stroke_cnt:,}명 ({ratio:.1f}%)"
                dot_lines.append(f'{node_id} [label="{label}", fillcolor="#ffffff"];')

                left_child = tree.children_left[node_id]
                right_child = tree.children_right[node_id]

                dot_lines.append(f'{node_id} -> {left_child} [label="예"];')
                dot_lines.append(f'{node_id} -> {right_child} [label="아니요"];')

                traverse(left_child)
                traverse(right_child)

        traverse(0)
        dot_lines.append('}')
        return "\n".join(dot_lines), leaf_count, normal_leaf_count, asked_features

    dot_str, leaf_cnt, normal_leaf_cnt, asked_feats = build_dot(tree_, selected_features)

    st.graphviz_chart(dot_str)

    # Tree Summary Info
    asked_str = ", ".join(list(asked_feats)) if asked_feats else "없음"
    st.write(f"📌 **답을 내는 마디(최종 잎 마디) 개수**: 총 **{leaf_cnt}**개 중 **{normal_leaf_cnt}**개 칸이 **'아님(0)'**으로 답합니다.")
    st.write(f"❓ **의사결정트리가 실제로 물어본 속성**: **{asked_str}**")
