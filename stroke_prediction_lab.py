import os
import json
import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# -----------------------------------------------------------------------------
# 1. Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="뇌졸중 예측 실습실",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# 2. Local Storage Setup (내_기록/notes.json)
# -----------------------------------------------------------------------------
SAVE_DIR = "내_기록"
SAVE_FILE = os.path.join(SAVE_DIR, "notes.json")

def ensure_save_directory():
    """Ensure the directory for saving records exists."""
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)

def load_user_notes():
    """Load user notes (column Korean meanings and data source text) from JSON."""
    if os.path.exists(SAVE_FILE):
        try:
            with open(SAVE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {"column_meanings": {}, "data_source": ""}

def save_user_notes(data):
    """Save user notes to JSON file."""
    ensure_save_directory()
    with open(SAVE_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

# Load saved notes into session state if available
saved_notes = load_user_notes()

# -----------------------------------------------------------------------------
# 3. Data Loading & Caching
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

# Sidebar Navigation
st.sidebar.header("📌 메뉴 Navigation")
menu = st.sidebar.radio(
    "원하는 화면을 선택하세요:",
    ["📊 데이터 소개 (Home)", "🔍 데이터 탐색 & 시각화", "🤖 뇌졸중 예측 모델링"]
)

# -----------------------------------------------------------------------------
# TAB 1: 데이터 소개 (Home)
# -----------------------------------------------------------------------------
if menu == "📊 데이터 소개 (Home)":
    st.title("🧠 뇌졸중 데이터셋 소개")
    st.markdown("머신러닝 기법을 활용하여 뇌졸중 발병 위험도를 분석하고 예측하기 위한 기본 데이터 정보입니다.")
    st.divider()

    if data_load_success:
        # Key Metrics Display (4 Big Cards)
        total_count = len(df)
        num_columns = len(df.columns)
        stroke_count = int(df['stroke'].sum()) if 'stroke' in df.columns else 0
        stroke_ratio = (stroke_count / total_count * 100) if total_count > 0 else 0

        c1, c2, c3, c4 = st.columns(4)
        c1.metric(label="👥 전체 사람 수", value=f"{total_count:,} 명")
        c2.metric(label="📐 열(특성) 개수", value=f"{num_columns} 개")
        c3.metric(label="🚨 뇌졸중 환자 수 (stroke=1)", value=f"{stroke_count:,} 명")
        c4.metric(label="📈 뇌졸중 발병 비율", value=f"{stroke_ratio:.2f} %")

        st.markdown("---")

        # Section: Column Description Table with Editable Korean Meanings
        st.write("### 📝 데이터 열(Column) 상세 정보")
        st.info("교재를 참고하여 각 열의 **우리말 뜻**을 직접 입력해 보세요! 작성한 내용은 `내_기록/notes.json`에 자동으로 저장됩니다.")

        # Compute information for each column
        cols_info = []
        saved_meanings = saved_notes.get("column_meanings", {})

        for col in df.columns:
            dtype_str = str(df[col].dtype)
            missing_count = int(df[col].isna().sum())
            korean_val = saved_meanings.get(col, "")
            
            cols_info.append({
                "열 이름 (Column)": col,
                "우리말 뜻": korean_val,
                "값의 종류 (Data Type)": dtype_str,
                "빈 값 개수 (Missing Values)": missing_count
            })

        info_df = pd.DataFrame(cols_info)

        # Editable data editor for column meanings
        edited_df = st.data_editor(
            info_df,
            column_config={
                "열 이름 (Column)": st.column_config.Column(disabled=True),
                "우리말 뜻": st.column_config.TextColumn(
                    "우리말 뜻 (직접 입력)",
                    help="교재를 보고 알맞은 한국어 의미를 채워 넣어 보세요.",
                    width="medium"
                ),
                "값의 종류 (Data Type)": st.column_config.Column(disabled=True),
                "빈 값 개수 (Missing Values)": st.column_config.NumberColumn(disabled=True),
            },
            hide_index=True,
            use_container_width=True,
            key="column_meanings_editor"
        )

        # Save button for Column Meanings
        if st.button("💾 우리말 뜻 저장하기", key="save_meanings_btn"):
            new_meanings = {}
            for idx, row in edited_df.iterrows():
                new_meanings[row["열 이름 (Column)"]] = str(row["우리말 뜻"]) if pd.notna(row["우리말 뜻"]) else ""
            
            saved_notes["column_meanings"] = new_meanings
            save_user_notes(saved_notes)
            st.success("우리말 뜻이 `내_기록/notes.json`에 성공적으로 저장되었습니다!")

        st.markdown("---")

        # Display raw data (First 5 rows)
        st.write("### 🔍 데이터 미리보기 (처음 5행)")
        st.dataframe(df.head(5), use_container_width=True)

        st.markdown("---")

        # Section: Data Source Field
        st.write("### 📌 데이터 출처")
        current_source_text = saved_notes.get("data_source", "")
        
        user_source_input = st.text_area(
            "데이터 출처를 작성해 주세요 (교재 내용 참고):",
            value=current_source_text,
            height=100,
            placeholder="예: Kaggle Stroke Prediction Dataset / 교재 p.123 참고..."
        )

        if st.button("💾 데이터 출처 저장하기", key="save_source_btn"):
            saved_notes["data_source"] = user_source_input
            save_user_notes(saved_notes)
            st.success("데이터 출처 정보가 `내_기록/notes.json`에 성공적으로 저장되었습니다!")

# -----------------------------------------------------------------------------
# TAB 2: 데이터 탐색 & 시각화 (Plotly)
# -----------------------------------------------------------------------------
elif menu == "🔍 데이터 탐색 & 시각화":
    st.title("🔍 데이터 탐색 & 시각화")
    st.markdown("Plotly 그래프를 활용하여 범주형 및 수치형 변수들과 뇌졸중의 관계를 탐색합니다.")
    st.divider()

    if data_load_success:
        tab1, tab2 = st.tabs(["범주형 데이터 분석", "수치형 데이터 분석"])

        with tab1:
            st.write("##### 뇌졸중 여부에 따른 범주형 변수 분포")
            cat_col = st.selectbox(
                "분석할 범주형 열을 선택하세요:",
                ["gender", "hypertension", "heart_disease", "ever_married", "work_type", "Residence_type", "smoking_status"]
            )
            
            fig_bar = px.histogram(
                df, 
                x=cat_col, 
                color="stroke", 
                barmode="group",
                title=f"{cat_col} 변수에 따른 뇌졸중(stroke) 발생 비율",
                color_discrete_sequence=px.colors.qualitative.Set2
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        with tab2:
            st.write("##### 수치형 변수 분포 및 상관관계")
            num_col = st.selectbox("분석할 수치형 열을 선택하세요:", ["age", "avg_glucose_level", "bmi"])
            
            fig_box = px.box(
                df, 
                x="stroke", 
                y=num_col, 
                color="stroke",
                title=f"뇌졸중 발병 유무에 따른 {num_col} 분포 (Box Plot)",
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            st.plotly_chart(fig_box, use_container_width=True)

# -----------------------------------------------------------------------------
# TAB 3: 뇌졸중 예측 모델링 (Scikit-learn)
# -----------------------------------------------------------------------------
elif menu == "🤖 뇌졸중 예측 모델링":
    st.title("🤖 뇌졸중 예측 모델링")
    st.markdown("Random Forest 모델을 사용하여 뇌졸중 위험도를 예측하고 중요 변수 및 평가 결과를 확인합니다.")
    st.divider()

    if data_load_success:
        st.write("간단한 전처리(결측치 대체 및 원-핫 인코딩)를 수행한 후 모델을 학습시킵니다.")

        if st.button("🚀 모델 학습 실행"):
            with st.spinner("모델을 학습 중입니다..."):
                # Preprocessing
                model_df = df.drop(columns=['id'], errors='ignore').copy()
                
                # Fill missing BMI values with median
                model_df['bmi'].fillna(model_df['bmi'].median(), inplace=True)
                
                # One-hot encoding for categorical values
                cat_cols = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
                model_df = pd.get_dummies(model_df, columns=cat_cols, drop_first=True)
                
                # Separate Features and Target
                X = model_df.drop(columns=['stroke'])
                y = model_df['stroke']

                # Train Test Split
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, random_state=42, test_size=0.2, stratify=y
                )

                # Random Forest Classifier
                rf = RandomForestClassifier(n_estimators=100, random_state=42)
                rf.fit(X_train, y_train)

                # Predict
                y_pred = rf.predict(X_test)
                acc = accuracy_score(y_test, y_pred)

                st.success(f"✅ 모델 학습 완료! 테스트 정확도(Accuracy): {acc:.4f}")

                col_res1, col_res2 = st.columns(2)
                
                with col_res1:
                    st.write("##### 혼동 행렬 (Confusion Matrix)")
                    cm = confusion_matrix(y_test, y_pred)
                    fig_cm = px.imshow(
                        cm, 
                        text_auto=True, 
                        labels=dict(x="예측값", y="실제값"),
                        x=['정상(0)', '뇌졸중(1)'],
                        y=['정상(0)', '뇌졸중(1)'],
                        color_continuous_scale="Viridis"
                    )
                    st.plotly_chart(fig_cm, use_container_width=True)

                with col_res2:
                    st.write("##### 특성 중요도 (Feature Importance)")
                    feat_imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False).head(10)
                    fig_imp = px.bar(
                        x=feat_imp.values, 
                        y=feat_imp.index, 
                        orientation='h',
                        labels={'x': '중요도', 'y': '특성'},
                        title="상위 10개 주요 변수"
                    )
                    fig_imp.update_layout(yaxis=dict(autorange="reversed"))
                    st.plotly_chart(fig_imp, use_container_width=True)
