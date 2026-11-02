import streamlit as st
import pandas as pd
import plotly.express as px

# -----------------------------------------------------------------------------
# 1. Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="뇌졸중 예측 실습실 - 데이터 탐색",
    page_icon="🧠",
    layout="wide"
)

# -----------------------------------------------------------------------------
# 2. Data Loading & Caching
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

# -----------------------------------------------------------------------------
# 3. 데이터 탐색 화면
# -----------------------------------------------------------------------------
st.title("🔍 데이터 탐색 (EDA)")
st.markdown("뇌졸중 데이터셋의 변수별 분포 및 위험요인과의 관계를 탐색합니다.")
st.divider()

if data_load_success:
    # -------------------------------------------------------------------------
    # 1) 나이와 평균 혈당 분포 (히스토그램 두 개 나란히)
    # -------------------------------------------------------------------------
    st.subheader("1. 나이 및 평균 혈당 분포")
    col1, col2 = st.columns(2)

    with col1:
        fig_age_hist = px.histogram(
            df,
            x="age",
            nbins=30,
            title="나이(age) 분포",
            color_discrete_sequence=["#636EFA"]
        )
        fig_age_hist.update_layout(xaxis_title="나이", yaxis_title="인원 수")
        st.plotly_chart(fig_age_hist, use_container_width=True)

    with col2:
        fig_glucose_hist = px.histogram(
            df,
            x="avg_glucose_level",
            nbins=30,
            title="평균 혈당 수치(avg_glucose_level) 분포",
            color_discrete_sequence=["#EF553B"]
        )
        fig_glucose_hist.update_layout(xaxis_title="평균 혈당", yaxis_title="인원 수")
        st.plotly_chart(fig_glucose_hist, use_container_width=True)

    st.markdown("---")

    # -------------------------------------------------------------------------
    # 2) 뇌졸중 발병 유무에 따른 나이 및 평균 혈당 비교 (상자그림 & 평균 표)
    # -------------------------------------------------------------------------
    st.subheader("2. 뇌졸중 여부에 따른 나이 및 평균 혈당 비교")
    
    # Label mapping for stroke plot
    df_box = df.copy()
    df_box["stroke_label"] = df_box["stroke"].map({0: "정상 (0)", 1: "뇌졸중 (1)"})

    col_box1, col_box2 = st.columns(2)

    with col_box1:
        fig_age_box = px.box(
            df_box,
            x="stroke_label",
            y="age",
            color="stroke_label",
            title="뇌졸중 여부별 나이 비교 (Box Plot)",
            color_discrete_map={"정상 (0)": "#00CC96", "뇌졸중 (1)": "#AB63FA"}
        )
        fig_age_box.update_layout(xaxis_title="뇌졸중 여부", yaxis_title="나이")
        st.plotly_chart(fig_age_box, use_container_width=True)

    with col_box2:
        fig_glucose_box = px.box(
            df_box,
            x="stroke_label",
            y="avg_glucose_level",
            color="stroke_label",
            title="뇌졸중 여부별 평균 혈당 비교 (Box Plot)",
            color_discrete_map={"정상 (0)": "#00CC96", "뇌졸중 (1)": "#AB63FA"}
        )
        fig_glucose_box.update_layout(xaxis_title="뇌졸중 여부", yaxis_title="평균 혈당")
        st.plotly_chart(fig_glucose_box, use_container_width=True)

    # Group Mean Table
    st.write("##### 📊 두 그룹의 평균값 비교 표")
    mean_table = df.groupby("stroke")[["age", "avg_glucose_level"]].mean().reset_index()
    mean_table["stroke"] = mean_table["stroke"].map({0: "정상 (0)", 1: "뇌졸중 (1)"})
    mean_table.columns = ["구분", "평균 나이(세)", "평균 혈당 수치"]
    mean_table["평균 나이(세)"] = mean_table["평균 나이(세)"].round(2)
    mean_table["평균 혈당 수치"] = mean_table["평균 혈당 수치"].round(2)
    st.dataframe(mean_table, use_container_width=True, hide_index=True)

    st.markdown("---")

    # -------------------------------------------------------------------------
    # 3) 고혈압 및 심장병 여부별 뇌졸중 비율 (막대그래프)
    # -------------------------------------------------------------------------
    st.subheader("3. 기저질환(고혈압, 심장병) 유무별 뇌졸중 비율")
    col_bar1, col_bar2 = st.columns(2)

    with col_bar1:
        # Hypertension Stroke Ratio Calculation
        ht_stroke = df.groupby("hypertension")["stroke"].mean().reset_index()
        ht_stroke["stroke_ratio"] = ht_stroke["stroke"] * 100
        ht_stroke["hypertension_label"] = ht_stroke["hypertension"].map({0: "없음 (0)", 1: "있음 (1)"})

        fig_ht = px.bar(
            ht_stroke,
            x="hypertension_label",
            y="stroke_ratio",
            text_auto=".2f",
            title="고혈압(hypertension) 유무에 따른 뇌졸중 비율 (%)",
            color="hypertension_label",
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        fig_ht.update_layout(xaxis_title="고혈압 여부", yaxis_title="뇌졸중 발병 비율 (%)")
        st.plotly_chart(fig_ht, use_container_width=True)

    with col_bar2:
        # Heart Disease Stroke Ratio Calculation
        hd_stroke = df.groupby("heart_disease")["stroke"].mean().reset_index()
        hd_stroke["stroke_ratio"] = hd_stroke["stroke"] * 100
        hd_stroke["hd_label"] = hd_stroke["heart_disease"].map({0: "없음 (0)", 1: "있음 (1)"})

        fig_hd = px.bar(
            hd_stroke,
            x="hd_label",
            y="stroke_ratio",
            text_auto=".2f",
            title="심장병(heart_disease) 유무에 따른 뇌졸중 비율 (%)",
            color="hd_label",
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        fig_hd.update_layout(xaxis_title="심장병 여부", yaxis_title="뇌졸중 발병 비율 (%)")
        st.plotly_chart(fig_hd, use_container_width=True)

    st.markdown("---")

    # -------------------------------------------------------------------------
    # 4) BMI 결측치 그룹 vs 전체 그룹 뇌졸중 비율 비교 표
    # -------------------------------------------------------------------------
    st.subheader("4. BMI 결측치 집단과 전체 집단의 뇌졸중 비율 비교")
    
    bmi_na_df = df[df["bmi"].isna()]
    bmi_na_total = len(bmi_na_df)
    bmi_na_stroke = int(bmi_na_df["stroke"].sum())
    bmi_na_stroke_ratio = (bmi_na_stroke / bmi_na_total * 100) if bmi_na_total > 0 else 0.0

    total_count = len(df)
    total_stroke = int(df["stroke"].sum())
    total_stroke_ratio = (total_stroke / total_count * 100) if total_count > 0 else 0.0

    bmi_compare_df = pd.DataFrame([
        {
            "구분": "BMI 결측치 그룹 (bmi is NA)",
            "전체 인원 수(명)": bmi_na_total,
            "뇌졸중 환자 수(명)": bmi_na_stroke,
            "뇌졸중 발병 비율(%)": round(bmi_na_stroke_ratio, 2)
        },
        {
            "구분": "전체 데이터 집단",
            "전체 인원 수(명)": total_count,
            "뇌졸중 환자 수(명)": total_stroke,
            "뇌졸중 발병 비율(%)": round(total_stroke_ratio, 2)
        }
    ])

    st.dataframe(bmi_compare_df, use_container_width=True, hide_index=True)

    st.markdown("---")

    # -------------------------------------------------------------------------
    # 5) 흡연 상태 (smoking_status) 값별 인원수 표
    # -------------------------------------------------------------------------
    st.subheader("5. 흡연 상태(smoking_status)별 인원수")
    
    smoke_counts = df["smoking_status"].value_counts().reset_index()
    smoke_counts.columns = ["흡연 상태 (smoking_status)", "인원 수 (명)"]
    
    st.dataframe(smoke_counts, use_container_width=True, hide_index=True)
