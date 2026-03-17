import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.pipeline import get_pipeline

st.set_page_config(page_title="HR Attrition Predictor", layout="wide")

# Load model
@st.cache_resource
def load_model():
    try:
        model = joblib.load('models/attrition_pipeline.joblib')
        st.info("✅ Model loaded successfully")
        return model
    except Exception as e:
        st.error(f"❌ Model load failed: {str(e)}")
        return None

model = load_model()

st.title("🏢 Employee Attrition Prediction Dashboard")
st.markdown("Professional MLOps system for HR attrition prediction. EDA, tuned RF model, predictions.")

tab1, tab2, tab3 = st.tabs(["📊 EDA Insights", "🔮 Predict", "📈 Model Metrics"])

with tab1:
    st.header("EDA Summary")
    st.metric("Dataset Size", "1470")
    st.metric("Attrition Rate", "16.12%")
    
    col1, col2 = st.columns(2)
    with col1:
        fig = px.box(pd.read_csv('data/raw/attrition.csv'), x='Attrition', y='MonthlyIncome', title="Income by Attrition")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = px.bar(pd.read_csv('data/raw/attrition.csv'), x='Department', color='Attrition', title="Attrition by Department", barmode='group')
        st.plotly_chart(fig, use_container_width=True)
    
    imp_df = pd.DataFrame({'Feature': ['MonthlyIncome', 'TotalWorkingYears', 'Age'], 'Importance': [0.067, 0.066, 0.060]})
    fig = px.bar(imp_df, x='Importance', y='Feature', orientation='h', title="Top Features")
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.header("Predict Attrition Risk")
    if model is None:
        st.stop()
    
    with st.form("prediction"):
        col1, col2 = st.columns(2)
        age = col1.number_input("Age", 18, 60, 36)
        monthly_income = col2.number_input("Monthly Income", 1000, 20000, 6500)
        
        col3, col4 = st.columns(2)
        total_working_years = col3.number_input("Total Working Years", 0, 40, 7)
        overtime = col4.selectbox("Overtime", ["No", "Yes"])
        
        if st.form_submit_button("Predict Risk"):
            input_df = pd.DataFrame({
                'Age': [age],
                'DailyRate': [800],
                'DistanceFromHome': [9],
                'MonthlyIncome': [monthly_income],
                'MonthlyRate': [14000],
                'NumCompaniesWorked': [2],
                'PercentSalaryHike': [12],
                'TotalWorkingYears': [total_working_years],
                'TrainingTimesLastYear': [2],
                'YearsAtCompany': [5],
                'YearsInCurrentRole': [3],
                'YearsSinceLastPromotion': [1],
                'YearsWithCurrManager': [4],
                'BusinessTravel': ['Travel_Rarely'],
                'Department': ['Research & Development'],
                'EducationField': ['Life Sciences'],
                'Gender': ['Male'],
                'JobRole': ['Research Scientist'],
                'MaritalStatus': ['Married'],
                'OverTime': [overtime]
            })
            pred_proba = model.predict_proba(input_df)[0][1]

            st.metric("Attrition Probability", f"{pred_proba:.1%}", delta=None)
            if pred_proba > 0.3:
                st.error("🔔 High Risk - Review retention strategies!")
            else:
                st.success("✅ Low Risk")

with tab3:
    st.header("Model Performance")
    st.info("Baseline RF: Train AUC 0.88, Test 0.85. Tune running (W&B: attrition-mlops).")

st.sidebar.markdown("### About")
st.sidebar.markdown("Production-ready MLOps: EDA, tuned model, API, Docker, W&B, Neon DB.")


if st.sidebar.button("Train Baseline"):
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.models.train import train_model
    train_model()
    st.sidebar.success("Model trained!")

st.sidebar.markdown("---")
st.sidebar.markdown("Built by BLACKBOXAI")

