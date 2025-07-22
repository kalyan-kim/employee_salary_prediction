
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px

st.set_page_config(page_title="Employee Salary Predictor", layout="wide")
st.title("💼 AI-Powered Employee Salary Predictor")

try:
    model = joblib.load("model.pkl")
    X_test, y_test = joblib.load("test_data.pkl")
except Exception as e:
    st.error(f"❌ Error loading model or test data: {e}")
    st.stop()

with st.container():
    col1, col2, col3 = st.columns(3)
    with col1:
        education = st.selectbox("Education", sorted(X_test['Education'].unique()))
        job_title = st.selectbox("Job Title", sorted(X_test['Job_Title'].unique()))
    with col2:
        experience = st.number_input("Experience (Years)", min_value=0, max_value=50, value=int(X_test['Experience'].mean()))
        location = st.selectbox("Location", sorted(X_test['Location'].unique()))
    with col3:
        age = st.number_input("Age", min_value=18, max_value=70, value=int(X_test['Age'].mean()))
        gender = st.selectbox("Gender", sorted(X_test['Gender'].unique()))

input_dict = {
    "Education": [education],
    "Experience": [experience],
    "Location": [location],
    "Job_Title": [job_title],
    "Age": [age],
    "Gender": [gender]
}
input_df = pd.DataFrame(input_dict)

if st.button("Predict Salary"):
    try:
        prediction = model.predict(input_df)[0]
        st.success(f"💰 Predicted Salary: ₹{prediction:,.2f}")
    except Exception as e:
        st.error(f"Prediction failed: {e}")

st.markdown("---")
st.subheader("📊 Model Evaluation")

try:
    y_pred = model.predict(X_test)
    results = X_test.copy()
    results['Actual Salary'] = y_test
    results['Predicted Salary'] = y_pred
    results['Error'] = y_pred - y_test
    results['Error %'] = (abs(results['Error']) / y_test) * 100

    st.dataframe(results.head(20).style.format({
        'Actual Salary': '₹{:.0f}',
        'Predicted Salary': '₹{:.0f}',
        'Error': '₹{:.0f}',
        'Error %': '{:.2f}%'
    }))

    fig1 = px.scatter(x=y_test, y=y_pred, labels={"x": "Actual Salary", "y": "Predicted Salary"}, title="Actual vs Predicted Salary")
    fig1.add_shape(type="line", x0=y_test.min(), y0=y_test.min(), x1=y_test.max(), y1=y_test.max(), line=dict(color="red", dash="dash"))
    st.plotly_chart(fig1)

    fig2 = px.histogram(results, x='Error', nbins=30, title="Residual Error Distribution")
    st.plotly_chart(fig2)
except Exception as e:
    st.warning(f"⚠️ Plotting failed: {e}")
