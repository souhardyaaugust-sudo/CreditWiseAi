import os
import sys
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# Add backend/src to path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(BASE_DIR, "backend", "src")
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from data_manager import download_dataset, load_raw_data, get_dataset_stats
from pipeline import LoanPreprocessor
from model_manager import ModelManager
from history_manager import HistoryManager

# Page Configuration
st.set_page_config(
    page_title="CreditAnalyzer AI",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.0rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #E2E8F0;
        text-align: center;
    }
    .approved-badge {
        background-color: #DCFCE7;
        color: #15803D;
        padding: 0.5rem 1rem;
        border-radius: 0.375rem;
        font-weight: 700;
        font-size: 1.25rem;
        text-align: center;
    }
    .rejected-badge {
        background-color: #FEE2E2;
        color: #B91C1C;
        padding: 0.5rem 1rem;
        border-radius: 0.375rem;
        font-weight: 700;
        font-size: 1.25rem;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Initialize Backend Managers
@st.cache_resource
def get_managers():
    download_dataset()
    history_mgr = HistoryManager()
    model_mgr = ModelManager()
    return history_mgr, model_mgr

history_manager, model_manager = get_managers()

# Helper to ensure baseline model exists
def ensure_models_trained():
    metadata = model_manager.load_metadata()
    if not metadata:
        df = load_raw_data()
        X = df.drop(columns=['Loan_ID', 'Loan_Status'])
        y = df['Loan_Status']
        preprocessor = LoanPreprocessor()
        preprocessor.fit(X)
        preprocessor.save()
        X_trans = preprocessor.transform(X)
        model_manager.train_all(X_trans, y, use_grid_search=False)
        metadata = model_manager.load_metadata()
    return metadata

# Main Header
st.markdown('<div class="main-header">💳 CreditAnalyzer AI</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Intelligent Credit Risk & Loan Approval Prediction Platform</div>', unsafe_allow_html=True)

# Sidebar Navigation
st.sidebar.title("Navigation")
menu = st.sidebar.radio(
    "Select Module",
    ["📊 Overview & Analytics", "🔍 Dataset Explorer", "🤖 Model Benchmarking", "🎯 Loan Predictor", "📜 Prediction History"]
)

st.sidebar.markdown("---")
st.sidebar.caption("WiseCredit AI • Machine Learning Platform")

# ----------------------------------------------------
# 1. OVERVIEW & ANALYTICS
# ----------------------------------------------------
if menu == "📊 Overview & Analytics":
    st.header("📊 Dataset Overview & Financial Analytics")
    
    df = load_raw_data()
    stats = get_dataset_stats()
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Applicants", len(df))
    with col2:
        approved_pct = (df['Loan_Status'] == 'Y').mean() * 100
        st.metric("Approval Rate", f"{approved_pct:.1f}%")
    with col3:
        avg_inc = df['ApplicantIncome'].mean()
        st.metric("Avg Applicant Income", f"${avg_inc:,.0f}")
    with col4:
        avg_loan = df['LoanAmount'].mean()
        st.metric("Avg Loan Amount ($k)", f"${avg_loan:.1f}k")
        
    st.markdown("---")
    
    chart_col1, chart_col2 = st.columns(2)
    
    with chart_col1:
        st.subheader("Loan Approval Distribution")
        fig, ax = plt.subplots(figsize=(6, 4))
        status_counts = df['Loan_Status'].value_counts()
        sns.barplot(x=status_counts.index, y=status_counts.values, palette=["#22c55e", "#ef4444"], ax=ax)
        ax.set_xticklabels(["Approved (Y)", "Rejected (N)"])
        ax.set_ylabel("Count")
        st.pyplot(fig)
        
    with chart_col2:
        st.subheader("Credit History vs Loan Approval")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.countplot(data=df, x='Credit_History', hue='Loan_Status', palette=["#ef4444", "#22c55e"], ax=ax)
        ax.set_xticklabels(["No Credit History (0)", "Has Credit History (1)"])
        ax.set_ylabel("Count")
        st.pyplot(fig)

    st.subheader("Numeric Feature Correlation Matrix")
    num_cols = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term']
    df_num = df[num_cols].dropna()
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.heatmap(df_num.corr(), annot=True, cmap="Blues", fmt=".2f", ax=ax)
    st.pyplot(fig)

# ----------------------------------------------------
# 2. DATASET EXPLORER
# ----------------------------------------------------
elif menu == "🔍 Dataset Explorer":
    st.header("🔍 Dataset Explorer")
    
    df = load_raw_data()
    
    col1, col2, col3 = st.columns(3)
    with col1:
        status_f = st.selectbox("Loan Status Filter", ["All", "Y", "N"])
    with col2:
        edu_f = st.selectbox("Education Filter", ["All"] + list(df['Education'].dropna().unique()))
    with col3:
        search_id = st.text_input("Search by Loan ID", "")
        
    filtered_df = df.copy()
    if status_f != "All":
        filtered_df = filtered_df[filtered_df['Loan_Status'] == status_f]
    if edu_f != "All":
        filtered_df = filtered_df[filtered_df['Education'] == edu_f]
    if search_id:
        filtered_df = filtered_df[filtered_df['Loan_ID'].str.contains(search_id, case=False, na=False)]
        
    st.markdown(f"Showing **{len(filtered_df)}** records:")
    st.dataframe(filtered_df, use_container_width=True)

# ----------------------------------------------------
# 3. MODEL BENCHMARKING
# ----------------------------------------------------
elif menu == "🤖 Model Benchmarking":
    st.header("🤖 Machine Learning Model Benchmarking")
    
    ensure_models_trained()
    metadata = model_manager.load_metadata()
    
    use_grid_search = st.checkbox("Enable GridSearchCV Hyperparameter Tuning (Takes a few seconds)", value=False)
    
    if st.button("⚡ Retrain & Benchmark Models"):
        with st.spinner("Preprocessing data and training all models..."):
            df = load_raw_data()
            X = df.drop(columns=['Loan_ID', 'Loan_Status'])
            y = df['Loan_Status']
            preprocessor = LoanPreprocessor()
            preprocessor.fit(X)
            preprocessor.save()
            X_trans = preprocessor.transform(X)
            metadata = model_manager.train_all(X_trans, y, use_grid_search=use_grid_search)
            st.success("All models successfully trained and benchmarked!")
            
    if metadata:
        st.subheader("Model Performance Comparison")
        records = []
        for name, metrics in metadata["models"].items():
            records.append({
                "Model Name": metrics["display_name"],
                "Accuracy": f"{metrics['accuracy']:.4f}",
                "Precision": f"{metrics['precision']:.4f}",
                "Recall": f"{metrics['recall']:.4f}",
                "F1 Score": f"{metrics['f1_score']:.4f}",
                "Training Time (s)": f"{metrics['training_time']:.3f}",
                "Best Model": "🌟 YES" if name == metadata["best_model_name"] else ""
            })
        st.table(pd.DataFrame(records))
        
        st.subheader("Confusion Matrix Visualizer")
        selected_model_key = st.selectbox(
            "Select Model to Inspect",
            list(metadata["models"].keys()),
            format_func=lambda x: metadata["models"][x]["display_name"]
        )
        
        cm = metadata["models"][selected_model_key]["confusion_matrix"]
        fig, ax = plt.subplots(figsize=(5, 3.5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Greens", 
                    xticklabels=["Rejected (0)", "Approved (1)"],
                    yticklabels=["Rejected (0)", "Approved (1)"], ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)

# ----------------------------------------------------
# 4. LOAN PREDICTOR
# ----------------------------------------------------
elif menu == "🎯 Loan Predictor":
    st.header("🎯 Real-Time Loan Approval Predictor")
    
    metadata = ensure_models_trained()
    
    with st.form("loan_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            gender = st.selectbox("Gender", ["Male", "Female"])
            married = st.selectbox("Married", ["Yes", "No"])
            dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
            education = st.selectbox("Education", ["Graduate", "Not Graduate"])
        with col2:
            self_employed = st.selectbox("Self Employed", ["No", "Yes"])
            applicant_income = st.number_input("Applicant Monthly Income ($)", value=5000, step=500)
            coapplicant_income = st.number_input("Coapplicant Monthly Income ($)", value=0, step=500)
            loan_amount = st.number_input("Loan Amount ($ in thousands)", value=150, step=10)
        with col3:
            loan_term = st.selectbox("Loan Term (Months)", [360, 180, 240, 120, 84, 60], index=0)
            credit_history = st.selectbox("Credit History", [1.0, 0.0], format_func=lambda x: "Good / Cleared (1.0)" if x == 1.0 else "Debts / Uncleared (0.0)")
            property_area = st.selectbox("Property Area", ["Semiurban", "Urban", "Rural"])
            
            model_options = {"best_model": f"Best Model ({metadata['models'][metadata['best_model_name']]['display_name']})"}
            for k, v in metadata["models"].items():
                model_options[k] = v["display_name"]
            selected_model = st.selectbox("Select Classification Model", list(model_options.keys()), format_func=lambda x: model_options[x])

        submit_btn = st.form_submit_button("🔮 Predict Loan Approval", type="primary", use_container_width=True)

    if submit_btn:
        input_data = {
            "Gender": gender,
            "Married": married,
            "Dependents": dependents,
            "Education": education,
            "Self_Employed": self_employed,
            "ApplicantIncome": float(applicant_income),
            "CoapplicantIncome": float(coapplicant_income),
            "LoanAmount": float(loan_amount),
            "Loan_Amount_Term": float(loan_term),
            "Credit_History": float(credit_history),
            "Property_Area": property_area
        }
        
        preprocessor = LoanPreprocessor.load()
        active_model_name = metadata["best_model_name"] if selected_model == "best_model" else selected_model
        model = model_manager.load_model(active_model_name)
        
        X_single = preprocessor.transform_single(input_data)
        pred = int(model.predict(X_single)[0])
        status = "Approved" if pred == 1 else "Rejected"
        
        if hasattr(model, "predict_proba"):
            prob = float(model.predict_proba(X_single)[0][1])
        else:
            prob = 1.0 if pred == 1 else 0.0
            
        explanations = model_manager.explain_prediction(
            active_model_name, input_data, pred, prob, preprocessor.feature_cols
        )
        
        history_manager.add_prediction(input_data, status, prob, active_model_name)
        
        st.markdown("---")
        st.subheader("Prediction Result")
        
        res_col1, res_col2 = st.columns(2)
        with res_col1:
            if status == "Approved":
                st.markdown('<div class="approved-badge">✅ LOAN APPROVED</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="rejected-badge">❌ LOAN REJECTED</div>', unsafe_allow_html=True)
            st.write("")
            st.metric("Approval Probability", f"{prob*100:.1f}%")
            st.progress(prob)
            st.caption(f"Model Used: **{metadata['models'][active_model_name]['display_name']}**")

        with res_col2:
            st.markdown("**💡 Key Decision Explanations:**")
            for exp in explanations:
                st.info(f"• {exp}")

# ----------------------------------------------------
# 5. PREDICTION HISTORY
# ----------------------------------------------------
elif menu == "📜 Prediction History":
    st.header("📜 Prediction Audit History Log")
    
    history = history_manager.get_history(limit=50)
    if history:
        df_hist = pd.DataFrame(history)
        st.dataframe(df_hist, use_container_width=True)
    else:
        st.info("No prediction history recorded yet. Make a prediction in the Loan Predictor module!")
