# app.py
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from data_processor import load_and_prepare_data, create_model_performance_data, create_feature_importance_data

# Page configuration
st.set_page_config(
    page_title="NZ OCR Prediction System",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
def load_css():
    st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 0.5rem 0;
    }
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
    </style>
    """, unsafe_allow_html=True)

def main():
    load_css()
    
    # Sidebar navigation
    st.sidebar.title("🏦 Navigation")
    page = st.sidebar.selectbox(
        "Choose a section:",
        ["Executive Summary", "Economic Analysis", "Model Performance", 
         "Feature Importance", "Predictions & Scenarios", "Technical Details"]
    )
    
    # Load data
    df = load_and_prepare_data()
    
    # Page routing
    if page == "Executive Summary":
        executive_summary_page()
    elif page == "Economic Analysis":
        economic_analysis_page(df)
    elif page == "Model Performance":
        model_performance_page()
    elif page == "Feature Importance":
        feature_importance_page()
    elif page == "Predictions & Scenarios":
        predictions_page(df)
    elif page == "Technical Details":
        technical_details_page()

if __name__ == "__main__":
    main()