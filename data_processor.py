# data_processor.py
import pandas as pd
import numpy as np
from datetime import datetime

def load_and_prepare_data():
    """Load and prepare data for visualization"""
    # Load your processed dataset
    df = pd.read_csv('ocr_prediction_enhanced.csv')
    
    # Ensure proper date formatting
    df['month'] = pd.to_datetime(df['month'])
    df = df.sort_values('month')
    
    # Calculate additional metrics for visualization
    df['OCR_change'] = df['OCR'].diff()
    df['CPI_target_deviation'] = df['CPI_pct'] - 2.0  # 2% target
    df['policy_aggressiveness'] = df['OCR_change'].abs()
    
    return df

# Create summary statistics for dashboard
def create_model_performance_data():
    """Create model performance data for visualization"""
    performance_data = {
        'Model': ['Linear Regression', 'Random Forest', 'Gradient Boosting', 'Ensemble'],
        'R2_Score': [0.9822, 0.9626, 0.9534, 0.9891],
        'RMSE': [0.1886, 0.2736, 0.2441, 0.1623],
        'MAE': [0.1582, 0.1966, 0.1734, 0.1445]
    }
    return pd.DataFrame(performance_data)

def create_feature_importance_data():
    """Create feature importance data"""
    features = {
        'Feature': ['OCR_lag1', 'OCR', 'CPI_pct', 'FloatingMortgage', 'CoreInflation',
                   'Mortgage_OCR_spread', 'TermDeposit6M', 'UnemploymentRate', 
                   'HousePriceGrowth', 'TWI'],
        'Importance': [0.287, 0.194, 0.156, 0.089, 0.067, 0.054, 0.048, 0.042, 0.038, 0.025],
        'Category': ['Policy Persistence', 'Policy Persistence', 'Inflation', 'Transmission',
                    'Inflation', 'Transmission', 'Transmission', 'Employment', 'Housing', 'Exchange Rate']
    }
    return pd.DataFrame(features)