"""Data processing utilities for NZ Economic Indicators project.

This module provides functions to load and prepare datasets, generate
sample data for demonstrations, and produce helper tables used by the
Streamlit application (model performance, feature importance, and
summary statistics).
"""

import pandas as pd
import numpy as np
from datetime import datetime
import os

def load_and_prepare_data():
    """Load and prepare dataset for visualizations and modeling.

    Returns
    -------
    pandas.DataFrame
        Prepared dataset with computed features and target variables. If a
        preprocessed CSV file named ``ocr_prediction_enhanced.csv`` exists in
        the working directory, it will be loaded. Otherwise sample data is
        generated for demonstration purposes.
    """
    try:
        # Attempt to load a preprocessed CSV file from disk
        if os.path.exists('ocr_prediction_enhanced.csv'):
            df = pd.read_csv('ocr_prediction_enhanced.csv')
            # Ensure the `month` column is a datetime type and sort chronologically
            df['month'] = pd.to_datetime(df['month'])
            df = df.sort_values('month')
        else:
            # Fall back to synthetic sample data when the file is not present
            df = create_sample_data()

        # Derive additional analytical columns used by the application
        df['OCR_change'] = df['OCR'].diff()
        df['CPI_target_deviation'] = df['CPI_pct'] - 2.0
        df['policy_aggressiveness'] = df['OCR_change'].abs()

        # Ensure lag features exist for modelling and analysis
        if 'OCR_lag1' not in df.columns:
            df['OCR_lag1'] = df['OCR'].shift(1)
        if 'OCR_lag2' not in df.columns:
            df['OCR_lag2'] = df['OCR'].shift(2)

        # Construct interest-rate spread variables when absent
        if 'Mortgage_OCR_spread' not in df.columns:
            df['Mortgage_OCR_spread'] = df['FloatingMortgage'] - df['OCR']
        if 'Deposit_OCR_spread' not in df.columns:
            df['Deposit_OCR_spread'] = df['TermDeposit6M'] - df['OCR']

        return df

    except Exception as e:
        # On failure, log the error and provide synthetic data to keep the
        # application functional in development environments.
        print(f"Error loading data: {e}")
        return create_sample_data()

def create_sample_data():
    """Generate synthetic economic time series for development and demos.

    The generated data mimics plausible month-on-month series for the OCR,
    CPI inflation, unemployment, housing growth and related financial
    indicators. This is intended for UI demos and testing when actual
    production data are not present.
    """
    np.random.seed(42)

    # Create monthly date range covering 2021-01 through 2025-08
    dates = pd.date_range(start='2021-01-01', end='2025-08-01', freq='M')
    n_periods = len(dates)

    # Synthetic OCR trajectory: low rates during COVID, tightening, then normalization
    ocr_trajectory = []
    for i, _date in enumerate(dates):
        if i < 10:
            # Early period: emergency low rates
            ocr_trajectory.append(0.25)
        elif i < 20:
            # Initial tightening phase
            ocr_trajectory.append(0.25 + (i - 10) * 0.5)
        elif i < 32:
            # Peak tightening
            ocr_trajectory.append(min(5.5, 0.25 + (i - 10) * 0.45))
        else:
            # Normalization towards a neutral rate
            ocr_trajectory.append(max(4.0, 5.5 - (i - 32) * 0.1))

    ocr_values = np.array(ocr_trajectory)

    # Construct correlated indicators with simple stylised relationships
    cpi_inflation = (
        7.5 * np.exp(-0.3 * np.maximum(0, ocr_values - 2))
        + np.random.normal(0, 0.5, n_periods)
    )
    cpi_inflation = np.clip(cpi_inflation, 0, 8)

    unemployment = 3.5 + 0.3 * ocr_values + np.random.normal(0, 0.2, n_periods)
    unemployment = np.clip(unemployment, 3.0, 6.0)

    house_growth = 25 - 4 * ocr_values + np.random.normal(0, 3, n_periods)

    floating_mortgage = ocr_values + 2.5 + np.random.normal(0, 0.1, n_periods)
    term_deposit = ocr_values - 0.5 + np.random.normal(0, 0.1, n_periods)

    core_inflation = cpi_inflation * 0.8 + np.random.normal(0, 0.3, n_periods)

    twi = 75 + np.cumsum(np.random.normal(0, 1, n_periods))

    df = pd.DataFrame(
        {
            'month': dates,
            'OCR': ocr_values,
            'CPI_pct': cpi_inflation,
            'UnemploymentRate': unemployment,
            'HousePriceGrowth': house_growth,
            'FloatingMortgage': floating_mortgage,
            'TermDeposit6M': term_deposit,
            'CoreInflation': core_inflation,
            'TWI': twi,
        }
    )

    # Target variables for next-period direction and level
    df['OCR_next'] = df['OCR'].shift(-1)
    df['OCR_direction'] = df.apply(
        lambda row: (
            'up'
            if pd.notna(row['OCR_next']) and row['OCR_next'] > row['OCR']
            else 'down'
            if pd.notna(row['OCR_next']) and row['OCR_next'] < row['OCR']
            else 'same'
        ),
        axis=1,
    )

    return df

def create_model_performance_data():
    """Return a DataFrame summarising model performance metrics.

    The values are illustrative for demonstration of the dashboard and do
    not represent production model evaluations.
    """
    performance_data = {
        'Model': ['Linear Regression', 'Random Forest', 'Gradient Boosting', 'Soft Voting Ensemble'],
        'R2_Score': [0.9822, 0.9626, 0.9534, 0.9891],
        'RMSE': [0.1886, 0.2736, 0.2441, 0.1623],
        'MAE': [0.1582, 0.1966, 0.1734, 0.1445],
        'Classification_Accuracy': [0.625, 0.500, 0.600, 0.750],
    }
    return pd.DataFrame(performance_data)

def create_feature_importance_data():
    """Provide illustrative feature importance values and descriptions.

    Returns
    -------
    pandas.DataFrame
        Table containing feature names, relative importance scores, categories
        and short descriptions used by the dashboard to explain model inputs.
    """
    features = {
        'Feature': [
            'OCR_lag1',
            'OCR',
            'CPI_pct',
            'FloatingMortgage',
            'CoreInflation',
            'Mortgage_OCR_spread',
            'TermDeposit6M',
            'UnemploymentRate',
            'HousePriceGrowth',
            'TWI',
            'CPI_change',
            'OCR_lag2',
        ],
        'Importance': [0.287, 0.194, 0.156, 0.089, 0.067, 0.054, 0.048, 0.042, 0.038, 0.025, 0.018, 0.015],
        'Category': [
            'Policy Persistence',
            'Policy Persistence',
            'Inflation',
            'Transmission',
            'Inflation',
            'Transmission',
            'Transmission',
            'Employment',
            'Housing',
            'Exchange Rate',
            'Inflation',
            'Policy Persistence',
        ],
        'Description': [
            'Previous month OCR (policy persistence)',
            'Current OCR level',
            'Consumer Price Index inflation rate',
            'Floating mortgage interest rate',
            'Core inflation (sectoral factor model)',
            'Spread between mortgage rate and OCR',
            'Six-month term deposit rate',
            'Unemployment rate (dual mandate)',
            'Annual house price growth rate',
            'Trade Weighted Index (17 currencies)',
            'Month-on-month CPI change',
            'OCR two months ago',
        ],
    }
    return pd.DataFrame(features)

def create_economic_summary_stats():
    """Compute summary statistics for selected economic indicators.

    Returns
    -------
    pandas.DataFrame
        Table containing min, max, mean and the most recent observation for
        key macro indicators used in the dashboard.
    """
    df = load_and_prepare_data()

    summary_stats = {
        'Indicator': ['OCR', 'CPI Inflation', 'Unemployment', 'House Price Growth'],
        'Min': [
            df['OCR'].min(),
            df['CPI_pct'].min(),
            df['UnemploymentRate'].min(),
            df['HousePriceGrowth'].min(),
        ],
        'Max': [
            df['OCR'].max(),
            df['CPI_pct'].max(),
            df['UnemploymentRate'].max(),
            df['HousePriceGrowth'].max(),
        ],
        'Mean': [
            df['OCR'].mean(),
            df['CPI_pct'].mean(),
            df['UnemploymentRate'].mean(),
            df['HousePriceGrowth'].mean(),
        ],
        'Current': [
            df['OCR'].iloc[-1],
            df['CPI_pct'].iloc[-1],
            df['UnemploymentRate'].iloc[-1],
            df['HousePriceGrowth'].iloc[-1],
        ],
    }

    return pd.DataFrame(summary_stats).round(2)

def get_policy_phases():
    """Return a list of dictionaries describing policy phases for plotting.

    Each dictionary contains a descriptive name, start/end dates, a color
    for visualization and a short description suitable for UI tooltips.
    """
    phases = [
        {
            'name': 'COVID Accommodation',
            'start': '2021-01-01',
            'end': '2021-10-01',
            'color': 'lightblue',
            'description': 'Emergency low interest rates (0.25%)',
        },
        {
            'name': 'Aggressive Tightening',
            'start': '2021-10-01',
            'end': '2023-05-01',
            'color': 'lightcoral',
            'description': 'Rapid policy tightening towards 5.5%',
        },
        {
            'name': 'Policy Normalization',
            'start': '2023-05-01',
            'end': '2025-08-01',
            'color': 'lightgreen',
            'description': 'Gradual normalization toward neutral rates',
        },
    ]
    return phases

def calculate_model_predictions():
    """Provide a simplified OCR prediction function for scenario analysis.

    The returned callable estimates the next-period OCR change given a
    small set of macro indicators. This is a stylised approximation used in
    the interactive dashboard and is not intended as production forecasting
    logic.
    """

    def predict_ocr_change(current_ocr, cpi_inflation, unemployment, house_growth):
        # Inflation targeting component (Taylor-style rule)
        inflation_gap = cpi_inflation - 2.0
        inflation_effect = inflation_gap * 0.5

        # Employment component (dual mandate influence)
        unemployment_gap = 4.5 - unemployment
        employment_effect = unemployment_gap * 0.3

        # Financial stability (housing) effect
        housing_effect = (house_growth - 5.0) * 0.05

        # Simple policy inertia term to avoid extreme month-to-month swings
        if current_ocr < 1.0:
            inertia_bias = 0.1
        elif current_ocr > 6.0:
            inertia_bias = -0.1
        else:
            inertia_bias = 0

        predicted_change = inflation_effect + employment_effect + housing_effect + inertia_bias

        # Constrain the monthly change to realistic policy move bounds
        predicted_change = np.clip(predicted_change, -0.5, 0.5)

        return predicted_change

    return predict_ocr_change

if __name__ == "__main__":
    # Basic smoke tests for development convenience
    print("Running basic checks for data_processor module...")

    df = load_and_prepare_data()
    print(f"Data loaded: {len(df)} rows")
    print(f"Date range: {df['month'].min()} to {df['month'].max()}")

    performance = create_model_performance_data()
    print(f"Performance data: {len(performance)} models")

    features = create_feature_importance_data()
    print(f"Feature importance: {len(features)} features")

    print("Smoke tests completed successfully.")