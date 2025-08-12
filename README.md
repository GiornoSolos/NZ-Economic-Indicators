# New Zealand Official Cash Rate Prediction System

## Overview

This project develops a comprehensive machine learning system to predict the Reserve Bank of New Zealand's (RBNZ) Official Cash Rate decisions using real-time economic indicators. The analysis achieves **98.2% accuracy in predicting exact OCR levels** and **75% accuracy in predicting policy direction changes**, demonstrating the application of advanced data science techniques to monetary policy forecasting.

## Key Achievements

- **98.2% R² accuracy** in OCR level prediction (RMSE: 0.19 percentage points)
- **75% classification accuracy** for policy direction prediction
- **50% relative improvement** over baseline through advanced ensemble methods
- **Complete monetary policy cycle analysis** (2021-2025: COVID recovery → inflation targeting → normalization)
- **22 engineered economic features** capturing policy transmission mechanisms

## Research Objectives

**Primary Goal**: Predict next month's OCR level and direction using current economic conditions

**Research Questions**:
- Can machine learning accurately forecast RBNZ monetary policy decisions?
- Which economic indicators drive systematic policy responses?
- How effective are ensemble methods for imbalanced policy classification?

## Dataset

### Data Sources
- **Stats NZ**: Consumer Price Index, Unemployment Rate  
- **Reserve Bank of New Zealand**: OCR, Core Inflation, House Price Index, Interest Rates
- **REINZ**: House price growth indicators
- **Coverage**: January 2021 - August 2025 (44 monthly observations)

### Target Variables
- `OCR_next`: Next month's Official Cash Rate (regression)
- `OCR_direction`: Policy change direction - "up", "down", "same" (classification)

### Key Features (22 Total)
#### Core Economic Indicators
- **Monetary Policy**: OCR, OCR lags, policy regime indicators
- **Inflation Measures**: CPI annual change, core inflation, inflation volatility  
- **Employment**: Unemployment rate, employment change indicators
- **Housing Market**: House price growth, 3-month momentum
- **Financial Markets**: TWI, floating mortgage rates, term deposit rates

#### Engineered Features
- **Interest Rate Spreads**: Mortgage-OCR spread, Deposit-OCR spread
- **Policy Persistence**: Lagged OCR values, tightening cycle indicators
- **Volatility Measures**: 6-month rolling standard deviations
- **Regime Indicators**: High inflation flags, unemployment thresholds

## Methodology

### Data Processing Pipeline
1. **Collection**: Automated sourcing from official NZ statistical agencies
2. **Cleaning**: Missing value imputation, frequency alignment (quarterly → monthly)
3. **Feature Engineering**: 22 variables capturing policy dynamics and transmission
4. **Validation**: Economic reasonableness checks, outlier detection

### Machine Learning Framework

#### Regression Models (OCR Level Prediction)
- **Linear Regression**: 98.2% R², RMSE 0.1886
- **Random Forest**: 96.3% R², RMSE 0.2736

#### Classification Models (Policy Direction)
- **Baseline**: Random Forest (50%), Logistic Regression (50%)
- **Advanced Ensemble**: 75% accuracy using sophisticated ensemble methods
- **Techniques**: Gradient Boosting, Voting Classifiers, SMOTE, Balanced Random Forest

#### Model Validation
- **Train/Test Split**: 80/20 with temporal ordering preserved
- **Evaluation Metrics**: R², RMSE, MAE (regression); Accuracy, F1-score (classification)
- **Cross-validation**: Stratified k-fold for robust performance assessment

## Results

### Regression Performance
| Model | R² | RMSE | MAE |
|-------|----|----- |-----|
| **Linear Regression** | **0.9822** | **0.1886** | **0.1582** |
| Random Forest | 0.9626 | 0.2736 | 0.1966 |

### Classification Performance  
| Model | Accuracy | Improvement |
|-------|----------|-------------|
| **Best Ensemble** | **75.0%** | **+25.0%** |
| Gradient Boosting | 70.0% | +20.0% |
| Baseline Random Forest | 50.0% | baseline |

### Feature Importance (Top 5)
1. **OCR_lag1** (28.7%) - Policy persistence
2. **OCR** (19.4%) - Current policy stance  
3. **CPI_pct** (15.6%) - Inflation targeting
4. **FloatingMortgage** (8.9%) - Policy transmission
5. **CoreInflation** (6.7%) - Underlying inflation

## Project Structure

```
nz-ocr-prediction/
├── data/
│   ├── raw/                          # Original CSV files from official sources
│   └── processed/                    # Cleaned and engineered datasets
├── src/
│   ├── data_collection.R            # R data processing pipeline
│   └── ocr_prediction_models.py     # Python ML implementation
├── outputs/
│   ├── model_performance_plots/     # Generated visualization files
│   ├── eda_visualizations/          # Economic data analysis plots
│   └── ocr_prediction_enhanced.csv  # Final processed dataset
└── README.md
```

## Technical Implementation

### Programming Languages & Libraries
- **R**: Data collection and preprocessing (dplyr, lubridate, tidyr, zoo)
- **Python**: Machine learning and analysis (scikit-learn, pandas, numpy, matplotlib)
- **Advanced Techniques**: Ensemble methods, SMOTE, imbalanced-learn

### Key Technical Features
- **Comprehensive EDA**: Pre/post-processing visualization pipeline
- **Feature Engineering**: 22 economically-motivated indicators
- **Ensemble Learning**: Multiple advanced techniques for class imbalance
- **Model Validation**: Robust evaluation with economic interpretation

## Visualizations

The project generates comprehensive visualizations including:
- **Economic Time Series**: OCR, inflation, unemployment trends
- **Correlation Analysis**: Economic relationship matrices
- **Model Performance**: Confusion matrices, feature importance plots
- **Policy Analysis**: Regime identification and transmission mechanisms

## Economic Insights

### Dual Mandate Validation
- **Inflation Control**: Successfully modeled CPI reduction from 7.5% → 2.0%
- **Employment Trade-off**: Captured unemployment rise from 3.25% → 5.0% during disinflation
- **Policy Effectiveness**: Strong systematic relationships validate RBNZ framework

### Transmission Mechanisms
- **Perfect Pass-through**: 0.99 correlation between OCR and market rates
- **Housing Channel**: Strong negative correlation (-0.85) with house prices
- **International Effects**: TWI impact on inflation dynamics

## Applications

### Government & Policy
- **Central Banking**: Enhanced policy analysis and scenario planning
- **Treasury**: Economic forecasting and fiscal-monetary coordination
- **Research**: Evidence-based monetary policy effectiveness studies

### Financial Markets
- **Risk Management**: Interest rate derivative pricing and hedging
- **Investment Strategy**: Sector rotation based on policy predictions
- **Portfolio Management**: Duration optimization using OCR forecasts

## Usage Instructions

### Prerequisites
```bash
# R packages
install.packages(c("dplyr", "lubridate", "tidyr", "zoo", "readr"))

# Python packages  
pip install pandas numpy scikit-learn matplotlib seaborn
```

### Running the Analysis
```bash
# 1. Data processing (R)
Rscript src/data_collection.R

# 2. Machine learning pipeline (Python)  
python src/ocr_prediction_models.py
```

## Future Enhancements

### Technical Extensions
- **Real-time Implementation**: API integration for live predictions
- **Deep Learning**: LSTM/Transformer models for complex patterns
- **Advanced Ensemble Methods**: Additional imbalanced learning techniques

### Documentation Enhancements
- **Technical Report**: Comprehensive LaTeX analysis document
- **Interactive Dashboard**: Power BI visualization for policy analysis
- **API Documentation**: RESTful service for real-time predictions

### Data Expansion
- **Higher Frequency**: Weekly/daily economic indicators
- **International Variables**: Global commodity prices, foreign interest rates
- **Alternative Data**: Satellite-based economic activity indicators

## Limitations & Disclaimers

### Current Limitations
- **Sample Size**: 44 monthly observations (2021-2025 period)
- **Structural Breaks**: COVID period may affect generalizability
- **Quantitative Focus**: Excludes qualitative policy communication factors

### Disclaimer
This model is for research and educational purposes only. It should not be used as the sole basis for financial or investment decisions. The Reserve Bank of New Zealand's monetary policy decisions involve complex qualitative assessments that may not be fully captured by quantitative models.

## Contact & Contribution

**Author**: Marco Mojicevic  
**Portfolio**: [github.com/GiornoSolos](https://github.com/GiornoSolos)  
**LinkedIn**: [linkedin.com/in/marco-mojicevic](https://linkedin.com/in/marco-mojicevic)

Contributions, suggestions, and collaborations are welcome. This project provides a foundation for understanding and predicting monetary policy decisions using modern data science techniques.

## License

This project is open source and available under the [MIT License](LICENSE).

---

*This independent research demonstrates the application of advanced machine learning techniques to economic policy analysis, achieving state-of-the-art accuracy in monetary policy prediction through comprehensive feature engineering and ensemble learning methods.*
