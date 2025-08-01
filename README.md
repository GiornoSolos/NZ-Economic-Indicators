# New Zealand OCR Prediction Project

## Overview

This project develops a predictive model for New Zealand's Official Cash Rate (OCR) using comprehensive economic indicators from 2021-2025. The model forecasts next month's OCR decisions by analyzing key macroeconomic variables that influence the Reserve Bank of New Zealand's (RBNZ) monetary policy decisions.

## Objective

**Primary Goal**: Predict next month's OCR level and direction (up/down/same) using current economic conditions.

**Research Question**: Can we accurately forecast RBNZ monetary policy decisions using publicly available economic indicators?

## Data Sources

### Official Data Providers
- **Stats NZ**: Consumer Price Index, Unemployment Rate
- **Reserve Bank of New Zealand (RBNZ)**: OCR, Core Inflation, House Price Index, Interest Rates
- **REINZ**: House price growth data
- **RBNZ Financial Markets**: Trade Weighted Index, Mortgage and Deposit Rates

### Time Period
- **Training Data**: January 2021 - Present
- **Frequency**: Monthly observations
- **Total Observations**: ~49 monthly data points

## Dataset Variables

### Target Variables
- `OCR_next`: Next month's Official Cash Rate
- `OCR_direction`: Direction of OCR change ("up", "down", "same")

### Predictor Variables

| Variable | Description | Source | Policy Relevance |
|----------|-------------|---------|------------------|
| `OCR` | Current Official Cash Rate (%) | RBNZ | Baseline rate |
| `CPI_pct` | Consumer Price Index annual % change | Stats NZ | Inflation mandate |
| `UnemploymentRate` | Unemployment rate (%) | Stats NZ | Employment mandate |
| `CoreInflation` | Core inflation (sectoral factor model) | RBNZ | True inflation signal |
| `HousePriceGrowth` | House price annual % change | RBNZ/REINZ | Financial stability |
| `TWI` | Trade Weighted Index | RBNZ | Exchange rate pressure |
| `FloatingMortgage` | Floating mortgage rate (%) | RBNZ | Policy transmission |
| `TermDeposit6M` | 6-month term deposit rate (%) | RBNZ | Market expectations |

## Methodology

### Data Processing Pipeline
1. **Data Collection**: Automated collection from official NZ statistical sources
2. **Frequency Alignment**: Convert quarterly data (unemployment, core inflation) to monthly using linear interpolation
3. **Missing Value Treatment**: Forward-fill and interpolation techniques
4. **Feature Engineering**: Create lagged variables and policy direction indicators
5. **Data Validation**: Ensure economic reasonableness of all variables

### Model Approach
- **Classification**: Predict OCR direction (up/down/same)
- **Regression**: Predict exact OCR level
- **Time Series**: Account for temporal dependencies in monetary policy

## Technical Implementation

### Programming Language
- **R** (primary analysis and data processing)
- Libraries: `dplyr`, `lubridate`, `tidyr`, `zoo`, `readr`

### File Structure
```
├── data/
│   ├── raw/                          # Original data files
│   └── processed/                    # Cleaned datasets
├── src/
│   ├── data_collection.R            # Data loading and cleaning
│   └── nz_economic_indicators.Rmd   # Main analysis notebook
├── output/
│   └── ocr_prediction_enhanced.csv  # Final processed dataset
└── README.md
```

## Key Features

### Economic Policy Focus
- **Dual Mandate Coverage**: Includes both price stability (inflation) and employment indicators
- **Financial Stability**: House price growth captures macroprudential concerns
- **Policy Transmission**: Mortgage and deposit rates show how OCR changes affect the economy
- **External Factors**: Trade-weighted index captures international pressures

### Data Quality
- **Official Sources Only**: All data from authoritative NZ government and RBNZ sources
- **Real-time Availability**: Variables available to policymakers when making decisions
- **Economic Validation**: All variables within reasonable economic bounds

## Usage Instructions

### Prerequisites
```r
install.packages(c("dplyr", "lubridate", "tidyr", "zoo", "readr", "stringr"))
```

### Data Preparation
1. Download raw data files from official sources (file names in notebook)
2. Place files in project directory
3. Run the R notebook: `nz_economic_indicators.Rmd`

### Output
- **Enhanced Dataset**: `ocr_prediction_enhanced.csv` with all processed variables
- **Summary Statistics**: Descriptive analysis of all variables
- **Data Validation**: Checks for missing values and outliers

## Model Performance Considerations

### Strengths
- **Policy-Relevant Variables**: Based on RBNZ's actual decision framework
- **High-Quality Data**: Official government sources with regular updates
- **Economic Intuition**: Variables align with established monetary policy theory

### Limitations
- **Limited Time Series**: Only 4+ years of data (49 observations)
- **Structural Breaks**: COVID-19 period may affect model generalizability
- **External Shocks**: Model may not capture unprecedented economic events
- **Policy Discretion**: Central bank decisions involve qualitative factors not captured in data

## Potential Applications

### Academic Research
- Monetary policy reaction function estimation
- Central bank predictability analysis
- Economic forecasting model development

### Financial Markets
- Interest rate forecasting for financial institutions
- Investment strategy development
- Risk management applications

### Policy Analysis
- Understanding RBNZ decision patterns
- Economic scenario analysis
- Policy communication effectiveness

## Future Enhancements

### Data Expansion
- **Additional Variables**: Business confidence, global commodity prices, international interest rates
- **Higher Frequency**: Weekly or daily indicators where available
- **Longer History**: Extend back to capture more policy cycles

### Modeling Improvements
- **Machine Learning**: Random forests, neural networks, ensemble methods
- **Time Series Models**: LSTM, ARIMA-X, Vector Autoregression (VAR)
- **Regime Switching**: Account for different economic environments

### Real-time Implementation
- **Automated Data Updates**: API connections to official sources
- **Live Predictions**: Real-time OCR forecasts
- **Performance Monitoring**: Track prediction accuracy over time

## Contact & Contribution

This project provides a foundation for understanding and predicting New Zealand monetary policy. Contributions, suggestions, and collaborations are welcome.

---

**Disclaimer**: This model is for research and educational purposes only. It should not be used as the sole basis for financial or investment decisions. The Reserve Bank of New Zealand's monetary policy decisions involve complex qualitative assessments that may not be fully captured by quantitative models.
