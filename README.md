# New Zealand Official Cash Rate Prediction System

## Overview

This project develops a comprehensive machine learning framework to predict the Reserve Bank of New Zealand's (RBNZ) Official Cash Rate decisions using real-time economic indicators. The system achieved exceptional predictive accuracy of 98.2% R² through advanced ensemble methods and comprehensive feature engineering during the complete monetary policy cycle from 2021-2025.

## Key Results

- **98.2% prediction accuracy** (R²) for OCR level forecasting with RMSE of 0.19 percentage points
- **75% classification accuracy** for policy direction prediction using advanced ensemble methods
- **Complete policy cycle analysis** covering COVID accommodation, aggressive tightening, and policy normalization
- **22 engineered economic features** capturing monetary policy transmission mechanisms
- **Quantitative validation** of RBNZ's dual mandate framework effectiveness

## Interactive Web Application

The project includes a comprehensive interactive web application built with Streamlit that provides:

- Real-time OCR prediction and scenario analysis
- Economic indicator correlation and time series analysis
- Model performance visualization and comparison
- Feature importance analysis with economic interpretation
- Mobile-responsive design with touch-friendly interactions

**Live Demo**: [Access the interactive dashboard](https://your-app.streamlit.app)

## Technical Architecture

### Machine Learning Framework
- **Linear Regression**: 98.2% R² (best performer for level prediction)
- **Random Forest**: 96.3% R² with feature importance analysis
- **Gradient Boosting**: 95.3% R² with advanced ensemble techniques
- **Classification Models**: 75% accuracy using voting classifiers and SMOTE

### Data Processing Pipeline
- **Data Sources**: RBNZ, Statistics New Zealand, Real Estate Institute of New Zealand
- **Feature Engineering**: 22 indicators including policy persistence, inflation targeting, and transmission mechanisms
- **Temporal Coverage**: January 2021 - August 2025 (44 monthly observations)
- **Validation Framework**: 80/20 train-test split with temporal ordering preserved

## Installation and Setup

### Prerequisites
- Python 3.8+ (tested with Python 3.13)
- Git for version control

### Installation Steps

1. Clone the repository:
```bash
git clone https://github.com/yourusername/nz-ocr-prediction-website.git
cd nz-ocr-prediction-website
```

2. Install required dependencies:
```bash
pip install streamlit plotly pandas numpy scikit-learn seaborn
```

3. Run the interactive web application:
```bash
streamlit run app.py
```

4. Access the dashboard at `http://localhost:8501`

### Network Access
To access from other devices on the same network:
```bash
streamlit run app.py --server.address=0.0.0.0
```

## Project Structure

```
nz-ocr-prediction-website/
├── app.py                           # Main Streamlit application
├── data_processor.py               # Data processing and feature engineering
├── requirements.txt                # Python dependencies
├── ocr_prediction_enhanced.csv     # Processed dataset (optional)
├── README.md                       # Project documentation
└── LICENSE                         # MIT License
```

## Data Sources and Methodology

### Official Data Sources
| Organization | Variables | Update Frequency |
|--------------|-----------|------------------|
| Reserve Bank of New Zealand | OCR, Core Inflation, Interest Rates | Monthly |
| Statistics New Zealand | Consumer Price Index, Unemployment Rate | Quarterly |
| Real Estate Institute of NZ | House Price Growth Indicators | Monthly |

### Feature Categories
- **Policy Persistence** (48.1% combined importance): OCR lags and current levels
- **Inflation Targeting** (22.3% combined importance): CPI and core inflation measures  
- **Transmission Mechanisms** (19.1% combined importance): Interest rate spreads and pass-through
- **Employment Indicators** (4.2% importance): Unemployment rate and changes
- **Housing Market** (3.8% importance): House price growth and momentum
- **Exchange Rate** (2.5% importance): Trade Weighted Index variations

## Model Performance

### Regression Results (OCR Level Prediction)
| Model | R² Score | RMSE | MAE | 
|-------|----------|------|-----|
| Linear Regression | 0.9822 | 0.1886 | 0.1582 |
| Random Forest | 0.9626 | 0.2736 | 0.1966 |
| Ensemble Method | 0.9891 | 0.1623 | 0.1445 |

### Classification Results (Policy Direction)
| Model | Accuracy | Improvement over Baseline |
|-------|----------|--------------------------|
| Soft Voting Ensemble | 75.0% | +25.0% |
| Gradient Boosting | 60.0% | +10.0% |
| Baseline Random Forest | 50.0% | baseline |

## Economic Insights

### Monetary Policy Validation
The exceptional predictive accuracy demonstrates that RBNZ follows highly systematic decision patterns based on economic fundamentals, supporting several key findings:

- **Policy Persistence**: Previous OCR decisions account for 48% of prediction power, validating gradual adjustment approaches
- **Inflation Primacy**: Combined inflation measures represent 22% of decision drivers, confirming systematic inflation targeting
- **Effective Transmission**: Strong importance of interest rate spreads (19%) validates monetary policy channel effectiveness
- **Dual Mandate Balance**: Employment considerations (4%) provide meaningful but secondary policy influence

### Policy Cycle Analysis
The 2021-2025 period captured a complete monetary policy cycle:
- **COVID Accommodation**: Emergency rates at 0.25% (early 2021)
- **Aggressive Tightening**: Rapid increases to 5.5% peak (2021-2023)
- **Policy Normalization**: Gradual reduction to sustainable levels (2024-2025)

## Applications and Use Cases

### Financial Markets
- Interest rate derivative pricing and hedging strategies
- Fixed income portfolio optimization and duration management
- Risk management for banking and financial institutions

### Policy Analysis
- Central bank decision support and scenario planning
- Economic forecasting and policy communication enhancement
- International monetary policy framework comparison

### Academic Research
- Quantitative validation of inflation targeting frameworks
- Machine learning applications in monetary economics
- Policy transmission mechanism analysis

## Future Enhancements

### Technical Development
- Real-time API integration for automated data updates
- Deep learning models (LSTM, Transformers) for complex temporal patterns
- Advanced ensemble optimization using AutoML techniques

### Research Extensions
- International comparison with other central banks (Fed, ECB, BoE)
- High-frequency analysis using daily economic indicators
- Natural language processing of central bank communications

## Contributing

Contributions are welcome for enhancing the analysis framework, improving visualization capabilities, or extending the methodology to other central banks. Please feel free to:

- Submit bug reports or feature requests via GitHub Issues
- Propose improvements to the machine learning models or data processing
- Suggest additional economic indicators or alternative methodologies

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact

**Marco Mojicevic**  
Data Scientist & ML Engineer  
Wellington, New Zealand

- GitHub: [github.com/GiornoSolos](https://github.com/GiornoSolos)
- LinkedIn: [linkedin.com/in/marco-mojicevic](https://linkedin.com/in/marco-mojicevic)

## Citation

If you use this work in academic research, please cite:

```
Mojicevic, M. (2025). Predicting New Zealand Official Cash Rate Decisions: 
A Machine Learning Approach to Monetary Policy Analysis. 
Independent Research, Wellington, New Zealand.
```

## Acknowledgments

This research acknowledges the Reserve Bank of New Zealand, Statistics New Zealand, and the Real Estate Institute of New Zealand for providing high-quality, publicly available economic data that made this analysis possible.

---

*This project demonstrates the application of modern machine learning techniques to economic policy analysis, achieving exceptional predictive accuracy while providing valuable insights into monetary policy effectiveness and systematic central bank decision-making.*
