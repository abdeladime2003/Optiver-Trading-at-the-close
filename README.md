<div align="center">

![Header](https://capsule-render.vercel.app/api?type=waving&color=0:1E88E5,50:43A047,100:FB8C00&height=200&section=header&text=OPTIVER%20TRADING%20SYSTEM&fontSize=40&fontColor=ffffff&animation=twinkling&fontAlignY=38&desc=LightGBM%20Stock%20Price%20Prediction&descAlignY=51&descSize=18&descColor=ffffff)

</div>

<div align="center">

[![Status](https://img.shields.io/badge/Status-Research_Project-00B894?style=for-the-badge&logo=code&logoColor=white)](https://github.com/abdeladime2003/Optiver-Trading-at-the-close)
![Machine Learning](https://img.shields.io/badge/ML-LightGBM-FF6B35?style=for-the-badge&logo=python&logoColor=white)
![Finance](https://img.shields.io/badge/Finance-Stock_Prediction-1E88E5?style=for-the-badge&logo=chart.js&logoColor=white)
![Competition](https://img.shields.io/badge/Competition-Optiver-43A047?style=for-the-badge&logo=tradingview&logoColor=white)

</div>

## PROJECT OVERVIEW

**Optiver Trading at the Close** is a **machine learning project** focused on predicting stock closing auction prices using **LightGBM regression**. This project implements systematic feature engineering and hyperparameter optimization to forecast target values based on historical market microstructure data from the **Optiver Trading Competition**.

### Core Objectives

- **Price Prediction**: Forecast closing auction price movements using market data
- **Feature Engineering**: Create meaningful financial indicators from raw market data  
- **Model Optimization**: Systematic hyperparameter tuning for optimal performance
- **Market Analysis**: Understanding bid-ask spreads, volume patterns, and price dynamics

### Technical Approach

- **Data Processing**: Handle missing values, outlier detection, and feature interpolation
- **Feature Creation**: Generate 20+ engineered features from market microstructure data
- **Model Training**: LightGBM with RandomizedSearchCV for hyperparameter optimization
- **Performance Evaluation**: Cross-validation with MAE and RMSE metrics

---

<div align="center">

![Tech Stack](https://capsule-render.vercel.app/api?type=rect&color=gradient&customColorList=12,20,2,28,0&height=60&section=header&text=TECHNICAL%20STACK&fontSize=20&fontColor=ffffff)

</div>

## TECHNOLOGY STACK

<div align="center">

### Machine Learning & Analytics

![LightGBM](https://img.shields.io/badge/LightGBM-Gradient_Boosting-FF6B35?style=flat-square&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-Data_Processing-150458?style=flat-square&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-Numerical_Computing-013243?style=flat-square&logo=numpy&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit_Learn-ML_Pipeline-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)

### Data Visualization & Analysis

![Matplotlib](https://img.shields.io/badge/Matplotlib-Plotting-11557C?style=flat-square&logo=matplotlib&logoColor=white)
![Seaborn](https://img.shields.io/badge/Seaborn-Statistical_Viz-8A2BE2?style=flat-square&logo=python&logoColor=white)

### Development Environment

![Jupyter](https://img.shields.io/badge/Jupyter_Notebook-Interactive_Development-F37626?style=flat-square&logo=jupyter&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=flat-square&logo=python&logoColor=white)

</div>

---

## FEATURE ENGINEERING

### **Raw Data Features**

<div align="center">

| **Feature** | **Description** | **Type** | **Usage** |
|:------------|:----------------|:---------|:----------|
| **stock_id** | Stock identifier | Categorical | Model training |
| **seconds_in_bucket** | Time within auction | Numerical | Temporal feature |
| **imbalance_size** | Order imbalance quantity | Numerical | Volume analysis |
| **reference_price** | Benchmark price | Numerical | Price normalization |
| **bid_price** | Best bid price | Numerical | Spread calculation |
| **ask_price** | Best ask price | Numerical | Spread calculation |
| **wap** | Weighted average price | Numerical | Price impact |
| **matched_size** | Matched volume | Numerical | Liquidity measure |

</div>

### **Engineered Features**

**Price-Based Features:**
```python
# Spread and mid-price calculations
df['spread'] = df['ask_price'] - df['bid_price'] 
df['mid_price'] = (df['ask_price'] + df['bid_price']) / 2
df['price_impact'] = df['reference_price'] - df['wap']
```

**Volume-Based Features:**
```python
# Volume analysis
df['total_volume'] = df['bid_size'] + df['ask_size']
df['volume_imbalance'] = df['bid_size'] - df['ask_size']
df['imbalance_buy_sell_ratio'] = df['imbalance_size'] / df['matched_size']
```

**Technical Indicators:**
```python
# RSI calculation (14-period)
def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Moving averages and volatility
df['rsi'] = compute_rsi(df['mid_price'])
df['sma_10'] = df['mid_price'].rolling(window=10).mean()
df['ema_10'] = df['mid_price'].ewm(span=10, adjust=False).mean()
df['price_volatility'] = df['mid_price'].rolling(window=10).std()
```

---

<div align="center">

![Performance](https://capsule-render.vercel.app/api?type=rect&color=gradient&customColorList=0,2,5,30,28&height=60&section=header&text=MODEL%20PERFORMANCE&fontSize=20&fontColor=ffffff)

</div>

## ACTUAL MODEL RESULTS

### **Model Configuration**

**Optimized LightGBM Parameters:**
```python
# Best parameters found through RandomizedSearchCV
best_params = {
    'num_leaves': 100,
    'n_estimators': 200, 
    'min_child_samples': 50,
    'learning_rate': 0.01
}
```

### **Performance Metrics**

<div align="center">

| **Metric** | **Value** | **Interpretation** |
|:-----------|:----------|:-------------------|
| **Mean Absolute Error (MAE)** | 6.30 | Average prediction error |
| **Root Mean Squared Error (RMSE)** | 9.29 | Penalizes larger errors |
| **Cross-Validation Score** | -86.89 | Negative MSE (lower is better) |

</div>

### **Feature Importance Analysis**

**Top Contributing Features:**
```python
# Feature importance from trained model
feature_importance = [
    ('spread', 336),                    # Highest importance
    ('seconds_in_bucket', 274),         # Temporal component  
    ('matched_size', 204),              # Volume measure
    ('imbalance_buy_sell_ratio', 190),  # Order flow imbalance
    ('imbalance_size', 179),            # Raw imbalance
]
```

**Model Selection Process:**
- Initially trained on all 26 features
- Selected features with importance > 100 for final model (12 features)
- Used 3-fold cross-validation for hyperparameter optimization
- 50 iterations of RandomizedSearchCV for parameter tuning

---

## DATA PREPROCESSING

### **Missing Value Treatment**

```python
# Handle missing values systematically
df = df.dropna(subset=["ask_price"], axis=0)

# Set near/far prices to 0 when seconds_in_bucket <= 300
df.loc[df['seconds_in_bucket'] <= 300, "near_price"] = 0
df.loc[df['seconds_in_bucket'] <= 300, "far_price"] = 0

# Interpolate remaining missing values
df['far_price'] = df['far_price'].interpolate()
df['near_price'] = df['near_price'].interpolate()
```

### **Outlier Detection and Treatment**

```python
def outlier_threshold(dataframe, variable):
    Q1 = dataframe[variable].quantile(0.01)
    Q3 = dataframe[variable].quantile(0.99)
    IQR = Q3 - Q1
    up_limit = Q3 + 1.5 * IQR
    low_limit = Q1 - 1.5 * IQR
    return low_limit, up_limit

def replace_with_threshold(dataframe, variable):
    low_limit, up_limit = outlier_threshold(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit
    return dataframe
```

---

## PROJECT STRUCTURE

```
Optiver-Trading-at-the-close/
├── Model.ipynb                    # Main analysis and model training
├── data/                          # Dataset directory
│   └── train.csv                 # Training dataset (5.2M+ rows)
├── requirements.txt               # Python dependencies
└── README.md                     # Project documentation
```

---

## GETTING STARTED

### **Installation**

```bash
# Clone repository
git clone https://github.com/abdeladime2003/Optiver-Trading-at-the-close.git
cd Optiver-Trading-at-the-close

# Install dependencies
pip install -r requirements.txt

# Key packages
pip install pandas numpy scikit-learn lightgbm matplotlib seaborn
```

### **Usage**

```python
# Load and preprocess data
import pandas as pd
import lightgbm as lgbm
from sklearn.model_selection import RandomizedSearchCV

# Load dataset
data = pd.read_csv("train.csv")

# Feature engineering pipeline
df = preprocess_data(data)
df = create_features(df)
df = handle_outliers(df)

# Model training
X = df.drop(["target"], axis=1)
y = df[["target"]]

# Hyperparameter optimization
param_grid = {
    'num_leaves': [31, 100],
    'learning_rate': [0.01, 0.15],
    'n_estimators': [100, 200],
    'min_child_samples': [20, 50]
}

lgb_model = lgbm.LGBMRegressor()
random_search = RandomizedSearchCV(
    estimator=lgb_model,
    param_distributions=param_grid,
    n_iter=50,
    scoring='neg_mean_squared_error',
    cv=3,
    random_state=42
)

# Train model
random_search.fit(X, y)
best_model = random_search.best_estimator_
```

---

## TECHNICAL INSIGHTS

### **Data Characteristics**

**Dataset Statistics:**
- **Size**: 5,237,980 rows × 17 original columns
- **Target Variable**: Continuous price movement predictions
- **Missing Values**: Significant in far_price (55%) and near_price (55%) columns
- **Feature Distribution**: Heavy-tailed distributions requiring outlier treatment

**Key Observations:**
- **Spread** is the most predictive feature (importance: 336)
- **Temporal features** (seconds_in_bucket) are highly relevant
- **Volume imbalances** provide significant predictive power
- **Technical indicators** (RSI, moving averages) add moderate value

### **Model Limitations**

**Performance Considerations:**
- MAE of 6.3 indicates moderate prediction accuracy
- RMSE of 9.29 suggests some large prediction errors
- Model performs best on typical market conditions
- Extreme market events may reduce prediction quality

**Improvements Needed:**
- Feature selection could be further refined
- Additional ensemble methods might improve robustness
- More sophisticated time-series validation approaches
- Integration of external market factors

---

## CONTRIBUTING

### **Development Guidelines**

```bash
# Fork and clone
git clone https://github.com/yourusername/Optiver-Trading-at-the-close.git

# Create feature branch
git checkout -b feature/model-improvement

# Implement changes
# - Add new features
# - Improve preprocessing
# - Enhance model architecture

# Submit pull request
git commit -m "feat: add volatility-based features"
git push origin feature/model-improvement
```

---

## COMPETITION CONTEXT

### **Optiver Trading Competition**

**Challenge Objective:**
- Predict closing auction price movements in stock markets
- Use market microstructure data for forecasting
- Compete against quantitative researchers and data scientists

**Dataset Characteristics:**
- Real market data from closing auctions
- High-frequency trading environment simulation
- Focus on market microstructure features

**Evaluation Methodology:**
- Mean Absolute Error (MAE) as primary metric
- Time-series validation to prevent data leakage
- Out-of-sample testing on unseen data

---

## LICENSE & CONTACT

**License**: MIT License - Educational and research purposes

**Developer**: Abdeladime Benali  
**Email**: abdeladimebenali2003@gmail.com  
**Institution**: INPT - National Institute of Posts & Telecommunications  
**LinkedIn**: [linkedin.com/in/abdeladime-benali](https://linkedin.com/in/abdeladime-benali)  
**GitHub**: [github.com/abdeladime2003](https://github.com/abdeladime2003)

**Disclaimer**: This project is for educational purposes. The model results shown are from academic research and should not be used for actual trading decisions without proper validation and risk management.

---

<div align="center">

![Repository Stats](https://github-readme-stats.vercel.app/api?username=abdeladime2003&repo=Optiver-Trading-at-the-close&show_icons=true&theme=tokyonight)

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://linkedin.com/in/abdeladime-benali)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/abdeladime2003)

**Academic Research Project | INPT 2025**

</div>

<div align="center">

![Footer](https://capsule-render.vercel.app/api?type=waving&color=0:1E88E5,50:43A047,100:FB8C00&height=120&section=footer)

</div>
