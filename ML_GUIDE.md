# ML Stock Predictor Guide

## Overview

The ML Stock Predictor identifies breakout candidates (NVDA, MU-like patterns) and crash risks using machine learning trained on historical data.

## How to Run the Model Trainer

You can train the ML model manually or automate it with GitHub Actions:

### Manual Run (Local)

1. Open a terminal in your project directory.
2. Run:
    ```bash
    python3 ml_predictor.py
    ```
3. This will read tickers from `data/tickers.csv`, fetch historical data, train the model, and save the model files to `data/ml_models/`.

### Automated Run (GitHub Actions)

If you want to automate retraining, use the provided workflow:

1. See `.github/workflows/mlbuild.yml` in your repo.
2. This workflow runs weekly (Sunday 5PM PST) and can be triggered manually from the Actions tab.
3. It will train the model and commit the updated model files to the repository automatically.

**Note:** The dashboard (`stocks.py`) will automatically use the latest model files for predictions if they exist.

## Features

- **Breakout Score (0-100%)**: Probability of >100% gain in 6-12 months
- **Crash Risk (0-100%)**: Probability of >50% decline in 6-12 months
- **Prediction**: BREAKOUT | CRASH | NEUTRAL classification
- **Confidence**: Model certainty (0-100%)

## Installation

```bash
# Install ML dependencies
pip install -r requirements.txt

# Or install manually
pip install scikit-learn xgboost
```

## Quick Start

### 1. Training the Model

The model needs historical data to learn patterns. You have two options:

**Option A: Use Synthetic Demo Data (for testing)**
```bash
# Generate synthetic training data and train model
python3 ml_predictor.py
```

**Option B: Use Real Historical Data (recommended for production)**

Create a training script (`train_ml_model.py`):

```python
import yfinance as yf
import pandas as pd
from ml_predictor import train_model
from datetime import datetime, timedelta

# 1. Fetch historical data
tickers = ['NVDA', 'MU', 'AMD', 'TSLA', 'GME', 'AAPL', 'MSFT', ...] # Add your tickers
historical_data = []
labels = []

for ticker in tickers:
    # Fetch data from 2 years ago
    end_date = datetime.now() - timedelta(days=365*2)
    stock = yf.Ticker(ticker)
    hist = stock.history(period="2y")
    
    # Calculate features at the midpoint (1 year ago)
    # Then check if price doubled (+100%) or crashed (-50%) in next 6-12 months
    # Label accordingly: 'BREAKOUT', 'CRASH', or 'NEUTRAL'
    
    # ... extract features and determine label ...
    historical_data.append(features_dict)
    labels.append(label)

# 2. Train model
model, scaler = train_model(historical_data, labels)
```

### 2. Running with ML Predictions

Once trained, ML scores appear automatically:

```bash
# Run dashboard (ML predictions included automatically if model exists)
python3 stocks.py data/tickers.csv
```

The dashboard will show:
- 🚀 **ML Column**: Breakout scores with color coding
  - **Green (≥70%)**: High breakout potential 🚀
  - **Orange (50-69%)**: Moderate potential
  - **Gray (<50%)**: Low potential
- ⚠️ **Crash warnings**: Displayed when crash risk ≥50%

## Understanding ML Scores

### Breakout Score Interpretation

| Score | Meaning | Action |
|-------|---------|--------|
| **80-100%** | Strong breakout signal (similar to past winners like NVDA) | Consider for watchlist |
| **60-79%** | Moderate potential | Monitor closely |
| **40-59%** | Mixed signals | Wait for confirmation |
| **<40%** | Low probability | Not a priority |

### Crash Risk Interpretation

| Risk | Meaning | Action |
|------|---------|--------|
| **≥70%** | High crash risk | Avoid or exit positions |
| **50-69%** | Elevated risk | Use tight stops |
| **30-49%** | Normal risk | Monitor |
| **<30%** | Low risk | - |

## Features Used by the Model

The ML model analyzes 18 technical and fundamental features:

### Technical Indicators
- RSI (momentum)
- Bollinger Band position and width
- MACD signal
- ATR (volatility)
- Volume bias and spikes
- Moving average crosses (Golden/Death Cross)

### Price Momentum
- Daily change %
- 5-day change %
- 1-month change %

### Fundamentals
- P/E ratio
- Market capitalization
- Put/Call ratio
- Short interest %

### Trading Signals
- Current active trading signal
- Trend score
- Squeeze indicators

## Real-World Training Example

To train on actual historical winners and losers:

```python
# Example: Identify breakout stocks from 2020-2023
breakout_stocks = {
    'NVDA': {'trained_date': '2020-01-01', 'outcome': 'BREAKOUT', 'gain': 450%},
    'MU': {'trained_date': '2020-06-01', 'outcome': 'BREAKOUT', 'gain': 180%},
    'AMD': {'trained_date': '2019-12-01', 'outcome': 'BREAKOUT', 'gain': 320%},
}

crash_stocks = {
    'COIN': {'trained_date': '2021-11-01', 'outcome': 'CRASH', 'loss': -75%},
    'PTON': {'trained_date': '2021-01-01', 'outcome': 'CRASH', 'loss': -80%},
}

# For each stock:
# 1. Get data from 'trained_date'
# 2. Calculate all technical indicators (RSI, BB, MACD, etc.)
# 3. Label as BREAKOUT/CRASH based on next 6-12 month performance
# 4. Train model
```

## Model Performance Metrics

After training, the model outputs:

```
📊 Model Performance:
              precision    recall  f1-score   support

   BREAKOUT       0.75      0.82      0.78       120
      CRASH       0.71      0.78      0.74       110
    NEUTRAL       0.88      0.85      0.87       170

   accuracy                           0.82       400

🔝 Top 10 Most Important Features:
           feature  importance
        rsi           0.18
        trend_score   0.15
        change_1m     0.12
        ...
```

## Best Practices

### 1. Combine with Human Judgment
- Don't rely solely on ML scores
- Use as one factor in multi-criteria analysis
- Consider macro trends and news

### 2. Regular Retraining
```bash
# Retrain monthly with new data
python3 train_ml_model.py
```

### 3. Backtesting
- Test model on historical data not used in training
- Validate predictions against actual outcomes
- Adjust features/parameters as needed

### 4. Risk Management
- Even 90% breakout score ≠ guaranteed gain
- Always use stop losses
- Position size appropriately

## Troubleshooting

### Model Not Loading
```
⚠️  ML predictions not available (model not trained)
```
**Solution**: Run `python3 ml_predictor.py` or train with real data

### ML Libraries Not Installed
```
❌ Please install ML libraries: pip install scikit-learn xgboost
```
**Solution**: `pip install -r requirements.txt`

### Low Accuracy
**Solutions**:
- Add more training data (>1000 samples recommended)
- Include more diverse market conditions (bull/bear markets)
- Tune hyperparameters (learning_rate, max_depth, etc.)
- Add more features (sector trends, market breadth, etc.)

## Advanced: Custom Features

Add your own features to `ml_predictor.py`:

```python
def extract_features(stock_data):
    features = [
        # ... existing features ...
        
        # Add custom features
        stock_data.get('your_custom_metric') or 0.0,
        calculate_custom_indicator(stock_data),
    ]
    return features

# Update FEATURE_NAMES accordingly
FEATURE_NAMES = [
    # ... existing names ...
    'your_custom_metric',
    'custom_indicator'
]
```

## FAQ

**Q: How often should I retrain?**  
A: Monthly with new data, or after major market regime changes.

**Q: Can it predict the next NVDA?**  
A: It identifies stocks with similar technical/fundamental patterns to past winners. No guarantee of future performance.

**Q: Why use ML vs traditional signals?**  
A: ML can identify complex multi-factor patterns humans might miss. Use together for best results.

**Q: What's the minimum training data needed?**  
A: Ideally 500+ labeled examples across different market conditions.

## Next Steps

1. ✅ Install dependencies
2. ✅ Train model (synthetic or real data)
3. ✅ Run dashboard and review ML scores
4. 📊 Backtest predictions
5. 🔄 Set up monthly retraining schedule
6. 🎯 Integrate into your trading workflow

For questions or issues, check the code comments in `ml_predictor.py` or create an issue in the repository.
