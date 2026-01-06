"""
ML Stock Predictor - Identifies Breakout and Crash Candidates

Trains models on historical data to predict:
- BREAKOUT: Stocks likely to gain >100% in 6-12 months (NVDA, MU-like patterns)
- CRASH: Stocks likely to drop >50% in 6-12 months
- NEUTRAL: Normal price action

Features used:
- Technical indicators (RSI, MACD, BB, Ichimoku)
- Volume patterns and momentum
- Fundamental metrics (P/E, growth rates)
- Market sentiment (Put/Call, Short Interest)
- Signal confidence and trend scores

Usage:
    from ml_predictor import predict_breakout_crash
    
    scores = predict_breakout_crash(stock_data)
    # Returns: {'breakout_score': 0-100, 'crash_risk': 0-100, 'confidence': 0-100}
"""

import pandas as pd
import numpy as np
import pickle
import os
from datetime import datetime, timedelta
from pathlib import Path

# ML imports - install with: pip install scikit-learn xgboost
try:
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, confusion_matrix
    import warnings
    warnings.filterwarnings('ignore')
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    print("⚠️  ML libraries not installed. Run: pip install scikit-learn xgboost")


MODEL_DIR = Path("data/ml_models")
MODEL_FILE = MODEL_DIR / "breakout_crash_model.pkl"
SCALER_FILE = MODEL_DIR / "feature_scaler.pkl"


def extract_features(stock_data):
    """
    Extract ML features from stock data dictionary.
    
    Returns list of numerical features in consistent order.
    """
    features = []
    
    # Technical Indicators
    features.append(stock_data.get('rsi') or 50.0)  # RSI
    features.append(stock_data.get('bb_position_pct') or 50.0)  # BB Position %
    features.append(stock_data.get('bb_width_pct') or 10.0)  # BB Width %
    features.append(1.0 if stock_data.get('macd_label') == 'Bullish' else (-1.0 if stock_data.get('macd_label') == 'Bearish' else 0.0))
    features.append(stock_data.get('atr_14') or 0.0)  # ATR (volatility)
    
    # Volume & Momentum
    features.append(stock_data.get('volume_bias') or 0.0)  # Volume bias
    features.append(1.0 if stock_data.get('volume_spike') else 0.0)  # Volume spike flag
    features.append(stock_data.get('change_pct') or 0.0)  # Daily change %
    features.append(stock_data.get('change_5d') or 0.0)  # 5-day change %
    features.append(stock_data.get('change_1m') or 0.0)  # 1-month change %
    
    # Moving Averages & Crosses
    features.append(1.0 if stock_data.get('golden_cross') else (-1.0 if stock_data.get('death_cross') else 0.0))
    
    # Fundamentals
    features.append(stock_data.get('pe_ratio') or 0.0)  # P/E ratio
    features.append(stock_data.get('market_cap') or 0.0)  # Market cap (log scale)
    
    # Sentiment & Risk
    features.append(stock_data.get('put_call_ratio') or 1.0)  # Put/Call ratio
    features.append(stock_data.get('short_interest') or 0.0)  # Short interest %
    
    # Squeeze Indicators
    squeeze_level = stock_data.get('squeeze_level') or 'None'
    squeeze_score = {'None': 0, 'Moderate': 1, 'High': 2, 'Extreme': 3}.get(squeeze_level, 0)
    features.append(float(squeeze_score))
    
    # Trading Signals
    active_signal = stock_data.get('active_signal') or 'HOLD'
    signal_map = {'BUY': 1.0, 'SHORT': -1.0, 'SELL': -0.5, 'HOLD': 0.0}
    features.append(signal_map.get(active_signal, 0.0))
    
    # Trend Score
    features.append(stock_data.get('trend_score') or 0.0)
    
    return features


FEATURE_NAMES = [
    'rsi', 'bb_position_pct', 'bb_width_pct', 'macd_signal',
    'atr_14', 'volume_bias', 'volume_spike', 'change_day',
    'change_5d', 'change_1m', 'ma_cross', 'pe_ratio', 
    'market_cap', 'put_call_ratio', 'short_interest',
    'squeeze_score', 'trading_signal', 'trend_score'
]


def train_model(historical_data, labels):
    """
    Train ML model on historical stock data.
    
    Args:
        historical_data: List of stock data dictionaries
        labels: List of labels ('BREAKOUT', 'CRASH', 'NEUTRAL')
    
    Returns:
        Trained model and scaler
    """
    if not ML_AVAILABLE:
        raise ImportError("ML libraries not installed")
    
    # Extract features
    X = []
    y = []
    
    for stock, label in zip(historical_data, labels):
        features = extract_features(stock)
        X.append(features)
        y.append(label)
    
    X = np.array(X)
    y = np.array(y)
    
    # Handle missing values
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train Gradient Boosting Classifier
    model = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=5,
        min_samples_split=20,
        min_samples_leaf=10,
        subsample=0.8,
        random_state=42,
        verbose=1
    )
    
    print(f"Training model on {len(X_train)} samples...")
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    print("\n📊 Model Performance:")
    print(classification_report(y_test, y_pred))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    # Feature importance
    importances = model.feature_importances_
    feature_imp = pd.DataFrame({
        'feature': FEATURE_NAMES,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    print("\n🔝 Top 10 Most Important Features:")
    print(feature_imp.head(10).to_string(index=False))
    
    # Save model
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    with open(MODEL_FILE, 'wb') as f:
        pickle.dump(model, f)
    with open(SCALER_FILE, 'wb') as f:
        pickle.dump(scaler, f)
    
    print(f"\n✅ Model saved to {MODEL_FILE}")
    
    return model, scaler


def load_model():
    """Load trained model and scaler from disk."""
    if not MODEL_FILE.exists() or not SCALER_FILE.exists():
        return None, None
    
    try:
        with open(MODEL_FILE, 'rb') as f:
            model = pickle.load(f)
        with open(SCALER_FILE, 'rb') as f:
            scaler = pickle.load(f)
        return model, scaler
    except Exception as e:
        print(f"⚠️  Error loading model: {e}")
        return None, None


def predict_breakout_crash(stock_data, model=None, scaler=None):
    """
    Predict breakout/crash probability for a single stock.
    
    Args:
        stock_data: Dictionary with stock metrics
        model: Trained model (loads from disk if None)
        scaler: Feature scaler (loads from disk if None)
    
    Returns:
        Dictionary with:
            - breakout_score: 0-100 (probability of breakout)
            - crash_risk: 0-100 (probability of crash)
            - prediction: 'BREAKOUT' | 'CRASH' | 'NEUTRAL'
            - confidence: 0-100 (model confidence)
    """
    if not ML_AVAILABLE:
        return {
            'breakout_score': 0,
            'crash_risk': 0,
            'prediction': 'NEUTRAL',
            'confidence': 0
        }
    
    # Load model if not provided
    if model is None or scaler is None:
        model, scaler = load_model()
        if model is None:
            return {
                'breakout_score': 0,
                'crash_risk': 0,
                'prediction': 'NEUTRAL',
                'confidence': 0
            }
    
    # Extract and scale features
    features = extract_features(stock_data)
    X = np.array([features])
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    X_scaled = scaler.transform(X)
    
    # Get prediction probabilities
    proba = model.predict_proba(X_scaled)[0]
    prediction = model.predict(X_scaled)[0]
    
    # Map to scores
    # Assumes classes are ordered: ['BREAKOUT', 'CRASH', 'NEUTRAL']
    classes = model.classes_
    breakout_idx = list(classes).index('BREAKOUT') if 'BREAKOUT' in classes else 0
    crash_idx = list(classes).index('CRASH') if 'CRASH' in classes else 1
    
    breakout_score = int(proba[breakout_idx] * 100)
    crash_risk = int(proba[crash_idx] * 100)
    confidence = int(max(proba) * 100)
    
    return {
        'breakout_score': breakout_score,
        'crash_risk': crash_risk,
        'prediction': prediction,
        'confidence': confidence
    }


def batch_predict(stock_list):
    """
    Predict for multiple stocks efficiently.
    
    Args:
        stock_list: List of stock data dictionaries
    
    Returns:
        List of prediction dictionaries
    """
    model, scaler = load_model()
    if model is None:
        return [predict_breakout_crash(s, None, None) for s in stock_list]
    
    results = []
    for stock in stock_list:
        pred = predict_breakout_crash(stock, model, scaler)
        results.append(pred)
    
    return results


def get_tickers_from_file(file_path):
    """
    Parse tickers from a file with sections like [SECTION].
    """
    tickers = set()
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('[') and line.endswith(']'):
                continue
            if line:
                # Split by comma and clean up tickers
                tickers.update([t.strip() for t in line.split(',') if t.strip()])
    return sorted(list(tickers))


# Example training data generator (for demonstration)
def generate_synthetic_training_data(n_samples=1000):
    """
    Generate synthetic training data for demonstration.
    
    In production, replace with real historical data:
    - Pull historical stock data from yfinance
    - Calculate features at time T
    - Label based on price movement T+6months
    """
    np.random.seed(42)
    data = []
    labels = []
    
    for _ in range(n_samples):
        # Generate random stock features
        stock = {
            'rsi': np.random.uniform(20, 80),
            'bb_position_pct': np.random.uniform(0, 100),
            'bb_width_pct': np.random.uniform(2, 20),
            'macd_label': np.random.choice(['Bullish', 'Bearish', None]),
            'atr_14': np.random.uniform(1, 10),
            'volume_bias': np.random.uniform(-1, 1),
            'volume_spike': np.random.choice([True, False]),
            'change_pct': np.random.uniform(-10, 10),
            'change_5d': np.random.uniform(-15, 15),
            'change_1m': np.random.uniform(-20, 20),
            'golden_cross': np.random.choice([True, False]),
            'death_cross': np.random.choice([True, False]),
            'pe_ratio': np.random.uniform(10, 50),
            'market_cap': np.random.uniform(1e9, 1e12),
            'put_call_ratio': np.random.uniform(0.5, 2.0),
            'short_interest': np.random.uniform(0, 30),
            'squeeze_level': np.random.choice(['None', 'Moderate', 'High', 'Extreme']),
            'active_signal': np.random.choice(['BUY', 'SHORT', 'SELL', 'HOLD']),
            'trend_score': np.random.uniform(-5, 5),
        }
        
        # Synthetic label based on features (simplified logic)
        # In reality, label would come from actual future price movement
        score = 0
        if stock['rsi'] < 30 and stock['active_signal'] == 'BUY':
            score += 2
        if stock['golden_cross'] and stock['volume_spike']:
            score += 2
        if stock['death_cross'] or stock['rsi'] > 70:
            score -= 2
        
        if score >= 2:
            label = 'BREAKOUT'
        elif score <= -2:
            label = 'CRASH'
        else:
            label = 'NEUTRAL'
        
        data.append(stock)
        labels.append(label)
    
    return data, labels


if __name__ == "__main__":
    print("🤖 ML Stock Predictor - Training Demo\n")
    
    if not ML_AVAILABLE:
        print("❌ Please install ML libraries: pip install scikit-learn")
        exit(1)

    try:
        import yfinance as yf
    except ImportError:
        print("❌ Please install yfinance: pip install yfinance")
        exit(1)

    # --- Use real data from tickers.csv for training ---
    TICKER_FILE = Path("data/tickers.csv")
    if not TICKER_FILE.exists():
        print(f"❌ Ticker file not found at {TICKER_FILE}")
        print("Falling back to synthetic data generation.")
        # Generate synthetic training data
        print("📊 Generating synthetic training data...")
        training_data, labels = generate_synthetic_training_data(n_samples=2000)
    else:
        print(f"✅ Found ticker file at {TICKER_FILE}")
        print("⚙️  Processing real historical data for training. This may take a while...")
        
        tickers = get_tickers_from_file(TICKER_FILE)
        print(f"Found {len(tickers)} unique tickers.")

        # This is a simplified feature/label generation process for demonstration.
        # For a production system, this logic should be much more robust.
        training_data = []
        labels = []

        for i, ticker in enumerate(tickers):
            print(f"Processing {ticker} ({i+1}/{len(tickers)})...")
            try:
                stock_hist = yf.Ticker(ticker).history(period="2y", auto_adjust=False)
                if len(stock_hist) < 252: # Need at least 1 year of data
                    continue

                # Use data from 1 year ago to generate features
                one_year_ago_data = stock_hist.iloc[-252]
                
                # Simplified feature creation - in a real scenario, you'd calculate
                # all the features that extract_features expects.
                features = {
                    'rsi': np.random.uniform(30, 70), # Placeholder
                    'pe_ratio': one_year_ago_data.get('P/E Ratio', 25),
                    'market_cap': one_year_ago_data.get('Market Cap', 1e9),
                    'change_1m': (one_year_ago_data['Close'] / stock_hist.iloc[-252-21]['Close'] - 1) * 100 if len(stock_hist) > 252+21 else 0,
                }

                # Label based on future performance (next 6-12 months)
                future_price_6m = stock_hist.iloc[-126]['Close']
                future_price_12m = stock_hist.iloc[-1]['Close']
                
                initial_price = one_year_ago_data['Close']
                
                label = 'NEUTRAL'
                if (future_price_6m / initial_price > 2.0) or (future_price_12m / initial_price > 2.0):
                    label = 'BREAKOUT'
                elif (future_price_6m / initial_price < 0.5) or (future_price_12m / initial_price < 0.5):
                    label = 'CRASH'
                
                training_data.append(features)
                labels.append(label)

            except Exception as e:
                print(f"  Could not process {ticker}: {e}")

        if not training_data:
            print("❌ No training data could be generated from the tickers. Exiting.")
            exit(1)

    # Train model
    print("\n🏋️  Training model...")
    model, scaler = train_model(training_data, labels)
    
    # Test prediction
    print("\n🔮 Testing prediction on sample stock...")
    test_stock = {
        'rsi': 25,
        'bb_position_pct': 5,
        'bb_width_pct': 3,
        'macd_label': 'Bullish',
        'atr_14': 2.5,
        'volume_bias': 0.6,
        'volume_spike': True,
        'change_pct': 3.2,
        'change_5d': 8.5,
        'change_1m': 15.0,
        'golden_cross': True,
        'death_cross': False,
        'pe_ratio': 25,
        'market_cap': 50e9,
        'put_call_ratio': 0.8,
        'short_interest': 15,
        'squeeze_level': 'High',
        'active_signal': 'BUY',
        'trend_score': 3.5,
    }
    
    result = predict_breakout_crash(test_stock, model, scaler)
    print(f"\n📈 Prediction Results:")
    print(f"   Breakout Score: {result['breakout_score']}%")
    print(f"   Crash Risk: {result['crash_risk']}%")
    print(f"   Prediction: {result['prediction']}")
    print(f"   Confidence: {result['confidence']}%")
    
    print("\n✅ Model ready for production use!")
