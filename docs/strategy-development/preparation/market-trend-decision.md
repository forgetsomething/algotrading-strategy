# Market Trend Decision

Market trend analysis forms the foundation of successful algorithmic trading strategies. This chapter explores how to leverage machine learning techniques to identify and predict market trends across different time horizons, enabling your trading algorithms to adapt to changing market conditions.

## Overview

Understanding market trends is crucial for:

- **Strategy Selection**: Choosing the right strategy for current market conditions
- **Risk Management**: Adjusting position sizes based on trend strength
- **Entry/Exit Timing**: Optimizing trade execution timing
- **Portfolio Allocation**: Dynamically adjusting asset allocation

## Time Horizon Analysis

### Short-Term Trends (1-30 days)

Short-term trend analysis focuses on immediate market movements and is primarily used for:

- **Day Trading Strategies**: Intraday position management
- **Swing Trading**: Capturing 2-10 day price movements
- **Risk Management**: Quick response to adverse movements

```python
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

class ShortTermTrendAnalyzer:
    def __init__(self, lookback_window=20):
        self.lookback_window = lookback_window
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        
    def create_features(self, price_data):
        """Create technical indicators for trend analysis"""
        features = pd.DataFrame()
        
        # Price-based features
        features['returns_1d'] = price_data['close'].pct_change(1)
        features['returns_5d'] = price_data['close'].pct_change(5)
        features['returns_10d'] = price_data['close'].pct_change(10)
        
        # Moving averages
        features['sma_5'] = price_data['close'].rolling(5).mean()
        features['sma_20'] = price_data['close'].rolling(20).mean()
        features['price_vs_sma5'] = price_data['close'] / features['sma_5'] - 1
        features['price_vs_sma20'] = price_data['close'] / features['sma_20'] - 1
        
        # Volatility features
        features['volatility_10d'] = features['returns_1d'].rolling(10).std()
        features['volatility_20d'] = features['returns_1d'].rolling(20).std()
        
        # Volume features
        features['volume_ratio'] = price_data['volume'] / price_data['volume'].rolling(20).mean()
        
        # RSI
        features['rsi'] = self.calculate_rsi(price_data['close'])
        
        return features.dropna()
    
    def calculate_rsi(self, prices, window=14):
        """Calculate Relative Strength Index"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def create_labels(self, price_data, horizon=5):
        """Create trend labels for supervised learning"""
        future_returns = price_data['close'].shift(-horizon) / price_data['close'] - 1
        
        # Define trend categories
        labels = pd.Series(index=price_data.index, dtype=int)
        labels[future_returns > 0.02] = 2  # Strong uptrend
        labels[(future_returns > 0.005) & (future_returns <= 0.02)] = 1  # Weak uptrend
        labels[(future_returns >= -0.005) & (future_returns <= 0.005)] = 0  # Sideways
        labels[(future_returns >= -0.02) & (future_returns < -0.005)] = -1  # Weak downtrend
        labels[future_returns < -0.02] = -2  # Strong downtrend
        
        return labels
    
    def train_model(self, price_data):
        """Train the trend prediction model"""
        features = self.create_features(price_data)
        labels = self.create_labels(price_data)
        
        # Align features and labels
        aligned_data = pd.concat([features, labels.rename('trend')], axis=1).dropna()
        
        X = aligned_data.drop('trend', axis=1)
        y = aligned_data['trend']
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train model
        self.model.fit(X_scaled, y)
        
        return self.model.score(X_scaled, y)
```

### Mid-Term Trends (1-6 months)

Mid-term trend analysis captures intermediate market cycles and is essential for:

- **Position Trading**: Holding positions for weeks to months
- **Portfolio Rebalancing**: Adjusting strategic allocations
- **Sector Rotation**: Moving between different market sectors

```python
class MidTermTrendAnalyzer:
    def __init__(self):
        self.regime_model = None
        self.trend_strength_model = None
        
    def detect_market_regimes(self, price_data):
        """Detect market regimes using Hidden Markov Models"""
        from hmmlearn import hmm
        
        # Calculate features for regime detection
        returns = price_data['close'].pct_change().dropna()
        volatility = returns.rolling(20).std()
        
        # Prepare data for HMM
        features = np.column_stack([
            returns.values,
            volatility.values
        ])[20:]  # Remove first 20 NaN values
        
        # Fit HMM with 3 states (bear, neutral, bull)
        model = hmm.GaussianHMM(n_components=3, covariance_type="full")
        model.fit(features)
        
        # Predict regimes
        regimes = model.predict(features)
        
        return regimes, model
    
    def calculate_trend_strength(self, price_data, window=60):
        """Calculate trend strength using multiple indicators"""
        # ADX (Average Directional Index)
        adx = self.calculate_adx(price_data, window)
        
        # Linear regression slope
        slopes = []
        for i in range(window, len(price_data)):
            y = price_data['close'].iloc[i-window:i].values
            x = np.arange(len(y))
            slope = np.polyfit(x, y, 1)[0]
            slopes.append(slope)
        
        slope_series = pd.Series(slopes, index=price_data.index[window:])
        
        # Combine indicators
        trend_strength = pd.DataFrame({
            'adx': adx,
            'slope': slope_series
        }).dropna()
        
        # Normalize and combine
        trend_strength_normalized = (trend_strength - trend_strength.mean()) / trend_strength.std()
        combined_strength = trend_strength_normalized.mean(axis=1)
        
        return combined_strength
```

### Long-Term Trends (6 months - 5 years)

Long-term trend analysis captures major market cycles and structural changes:

- **Strategic Asset Allocation**: Long-term portfolio positioning
- **Macroeconomic Analysis**: Understanding economic cycles
- **Fundamental Trends**: Identifying secular market shifts

```python
class LongTermTrendAnalyzer:
    def __init__(self):
        self.cycle_length = 252 * 4  # 4 years of daily data
        
    def detect_secular_trends(self, price_data, economic_data=None):
        """Detect long-term secular trends"""
        # Calculate long-term moving averages
        price_data['ma_200'] = price_data['close'].rolling(200).mean()
        price_data['ma_500'] = price_data['close'].rolling(500).mean()
        
        # Trend direction
        trend_direction = np.where(
            price_data['close'] > price_data['ma_200'], 1, -1
        )
        
        # Trend strength based on distance from long-term MA
        trend_strength = (
            price_data['close'] / price_data['ma_500'] - 1
        ).abs()
        
        # Economic cycle integration (if available)
        if economic_data is not None:
            cycle_features = self.integrate_economic_cycles(economic_data)
            return trend_direction, trend_strength, cycle_features
        
        return trend_direction, trend_strength
    
    def integrate_economic_cycles(self, economic_data):
        """Integrate macroeconomic indicators"""
        cycle_features = pd.DataFrame()
        
        # Interest rate environment
        if 'interest_rates' in economic_data.columns:
            cycle_features['rate_trend'] = economic_data['interest_rates'].diff(12)
        
        # Economic growth indicators
        if 'gdp_growth' in economic_data.columns:
            cycle_features['growth_trend'] = economic_data['gdp_growth'].rolling(4).mean()
        
        # Inflation trends
        if 'inflation' in economic_data.columns:
            cycle_features['inflation_trend'] = economic_data['inflation'].rolling(6).mean()
        
        return cycle_features
```

## Machine Learning Implementation

### Feature Engineering for Trend Detection

```python
class TrendFeatureEngineer:
    def __init__(self):
        self.feature_names = []
        
    def engineer_features(self, price_data, volume_data=None, sentiment_data=None):
        """Comprehensive feature engineering for trend detection"""
        features = pd.DataFrame(index=price_data.index)
        
        # 1. Price-based features
        features.update(self._price_features(price_data))
        
        # 2. Volume-based features (if available)
        if volume_data is not None:
            features.update(self._volume_features(volume_data))
        
        # 3. Sentiment features (if available)
        if sentiment_data is not None:
            features.update(self._sentiment_features(sentiment_data))
        
        # 4. Cross-asset features
        features.update(self._cross_asset_features(price_data))
        
        self.feature_names = features.columns.tolist()
        return features.dropna()
    
    def _price_features(self, price_data):
        """Extract price-based technical features"""
        features = {}
        
        # Multiple timeframe returns
        for period in [1, 3, 5, 10, 20, 50]:
            features[f'return_{period}d'] = price_data['close'].pct_change(period)
        
        # Moving average ratios
        for ma_period in [5, 10, 20, 50, 200]:
            ma = price_data['close'].rolling(ma_period).mean()
            features[f'price_ma{ma_period}_ratio'] = price_data['close'] / ma - 1
        
        # Volatility features
        returns = price_data['close'].pct_change()
        for vol_period in [5, 10, 20, 50]:
            features[f'volatility_{vol_period}d'] = returns.rolling(vol_period).std()
        
        # Technical indicators
        features['rsi_14'] = self._calculate_rsi(price_data['close'])
        features['bb_position'] = self._bollinger_band_position(price_data['close'])
        features['macd_signal'] = self._macd_signal(price_data['close'])
        
        return pd.DataFrame(features, index=price_data.index)
```

### Model Training and Validation

```python
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix

class TrendPredictionPipeline:
    def __init__(self):
        self.models = {
            'random_forest': RandomForestClassifier(random_state=42),
            'gradient_boosting': GradientBoostingClassifier(random_state=42),
            'neural_network': MLPClassifier(random_state=42, max_iter=1000)
        }
        self.best_model = None
        self.scaler = StandardScaler()
        
    def train_models(self, X, y, test_size=0.2):
        """Train and validate multiple models"""
        # Time series split for validation
        tscv = TimeSeriesSplit(n_splits=5)
        
        results = {}
        
        for name, model in self.models.items():
            print(f"Training {name}...")
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Cross-validation
            cv_scores = []
            for train_idx, val_idx in tscv.split(X_scaled):
                X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                
                model.fit(X_train, y_train)
                score = model.score(X_val, y_val)
                cv_scores.append(score)
            
            results[name] = {
                'mean_cv_score': np.mean(cv_scores),
                'std_cv_score': np.std(cv_scores),
                'model': model
            }
        
        # Select best model
        best_model_name = max(results.keys(), key=lambda k: results[k]['mean_cv_score'])
        self.best_model = results[best_model_name]['model']
        
        print(f"Best model: {best_model_name}")
        print(f"CV Score: {results[best_model_name]['mean_cv_score']:.4f} (+/- {results[best_model_name]['std_cv_score']:.4f})")
        
        return results
    
    def predict_trend(self, X_new):
        """Predict trend for new data"""
        if self.best_model is None:
            raise ValueError("Model not trained yet!")
        
        X_scaled = self.scaler.transform(X_new)
        predictions = self.best_model.predict(X_scaled)
        probabilities = self.best_model.predict_proba(X_scaled)
        
        return predictions, probabilities
```

## Integration with Trading Strategy

### Real-time Trend Monitoring

```python
class RealTimeTrendMonitor:
    def __init__(self, models_dict):
        self.short_term_model = models_dict['short_term']
        self.mid_term_model = models_dict['mid_term']
        self.long_term_model = models_dict['long_term']
        
    def get_trend_signals(self, current_data):
        """Get trend signals across all timeframes"""
        signals = {}
        
        # Short-term signal
        short_features = self.short_term_model.engineer_features(current_data[-30:])
        signals['short_term'] = self.short_term_model.predict_trend(short_features[-1:])
        
        # Mid-term signal
        mid_features = self.mid_term_model.engineer_features(current_data[-120:])
        signals['mid_term'] = self.mid_term_model.predict_trend(mid_features[-1:])
        
        # Long-term signal
        long_features = self.long_term_model.engineer_features(current_data[-500:])
        signals['long_term'] = self.long_term_model.predict_trend(long_features[-1:])
        
        # Combine signals
        combined_signal = self._combine_signals(signals)
        
        return signals, combined_signal
    
    def _combine_signals(self, signals):
        """Combine multi-timeframe signals"""
        weights = {
            'short_term': 0.3,
            'mid_term': 0.4,
            'long_term': 0.3
        }
        
        weighted_signal = sum(
            signals[timeframe][0][0] * weight 
            for timeframe, weight in weights.items()
        )
        
        return weighted_signal
```

## Practical Implementation

### Example: Complete Trend Analysis System

```python
def main_trend_analysis():
    """Complete trend analysis workflow"""
    
    # 1. Load data
    price_data = load_market_data('SPY', start_date='2015-01-01')
    
    # 2. Initialize analyzers
    short_analyzer = ShortTermTrendAnalyzer()
    mid_analyzer = MidTermTrendAnalyzer()
    long_analyzer = LongTermTrendAnalyzer()
    
    # 3. Train models
    print("Training short-term model...")
    short_accuracy = short_analyzer.train_model(price_data)
    print(f"Short-term model accuracy: {short_accuracy:.4f}")
    
    print("Training mid-term model...")
    regimes, regime_model = mid_analyzer.detect_market_regimes(price_data)
    trend_strength = mid_analyzer.calculate_trend_strength(price_data)
    
    print("Analyzing long-term trends...")
    long_trend_dir, long_trend_strength = long_analyzer.detect_secular_trends(price_data)
    
    # 4. Generate current predictions
    latest_data = price_data.tail(100)
    
    # Get short-term prediction
    short_features = short_analyzer.create_features(latest_data)
    latest_features = short_features.tail(1)
    short_prediction = short_analyzer.model.predict(
        short_analyzer.scaler.transform(latest_features)
    )[0]
    
    print(f"Current trend predictions:")
    print(f"Short-term: {short_prediction}")
    print(f"Mid-term regime: {regimes[-1]}")
    print(f"Long-term direction: {long_trend_dir[-1]}")
    
    return {
        'short_term': short_prediction,
        'mid_term': regimes[-1],
        'long_term': long_trend_dir[-1]
    }

if __name__ == "__main__":
    trend_signals = main_trend_analysis()
```

## Key Takeaways

1. **Multi-timeframe Analysis**: Combine short, mid, and long-term trend signals for robust decision making
2. **Machine Learning Enhancement**: ML models can capture complex patterns that traditional technical analysis might miss
3. **Feature Engineering**: Comprehensive feature engineering is crucial for model performance
4. **Model Validation**: Use time-series cross-validation to ensure models generalize to future data
5. **Real-time Integration**: Design systems that can process new data and update predictions continuously

## Next Steps

In the next chapter, we'll explore [Regime Detection](regime-detection.md) techniques that help identify distinct market environments and adapt trading strategies accordingly.

---

**Previous**: [Why Algorithmic Trading Matters ←](../../introduction/why-algo-trading.md)

**Next**: [Regime Detection →](regime-detection.md)

**Related Topics**:
- [Feature Selection](../features/feature-selection.md)
- [Machine Learning Trading](../../strategies/ml-trading.md)
- [Risk Management](../../management/risk/model-based-var.md)
