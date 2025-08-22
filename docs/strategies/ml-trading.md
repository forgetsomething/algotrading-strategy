# Machine Learning Trading Strategies

This chapter explores how to integrate machine learning techniques into trading strategy development, focusing on practical implementations that can generate alpha in real market conditions. We'll cover everything from feature engineering to model deployment and real-time prediction systems.

## Overview

Machine learning has revolutionized algorithmic trading by enabling:

- **Pattern Recognition**: Identifying complex market patterns invisible to traditional analysis
- **Adaptive Strategies**: Models that learn and adapt to changing market conditions
- **Feature Discovery**: Automatic identification of predictive market signals
- **Risk Management**: Dynamic risk assessment and position sizing

## Feature Engineering for Trading

### Price-Based Features

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import talib

class TradingFeatureEngineer:
    """
    Comprehensive feature engineering for ML trading strategies
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_names = []
    
    def create_price_features(self, ohlcv_data):
        """Create price-based technical features"""
        features = pd.DataFrame(index=ohlcv_data.index)
        
        # Basic price relationships
        features['open_close_ratio'] = ohlcv_data['open'] / ohlcv_data['close']
        features['high_low_ratio'] = ohlcv_data['high'] / ohlcv_data['low']
        features['close_volume_ratio'] = ohlcv_data['close'] / ohlcv_data['volume']
        
        # Returns across multiple timeframes
        for period in [1, 2, 3, 5, 10, 20]:
            features[f'return_{period}d'] = ohlcv_data['close'].pct_change(period)
            features[f'volatility_{period}d'] = features[f'return_1d'].rolling(period).std()
        
        # Moving averages and ratios
        for ma_period in [5, 10, 20, 50, 200]:
            features[f'sma_{ma_period}'] = ohlcv_data['close'].rolling(ma_period).mean()
            features[f'price_sma_{ma_period}_ratio'] = ohlcv_data['close'] / features[f'sma_{ma_period}']
        
        # Exponential moving averages
        for ema_period in [12, 26, 50]:
            features[f'ema_{ema_period}'] = ohlcv_data['close'].ewm(span=ema_period).mean()
            features[f'price_ema_{ema_period}_ratio'] = ohlcv_data['close'] / features[f'ema_{ema_period}']
        
        return features
    
    def create_technical_indicators(self, ohlcv_data):
        """Create traditional technical indicators"""
        features = pd.DataFrame(index=ohlcv_data.index)
        
        # RSI
        features['rsi_14'] = talib.RSI(ohlcv_data['close'].values, timeperiod=14)
        features['rsi_30'] = talib.RSI(ohlcv_data['close'].values, timeperiod=30)
        
        # MACD
        macd, macdsignal, macdhist = talib.MACD(ohlcv_data['close'].values)
        features['macd'] = macd
        features['macd_signal'] = macdsignal
        features['macd_histogram'] = macdhist
        
        # Bollinger Bands
        bb_upper, bb_middle, bb_lower = talib.BBANDS(ohlcv_data['close'].values)
        features['bb_upper'] = bb_upper
        features['bb_lower'] = bb_lower
        features['bb_width'] = (bb_upper - bb_lower) / bb_middle
        features['bb_position'] = (ohlcv_data['close'] - bb_lower) / (bb_upper - bb_lower)
        
        # Stochastic Oscillator
        slowk, slowd = talib.STOCH(
            ohlcv_data['high'].values,
            ohlcv_data['low'].values,
            ohlcv_data['close'].values
        )
        features['stoch_k'] = slowk
        features['stoch_d'] = slowd
        
        # Average True Range
        features['atr'] = talib.ATR(
            ohlcv_data['high'].values,
            ohlcv_data['low'].values,
            ohlcv_data['close'].values
        )
        
        # Volume indicators
        features['volume_sma_20'] = ohlcv_data['volume'].rolling(20).mean()
        features['volume_ratio'] = ohlcv_data['volume'] / features['volume_sma_20']
        
        # On Balance Volume
        features['obv'] = talib.OBV(ohlcv_data['close'].values, ohlcv_data['volume'].values)
        
        return features
    
    def create_advanced_features(self, ohlcv_data):
        """Create advanced ML-specific features"""
        features = pd.DataFrame(index=ohlcv_data.index)
        
        # Fractal features
        features['high_fractal'] = self._detect_fractals(ohlcv_data['high'], 'high')
        features['low_fractal'] = self._detect_fractals(ohlcv_data['low'], 'low')
        
        # Support and resistance levels
        features['support_level'] = self._calculate_support_resistance(ohlcv_data, 'support')
        features['resistance_level'] = self._calculate_support_resistance(ohlcv_data, 'resistance')
        
        # Price acceleration
        returns = ohlcv_data['close'].pct_change()
        features['price_acceleration'] = returns.diff()
        
        # Volatility regimes
        features['vol_regime'] = self._detect_volatility_regimes(returns)
        
        # Time-based features
        features['hour'] = ohlcv_data.index.hour
        features['day_of_week'] = ohlcv_data.index.dayofweek
        features['month'] = ohlcv_data.index.month
        
        # Lagged features
        for lag in [1, 2, 3, 5]:
            features[f'close_lag_{lag}'] = ohlcv_data['close'].shift(lag)
            features[f'volume_lag_{lag}'] = ohlcv_data['volume'].shift(lag)
        
        return features
```

### Multi-Asset and Cross-Asset Features

```python
class CrossAssetFeatureEngineer:
    """
    Create features based on relationships between multiple assets
    """
    
    def __init__(self, reference_assets=['SPY', 'QQQ', 'VIX', 'TLT']):
        self.reference_assets = reference_assets
    
    def create_correlation_features(self, target_data, reference_data_dict, window=20):
        """Create rolling correlation features with reference assets"""
        features = pd.DataFrame(index=target_data.index)
        
        target_returns = target_data['close'].pct_change()
        
        for asset_name, asset_data in reference_data_dict.items():
            if asset_name in self.reference_assets:
                ref_returns = asset_data['close'].pct_change()
                
                # Rolling correlation
                features[f'corr_{asset_name}_{window}d'] = (
                    target_returns.rolling(window).corr(ref_returns)
                )
                
                # Rolling beta
                covariance = target_returns.rolling(window).cov(ref_returns)
                ref_variance = ref_returns.rolling(window).var()
                features[f'beta_{asset_name}_{window}d'] = covariance / ref_variance
        
        return features
    
    def create_relative_strength_features(self, target_data, reference_data_dict):
        """Create relative strength features"""
        features = pd.DataFrame(index=target_data.index)
        
        for asset_name, asset_data in reference_data_dict.items():
            if asset_name in self.reference_assets:
                # Relative performance
                target_norm = target_data['close'] / target_data['close'].iloc[0]
                ref_norm = asset_data['close'] / asset_data['close'].iloc[0]
                
                features[f'relative_strength_{asset_name}'] = target_norm / ref_norm
                
                # Price ratio
                features[f'price_ratio_{asset_name}'] = (
                    target_data['close'] / asset_data['close']
                )
        
        return features
```

## Model Architecture and Training

### Multi-Model Ensemble System

```python
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import TimeSeriesSplit
import xgboost as xgb
import lightgbm as lgb

class MLTradingEnsemble:
    """
    Ensemble of multiple ML models for trading signal generation
    """
    
    def __init__(self):
        self.models = {
            'random_forest': RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                random_state=42
            ),
            'xgboost': xgb.XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            ),
            'lightgbm': lgb.LGBMClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            ),
            'neural_network': MLPClassifier(
                hidden_layer_sizes=(100, 50),
                max_iter=1000,
                random_state=42
            ),
            'logistic_regression': LogisticRegression(
                random_state=42,
                max_iter=1000
            )
        }
        
        self.trained_models = {}
        self.feature_importance = {}
        self.ensemble_weights = {}
    
    def create_labels(self, price_data, method='forward_returns', threshold=0.02):
        """Create trading labels based on future returns"""
        if method == 'forward_returns':
            # Multi-class classification based on forward returns
            forward_returns = price_data['close'].shift(-5) / price_data['close'] - 1
            
            labels = pd.Series(index=price_data.index, dtype=int)
            labels[forward_returns > threshold] = 2      # Strong buy
            labels[(forward_returns > 0.005) & (forward_returns <= threshold)] = 1  # Buy
            labels[abs(forward_returns) <= 0.005] = 0   # Hold
            labels[(forward_returns < -0.005) & (forward_returns >= -threshold)] = -1  # Sell
            labels[forward_returns < -threshold] = -2   # Strong sell
            
        elif method == 'trend_following':
            # Trend following labels
            sma_short = price_data['close'].rolling(10).mean()
            sma_long = price_data['close'].rolling(30).mean()
            
            labels = pd.Series(index=price_data.index, dtype=int)
            labels[sma_short > sma_long] = 1  # Uptrend
            labels[sma_short <= sma_long] = -1  # Downtrend
        
        return labels.dropna()
    
    def train_ensemble(self, X, y, validation_split=0.2):
        """Train ensemble of models with time series validation"""
        # Time series split for validation
        tscv = TimeSeriesSplit(n_splits=5)
        
        model_scores = {}
        
        for model_name, model in self.models.items():
            print(f"Training {model_name}...")
            
            cv_scores = []
            feature_importances = []
            
            for train_idx, val_idx in tscv.split(X):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                
                # Train model
                model.fit(X_train, y_train)
                
                # Validate
                val_score = model.score(X_val, y_val)
                cv_scores.append(val_score)
                
                # Feature importance (if available)
                if hasattr(model, 'feature_importances_'):
                    feature_importances.append(model.feature_importances_)
            
            # Store results
            model_scores[model_name] = {
                'mean_cv_score': np.mean(cv_scores),
                'std_cv_score': np.std(cv_scores)
            }
            
            if feature_importances:
                self.feature_importance[model_name] = np.mean(feature_importances, axis=0)
            
            # Final training on full dataset
            model.fit(X, y)
            self.trained_models[model_name] = model
            
            print(f"{model_name} CV Score: {model_scores[model_name]['mean_cv_score']:.4f} "
                  f"(+/- {model_scores[model_name]['std_cv_score']:.4f})")
        
        # Calculate ensemble weights based on performance
        scores = [model_scores[name]['mean_cv_score'] for name in self.models.keys()]
        total_score = sum(scores)
        self.ensemble_weights = {
            name: score / total_score 
            for name, score in zip(self.models.keys(), scores)
        }
        
        return model_scores
    
    def predict_ensemble(self, X):
        """Generate ensemble predictions"""
        predictions = {}
        probabilities = {}
        
        for model_name, model in self.trained_models.items():
            pred = model.predict(X)
            pred_proba = model.predict_proba(X)
            
            predictions[model_name] = pred
            probabilities[model_name] = pred_proba
        
        # Weighted ensemble prediction
        ensemble_proba = np.zeros_like(probabilities[list(self.models.keys())[0]])
        
        for model_name, weight in self.ensemble_weights.items():
            ensemble_proba += weight * probabilities[model_name]
        
        ensemble_pred = np.argmax(ensemble_proba, axis=1)
        
        return {
            'individual_predictions': predictions,
            'individual_probabilities': probabilities,
            'ensemble_prediction': ensemble_pred,
            'ensemble_probability': ensemble_proba
        }
```

### Deep Learning Implementation

```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

class DeepLearningTradingModel:
    """
    Deep learning models for trading signal generation
    """
    
    def __init__(self, sequence_length=60):
        self.sequence_length = sequence_length
        self.models = {}
        self.scalers = {}
    
    def create_lstm_model(self, input_shape, n_classes=5):
        """Create LSTM model for sequence prediction"""
        model = Sequential([
            LSTM(100, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(100, return_sequences=True),
            Dropout(0.2),
            LSTM(50),
            Dropout(0.2),
            Dense(25, activation='relu'),
            Dense(n_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def create_cnn_lstm_model(self, input_shape, n_classes=5):
        """Create CNN-LSTM hybrid model"""
        model = Sequential([
            Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape),
            Conv1D(filters=64, kernel_size=3, activation='relu'),
            MaxPooling1D(pool_size=2),
            Dropout(0.25),
            LSTM(100, return_sequences=True),
            Dropout(0.2),
            LSTM(50),
            Dropout(0.2),
            Dense(25, activation='relu'),
            Dense(n_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def prepare_sequences(self, X, y):
        """Prepare sequences for deep learning models"""
        X_sequences = []
        y_sequences = []
        
        for i in range(self.sequence_length, len(X)):
            X_sequences.append(X.iloc[i-self.sequence_length:i].values)
            y_sequences.append(y.iloc[i])
        
        return np.array(X_sequences), np.array(y_sequences)
    
    def train_deep_models(self, X, y, validation_split=0.2):
        """Train deep learning models"""
        # Prepare sequences
        X_seq, y_seq = self.prepare_sequences(X, y)
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_seq.reshape(-1, X_seq.shape[-1]))
        X_scaled = X_scaled.reshape(X_seq.shape)
        self.scalers['features'] = scaler
        
        # Split data
        split_idx = int(len(X_scaled) * (1 - validation_split))
        X_train, X_val = X_scaled[:split_idx], X_scaled[split_idx:]
        y_train, y_val = y_seq[:split_idx], y_seq[split_idx:]
        
        # Callbacks
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=20,
            restore_best_weights=True
        )
        
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=10,
            min_lr=0.0001
        )
        
        # Train LSTM model
        print("Training LSTM model...")
        lstm_model = self.create_lstm_model(
            input_shape=(X_scaled.shape[1], X_scaled.shape[2])
        )
        
        lstm_history = lstm_model.fit(
            X_train, y_train,
            epochs=100,
            batch_size=32,
            validation_data=(X_val, y_val),
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )
        
        self.models['lstm'] = lstm_model
        
        # Train CNN-LSTM model
        print("Training CNN-LSTM model...")
        cnn_lstm_model = self.create_cnn_lstm_model(
            input_shape=(X_scaled.shape[1], X_scaled.shape[2])
        )
        
        cnn_lstm_history = cnn_lstm_model.fit(
            X_train, y_train,
            epochs=100,
            batch_size=32,
            validation_data=(X_val, y_val),
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )
        
        self.models['cnn_lstm'] = cnn_lstm_model
        
        return {
            'lstm_history': lstm_history,
            'cnn_lstm_history': cnn_lstm_history
        }
```

## Real-Time Trading System

### Live Prediction Pipeline

```python
class RealTimeTradingSystem:
    """
    Real-time trading system integrating ML predictions
    """
    
    def __init__(self, ensemble_model, deep_model, feature_engineer):
        self.ensemble_model = ensemble_model
        self.deep_model = deep_model
        self.feature_engineer = feature_engineer
        
        self.position = 0
        self.cash = 100000
        self.portfolio_value = 100000
        self.trades = []
        
    def get_live_prediction(self, current_data):
        """Get prediction from both ensemble and deep learning models"""
        # Engineer features
        features = self.feature_engineer.engineer_all_features(current_data)
        
        # Ensemble prediction
        ensemble_result = self.ensemble_model.predict_ensemble(features.tail(1))
        
        # Deep learning prediction (if enough data)
        if len(features) >= self.deep_model.sequence_length:
            X_seq, _ = self.deep_model.prepare_sequences(
                features.tail(self.deep_model.sequence_length + 1).iloc[:-1],
                pd.Series([0])  # Dummy label
            )
            X_scaled = self.deep_model.scalers['features'].transform(
                X_seq.reshape(-1, X_seq.shape[-1])
            ).reshape(X_seq.shape)
            
            deep_pred_lstm = self.deep_model.models['lstm'].predict(X_scaled[-1:])
            deep_pred_cnn = self.deep_model.models['cnn_lstm'].predict(X_scaled[-1:])
            
            # Combine deep learning predictions
            deep_pred_combined = (deep_pred_lstm + deep_pred_cnn) / 2
        else:
            deep_pred_combined = None
        
        return {
            'ensemble': ensemble_result,
            'deep_learning': deep_pred_combined,
            'features': features.tail(1)
        }
    
    def generate_trading_signal(self, predictions, confidence_threshold=0.6):
        """Generate trading signal based on model predictions"""
        ensemble_pred = predictions['ensemble']['ensemble_prediction'][0]
        ensemble_proba = predictions['ensemble']['ensemble_probability'][0]
        
        # Get confidence (max probability)
        ensemble_confidence = np.max(ensemble_proba)
        
        # Combine with deep learning if available
        if predictions['deep_learning'] is not None:
            deep_pred = np.argmax(predictions['deep_learning'][0])
            deep_confidence = np.max(predictions['deep_learning'][0])
            
            # Weighted combination
            combined_signal = (ensemble_pred + deep_pred) / 2
            combined_confidence = (ensemble_confidence + deep_confidence) / 2
        else:
            combined_signal = ensemble_pred
            combined_confidence = ensemble_confidence
        
        # Generate signal if confidence is high enough
        if combined_confidence > confidence_threshold:
            if combined_signal >= 1.5:  # Buy signals (1, 2)
                return 'BUY'
            elif combined_signal <= -0.5:  # Sell signals (-1, -2)
                return 'SELL'
        
        return 'HOLD'
    
    def execute_trade(self, signal, current_price, quantity_pct=0.1):
        """Execute trade based on signal"""
        if signal == 'BUY' and self.cash > 0:
            quantity = int((self.cash * quantity_pct) / current_price)
            if quantity > 0:
                cost = quantity * current_price
                self.position += quantity
                self.cash -= cost
                
                trade = {
                    'timestamp': pd.Timestamp.now(),
                    'signal': signal,
                    'quantity': quantity,
                    'price': current_price,
                    'cost': cost
                }
                self.trades.append(trade)
                
                return trade
        
        elif signal == 'SELL' and self.position > 0:
            quantity = min(int(self.position * quantity_pct), self.position)
            if quantity > 0:
                proceeds = quantity * current_price
                self.position -= quantity
                self.cash += proceeds
                
                trade = {
                    'timestamp': pd.Timestamp.now(),
                    'signal': signal,
                    'quantity': -quantity,
                    'price': current_price,
                    'proceeds': proceeds
                }
                self.trades.append(trade)
                
                return trade
        
        return None
    
    def update_portfolio_value(self, current_price):
        """Update current portfolio value"""
        self.portfolio_value = self.cash + (self.position * current_price)
        return self.portfolio_value
```

## Performance Evaluation

### Comprehensive Backtesting

```python
class MLTradingBacktester:
    """
    Comprehensive backtesting system for ML trading strategies
    """
    
    def __init__(self, initial_capital=100000):
        self.initial_capital = initial_capital
        self.results = {}
    
    def backtest_strategy(self, price_data, predictions, transaction_cost=0.001):
        """Backtest the ML trading strategy"""
        portfolio = {
            'cash': self.initial_capital,
            'position': 0,
            'portfolio_value': [],
            'trades': [],
            'signals': []
        }
        
        for i in range(len(predictions)):
            current_price = price_data['close'].iloc[i]
            signal = predictions[i]
            
            # Execute trades based on signals
            if signal == 1 and portfolio['cash'] > 0:  # Buy
                shares = int(portfolio['cash'] * 0.95 / current_price)  # 95% allocation
                cost = shares * current_price * (1 + transaction_cost)
                
                if cost <= portfolio['cash']:
                    portfolio['position'] += shares
                    portfolio['cash'] -= cost
                    portfolio['trades'].append({
                        'date': price_data.index[i],
                        'action': 'BUY',
                        'shares': shares,
                        'price': current_price
                    })
            
            elif signal == -1 and portfolio['position'] > 0:  # Sell
                proceeds = portfolio['position'] * current_price * (1 - transaction_cost)
                portfolio['cash'] += proceeds
                portfolio['trades'].append({
                    'date': price_data.index[i],
                    'action': 'SELL',
                    'shares': portfolio['position'],
                    'price': current_price
                })
                portfolio['position'] = 0
            
            # Update portfolio value
            current_value = portfolio['cash'] + (portfolio['position'] * current_price)
            portfolio['portfolio_value'].append(current_value)
            portfolio['signals'].append(signal)
        
        return self._calculate_performance_metrics(portfolio, price_data)
    
    def _calculate_performance_metrics(self, portfolio, price_data):
        """Calculate comprehensive performance metrics"""
        portfolio_values = np.array(portfolio['portfolio_value'])
        returns = np.diff(portfolio_values) / portfolio_values[:-1]
        
        # Basic metrics
        total_return = (portfolio_values[-1] / self.initial_capital) - 1
        annualized_return = (1 + total_return) ** (252 / len(returns)) - 1
        volatility = np.std(returns) * np.sqrt(252)
        sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
        
        # Drawdown analysis
        cumulative = portfolio_values / self.initial_capital
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = np.min(drawdown)
        
        # Win rate and profit factor
        winning_trades = [t for t in portfolio['trades'] if t['action'] == 'SELL']
        if len(winning_trades) > 1:
            trade_returns = []
            for i in range(1, len(winning_trades)):
                buy_price = winning_trades[i-1]['price'] if i > 0 else self.initial_capital
                sell_price = winning_trades[i]['price']
                trade_return = (sell_price - buy_price) / buy_price
                trade_returns.append(trade_return)
            
            win_rate = len([r for r in trade_returns if r > 0]) / len(trade_returns)
            avg_win = np.mean([r for r in trade_returns if r > 0]) if any(r > 0 for r in trade_returns) else 0
            avg_loss = np.mean([r for r in trade_returns if r < 0]) if any(r < 0 for r in trade_returns) else 0
            profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else np.inf
        else:
            win_rate = 0
            profit_factor = 0
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'total_trades': len(portfolio['trades']),
            'portfolio_values': portfolio_values,
            'trades': portfolio['trades']
        }
```

## Key Takeaways

1. **Feature Engineering is Critical**: The quality of features determines model performance
2. **Ensemble Methods Work Best**: Combining multiple models reduces overfitting
3. **Deep Learning for Sequences**: LSTM and CNN-LSTM excel at temporal pattern recognition
4. **Real-time Implementation**: Consider latency and computational requirements
5. **Comprehensive Backtesting**: Evaluate multiple metrics beyond just returns

## Next Steps

- [Portfolio Optimization](../optimization/portfolio/multi-objective.md): Integrate ML signals into portfolio construction
- [Risk Management](../management/risk/model-based-var.md): Dynamic risk management with ML
- [Market Regime Detection](../strategy-development/preparation/regime-detection.md): Adapt strategies to market conditions

---

**Previous**: [Reinforcement Learning ←](reinforcement-learning.md)

**Next**: [Single Objective Optimization →](../optimization/single-objective/ga-pso-es.md)

**Related Topics**:
- [Feature Selection](../strategy-development/features/feature-selection.md)
- [Multi-Objective Optimization](../optimization/multi-objective/nsga3.md)
