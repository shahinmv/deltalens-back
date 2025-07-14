import numpy as np
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score
import matplotlib.pyplot as plt

class TunedBitcoinLSTMPredictor:
    def __init__(self, df_daily):
        self.df = df_daily.copy()
        self.price_scaler = MinMaxScaler(feature_range=(0, 1))
        self.feature_scaler = RobustScaler()
        self.target_scaler = MinMaxScaler(feature_range=(0, 1))
        self.sequence_length = 30
        self.lstm_units_1 = 64
        self.lstm_units_2 = 16
        self.dense_units = 8
        self.dropout_rate = 0.1
        self.learning_rate = 0.01
        self.batch_size = 16
        self.price_model = None
        self.direction_model = None
        self.feature_names = None
    def engineer_features(self):
        raise NotImplementedError("Feature engineering should be done before initializing this class.")
    def create_sequences(self, data, target, sequence_length):
        X, y = [], []
        for i in range(sequence_length, len(data)):
            X.append(data[i-sequence_length:i])
            y.append(target[i])
        return np.array(X), np.array(y)
    def prepare_data(self, test_size=0.2, val_size=0.1):
        df = self.df
        feature_cols = [
            'high_close_ratio', 'low_close_ratio', 'open_close_ratio',
            'returns_1d', 'returns_3d', 'returns_7d',
            'price_ma_5_ratio', 'price_ma_10_ratio', 'price_ma_20_ratio',
            'macd_normalized', 'macd_signal_normalized',
            'rsi_normalized', 'bb_position', 'bb_width',
            'volatility_10', 'volatility_20',
            'volume_avg_ratio', 'volume_change',
            'vader_ma_3', 'vader_ma_7', 'article_count_norm',
            'funding_rate', 'funding_rate_ma',
            'momentum_5', 'momentum_10',
            'day_sin', 'day_cos'
        ]
        available_cols = [col for col in feature_cols if col in df.columns]
        feature_data = df[available_cols].values
        feature_data = np.nan_to_num(feature_data, nan=0.0, posinf=1.0, neginf=-1.0)
        feature_data_scaled = self.feature_scaler.fit_transform(feature_data)
        target_returns = df['target_return'].values
        target_directions = df['target_direction'].values
        target_returns_scaled = self.target_scaler.fit_transform(target_returns.reshape(-1, 1)).flatten()
        X, y_returns = self.create_sequences(feature_data_scaled, target_returns_scaled, self.sequence_length)
        _, y_directions = self.create_sequences(feature_data_scaled, target_directions, self.sequence_length)
        total_samples = len(X)
        train_size = int(total_samples * (1 - test_size - val_size))
        val_size_samples = int(total_samples * val_size)
        self.X_train = X[:train_size]
        self.y_return_train = y_returns[:train_size]
        self.y_dir_train = y_directions[:train_size]
        self.X_val = X[train_size:train_size + val_size_samples]
        self.y_return_val = y_returns[train_size:train_size + val_size_samples]
        self.y_dir_val = y_directions[train_size:train_size + val_size_samples]
        self.X_test = X[train_size + val_size_samples:]
        self.y_return_test = y_returns[train_size + val_size_samples:]
        self.y_dir_test = y_directions[train_size + val_size_samples:]
        self.test_current_prices = df['close'].iloc[train_size + val_size_samples + self.sequence_length:].values
        self.test_dates = df.index[train_size + val_size_samples + self.sequence_length:]
        self.last_date = df.index[-1]
        self.feature_names = available_cols
    def build_return_model(self):
        model = Sequential([
            LSTM(self.lstm_units_1, return_sequences=True, input_shape=(self.sequence_length, self.X_train.shape[2])),
            Dropout(self.dropout_rate),
            BatchNormalization(),
            LSTM(self.lstm_units_2, return_sequences=False),
            Dropout(self.dropout_rate),
            Dense(self.dense_units, activation='relu'),
            Dropout(self.dropout_rate),
            Dense(1, activation='tanh')
        ])
        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss='mse',
            metrics=['mae']
        )
        return model
    def build_direction_model(self):
        model = Sequential([
            LSTM(self.lstm_units_1, return_sequences=True, input_shape=(self.sequence_length, self.X_train.shape[2])),
            Dropout(self.dropout_rate + 0.2),
            BatchNormalization(),
            LSTM(self.lstm_units_2, return_sequences=False),
            Dropout(self.dropout_rate + 0.2),
            Dense(self.dense_units, activation='relu'),
            Dropout(self.dropout_rate + 0.1),
            Dense(1, activation='sigmoid')
        ])
        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        return model
    def train_models(self, epochs=50, batch_size=None):
        if batch_size is not None:
            training_batch_size = batch_size
        else:
            training_batch_size = self.batch_size
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6
        )
        self.return_model = self.build_return_model()
        return_history = self.return_model.fit(
            self.X_train, self.y_return_train,
            validation_data=(self.X_val, self.y_return_val),
            epochs=epochs,
            batch_size=training_batch_size,
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )
        self.direction_model = self.build_direction_model()
        direction_history = self.direction_model.fit(
            self.X_train, self.y_dir_train,
            validation_data=(self.X_val, self.y_dir_val),
            epochs=epochs,
            batch_size=training_batch_size,
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )
        return return_history, direction_history
    def evaluate_models(self):
        y_return_pred_scaled = self.return_model.predict(self.X_test)
        y_return_pred = self.target_scaler.inverse_transform(y_return_pred_scaled).flatten()
        y_return_actual = self.target_scaler.inverse_transform(self.y_return_test.reshape(-1, 1)).flatten()
        y_price_pred = self.test_current_prices * (1 + y_return_pred)
        y_price_actual = self.test_current_prices * (1 + y_return_actual)
        mse_return = mean_squared_error(y_return_actual, y_return_pred)
        mae_return = mean_absolute_error(y_return_actual, y_return_pred)
        mse_price = mean_squared_error(y_price_actual, y_price_pred)
        mae_price = mean_absolute_error(y_price_actual, y_price_pred)
        mape_price = np.mean(np.abs((y_price_actual - y_price_pred) / y_price_actual)) * 100
        y_dir_pred_proba = self.direction_model.predict(self.X_test)
        y_dir_pred = (y_dir_pred_proba > 0.5).astype(int).flatten()
        accuracy = accuracy_score(self.y_dir_test, y_dir_pred)
        tp = np.sum((self.y_dir_test == 1) & (y_dir_pred == 1))
        fp = np.sum((self.y_dir_test == 0) & (y_dir_pred == 1))
        fn = np.sum((self.y_dir_test == 1) & (y_dir_pred == 0))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        return {
            'return_predictions': y_return_pred,
            'return_actual': y_return_actual,
            'price_predictions': y_price_pred,
            'price_actual': y_price_actual,
            'direction_predictions': y_dir_pred,
            'direction_probabilities': y_dir_pred_proba.flatten(),
            'return_metrics': {'mse': mse_return, 'mae': mae_return},
            'price_metrics': {'mse': mse_price, 'mae': mae_price, 'mape': mape_price},
            'direction_accuracy': accuracy,
            'direction_metrics': {'precision': precision, 'recall': recall, 'f1': f1}
        }
    def predict_next_day(self):
        last_sequence = self.X_test[-1:]
        predicted_return_scaled = self.return_model.predict(last_sequence)[0][0]
        predicted_return = self.target_scaler.inverse_transform([[predicted_return_scaled]])[0][0]
        current_price = self.test_current_prices[-1]
        import pandas as pd
        prediction_date = self.last_date + pd.Timedelta(days=1)
        if prediction_date.weekday() >= 5:
            days_to_add = 7 - prediction_date.weekday()
            prediction_date = prediction_date + pd.Timedelta(days=days_to_add)
        predicted_price = current_price * (1 + predicted_return)
        if predicted_price <= 0:
            predicted_price = current_price * 1.001
            predicted_return = 0.001
        direction_prob = self.direction_model.predict(last_sequence)[0][0]
        predicted_direction = "UP" if direction_prob > 0.5 else "DOWN"
        confidence = max(direction_prob, 1 - direction_prob) * 100
        expected_return_pct = predicted_return * 100
        return {
            'last_known_date': self.last_date,
            'prediction_date': prediction_date,
            'current_price': current_price,
            'predicted_price': predicted_price,
            'expected_return': predicted_return,
            'predicted_direction': predicted_direction,
            'confidence': confidence,
            'prob_up': direction_prob,
            'prob_down': 1 - direction_prob,
            'model_config': {
                'lstm_units_1': self.lstm_units_1,
                'lstm_units_2': self.lstm_units_2,
                'dense_units': self.dense_units,
                'dropout_rate': self.dropout_rate,
                'learning_rate': self.learning_rate,
                'batch_size': self.batch_size,
                'sequence_length': self.sequence_length
            }
        }
    def plot_predictions(self, results, n_days=30):
        import matplotlib.pyplot as plt
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        recent_actual = results['price_actual'][-n_days:]
        recent_pred = results['price_predictions'][-n_days:]
        recent_dates = self.test_dates[-n_days:]
        ax1.plot(recent_dates, recent_actual, label='Actual Price', color='blue', linewidth=2)
        ax1.plot(recent_dates, recent_pred, label='Predicted Price', color='red', linewidth=2, alpha=0.7)
        ax1.set_title('Recent Price Predictions vs Actual (Optimized Model)')
        ax1.set_ylabel('Price ($)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis='x', rotation=45)
        recent_return_actual = results['return_actual'][-n_days:]
        recent_return_pred = results['return_predictions'][-n_days:]
        ax2.plot(recent_dates, recent_return_actual, label='Actual Return', color='blue', linewidth=2)
        ax2.plot(recent_dates, recent_return_pred, label='Predicted Return', color='red', linewidth=2, alpha=0.7)
        ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax2.set_title('Recent Return Predictions vs Actual (Optimized Model)')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Return')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.tick_params(axis='x', rotation=45)
        plt.tight_layout()
        plt.show()
    def get_model_summary(self):
        summary = {
            'optimized_hyperparameters': {
                'batch_size': self.batch_size,
                'dense_units': self.dense_units,
                'dropout_rate': self.dropout_rate,
                'learning_rate': self.learning_rate,
                'lstm_units_1': self.lstm_units_1,
                'lstm_units_2': self.lstm_units_2,
                'sequence_length': self.sequence_length
            },
            'expected_performance': {
                'mse': 0.007947,
                'mae': 0.06260,
                'optimization_method': 'grid_search',
                'total_evaluations': 300
            }
        }
        return summary 