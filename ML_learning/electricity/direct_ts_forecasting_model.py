import os
import pandas as pd
import lightgbm as lbgb
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import numpy as np

try:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_dir, "electricity.csv")
    data = pd.read_csv(data_path)
except NameError:
    data = pd.read_csv("electricity.csv")

data['Time'] = pd.to_datetime(data['Time'])
data = data.set_index('Time')
data['Holiday'] = data['Holiday'].astype(int)
data = data.drop('Date', axis=1)
data = data.dropna(axis=0)

def create_time_features(df):
    """Tạo các feature liên quan đến thời gian từ index của DataFrame."""
    df['hour'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek # Monday=0, Sunday=6
    df['month'] = df.index.month
    df['quarter'] = df.index.quarter
    df['day_of_year'] = df.index.dayofyear
    df['is_weekend'] = (df.index.dayofweek >= 5).astype(int)
    return df

data = create_time_features(data)

def create_multivariate_ts_data_direct(data, columns_to_lag, target_column, window_size=10, target_size=6):
    """
    Tạo dữ liệu cho chiến lược trực tiếp từ bộ dữ liệu đa biến.
    """
    df = data.copy()
    lagged_features = []
    for col in columns_to_lag:
        lagged = pd.concat(
            [df[col].shift(i).rename(f'{col}_lag_{i}') for i in range(1, window_size + 1)],
            axis=1
        )
        lagged_features.append(lagged)
    lagged_df = pd.concat(lagged_features, axis=1)
    df = pd.concat([df, lagged_df], axis=1)
    for i in range(1, target_size + 1):
        df[f'target_t+{i}'] = df[target_column].shift(-i)
    df = df.dropna(axis=0)
    
    target = [f'target_t+{i}' for i in range(1, target_size + 1)]
    y = df[target]
    X = df.drop(target, axis=1)
    
    return X, y

print("Chuẩn bị dữ liệu xong.")
# --- Phần 2: Tạo Feature dạng Lag (Supervised Learning Format) ---
print("Bắt đầu tạo feature dạng lag...")
column_to_lag = ['Demand','Temperature']
window_size = 48*2
target_size = 5
target_column = 'Demand'
X, y = create_multivariate_ts_data_direct(data, column_to_lag, target_column, window_size, target_size)

print(f"X shape: {X.shape}, y shape: {y.shape}")
split_point = int(0.8 * len(X))
X_train, X_test = X[:split_point], X[split_point:]
y_train, y_test = y[:split_point], y[split_point:]
print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")

models = {}

y_pred_all = pd.DataFrame(index=y_test.index)

for i in range(1, target_size + 1):
    target_column = f'target_t+{i}'

    model = lbgb.LGBMRegressor(
        random_state=42,
        n_estimators=5000,
        learning_rate=0.05,
        num_leaves=31,
    )

    model.fit(
        X_train, y_train[target_column],
        eval_set=[(X_test, y_test[target_column])],
        eval_metric='l1',
        callbacks=[lbgb.early_stopping(stopping_rounds=100, verbose=False)]
    )

    models[target_column] = model
    y_pred_all[target_column] = model.predict(X_test)

# --- Đánh giá ---
for i in range(1, target_size+1):
    col = f"target_t+{i}"
    mae = mean_absolute_error(y_test[col], y_pred_all[col])
    rmse = np.sqrt(mean_squared_error(y_test[col], y_pred_all[col]))
    print(f"{col}: MAE={mae:.2f}, RMSE={rmse:.2f}")

# --- Trực quan hóa ---
plt.figure(figsize=(15,7))
plt.plot(y_test.index[:300], y_test['target_t+1'][:300], label="Actual t+1", color="blue")
plt.plot(y_pred_all.index[:300], y_pred_all['target_t+1'][:300], label="Predicted t+1", color="red", linestyle="--")
plt.title("Direct Forecasting - Horizon t+1 (300 điểm test đầu tiên)")
plt.xlabel("Time")
plt.ylabel("Demand")
plt.legend()
plt.show()