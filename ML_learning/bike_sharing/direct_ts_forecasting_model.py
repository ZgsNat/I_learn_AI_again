import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

try:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_dir, "bike-sharing-dataset.csv")
    data = pd.read_csv(data_path)
except NameError:
    data = pd.read_csv("bike-sharing-dataset.csv")

# profile = ProfileReport(data, title="bike-sharing-dataset Data Profiling Report",explorative=True)
# profile.to_file("bike-sharing-dataset.html")

data = data.rename(columns={'users': 'Demand', 'temp': 'Temperature'})
data['date_time'] = pd.to_datetime(data['date_time'])
data = data.set_index('date_time')

data['hour_sin'] = np.sin(2 * np.pi * data['hour'] / 24.0)
data['hour_cos'] = np.cos(2 * np.pi * data['hour'] / 24.0)
data['weekday_sin'] = np.sin(2 * np.pi * data['weekday'] / 7.0)
data['weekday_cos'] = np.cos(2 * np.pi * data['weekday'] / 7.0)
data = data.drop(['hour', 'weekday', 'month'], axis=1)


def create_multivariate_ts_data_direct(data, column_to_lag, target_column, window_size=10, target_size=6):
    df = data.copy()
    lagged_features = []
    for col in column_to_lag:
        if col not in df.columns:
            print(f"Cảnh báo: Cột '{col}' không tồn tại. Bỏ qua.")
            continue
        lagged = pd.concat(
            [df[col].shift(i).rename(f'{col}_lag_{i}') for i in range(1, window_size + 1)],
            axis=1
        )
        lagged_features.append(lagged)
    if not lagged_features:
        raise ValueError("Không có cột hợp lệ để tạo đặc trưng trễ.")
    lagged_df = pd.concat(lagged_features, axis=1)
    df = pd.concat([df, lagged_df], axis=1)
    for i in range(1, target_size + 1):
        df[f'target_t+{i}'] = df[target_column].shift(-i)
    df = df.dropna(axis=0)
    target_cols = [f'target_t+{i}' for i in range(1, target_size + 1)]
    y = df[target_cols]
    # Bỏ các cột target ban đầu khỏi X. Cột target gốc vẫn giữ lại để tạo lag features.
    X = df.drop(target_cols, axis=1)
    return X, y

column_to_lag = ['Demand', 'Temperature']
window_size = 48 * 2
target_size = 5
target_column = 'Demand'

X, y = create_multivariate_ts_data_direct(data, column_to_lag, target_column, window_size, target_size)

split_point = int(0.8 * len(X))
X_train, X_test = X.iloc[:split_point], X.iloc[split_point:]
y_train, y_test = y.iloc[:split_point], y.iloc[split_point:]

categorical_features = ['weather']

numerical_features = [col for col in X_train.columns if col not in categorical_features]

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ],
    remainder='passthrough' # Giữ lại các cột không được chỉ định (nếu có)
)

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1, min_samples_leaf=5))
])

pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)
y_pred = pd.DataFrame(y_pred, index=y_test.index, columns=y_test.columns)

print("\n--- Kết quả đánh giá trên tập Test ---")
for i in range(target_size):
    target_col_name = f'target_t+{i+1}'
    mae = mean_absolute_error(y_test[target_col_name], y_pred[target_col_name])
    rmse = np.sqrt(mean_squared_error(y_test[target_col_name], y_pred[target_col_name]))
    print(f"Dự đoán cho giờ thứ {i+1} (t+{i+1}): MAE: {mae:.2f}, RMSE: {rmse:.2f}")

plot_range = 168
plt.figure(figsize=(20, 8))
plt.plot(y_test.index[:plot_range], y_test['target_t+1'].iloc[:plot_range], label='Giá trị thực tế (t+1)', color='blue', marker='.')
plt.plot(y_pred.index[:plot_range], y_pred['target_t+1'].iloc[:plot_range], label='Giá trị dự đoán (t+1)', color='red', linestyle='--')
plt.title('So sánh giá trị Thực tế và Dự đoán (sử dụng OneHotEncoder)')
plt.xlabel('Thời gian')
plt.ylabel('Số lượng người dùng (Demand)')
plt.legend()
plt.grid(True)
plt.show()