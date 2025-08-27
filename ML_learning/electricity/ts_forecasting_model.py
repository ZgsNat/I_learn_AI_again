import pandas as pd
import os
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns


try:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_dir, "electricity.csv")
    data = pd.read_csv(data_path)
except NameError:
    # Chạy trong môi trường không có __file__ (ví dụ: Jupyter Notebook)
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
print("Chuẩn bị dữ liệu xong.")

# --- Phần 2: Tạo Feature dạng Lag (Supervised Learning Format) ---
print("Bắt đầu tạo feature dạng lag...")
def create_multivariate_ts_data_recursion(data, columns_to_lag, target_column, window_size=10):
    """Tạo dữ liệu cho chiến lược đệ quy từ bộ dữ liệu đa biến."""
    df = data.copy()
    for col in columns_to_lag:
        for i in range(1, window_size + 1):
            df[f'{col}_lag_{i}'] = df[col].shift(i)
    
    df = df.rename(columns={target_column: 'target'})
    df = df.dropna(axis=0)
    
    y = df['target']
    # Loại bỏ các cột gốc đã được tạo lag
    cols_to_drop = ['target'] + columns_to_lag
    X = df.drop(columns=cols_to_drop)
    return X, y

# Định nghĩa các tham số
columns_to_lag = ['Demand', 'Temperature']
target_column = 'Demand'
# Dữ liệu 30 phút -> 48 điểm/ngày. Lấy lag của 2 ngày trước.
WINDOW_SIZE = 48 * 2 

X, y = create_multivariate_ts_data_recursion(
    data=data,
    columns_to_lag=columns_to_lag,
    target_column=target_column,
    window_size=WINDOW_SIZE
)
print(f"Tạo xong feature. Kích thước X: {X.shape}")

# --- Phần 3: Phân chia Dữ liệu Train/Test theo thời gian ---
print("Bắt đầu phân chia dữ liệu...")
# 80% cho huấn luyện, 20% cho kiểm tra
split_point = int(len(X) * 0.8)
X_train, X_test = X.iloc[:split_point], X.iloc[split_point:]
y_train, y_test = y.iloc[:split_point], y.iloc[split_point:]
print(f"Kích thước tập train: {X_train.shape}, Kích thước tập test: {X_test.shape}")

# --- Phần 4: Huấn luyện Mô hình LightGBM ---
print("Bắt đầu huấn luyện mô hình...")
model = lgb.LGBMRegressor(
    random_state=42,
    n_estimators=500,  # Số lượng cây
    learning_rate=0.05,
    num_leaves=31
)

model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    eval_metric='l1', # l1 là MAE
    callbacks=[lgb.early_stopping(10, verbose=True)] # Dừng sớm nếu không cải thiện
)
print("Huấn luyện xong.")

# --- Phần 5: Dự báo và Đánh giá ---
print("Bắt đầu dự báo và đánh giá...")
y_pred = model.predict(X_test)

# Tính toán các chỉ số lỗi
mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)

print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")

# --- Phần 6: Trực quan hóa Kết quả ---
print("Bắt đầu trực quan hóa...")
# Tạo một DataFrame để dễ dàng vẽ biểu đồ
results_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
results_df.index = y_test.index

# Vẽ một phần của tập test để dễ quan sát hơn (ví dụ: 500 điểm đầu tiên)
plt.style.use('seaborn-v0_8-whitegrid')
fig, ax = plt.subplots(figsize=(15, 7))
results_df.head(500).plot(ax=ax, style=['-', '--'])
ax.set_title('So sánh Giá trị Thực tế và Dự báo (500 điểm đầu tiên của tập Test)', fontsize=16)
ax.set_ylabel('Demand', fontsize=12)
ax.set_xlabel('Time', fontsize=12)
ax.legend()
plt.show()