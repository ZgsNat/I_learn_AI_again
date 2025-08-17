import pickle
import os
import pandas as pd
import numpy as np

# --- Tải mô hình đã được huấn luyện ---
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, "high_model_csgo.pkl")
with open(model_path, "rb") as f:
    model = pickle.load(f)

# --- Dữ liệu thô để dự đoán ---
# Dữ liệu này phải tương ứng với các cột ban đầu TRƯỚC khi bạn bỏ cột trong file training
# Cụ thể ở đây thiếu 2 cột "team_a_rounds" và "team_b_rounds"
# Kết quả thật: Lost
test_data_raw = [
    [
        "Dust II",      # map
        28,             # day
        7,              # month
        2018,           # year
        "28/7/2018",    # date
        274,            # wait_time_s
        3194,           # match_time_s
        16,             # team_a_rounds
        14,             # team_b_rounds
        89,             # ping
        19,             # kills
        5,              # assists
        24,             # deaths
        2,              # mvps
        15,             # hs_percent
        52              # points
    ]
]
# --- Biến đổi dữ liệu đầu vào GIỐNG HỆT file training ---

# 1. Tạo DataFrame với đúng tên cột ban đầu (trừ cột target 'result')
original_columns = [
    "map", "day", "month", "year", "date", "wait_time_s", "match_time_s",
    "team_a_rounds", "team_b_rounds", "ping", "kills", "assists", "deaths",
    "mvps", "hs_percent", "points"
]
test_df = pd.DataFrame(test_data_raw, columns=original_columns)


# 2. Thực hiện Feature Engineering y hệt file training
test_df['deaths_safe'] = test_df['deaths'].replace(0, 1)
test_df['kd_ratio'] = test_df['kills'] / test_df['deaths_safe']
test_df['kda_ratio'] = (test_df['kills'] + test_df['assists']) / test_df['deaths_safe']
test_df['points_per_minute'] = (test_df['points'] / (test_df['match_time_s'] / 60)).replace([np.inf, -np.inf], 0).fillna(0)
test_df["date"] = pd.to_datetime(test_df["date"], dayfirst=True)
test_df["weekday"] = test_df["date"].dt.weekday

# 3. Chọn ra đúng 14 cột mà mô hình mong đợi (giữ đúng thứ tự)
# Đây là các cột của X_train trong file training
final_features = [
    "wait_time_s", "match_time_s", "ping", "kills", "assists", "deaths",
    "mvps", "hs_percent", "points", "weekday", "kd_ratio", "kda_ratio",
    "points_per_minute", "map"
]
# Sắp xếp lại các cột của test_df để khớp với thứ tự này
test_df_final = test_df[final_features]


# 4. Thực hiện dự đoán trên dữ liệu đã được xử lý
prediction = model.predict(test_df_final)

print("Kết quả dự đoán:", prediction)

# Dự đoán xác suất (nếu bạn muốn)
# prediction_proba = model.predict_proba(test_df_final)
# print("Xác suất dự đoán:", prediction_proba)