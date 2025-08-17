import pandas as pd
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier  # Thử nghiệm mô hình mới
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
import pickle
import numpy as np # Thêm thư viện numpy
from sklearn.metrics import classification_report,f1_score, accuracy_score, precision_score, recall_score

# Đọc dữ liệu
current_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(current_dir, "csgo.csv")
data = pd.read_csv(data_path)

# --- BƯỚC 1: FEATURE ENGINEERING ---
# Xử lý trường hợp chia cho 0
data['deaths_safe'] = data['deaths'].replace(0, 1)

# Tạo các feature mới
data['kd_ratio'] = data['kills'] / data['deaths_safe']
data['kda_ratio'] = (data['kills'] + data['assists']) / data['deaths_safe']
data['points_per_minute'] = (data['points'] / (data['match_time_s'] / 60)).replace([np.inf, -np.inf], 0).fillna(0)

# Xử lý cột date
data["date"] = pd.to_datetime(data["date"], dayfirst=True)
data["weekday"] = data["date"].dt.weekday

# --- BƯỚC 2: LOẠI BỎ CỘT GÂY RÒ RỈ VÀ CỘT KHÔNG CẦN THIẾT ---
# Bỏ 'team_a_rounds' và 'team_b_rounds'
data = data.drop(columns=["date", "day", "month", "year", "team_a_rounds", "team_b_rounds", "deaths_safe"])
target = "result"

X = data.drop(target, axis=1)
y = data[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1009, stratify=y # Thêm stratify để giữ tỉ lệ các lớp
)

# Cập nhật lại danh sách các feature
num_features = ["wait_time_s", "match_time_s", "ping", "kills", "assists", "deaths", 
                "mvps", "hs_percent", "points", "weekday", 
                "kd_ratio", "kda_ratio", "points_per_minute"] # Thêm feature mới
cat_features = ["map"]

# Pipeline cho numeric
num_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer()),
    ("scaler", StandardScaler())
])

# Pipeline cho categorical
cat_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

# Column transformer
preprocessor = ColumnTransformer(transformers=[
    ("num", num_transformer, num_features),
    ("cat", cat_transformer, cat_features)
])

# Pipeline với SMOTE + RandomForest
clf = ImbPipeline(steps=[
    ("preprocessor", preprocessor),
    ("smote", SMOTE(random_state=1009)),
    ("clf", RandomForestClassifier(random_state=1009, class_weight='balanced')) # Thêm class_weight
])

# Hyperparameters
params = {
    "clf__n_estimators": [100, 200, 300],
    "clf__criterion": ["gini", "entropy"],
    "clf__max_depth": [10, 20, None],
    "preprocessor__num__imputer__strategy": ["mean", "median"]
}

model = GridSearchCV(
    estimator=clf,
    param_grid=params,
    scoring="f1_macro",
    verbose=1,
    cv=6,
    n_jobs=-1
)

# Train
model.fit(X_train, y_train)
print("Best params:", model.best_params_)
print("Best score:", model.best_score_)

# Lưu ý: Sau khi bỏ các cột gây rò rỉ, điểm F1-score có thể sẽ *giảm* ban đầu. 
# Tuy nhiên, đây mới là điểm số trung thực và mô hình của bạn giờ đây mới thực sự "học" 
# để dự đoán dựa trên hiệu suất, chứ không phải dựa vào kết quả có sẵn.

# y_predict = model.predict(X_test)
# print(classification_report(y_test, y_predict))

# with open("high_model_csgo.pkl","wb") as f:
#     pickle.dump(model, f)