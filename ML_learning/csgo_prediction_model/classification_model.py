import pandas as pd
import os
from ydata_profiling import ProfileReport
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier


current_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(current_dir,"csgo.csv")
data = pd.read_csv(data_path)

# profile = ProfileReport(data,title="Csgo Report", explorative=True)
# profile.to_file("csgo.html")
# Xóa các cột không cần thiết: "date", "day", "month", "year".
# Vì giờ ta chỉ cần ngày trong tuần (weekday) để mô hình học.
# Cột date gốc bị bỏ vì dạng datetime khó dùng trực tiếp trong mô hình.
# data["result"] = data["result"].map({"lost": 0, "tie": 1, "win": 2})
data["date"] = pd.to_datetime(data["date"],dayfirst=True)
# Giữ lại weekday vì có thể cuối tuần chất lượng tốt hơn
data["weekday"] = data["date"].dt.weekday
data = data.drop(columns=["date","day","month","year"])
target = "result"


x = data.drop(target,axis=1)
y = data[target]

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=1009)

num_features = ["wait_time_s", "match_time_s", "team_a_rounds", "team_b_rounds",
                "ping", "kills", "assists", "deaths", "mvps", "hs_percent", "points", "weekday"]
cat_features = ["map"]

num_transformer = Pipeline(steps=[
    ("imputer",SimpleImputer()),
    ("scaler",StandardScaler())
])
cat_transformer = Pipeline(steps=[
    ("imputer",SimpleImputer(strategy="most_frequent")),
    ("onehot",OneHotEncoder(handle_unknown="ignore"))
])
preprocessor = ColumnTransformer(transformers=[
    ("num", num_transformer, num_features),
    ("cat", cat_transformer, cat_features)
])

clf = Pipeline(steps=[
    ("preprocessor",preprocessor),
    ("clf",RandomForestClassifier())
])

params = {
    "clf__n_estimators": [50, 100, 200, 300],
    "clf__criterion": ["gini", "entropy", "log_loss"],
    "clf__max_depth": [None, 5, 10, 20],
    "preprocessor__num__imputer__strategy": ["mean", "median"]
}

model = RandomizedSearchCV(
    estimator=clf,
    param_distributions=params,
    n_iter=20,
    scoring="accuracy",
    verbose=1,
    cv=6,
    n_jobs=-1
)

model.fit(x_train,y_train)
print("Best params:", model.best_params_)
print("Best score:", model.best_score_)