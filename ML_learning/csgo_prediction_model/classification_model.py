import pandas as pd
import os
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from imblearn.pipeline import Pipeline as ImbPipeline  # quan trọng
from imblearn.over_sampling import SMOTE
import pickle
from sklearn.metrics import classification_report
# Đọc dữ liệu
current_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(current_dir, "csgo.csv")
data = pd.read_csv(data_path)

# Xử lý cột date
data["date"] = pd.to_datetime(data["date"], dayfirst=True)
data["weekday"] = data["date"].dt.weekday
data = data.drop(columns=["date", "day", "month", "year"])
target = "result"

X = data.drop(target, axis=1)
y = data[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1009
)

# Các feature
num_features = ["wait_time_s", "match_time_s", "team_a_rounds", "team_b_rounds",
                "ping", "kills", "assists", "deaths", "mvps", "hs_percent", "points", "weekday"]
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
    ("smote", SMOTE(random_state=1009)),   # Thêm oversampling ở đây
    ("clf", RandomForestClassifier())    
])

# Hyperparameters
params = {
    "clf__n_estimators": [50, 100, 200, 300],
    "clf__criterion": ["gini", "entropy", "log_loss"],
    "clf__max_depth": [None, 5, 10, 20],
    "preprocessor__num__imputer__strategy": ["mean", "median"]
}

# RandomizedSearchCV
model = GridSearchCV(
    estimator=clf,
    param_grid=params,
    # n_iter=20,
    scoring="f1_macro",   # vẫn tối ưu macro F1
    verbose=1,
    cv=6,
    n_jobs=-1
)

# Train
model.fit(X_train, y_train)
print("Best params:", model.best_params_)
print("Best score:", model.best_score_)
y_predict = model.predict(X_test)
print(classification_report(y_test, y_predict))
# Save model nếu cần
# with open("model_csgo.pkl","wb") as f:
#     pickle.dump(model, f)
