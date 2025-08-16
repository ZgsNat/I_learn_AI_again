import pandas as pd
import os
from ydata_profiling import ProfileReport
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler,OrdinalEncoder, OneHotEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, root_mean_squared_error, r2_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

current_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(current_dir, "StudentScore.xls")
data = pd.read_csv(data_path)

target = "writing score"

# profile = ProfileReport(data, title = "Student Score", explorative = True)
# profile.to_file("StudentScore.html")

# print(data[["math score", "reading score", "writing score"]].corr())

x = data.drop(target, axis=1)
y = data[target]

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=1009)
num_transformer = Pipeline(steps=[
    ("imputer",SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
]) 
education_levels = ["some high school","high school","some college","associate's degree","bachelor's degree","master's degree"]
gender = data["gender"].unique()
lunch_values = data["lunch"].unique()
test_values = data["test preparation course"].unique()
ordinal_transformer = Pipeline(steps=[
    ("imputer",SimpleImputer(strategy="most_frequent")),
    ("encoder", OrdinalEncoder(categories=[education_levels, gender, lunch_values, test_values]))
    
])
nominal_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(sparse_output=False))
])
transformer = ColumnTransformer(transformers=[
    ("num_feature", num_transformer, ["reading score","math score"]),
    ("ordinal_feature", ordinal_transformer, ["parental level of education", "gender", "lunch", "test preparation course"]),
    ("nominal_features", nominal_transformer, ["race/ethnicity"])
])
# output = transformer.fit_transform(x_train)
reg = Pipeline(steps=[
    ("preprocessor", transformer),
    ("regressor", RandomForestRegressor())
])

# model.fit(x_train,y_train)

# y_predict = model.predict(x_test)

# # for i, j in zip(y_predict, y_test):
# #     print("test:{} and predict:{}".format(j, i))

# print("MAE: {}".format(mean_absolute_error(y_test, y_predict)))
# print("MSE: {}".format(mean_squared_error(y_test, y_predict)))
# print("RMSE: {}".format(root_mean_squared_error(y_test, y_predict)))
# print("R2: {}".format(r2_score(y_test, y_predict)))
# # 
# MAE: 2.7351944088586113
# MSE: 11.57965112322608
# RMSE: 3.402888643965018
# R2: 0.9499669196053658
 
params = {
    "regressor__n_estimators": [50, 100, 200],
    "regressor__criterion": ["squared_error", "absolute_error", "friedman_mse","poisson"],
    "preprocessor__num_feature__imputer__strategy":["median", "mean"]
}

# model = GridSearchCV(
#     estimator=reg,
#     param_grid= params,
#     scoring="r2",
#     cv=6,
#     verbose=1,
#     n_jobs=1 
# )
model = RandomizedSearchCV(
    estimator=reg,
    param_distributions=params,
    n_iter=30,
    scoring="r2",
    cv=6,
    verbose=1,
    n_jobs=1 
)
model.fit(x_train, y_train)
print(model.best_score_)
print(model.best_params_)
y_predict = model.predict(x_test)
print("MAE: {}".format(mean_absolute_error(y_test, y_predict)))
print("MSE: {}".format(mean_squared_error(y_test, y_predict)))
print("RMSE: {}".format(root_mean_squared_error(y_test, y_predict)))
print("R2: {}".format(r2_score(y_test, y_predict)))