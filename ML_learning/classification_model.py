import pandas as pd
import os
from ydata_profiling import ProfileReport
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC 
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from lazypredict.Supervised import LazyClassifier
import pickle

current_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(current_dir, "diabetes.csv")
data = pd.read_csv(data_path)

# print(data.describe())
# print(data.info())
# print(data.corr())

# create report

# profile = ProfileReport(data, title="Diabetes Report", explorative=True)
# profile.to_file("diabetes.html")

target = "Outcome"
x = data.drop(target, axis=1)
y = data[target] 


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1009) 
print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.25, random_state=1009)
scaler = StandardScaler()
# Giống như ông chủ tiệm may, chúng ta đến, ông đo đạc rồi bắt đầu may
# scaler.fit(x_train)
# scaler.transform(x_train)
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
# scaler.fit_transform() cái này chỉ dùng cho bộ train, test thì có số liệu rồi, không cần đo lại nữa
# chỉ được cân bằng sau khi phân chia dữ liệu rồi!!!  

# model = SVC(probability=True)
# model = RandomForestClassifier(n_estimators=200, criterion="gini",random_state = 1009)
params = {
    "n_estimators": [50, 100, 200],
    "criterion": ["gini", "entropy", "log_loss"],
    # "max_depth": [None, 2, 5]
}
model = GridSearchCV(
    estimator=RandomForestClassifier(random_state=1009),
    param_grid= params,
    scoring="recall",
    cv=6,
    verbose=1,
    n_jobs=1 
)
model.fit(x_train, y_train)
# # y_predict = model.predict(x_test)
# y_predict = model.predict_proba(x_test)
y_predict = model.predict(x_test)
# # threshold = 0.3
# # for score in y_predict:
# #     if score[1] > threshold:
# #         print("class 1")
# #     else:
# #         print("class 2")
# # print("acc: {}".format(accuracy_score(y_test, y_predict)))
# # print("recalls: {}".format(recall_score(y_test, y_predict)))
# # print("pre: {}".format(precision_score(y_test, y_predict)))
# # print("f1: {}".format(f1_score(y_test, y_predict)))
print(model.best_score_)
print(model.best_params_)
print(classification_report(y_test, y_predict))

# Có những thuật toán không cần phải tiền xử lý dữ liệu, như decision Tree, liên quan đến cây quyết định
# clf = LazyClassifier(verbose=0, ignore_warnings=True, custom_metric=None)
# models, predictions = clf.fit(x_train, x_test, y_train, y_test)

with open("model.pkl","wb") as f:
    pickle.dump(model, f)