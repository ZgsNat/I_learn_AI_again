import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
current_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(current_dir, "web-traffic.csv")


def create_direct_ts_data(data, window_size=10, target_size=6) -> object:
    df = data.copy()
    for i in range(1, window_size + 1):
        df[f'users_{i}'] = df['users'].shift(i)
    for i in range(1, target_size+1):
        df[f'target_t+{i}'] = df['users'].shift(-i)
    df = df.dropna(axis=0)
    
    return df

data = pd.read_csv(data_path)
window_size = 25
target_size = 5
train_ratio = 0.8
data = create_direct_ts_data(data, window_size, target_size)
target = [f"target_t+{i}" for i in range(1, target_size + 1)]
x = data.drop(['date'] + target, axis = 1)
y = data[target]
num_samples = len(data)
x_train = x[:int(train_ratio*num_samples)]
y_train = y[:int(train_ratio*num_samples)]
x_test = x[int(train_ratio*num_samples):]
y_test = y[int(train_ratio*num_samples):]

models = [LinearRegression() for _ in range(target_size)]

r2 = []
mae = []
mse = []
for i, model in enumerate(models):
    model.fit(x_train, y_train[f"target_t+{i+1}"])
    y_predict = model.predict(x_test)
    r2.append(r2_score(y_test[f"target_t+{i+1}"], y_predict))
    mae.append(mean_absolute_error(y_test[f"target_t+{i+1}"], y_predict))
    mse.append(mean_squared_error(y_test[f"target_t+{i+1}"], y_predict))
print("MAE: {}".format(mae))
print("MSE: {}".format(mse))
print("R2: {}".format(r2))
