import os
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(current_dir, "web-traffic.csv")

def create_ts_data_recursion(data, window_size = 10) -> tuple[pd.DataFrame, pd.Series, object]:
    df = data.copy()

    for i in range(1, window_size + 1):
        df[f'users_{i}'] = df['users'].shift(i)
        
    df = df.rename(columns={'users' : 'target'})
    df = df.dropna(axis=0)

    y = df['target']
    X = df.drop(columns={'date','target'})
    return X, y, df

data = pd.read_csv(data_path)
data["date"] = pd.to_datetime(data["date"], format="%d/%m/%y")

window_size = 25
train_ratio = 0.8
X, y, data = create_ts_data_recursion(data, window_size)

num_samples = len(data)
train_size = int(train_ratio * num_samples)

x_train = X[:train_size]
x_test = X[train_size:]
y_train = y[:train_size]
y_test = y[train_size:]

model = RandomForestRegressor()
model.fit(x_train, y_train)

y_predict = model.predict(x_test)

print("MAE: {}".format(mean_absolute_error(y_test, y_predict)))
print("MSE: {}".format(mean_squared_error(y_test, y_predict)))
print("RMSE: {}".format(np.sqrt(mean_squared_error(y_test, y_predict))))
print("R2: {}".format(r2_score(y_test, y_predict)))


