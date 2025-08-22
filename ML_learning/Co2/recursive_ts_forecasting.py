import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error, r2_score
current_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(current_dir,"co2.csv")
def create_ts_data(data, window_size=10):
    i = 1
    while i < window_size:
        data["co2_{}".format(i)] = data["co2"].shift(-i)
        i += 1
    data["target"] = data["co2"].shift(-i)
    data = data.dropna(axis=0)
    return data

data = pd.read_csv(data_path)
data["time"] = pd.to_datetime(data["time"])
data["co2"] = data["co2"].interpolate()
# fig, ax = plt.subplots()
# ax.plot(data["time"], data["co2"])
# ax.set_xlabel("Time")
# ax.set_ylabel("CO2")
# plt.show()

window_size = 5
train_ratio = 0.8
data = create_ts_data(data, window_size)
num_samples = len(data)
x = data.drop(["time", "target"], axis=1)
y = data["target"]
x_train = x[:int(train_ratio*num_samples)]
y_train = y[:int(train_ratio*num_samples)]
x_test = x[int(train_ratio*num_samples):]
y_test = y[int(train_ratio*num_samples):]

model = LinearRegression()
model.fit(x_train, y_train)

y_predict = model.predict(x_test)

print("MAE: {}".format(mean_absolute_error(y_test, y_predict)))
print("MSE: {}".format(mean_squared_error(y_test, y_predict)))
print("RMSE: {}".format(root_mean_squared_error(y_test, y_predict)))
print("R2: {}".format(r2_score(y_test, y_predict)))
sample_input = [340, 340.2, 340.5, 350, 350.2]
for i in range(10):
    print("Input: {}".format(sample_input))
    prediction = model.predict([sample_input])
    print(prediction)
    sample_input = sample_input[1:] + prediction.tolist()
    print("----------")

# fig, ax = plt.subplots()
# # ax.plot(data["time"], data["co2"])
# ax.plot(data["time"][:int(train_ratio*num_samples)], y_train, label="train")
# ax.plot(data["time"][int(train_ratio*num_samples):], y_test, label="test")
# ax.plot(data["time"][int(train_ratio*num_samples):], y_predict, label="prediction")
# ax.set_xlabel("Time")
# ax.set_ylabel("CO2")
# ax.legend()
# ax.grid()
# plt.show()
# Linear Regression
# MAE: 0.3605603788359208
# MSE: 0.22044947360346367
# RMSE: 0.4695204719748263
# R2: 0.9907505918201437