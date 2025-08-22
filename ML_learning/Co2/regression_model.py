import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

def create_ts_data(data, window_size = 10):
    i = 1
    while i < window_size:
        data["co2_{}".format(i)] = data["co2"].shift(-i)
        i += 1
        data["target"] = data["co2"].shift(-i)
        data.dropna(axis=0)
    return data

current_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(current_dir,"co2.csv")

data = pd.read_csv(data_path)
data["time"] = pd.to_datetime(data["time"])
data["co2"] = data["co2"].interpolate()
# fig, ax = plt.subplots()
# print(data.info())
# ax.plot(data["time"],data["co2"])
# ax.set_xlabel("Time")
# ax.set_ylabel("CO2")
# plt.show()

window_size = 5
data = create_ts_data(data, window_size)
x = data.drop(["time","target"], axis=1)
y = data["target"] 

train_ratio = 0.8
num_sample = len(data)

x_train = x[:int(train_ratio*num_sample)]
y_train = y[:int(train_ratio*num_sample)]
x_test = x[int(train_ratio*num_sample):]
y_test = y[int(train_ratio*num_sample):]

# scaler = StandardScaler()
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)

model = LinearRegression()
model.fit(x_train,y_train)

y_pred = model.predict(x_test)