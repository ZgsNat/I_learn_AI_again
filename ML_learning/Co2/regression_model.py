import os
import pandas as pd
import matplotlib.pyplot as plt

def create_ts_data(data, window_size = 10):
    i = 1
    while i < window_size:
        data["co2_{}".format(i)] = data["co2"].shift(-i)
        i += 1
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