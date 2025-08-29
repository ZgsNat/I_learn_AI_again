import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from ydata_profiling import ProfileReport
from sklearn.preprocessing import OneHotEncoder
try:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_dir, "bike-sharing-dataset.csv")
    data = pd.read_csv(data_path)
except NameError:
    data = pd.read_csv("bike-sharing-dataset.csv")

# profile = ProfileReport(data, title="bike-sharing-dataset Data Profiling Report",explorative=True)
# profile.to_file("bike-sharing-dataset.html")

data['date_time'] = pd.to_datetime(data['date_time'])
data = data.set_index('date_time')
data['weather'] = OneHotEncoder(sparse_output=False).fit_transform(data[['weather']])
data = data.drop('date_time', axis=1)
data = data.dropna(axis=0)

def create_ts_data_direct(data,column_to_lag, target_column, window_size=10, target_size=6):
    df = data.copy()
    lagged_features = []
    for col in column_to_lag:
        lagged = pd.concat(
            [df[col].shift(i).rename(f'{col}_lag_{i}') for i in range(1, window_size + 1)],
            axis=1
        )
        lagged_features.append(lagged)
    lagged_df = pd.concat(lagged_features, axis=1)
    df = pd.concat([df, lagged_df], axis=1)

    for i in range(1, target_size + 1):
        df[f'target_t+{i}'] = df[target_column].shift(-i)
    df = df.dropna(axis=0)
    target = [f'target_t+{i}' for i in range(1, target_size + 1)]
    y = df[target]
    X = df.drop(target, axis=1)
    return X, y
