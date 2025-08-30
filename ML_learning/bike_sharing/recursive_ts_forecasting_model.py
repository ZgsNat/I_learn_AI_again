import os
import pandas as pd
import numpy as np

try:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_dir, "bike-sharing-dataset.csv")
    data = pd.read_csv(data_path)
except NameError:
    data = pd.read_csv("bike-sharing-dataset.csv")

data = data.rename(columns={'users':'Demand', 'temp':'Temperature'})
data['datetime'] = pd.to_datetime(data['datetime'])
data.set_index('datetime', inplace=True)

data['hour_sin'] = np.sin(2 * np.pi * data['hour'] / 24.0)
data['hour_cos'] = np.cos(2 * np.pi * data['hour'] / 24.0)
data['weekday_sin'] = np.sin(2 * np.pi * data['weekday'] / 7.0)
data['weekday_cos'] = np.cos(2 * np.pi * data['weekday'] / 7.0)
data = data.drop(['hour', 'weekday', 'month'], axis=1)


def create_multivariate_ts_data_recursive(data, window_size=10, target_size=6, target_column='Demand'):
    """
    Tạo dữ liệu cho chiến lược đệ quy từ bộ dữ liệu đa biến.
    """
    df = data.copy()
    lagged_features = []
    


