import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from ydata_profiling import ProfileReport

try:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_dir, "bike-sharing-dataset.csv")
    data = pd.read_csv(data_path)
except NameError:
    data = pd.read_csv("bike-sharing-dataset.csv")

profile = ProfileReport(data, title="bike-sharing-dataset Data Profiling Report",explorative=True)
profile.to_file("bike-sharing-dataset.html")