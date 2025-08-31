import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer


try:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_dir, "bike-sharing-dataset.csv")
    data = pd.read_csv(data_path)
except NameError:
    data = pd.read_csv("bike-sharing-dataset.csv")

data = data.rename(columns={'users':'Demand', 'temp':'Temperature'})
data['date_time'] = pd.to_datetime(data['date_time'])
data.set_index('date_time', inplace=True)

data['hour_sin'] = np.sin(2 * np.pi * data['hour'] / 24.0)
data['hour_cos'] = np.cos(2 * np.pi * data['hour'] / 24.0)
data['weekday_sin'] = np.sin(2 * np.pi * data['weekday'] / 7.0)
data['weekday_cos'] = np.cos(2 * np.pi * data['weekday'] / 7.0)
data = data.drop(['hour', 'weekday', 'month'], axis=1)

data.dropna(axis=0, inplace=True)

def create_ts_data_recursion(data: object, columns_to_lag: list, target_column: list, window_size:int=10) -> tuple[pd.DataFrame, pd.Series]:
    
    df = data.copy()
    lagged_data = []

    for col in columns_to_lag:
        for i in range(1, window_size + 1):
            lag_col = df[col].shift(i).rename(f'{col}_lag_{i}')
            lagged_data.append(lag_col)
    lagged_df = pd.concat(lagged_data, axis=1)
    df = pd.concat([df, lagged_df], axis=1)
    df = df.rename(columns={target_column: 'target'})
    df = df.dropna(axis=0)

    y = df['target']
    X = df.drop('target', axis=1)

    return X, y

X, y = create_ts_data_recursion(data, columns_to_lag=['Demand', 'Temperature'], target_column='Demand', window_size=48*2)

split_point = int(0.8 * len(X))
X_train, X_test = X.iloc[:split_point], X.iloc[split_point:]
y_train, y_test = y.iloc[:split_point], y.iloc[split_point:]


categorical_features = ['weather']

numerical_features = [col for col in X_train.columns if col not in categorical_features]

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ]   
)

model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1))
])

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"Mean Absolute Error: {mae}")
print(f"R^2 Score: {r2}")