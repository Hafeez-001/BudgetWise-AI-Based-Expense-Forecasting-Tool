import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split

# -------------------------------------
# Prophet Model
# -------------------------------------
def forecast_prophet(df: pd.DataFrame, periods: int = 90) -> pd.DataFrame:
    ts = df.groupby('date')['amount'].sum().reset_index()
    ts = ts.rename(columns={'date': 'ds', 'amount': 'y'})
    model = Prophet(daily_seasonality=True)
    model.fit(ts)
    future = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)
    return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]

# -------------------------------------
# Random Forest Model
# -------------------------------------
def forecast_random_forest(df: pd.DataFrame, periods: int = 90) -> pd.DataFrame:
    ts = df.groupby('date')['amount'].sum().reset_index()
    ts['dayofyear'] = ts['date'].dt.dayofyear
    ts['month'] = ts['date'].dt.month
    ts['dayofweek'] = ts['date'].dt.dayofweek
    ts['year'] = ts['date'].dt.year

    X = ts[['dayofyear', 'month', 'dayofweek', 'year']]
    y = ts['amount']

    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X, y)

    last_date = ts['date'].max()
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=periods)
    future_df = pd.DataFrame({
        'ds': future_dates,
        'dayofyear': future_dates.dayofyear,
        'month': future_dates.month,
        'dayofweek': future_dates.dayofweek,
        'year': future_dates.year
    })

    y_pred = model.predict(future_df[['dayofyear', 'month', 'dayofweek', 'year']])
    forecast_df = pd.DataFrame({'ds': future_dates, 'yhat': y_pred})
    forecast_df['yhat_lower'] = forecast_df['yhat'] * 0.9
    forecast_df['yhat_upper'] = forecast_df['yhat'] * 1.1
    return forecast_df

# -------------------------------------
# XGBoost Model
# -------------------------------------
def forecast_xgboost(df: pd.DataFrame, periods: int = 90) -> pd.DataFrame:
    ts = df.groupby('date')['amount'].sum().reset_index()
    ts['dayofyear'] = ts['date'].dt.dayofyear
    ts['month'] = ts['date'].dt.month
    ts['dayofweek'] = ts['date'].dt.dayofweek
    ts['year'] = ts['date'].dt.year

    X = ts[['dayofyear', 'month', 'dayofweek', 'year']]
    y = ts['amount']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    model = XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=5, random_state=42)
    model.fit(X_train, y_train)

    last_date = ts['date'].max()
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=periods)
    future_df = pd.DataFrame({
        'ds': future_dates,
        'dayofyear': future_dates.dayofyear,
        'month': future_dates.month,
        'dayofweek': future_dates.dayofweek,
        'year': future_dates.year
    })

    y_pred = model.predict(future_df[['dayofyear', 'month', 'dayofweek', 'year']])
    forecast_df = pd.DataFrame({'ds': future_dates, 'yhat': y_pred})
    forecast_df['yhat_lower'] = forecast_df['yhat'] * 0.9
    forecast_df['yhat_upper'] = forecast_df['yhat'] * 1.1
    return forecast_df
