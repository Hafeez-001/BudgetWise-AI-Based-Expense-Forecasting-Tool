from sklearn.metrics import mean_absolute_error, root_mean_squared_error
import numpy as np

def evaluate_forecast(y_true, y_pred):
    """
    Compute common metrics: MAE, RMSE, MAPE
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred, squared=False)
    mape = np.mean(np.abs((y_true - y_pred)/np.maximum(y_true,1e-6))) * 100
    return {'MAE': mae, 'RMSE': rmse, 'MAPE': mape}
