import numpy as np

def _check_y_inputs(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if y_true.size == 0 or y_pred.size == 0:
        raise ValueError("y_true and y_pred cannot be empty arrays")

    if y_true.ndim != 1 or y_pred.ndim != 1:
        raise ValueError("y_true and y_pred must be 1-dimensional arrays (n_samples,)")

    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError(f"y_true samples ({y_true.shape[0]}) != y_pred samples ({y_pred.shape[0]})")

    return y_true, y_pred


def mse(y_true, y_pred):
    """
    MSE = (1/n) * Σ(y_true - y_pred)²
    """
    y_true, y_pred = _check_y_inputs(y_true, y_pred)
    return (np.mean((y_true - y_pred) ** 2)).item()


def rmse(y_true, y_pred):
    """
    RMSE = √MSE
    """
    return (np.sqrt(mse(y_true, y_pred))).item()


def mae(y_true, y_pred):
    """
    MAE = (1/n) * Σ|y_true - y_pred|
    """
    y_true, y_pred = _check_y_inputs(y_true, y_pred)
    return (np.mean(np.abs(y_true - y_pred))).item()


def r2_score(y_true, y_pred):
    """
    R² = 1 - (Σ(y_true - y_pred)² / Σ(y_true - y_true_mean)²)
    """
    y_true, y_pred = _check_y_inputs(y_true, y_pred)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)

    if ss_tot == 0:
        raise ValueError("All values in y_true are the same, R² score is undefined")

    return (1 - (ss_res / ss_tot)).item()


def mape(y_true, y_pred):
    """
    MAPE = (1/n) * Σ(|(y_true - y_pred)/y_true|) * 100
    """
    y_true, y_pred = _check_y_inputs(y_true, y_pred)

    if np.any(y_true == 0):
        raise ValueError("y_true contains zero values, MAPE is undefined (division by zero)")

    return (np.mean(np.abs((y_true - y_pred) / y_true)) * 100).item()


def evaluate(y_true, y_pred, decimals=4):
    metrics = {
        "MSE": mse(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "MAE": mae(y_true, y_pred),
        "R²": r2_score(y_true, y_pred),
        "MAPE(%)": mape(y_true, y_pred)
    }
    metrics_rounded = {k: round(v, decimals) for k, v in metrics.items()}
    return metrics_rounded