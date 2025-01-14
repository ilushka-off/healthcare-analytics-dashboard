import os
import pandas as pd

def calculate_metrics():
    # Проверка существования файла с предсказаниями
    if not os.path.exists('data/predictions.csv'):
        raise FileNotFoundError("Файл 'predictions.csv' не найден в папке 'data'.")

    # Загрузка данных из CSV файла
    pred_df = pd.read_csv('data/predictions.csv')

    # Вычисление метрик
    mae = (pred_df['y_pred'] - pred_df['y_true']).abs().mean()
    mse = ((pred_df['y_pred'] - pred_df['y_true']) ** 2).mean()
    rmse = mse ** 0.5
    r2 = 1 - (mse / ((pred_df['y_true'] - pred_df['y_true'].mean()) ** 2).mean())

    return mae, mse, rmse, r2
