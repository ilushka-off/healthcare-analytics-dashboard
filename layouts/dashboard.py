import os
from dash import html, dcc
from callbacks.figures import create_figures
from callbacks.metrics import calculate_metrics
from utils.data_loader import load_data
from callbacks.model import train_model
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Логирование начала работы дашборда
log_file = 'data/dashboard_log.txt'
with open(log_file, 'w') as log:
    log.write("Запуск дашборда...\n")

# Проверка наличия нужных файлов
if not os.path.exists('data/predictions.csv'):
    with open(log_file, 'a') as log:
        log.write("Ошибка: файл predictions.csv не найден.\n")
if not os.path.exists('data/training_history.csv'):
    with open(log_file, 'a') as log:
        log.write("Ошибка: файл training_history.csv не найден.\n")
else:
    with open(log_file, 'a') as log:
        log.write("Все необходимые файлы присутствуют.\n")

# Загрузка данных и обучение модели
data = load_data()
best_model, pred_df, hist_df = train_model(data)

# Вычисление метрик
mae, mse, rmse, r2 = calculate_metrics()

# Создание копии данных для категориальных признаков
categorical_data = data.select_dtypes(include=['object', 'bool']).copy()

# Нормализация данных для корректного распределения признаков
numeric_columns = data.select_dtypes(include=['int64', 'float64']).columns
scaler = StandardScaler()
data_scaled = pd.DataFrame(scaler.fit_transform(data[numeric_columns]), columns=numeric_columns)
data_melted = data_scaled.melt()

# Генерация графиков
figures = create_figures(pred_df, categorical_data)
(fig_age_distribution, fig_age_scatter, fig_billing_by_admission, 
 fig_billing_scatter, fig_age_error, fig_billing_error, 
 fig_gender_pie, fig_medical_condition_pie, fig_corr_matrix, 
 feature_distribution, fig_error_correlation) = figures

# Layout приложения
layout = html.Div(style={'font-family': 'Arial, sans-serif', 'backgroundColor': '#f4f4f4', 'padding': '20px'}, children=[
    html.H1("Healthcare Analytics Dashboard", style={'textAlign': 'center'}),
    
    # Метрики
    html.Div(style={'display': 'flex', 'justifyContent': 'space-around'}, children=[
        html.Div([
            html.H3("MAE"),
            html.P(f"{mae:.4f}"),
            html.P("Средняя абсолютная ошибка (MAE) измеряет среднюю величину отклонения предсказанных значений от истинных значений.")
        ]),
        html.Div([
            html.H3("MSE"),
            html.P(f"{mse:.4f}"),
            html.P("Среднеквадратичная ошибка (MSE) измеряет средний квадрат отклонений предсказанных значений от истинных.")
        ]),
        html.Div([
            html.H3("RMSE"),
            html.P(f"{rmse:.4f}"),
            html.P("Корень из среднеквадратичной ошибки (RMSE) позволяет интерпретировать ошибку в тех же единицах, что и исходная целевая переменная.")
        ]),
        html.Div([
            html.H3("R²"),
            html.P(f"{r2:.4f}"),
            html.P("Коэффициент детерминации (R²) показывает, какая доля дисперсии в целевой переменной объясняется моделью. Чем ближе к 1, тем лучше модель.")
        ])
    ]),
    
    # Логи обучения
    html.Div([
        html.H3("Логи обучения"),
        html.P("В этом разделе представлены логи обучения модели, включая значения потерь и точности (MAE) на тренировочной и валидационной выборках."),
        dcc.Graph(figure={
            'data': [
                {'x': hist_df['epoch'], 'y': hist_df['loss'], 'type': 'line', 'name': 'Loss'},
                {'x': hist_df['epoch'], 'y': hist_df['val_loss'], 'type': 'line', 'name': 'Validation Loss'},
                {'x': hist_df['epoch'], 'y': hist_df['mae'], 'type': 'line', 'name': 'MAE'},
                {'x': hist_df['epoch'], 'y': hist_df['val_mae'], 'type': 'line', 'name': 'Validation MAE'}
            ],
            'layout': {
                'title': 'Логи обучения: Loss и MAE',
                'xaxis': {'title': 'Эпоха'},
                'yaxis': {'title': 'Значение'}
            }
        }),
    ]),
    
    # Графики
    html.Div([dcc.Graph(figure=fig_age_distribution), html.P("Распределение истинных значений возраста.")]),
    html.Div([dcc.Graph(figure=fig_age_scatter), html.P("Сравнение предсказанных и истинных значений возраста.")]),
    html.Div([dcc.Graph(figure=fig_billing_by_admission), html.P("Распределение стоимости лечения.")]),
    html.Div([dcc.Graph(figure=fig_billing_scatter), html.P("Сравнение предсказанных и истинных значений стоимости лечения.")]),
    html.Div([dcc.Graph(figure=fig_age_error), html.P("Распределение ошибок предсказания возраста.")]),
    html.Div([dcc.Graph(figure=fig_billing_error), html.P("Распределение ошибок предсказания стоимости лечения.")]),
    html.Div([dcc.Graph(figure=fig_gender_pie), html.P("Распределение пациентов по полу.")]),
    html.Div([dcc.Graph(figure=fig_medical_condition_pie), html.P("Распределение заболеваний пациентов.")]),
    html.Div([dcc.Graph(figure=fig_corr_matrix), html.P("Матрица корреляции признаков, визуализирующая взаимосвязи между переменными.")]),
    html.Div([dcc.Graph(figure=feature_distribution), html.P("Гистограмма распределения признаков (нормализованные данные).")]),
    html.Div([dcc.Graph(figure=fig_error_correlation), html.P("Анализ корреляции между ошибками предсказания возраста и стоимости лечения.")]),
])
