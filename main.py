from scipy.stats import norm  # Импортируем norm для кривой Гаусса
import dash
from dash import dcc, html, Input, Output, dash_table
import plotly.express as px
import pandas as pd 
import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import dash_bootstrap_components as dbc
from datetime import datetime
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
import io
import os
import base64
from sklearn.cluster import KMeans, DBSCAN
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier 
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# Загрузка данных
data = pd.read_csv("healthcare_dataset.csv")

# Преобразуем категориальные признаки в числовые
label_encoders = {}
for column in ["Gender", "Blood Type", "Medical Condition", "Doctor", "Hospital", "Insurance Provider", "Admission Type", "Medication", "Test Results"]:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le

# Разделение данных на признаки и целевую переменную
X = data.drop(columns=["Test Results", "Name", "Date of Admission", "Discharge Date"])
y = data["Test Results"]

# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Масштабирование данных
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Обучение моделей
models = {
    "Logistic Regression": LogisticRegression(),
    "Support Vector Machines": SVC(probability=True),
    "k-Nearest Neighbors": KNeighborsClassifier(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "Gradient Boosting": GradientBoostingClassifier()
}

for name, model in models.items():
    model_filename = f"{name.replace(' ', '_')}_model.pkl"
    if not os.path.exists(model_filename):  # Проверяем, существует ли файл модели
        print(f"Обучение модели: {name} начато в {datetime.now().time()}")
        model.fit(X_train_scaled if name in ["Logistic Regression", "Support Vector Machines",
                                            "k-Nearest Neighbors", "Gradient Boosting"] else X_train, y_train)
        joblib.dump(model, model_filename)
        print(f"Обучение модели: {name} завершено в {datetime.now().time()}")
    else:
        print(f"Модель {name} уже обучена и загружена из файла.")

# Функция для вычисления основных статистик
def calculate_statistics(values):
    n = len(values)
    mean, std_dev = values.mean(), values.std()
    std_err = std_dev / np.sqrt(n)
    skewness, kurtosis = stats.skew(values), stats.kurtosis(values)
    skew_err = np.sqrt(6 * n / ((n - 1) * (n - 2)))
    kurt_err = np.sqrt(24 * n * (n - 1)**2 / ((n - 3) * (n - 2) * n))
    conf_interval = stats.norm.interval(0.95, loc=mean, scale=std_err)

    return {
        "Среднее значение": f"{mean:.2f}",
        "Стандартное отклонение": f"{std_dev:.2f}",
        "Ошибка среднего значения": f"{std_err:.2f}",
        "Медиана": f"{values.median():.2f}",
        "Асимметрия": f"{skewness:.2f} ± {skew_err:.2f}",
        "Эксцесс": f"{kurtosis:.2f} ± {kurt_err:.2f}",
        "95% Доверительный интервал": f"({conf_interval[0]:.2f}, {conf_interval[1]:.2f})"
    }

# Функция для загрузки моделей и предсказаний
def load_models_and_predictions(data):
    models = {
        name: joblib.load(f"{name.replace(' ', '_')}_model.pkl")
        for name in ["Logistic Regression", "Support Vector Machines", "k-Nearest Neighbors",
                     "Decision Tree", "Random Forest", "Gradient Boosting"]
    }

    X, y = data.drop(columns=["Test Results", "Name", "Date of Admission", "Discharge Date"]), data["Test Results"]
    X_scaled = StandardScaler().fit_transform(X)

    precomputed_predictions = {
        name: {
            "y_pred": model.predict(X_scaled if name in ["Logistic Regression", "Support Vector Machines",
                                                         "k-Nearest Neighbors", "Gradient Boosting"] else X),
            "y_proba": (model.predict_proba(X_scaled if name in ["Logistic Regression", "Support Vector Machines",
                                                                 "k-Nearest Neighbors", "Gradient Boosting"] else X)
                         if hasattr(model, "predict_proba") else np.zeros((len(y), len(np.unique(y)))))
        }
        for name, model in models.items()
    }

    metrics = {
        name: {
            "Accuracy": accuracy_score(y, preds["y_pred"]),
            "Precision": precision_score(y, preds["y_pred"], average='macro'),
            "Recall": recall_score(y, preds["y_pred"], average='macro'),
            "F1-Score": f1_score(y, preds["y_pred"], average='macro'),
            "ROC-AUC": roc_auc_score(y, preds["y_proba"], multi_class='ovr', average='macro')
        }
        for name, preds in precomputed_predictions.items()
    }

    return models, precomputed_predictions, metrics, y 

# Инициализация моделей, предсказаний и метрик
models, precomputed_predictions, metrics, y = load_models_and_predictions(data)

# Подготавливает данные для кластеризации и визуализации.
def prepare_clustering_Kmeans_data(data, n_clusters=4):
    # Используем только возраст и сумму счета для кластеризации
    selected_features = ['Age', 'Billing Amount']
    data_selected = data[selected_features]

    # Кластеризация методом K-средних
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    data['Cluster'] = kmeans.fit_predict(data_selected)

    # Подготовка данных для визуализации
    reduced_data = pd.DataFrame(data_selected, columns=selected_features)
    reduced_data['Cluster'] = data['Cluster']

    # График кластеризации
    fig = px.scatter(reduced_data, x='Age', y='Billing Amount', color='Cluster',
                     title="K-Means Clustering (Age vs Billing Amount)", 
                     labels={'Cluster': 'Cluster Group', 'Age': 'Age', 'Billing Amount': 'Billing Amount'})

    # Удаление столбца Cluster из исходных данных
    data.drop(columns=['Cluster'], inplace=True)

    return data, reduced_data, fig

# Вызываем функцию для кластеризации
n_clusters = 5
data, reduced_data, fig = prepare_clustering_Kmeans_data(data, n_clusters=n_clusters)

# Подготавливает данные для кластеризации и визуализации с использованием DBSCAN
def prepare_clustering_data_dbscan(data, eps=0.5, min_samples=5):
    # Используем только возраст и сумму счета для кластеризации
    selected_features = ['Age', 'Billing Amount']
    data_selected = data[selected_features]

    # Кластеризация методом DBSCAN
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    clusters = dbscan.fit_predict(data_selected)
    data['Cluster'] = clusters

    # Подготовка данных для визуализации
    reduced_data = pd.DataFrame(data_selected, columns=selected_features)
    reduced_data['Cluster'] = clusters

    # График кластеризации
    fig = px.scatter(reduced_data, x='Age', y='Billing Amount', color='Cluster',
                     title="DBSCAN Clustering (Age vs Billing Amount)",
                     labels={'Cluster': 'Cluster Group', 'Age': 'Age', 'Billing Amount': 'Billing Amount'})

    # Удаление столбца Cluster из исходных данных
    data.drop(columns=['Cluster'], inplace=True)

    return data, reduced_data, fig

# Параметры DBSCAN
eps = 0.5  # Максимальное расстояние для объединения точек
min_samples = 5 # Минимальное число точек в кластере

# Вызываем функцию для подготовки данных с DBSCAN
data1, reduced_data1, fig1 = prepare_clustering_data_dbscan(data, eps=eps, min_samples=min_samples)

# Общая функция для создания тепловой карты
def create_correlation_heatmap(df, column_to_drop="Test Results"):
    print("Создание тепловой карты", datetime.now().time())
    
    # Удаляем указанный столбец
    df_corr = df.drop(columns=[column_to_drop])
    
    # Удаляем нечисловые столбцы
    df_corr = df_corr.select_dtypes(include=[np.number])
    
    # Вычисление корреляционной матрицы
    correlation_matrix = df_corr.corr()
    
    # Построение тепловой карты
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
    
    # Сохранение изображения в буфер
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    
    # Преобразование изображения в base64
    return base64.b64encode(buf.getvalue()).decode('utf-8')

def analyze_medical_conditions(data):
    condition_counts = data["Medical Condition"].value_counts().reset_index()
    condition_counts.columns = ["Medical Condition", "Count"]
    fig_conditions = px.bar(condition_counts, x="Medical Condition", y="Count", 
                            title="Распространённость медицинских условий",
                            labels={"Medical Condition": "Медицинское условие", "Count": "Количество пациентов"})
    return fig_conditions

def analyze_conditions_by_age(data):
    fig_age = px.box(data, x="Medical Condition", y="Age", 
                     title="Зависимость заболеваний от возраста",
                     labels={"Medical Condition": "Медицинское условие", "Age": "Возраст"})
    return fig_age

def analyze_conditions_by_gender(data):
    gender_condition_counts = data.groupby(["Medical Condition", "Gender"]).size().reset_index(name="Count")
    fig_gender = px.bar(gender_condition_counts, x="Medical Condition", y="Count", color="Gender",
                        title="Зависимость заболеваний от пола",
                        labels={"Medical Condition": "Медицинское условие", "Count": "Количество пациентов", "Gender": "Пол"})
    return fig_gender

def analyze_age_distribution(data):
    # Гистограмма
    fig = px.histogram(data, x='Age', nbins=20, title="Распределение возраста пациентов",
                       labels={'Age': 'Возраст', 'count': 'Количество пациентов'},
                       opacity=0.7)
    return fig

def analyze_billing_amount_distribution(data):
    # Гистограмма
    fig = px.histogram(data, x='Billing Amount', nbins=20, title="Распределение суммы счета",
                       labels={'Billing Amount': 'Сумма счета', 'count': 'Количество пациентов'},
                       opacity=0.7)
    return fig

# Вызов функции для получения изображения тепловой карты
img_str = create_correlation_heatmap(data)

print("Инициализация Dash-приложения", datetime.now().time())
# Инициализация Dash-приложения
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
print("app.layout", datetime.now().time())

# Определение макета приложения
app.layout = html.Div([
    html.H1("Анализ медицинских данных", style={'textAlign': 'center'}),
    html.H2("Работа студенток гр. 4296 Мейзер М.В., Северьянова Е.Д., Салимзянова Р.Р.", style={'textAlign': 'center'}),
    
    # Таблица с данными
    html.Div([
        html.H3("Таблица с данными", style={'textAlign': 'center'}),
        dash_table.DataTable(
            id='table',
            columns=[{"name": i, "id": i} for i in data.columns],
            data=data.to_dict('records'),
            page_size=10,
            style_table={'height': '300px', 'overflowY': 'auto'}
        )
    ]),
    
    # Тепловая карта корреляций
    html.Div([
        html.H3("Тепловая карта корреляций", style={'textAlign': 'center'}),
        html.Img(src=f'data:image/png;base64,{img_str}', style={'display': 'block', 'margin': 'auto'})
    ]),
    
    # Распределение возраста пациентов
    html.Div([
        html.H3("Распределение возраста пациентов", style={'textAlign': 'center'}),
        dcc.Graph(figure=analyze_age_distribution(data))
    ]),
    
    # Распределение суммы счета
    html.Div([
        html.H3("Распределение суммы счета", style={'textAlign': 'center'}),
        dcc.Graph(figure=analyze_billing_amount_distribution(data))
    ]),
    
    # Распространённость медицинских условий
    html.Div([
        html.H3("Распространённость медицинских условий", style={'textAlign': 'center'}),
        dcc.Graph(figure=analyze_medical_conditions(data))
    ]),
    
    # Зависимость заболеваний от возраста
    html.Div([
        html.H3("Зависимость заболеваний от возраста", style={'textAlign': 'center'}),
        dcc.Graph(figure=analyze_conditions_by_age(data))
    ]),
    
    # Зависимость заболеваний от пола
    html.Div([
        html.H3("Зависимость заболеваний от пола", style={'textAlign': 'center'}),
        dcc.Graph(figure=analyze_conditions_by_gender(data))
    ]),
    
    # Кластеризация методом K-средних
    html.Div([
        html.H3("Кластеризация методом K-средних (Age vs Billing Amount)", style={'textAlign': 'center'}),
        dcc.Graph(figure=fig)
    ]),
    
    # Кластеризация методом DBSCAN
    html.Div([
        html.H3("Кластеризация методом DBSCAN (Age vs Billing Amount)", style={'textAlign': 'center'}),
        dcc.Graph(figure=fig1)
    ])
])

# Запуск приложения
if __name__ == '__main__':
    app.run_server(debug=True)