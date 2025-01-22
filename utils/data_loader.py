import pandas as pd
import os

def load_data():
    # Загрузка данных из healthcare_dataset
    data_path = os.path.join(os.path.dirname(__file__), '../data/healthcare_dataset.csv')
    data = pd.read_csv(data_path)

    # Преобразование дат в количество дней (для каждого столбца с датой)
    data['Date of Admission'] = pd.to_datetime(data['Date of Admission'])
    data['Discharge Date'] = pd.to_datetime(data['Discharge Date'])
    data['Stay Duration (days)'] = (data['Discharge Date'] - data['Date of Admission']).dt.days

    # Убираем текстовые данные, такие как имена и медицинские назначения
    data = data.drop(columns=['Name', 'Doctor', 'Hospital', 'Medication', 'Test Results'])

    # Преобразуем даты в количество дней с 01-01-1970
    data['Date of Admission'] = (data['Date of Admission'] - pd.Timestamp('1970-01-01')) // pd.Timedelta('1D')
    data['Discharge Date'] = (data['Discharge Date'] - pd.Timestamp('1970-01-01')) // pd.Timedelta('1D')

    # Применение One-Hot Encoding для категориальных признаков
    data = pd.get_dummies(data, columns=['Gender', 'Blood Type', 'Medical Condition', 'Admission Type', 'Insurance Provider'], drop_first=True)
    
    # Возвращаем данные
    return data
