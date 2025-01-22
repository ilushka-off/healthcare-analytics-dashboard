import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import load_model

def train_model(data):
    # Убедимся, что папка 'data' существует
    if not os.path.exists('data'):
        os.makedirs('data')

    # Разделение данных для возраста
    X = data.drop(columns=['Age', 'Billing Amount'])
    y = data['Age']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = Sequential()
    model.add(Dense(64, input_dim=X_train_scaled.shape[1], activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1))

    model.compile(loss='mse', optimizer=Adam(learning_rate=0.001), metrics=['mae'])

    es = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    mc = ModelCheckpoint('model_age.keras', monitor='val_loss', save_best_only=True)

    # Обучение модели
    history = model.fit(X_train_scaled, y_train, validation_split=0.2, epochs=100, batch_size=32, callbacks=[es, mc])

    # Добавление столбца 'epoch' в DataFrame истории
    hist_df = pd.DataFrame(history.history)
    hist_df['epoch'] = hist_df.index + 1  # Добавляем эпоки как столбец

    # Загрузка лучшей модели
    best_model = load_model('model_age.keras')

    y_pred = best_model.predict(X_test_scaled)
    y_pred = y_pred.flatten()

    pred_df = pd.DataFrame({'y_true': y_test, 'y_pred': y_pred})
    pred_df['error'] = pred_df['y_pred'] - pred_df['y_true']
    
    # Сохранение предсказаний в файл
    pred_df.to_csv('data/predictions.csv', index=False)

    # Повторяем для 'Billing Amount'
    y_billing = data['Billing Amount']
    X_train, X_test, y_train, y_test = train_test_split(X, y_billing, test_size=0.2, random_state=42)
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model.compile(loss='mse', optimizer=Adam(learning_rate=0.001), metrics=['mae'])

    mc_billing = ModelCheckpoint('model_billing.keras', monitor='val_loss', save_best_only=True)
    history_billing = model.fit(X_train_scaled, y_train, validation_split=0.2, epochs=100, batch_size=32, callbacks=[es, mc_billing])

    best_model_billing = load_model('model_billing.keras')

    y_pred_billing = best_model_billing.predict(X_test_scaled)
    y_pred_billing = y_pred_billing.flatten()

    pred_df['y_true_billing'] = y_test
    pred_df['y_pred_billing'] = y_pred_billing
    pred_df['error_billing'] = pred_df['y_pred_billing'] - pred_df['y_true_billing']
    pred_df.to_csv('data/predictions_billing.csv', index=False)

    # Добавление эпок в историю обучения для 'Billing Amount'
    hist_billing_df = pd.DataFrame(history_billing.history)
    hist_billing_df['epoch'] = hist_billing_df.index + 1  # Добавляем эпоки как столбец

    hist_billing_df.to_csv('data/training_history.csv', index=False)

    return best_model, pred_df, hist_df
