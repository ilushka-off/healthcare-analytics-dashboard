# Healthcare Analytics Dashboard

Этот проект представляет собой дашборд для анализа данных в сфере здравоохранения. Он использует нейронные сети для предсказания возраста пациентов и стоимости их лечения. Дашборд отображает результаты работы модели в виде графиков и статистик, таких как распределение ошибок, сравнительные графики и диаграммы по медицинским условиям.

## Датасет

Датасет содержит информацию о пациентах, их медицинских условиях и стоимости лечения. Основные столбцы в датасете:

| **Атрибут**           | **Тип**      | **Описание**                                                                 | **Пример значений**                |
|-----------------------|--------------|-------------------------------------------------------------------------------|------------------------------------|
| **Name**              | Строка       | Имя пациента                                                                 | Bobby JacksOn, LesLie TErRy       |
| **Age**               | Число        | Возраст пациента                                                              | 30, 62, 76                        |
| **Gender**            | Строка       | Пол пациента                                                                  | Male, Female                      |
| **Medical Condition** | Строка       | Медицинское состояние пациента (например, рак, диабет, ожирение и т.д.)       | Cancer, Obesity                   |


## Установка

1. Клонируйте репозиторpoetry shellий:

    ```bash
    git clone https://github.com/yourusername/healthcare-analytics-dashboard.git
    cd healthcare-analytics-dashboard
    ```

2. Установите зависимости с помощью Poetry:

    ```bash
    poetry install
    ```

3. Для активации виртуальной среды:

    ```bash
    poetry shell
    ```

## Запуск приложения

1. После того как вы активировали виртуальную среду и установили все зависимости, выполните следующую команду для запуска приложения:

    ```bash
    poetry run python main.py
    ```

2. Перейдите в браузер и откройте http://127.0.0.1:8050 для доступа к дашборду.

## Структура проекта

- main.py: Главный файл для запуска приложения.
- layouts/dashboard.py: Разметка дашборда с графиками и метриками.
- callbacks/model.py: Обучение нейронной сети и сохранение модели.    
- callbacks/metrics.py: Функции для вычисления метрик.
- callbacks/figures.py: Генерация графиков для дашборда.
- utils/data_loader.py: Загрузка и обработка данных.

## Логи и диагностика

В проекте настроено логирование, и все важные сообщения о процессе обучения и работе дашборда записываются в файлы:

- data/training_log.txt: Лог обучения модели.
- data/dashboard_log.txt: Лог работы дашборда.
- data/training_history.csv: История обучения модели.
- data/predictions.csv: Файл с предсказаниями модели.
- data/predictions_billing.csv: Файл с предсказаниями стоимости лечения.