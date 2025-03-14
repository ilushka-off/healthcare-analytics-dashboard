# Группа 4296

## Авторы:
### Горбунов Илья Дмитриевич
### Шарипов Адель Сиринович
### Мальковский Никита Сергеевич 

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

1. Клонируйте репозиторий:

    ```bash
    git clone git@github.com:ilushka-off/healthcare-analytics-dashboard.git
    cd healthcare-analytics-dashboard
    ```

2. Шаги установки (при помощи pip)
#### Активируем виртуальное окружение
    
    ```
    source venv/bin/activate
    ```

#### Устанавливем зависимости

    ```
    pip install -r requirements.txt
    ```

## Запуск приложения

1. После того как вы активировали виртуальную среду и установили все зависимости, выполните следующую команду для запуска приложения:

    ```bash
    python main.py
    ```

2. Перейдите в браузер и откройте http://127.0.0.1:8050 для доступа к дашборду.