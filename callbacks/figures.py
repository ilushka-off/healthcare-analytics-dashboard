import plotly.express as px

def create_figures(pred_df, data):
    # Гистограмма для распределения истинных значений возраста (y_true)
    fig_age_distribution = px.histogram(
        pred_df, x='y_true', nbins=10, 
        title="Распределение истинных значений возраста"
    )
    
    # Scatter plot для y_true и y_pred (Предсказания vs Истинные значения)
    fig_age_scatter = px.scatter(
        pred_df, x='y_true', y='y_pred',
        title="Предсказания возраста vs Истинные значения возраста",
        labels={'y_true': 'Истинное значение возраста', 'y_pred': 'Предсказанное значение возраста'}
    )
    fig_age_scatter.add_shape(
        type='line', line=dict(dash='dash', color='red'),
        x0=pred_df['y_true'].min(), x1=pred_df['y_true'].max(), 
        y0=pred_df['y_true'].min(), y1=pred_df['y_true'].max()
    )

    # Boxplot для стоимости лечения
    fig_billing_by_admission = px.box(
        pred_df, x='y_true', y='y_pred',
        title="Распределение истинных и предсказанных значений стоимости лечения"
    )

    # Scatter plot для предсказаний стоимости лечения
    fig_billing_scatter = px.scatter(
        pred_df, x='y_true_billing', y='y_pred_billing',
        title="Предсказания стоимости лечения vs Истинные значения стоимости лечения",
        labels={'y_true_billing': 'Истинная стоимость лечения', 'y_pred_billing': 'Предсказанная стоимость лечения'}
    )
    fig_billing_scatter.add_shape(
        type='line', line=dict(dash='dash', color='red'),
        x0=pred_df['y_true_billing'].min(), x1=pred_df['y_true_billing'].max(), 
        y0=pred_df['y_true_billing'].min(), y1=pred_df['y_true_billing'].max()
    )
    
    # Ошибка для возраста
    fig_age_error = px.histogram(
        pred_df, x='error', nbins=50, 
        title="Ошибка предсказаний для возраста"
    )

    # Ошибка для стоимости лечения
    fig_billing_error = px.histogram(
        pred_df, x='error_billing', nbins=50, 
        title="Ошибка предсказаний для стоимости лечения"
    )

    # Диаграмма пирога для анализа категориальных переменных (Gender)
    # Используем one-hot encoded столбец для Gender
    fig_gender_pie = px.pie(
        data, names='Gender_Male', title="Распределение по полу пациентов"
    )

    # Диаграмма пирога для анализа категориальных переменных (Medical Condition)
    fig_medical_condition_pie = px.pie(
        data, names='Medical Condition_Obesity', title="Распределение заболеваний пациентов"
    )

    return fig_age_distribution, fig_age_scatter, fig_billing_by_admission, fig_billing_scatter, fig_age_error, fig_billing_error, fig_gender_pie, fig_medical_condition_pie
