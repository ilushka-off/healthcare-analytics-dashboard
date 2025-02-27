import plotly.express as px
import plotly.figure_factory as ff
import pandas as pd

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
    fig_gender_pie = px.pie(
        data, names='Gender_Male', title="Распределение по полу пациентов"
    )

    # Диаграмма пирога для анализа категориальных переменных (Medical Condition)
    fig_medical_condition_pie = px.pie(
        data, names='Medical Condition_Obesity', title="Распределение заболеваний пациентов"
    )
    
    # Матрица корреляции
    correlation_matrix = pred_df.corr()
    fig_corr_matrix = ff.create_annotated_heatmap(
        z=correlation_matrix.values,
        x=list(correlation_matrix.columns),
        y=list(correlation_matrix.index),
        annotation_text=correlation_matrix.round(2).values,
        colorscale='Viridis',
        showscale=True,
    )
    fig_corr_matrix.update_layout(title="Матрица корреляции признаков")
    
    # Гистограмма распределения признаков
    feature_distribution = px.histogram(
        data.melt(), x='value', color='variable',
        title="Распределение признаков"
    )
    
    # Scatter plot корреляции между ошибками предсказаний
    fig_error_correlation = px.scatter(
        pred_df, x='error', y='error_billing',
        title="Корреляция ошибок предсказаний возраста и стоимости лечения"
    )
    
    return (
        fig_age_distribution, fig_age_scatter, fig_billing_by_admission, 
        fig_billing_scatter, fig_age_error, fig_billing_error, 
        fig_gender_pie, fig_medical_condition_pie, fig_corr_matrix, 
        feature_distribution, fig_error_correlation
    )
