import streamlit as st
import pandas as pd
import joblib
import numpy as np
from datetime import datetime, timedelta
from catboost import CatBoostRegressor

# Функция корректировки прогноза с учётом даты и введённой цены
def adjust_price(base_pred, input_price, tolerance=0.2, user_weight=0.9):
    """Корректировка в сторону введённой цены"""
    if abs(input_price - base_pred) / input_price > tolerance:
        return (1 - user_weight) * base_pred + user_weight * input_price
    return base_pred

# Загрузка данных и модели
data_oned = pd.read_csv('intensive3/data/price_data.csv', parse_dates=['dt'])
data_oned = data_oned.sort_values('dt').reset_index(drop=True)

model = joblib.load('intensive3/models/model_oned.pkl')

st.title("Прогноз цены и уровень закупки арматуры")

# Пользовательский ввод
input_date = st.date_input("Выберите дату для прогноза", datetime.today())
current_price = st.number_input("Введите актуальную цену на выбранную дату", min_value=0.0, format="%.2f", value=1000.0)

if st.button("Сделать прогноз"):
    try:
        # Подготовка истории для лагов
        if len(data_oned) >= 6:
            hist_data = data_oned[data_oned['dt'] <= pd.to_datetime(input_date)]
            if len(hist_data) >= 6:
                history = list(hist_data['Цена на арматуру'].tail(6))
            else:
                history = [current_price] * 6
        else:
            history = [current_price] * 6

        dates = [pd.to_datetime(input_date)]
        prices = [current_price]
        max_multiplier = 1
        current_multiplier = 1
        stop_multiplier = False  # Флаг остановки расчёта множителя

        # Прогноз на 6 недель
        for i in range(6):
            # Формируем датафрейм с лагами
            X_pred = pd.DataFrame({
                'lag_1': [history[-1]],
                'lag_2': [history[-2]],
                'lag_3': [history[-3]],
                'lag_4': [history[-4]],
                'lag_5': [history[-5]],
                'lag_6': [history[-6]]
            })

            # Прогноз модели
            base_pred = model.predict(X_pred)[0]
            pred = adjust_price(base_pred, current_price)

            # Определение даты
            pred_date = dates[-1] + timedelta(weeks=1)

            # Логика расчёта множителя X
            if not stop_multiplier:
                if pred > prices[-1]:
                    current_multiplier += 1
                else:
                    stop_multiplier = True  # Как только цена идёт вверх — фиксируем X

            max_multiplier = max(max_multiplier, current_multiplier)

            dates.append(pred_date)
            prices.append(pred)

            # Обновляем историю
            history.pop(0)
            history.append(pred)

        # Формирование таблицы
        df_result = pd.DataFrame({
            "Дата": [d.strftime("%Y-%m-%d") for d in dates],
            "Прогноз цены": [round(p, 2) for p in prices]
        })

        st.subheader("Прогноз на 6 недель")
        st.table(df_result)
        st.line_chart(df_result.set_index('Дата')['Прогноз цены'])

        # Вывод итогового множителя X
        st.subheader("Итоговый уровень закупки")
        st.write(f"🔹 Уровень закупки: **{max_multiplier}X**")

    except Exception as e:
        st.error(f"Ошибка при выполнении прогноза: {str(e)}")
