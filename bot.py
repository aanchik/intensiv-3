import logging
import pandas as pd
import asyncio
from aiogram import Bot, Dispatcher, types
import numpy as np
from aiogram.filters import Command
from aiogram.types import Message
import joblib
import nbformat
from nbconvert import PythonExporter
from nbclient import NotebookClient

API_TOKEN = "7653341979:AAHAntVU1fqDOcjSlzqtDHOS0akt3LTYQuQ"
dp = Dispatcher()

# Включаем логирование
logging.basicConfig(level=logging.INFO)
# Создаем объекты бота и диспетчера
bot = Bot(token=API_TOKEN)
# Загружаем модель
model = joblib.load("LGBMRegressor.pkl")  # Замените на путь к вашей модели

def run_notebook(price):
    """Запускает main.ipynb с переданной ценой и получает прогноз."""
    with open("main.ipynb") as f:
        notebook = nbformat.read(f, as_version=4)
    feature_columns = ["Цена на арматуру", 0, 0, 0, 0, 0]
    input_features = np.zeros(len(feature_columns))
    input_features[0] = price  # Первым идет переданная цена 
    
    # Передаем цену как глобальную переменную
    client = NotebookClient(notebook)
    client.kernelspec_name = "python3"
    client.execute()
    
    # Предполагаем, что в конце main.ipynb сохраняет прогноз в файл
    df = pd.read_excel("predicted_procurement.xlsx")
    forecast_weeks = df.iloc[0]["Закупка_на_колво_недель"]
    return int(forecast_weeks)

@dp.message(Command("start"))  # ✅ Новый синтаксис
async def send_welcome(message: Message):
    """Обработчик команды /start"""
    await message.answer("Привет! Отправь мне текущую цену на арматуру, и я сделаю прогноз на количество недель для закупки.")

@dp.message()
async def handle_price(message: Message):
    """Обработчик отправленной цены"""
    try:
        price = float(message.text)

        # ✅ Создаем DataFrame с нужными признаками
        # Дополняем остальные признаки средними значениями или заглушками
        additional_features = np.zeros(len(feature_columns) - 1)  # 0 вместо неизвестных значений
        df = pd.DataFrame([[price] + list(additional_features)], columns=feature_columns)

        # ✅ Делаем прогноз
        forecast_weeks = int(model.predict(df)[0])
        await message.answer(f"Рекомендуем закупку на {forecast_weeks} недель.")

    except ValueError:
        await message.answer("Пожалуйста, отправьте корректную числовую цену на арматуру.")
async def main():
    """Функция для запуска бота"""
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
