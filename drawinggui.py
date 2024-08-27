import tkinter as tk
from PIL import Image, ImageDraw
import numpy as np
from tensorflow.keras.models import load_model

# Загрузка обученной модели
# Убедись, что у тебя есть обученная модель. Если модель сохранена в файл, можно её загрузить так:
model = load_model('mnist_model.h5')
# Если модель создавалась и обучалась ранее в этом скрипте, то пропускаем загрузку.


# Параметры окна
window_size = 280  # Размер окна (10x увеличение)
canvas_size = 28  # Размер холста (28x28 пикселей)
brush_size = 10  # Размер кисти

# Создание окна
root = tk.Tk()
root.title("Рисование цифры")

# Переменные для хранения изображения и рисования
image = Image.new("L", (canvas_size, canvas_size), color=0)
draw = ImageDraw.Draw(image)

# Функция для обработки рисования на холсте
def paint(event):
    x, y = event.x // 10, event.y // 10  # Преобразуем координаты в 28x28
    draw.rectangle([x, y, x + brush_size // 10, y + brush_size // 10], fill=255)
    canvas.create_oval(event.x - brush_size, event.y - brush_size, event.x + brush_size, event.y + brush_size, fill="white", width=0)
    predict_digit()

# Функция для предсказания цифры
def predict_digit():
    img_array = np.array(image)
    img_array = img_array / 255.0  # Нормализация
    img_array = img_array.reshape(1, 28, 28)  # Преобразование для подачи в модель
    prediction = model.predict(img_array)
    predicted_label = np.argmax(prediction)
    label.config(text=f"Предсказано: {predicted_label}")

# Функция для очистки холста
def clear_canvas():
    global image, draw
    canvas.delete("all")
    image = Image.new("L", (canvas_size, canvas_size), color=0)
    draw = ImageDraw.Draw(image)
    label.config(text="Предсказано: ")

# Создание холста для рисования
canvas = tk.Canvas(root, width=window_size, height=window_size, bg="black")
canvas.pack()

# Привязка событий мыши к холсту
canvas.bind("<B1-Motion>", paint)

# Добавление кнопки для очистки холста
clear_button = tk.Button(root, text="Очистить", command=clear_canvas)
clear_button.pack()

# Метка для вывода предсказания
label = tk.Label(root, text="Предсказано: ", font=("Helvetica", 24))
label.pack()

# Запуск интерфейса
root.mainloop()
