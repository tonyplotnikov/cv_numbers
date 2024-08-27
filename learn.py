import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Flatten

# Шаг 1: Загрузка и подготовка данных
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train / 255.0
x_test = x_test / 255.0

model = Sequential([
    Flatten(input_shape=(28, 28)),       # Преобразуем изображение 28x28 в вектор
    Dense(128, activation='relu'),       # Полносвязный слой с 128 нейронами и активацией ReLU
    Dense(10, activation='softmax')      # Выходной слой с 10 нейронами для каждой цифры (0-9)
])

# Шаг 3: Компиляция модели
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Шаг 4: Обучение модели
model.fit(x_train, y_train, epochs=5, validation_split=0.1)

# Шаг 5: Оценка модели
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"\nTest accuracy: {test_acc:.4f}")

# Шаг 6: Сохранение модели
model.save('mnist_model.h5')

# Пример загрузки модели (не обязательно сразу использовать)
# model = load_model('mnist_model.h5')
