import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

# Passo 1: Carregar o conjunto de dados MNIST
(x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()

# Passo 2: Normalizar as imagens para o intervalo de 0 a 1
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Passo 3: Visualizar algumas imagens do conjunto de dados
plt.figure(figsize=(10, 10))
for i in range(9):
    plt.subplot(3, 3, i + 1)
    plt.imshow(x_train[i], cmap='gray')
    plt.title(f"Label: {y_train[i]}")
    plt.axis('off')
plt.show()

# Passo 4: Criar o modelo
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# Passo 5: Compilar o modelo
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Passo 6: Ajustar a forma das entradas
x_train = x_train.reshape((x_train.shape[0], 28, 28, 1))
x_test = x_test.reshape((x_test.shape[0], 28, 28, 1))

# Passo 7: Treinar o modelo
model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))

# Passo 8: Avaliar o modelo
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Test accuracy: {test_acc}")

# Passo 9: Fazer previsões
predictions = model.predict(x_test)
predicted_classes = tf.argmax(predictions, axis=1)

# Passo 10: Exibir uma imagem de teste e sua previsão
plt.imshow(x_test[0].reshape(28, 28), cmap='gray')
plt.title(f"Predicted: {predicted_classes[0].numpy()}")
plt.axis('off')
plt.show()