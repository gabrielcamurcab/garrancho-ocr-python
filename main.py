import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

# Passo 1: Carregar o conjunto de dados MNIST
(x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()

# Passo 2: Normalizar as imagens para o intervalo de 0 a 1
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Passo 3: Ajustar a forma das entradas
x_train = x_train.reshape((x_train.shape[0], 28, 28, 1))
x_test = x_test.reshape((x_test.shape[0], 28, 28, 1))

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

# Passo 6: Treinar o modelo e armazenar o histórico
history = model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))

# Passo 7: Avaliar o modelo
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Test accuracy: {test_acc}")

# Passo 8: Fazer previsões
predictions = model.predict(x_test)
predicted_classes = tf.argmax(predictions, axis=1)

# Passo 9: Exibir as imagens de teste e previsões
def plot_predictions_paginated(images, labels, predictions, images_per_page=10):
    num_images = len(images)
    num_pages = (num_images + images_per_page - 1) // images_per_page  # Calcula o número de páginas
    current_page = 0

    while True:
        plt.figure(figsize=(10, 5))
        start_idx = current_page * images_per_page
        end_idx = min(start_idx + images_per_page, num_images)

        for i in range(start_idx, end_idx):
            plt.subplot(2, 5, i - start_idx + 1)
            plt.imshow(images[i].reshape(28, 28), cmap='gray')
            plt.title(f"Pred: {predictions[i].numpy()}\nTrue: {labels[i]}")
            plt.axis('off')

        plt.tight_layout()
        plt.show()

        if current_page < num_pages - 1:
            next_page = input("Press Enter to go to the next page, or 'q' to quit: ")
            if next_page.lower() == 'q':
                break
            current_page += 1
        else:
            print("You are on the last page. Exiting.")
            break

# Chame a função para exibir as previsões paginadas
plot_predictions_paginated(x_test, y_test, predicted_classes, images_per_page=10)

# Passo 10: Gerar gráficos de acurácia e perda
def plot_history(history):
    plt.figure(figsize=(12, 5))
    
    # Gráfico de perda
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Perda de Treinamento')
    plt.plot(history.history['val_loss'], label='Perda de Validação')
    plt.title('Perda durante o Treinamento')
    plt.xlabel('Épocas')
    plt.ylabel('Perda')
    plt.legend()
    
    # Gráfico de acurácia
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Acurácia de Treinamento')
    plt.plot(history.history['val_accuracy'], label='Acurácia de Validação')
    plt.title('Acurácia durante o Treinamento')
    plt.xlabel('Épocas')
    plt.ylabel('Acurácia')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

plot_history(history)