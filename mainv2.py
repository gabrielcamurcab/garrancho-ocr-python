import tensorflow as tf
import os
import numpy as np
from tensorflow.keras.preprocessing import image  # type: ignore
from tensorflow.keras import datasets, layers, models  # type: ignore
from tensorflow.keras.optimizers import Adam  # type: ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
import matplotlib.pyplot as plt

def load_and_preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(28, 28), color_mode='grayscale')
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0  # Normalizando
    return img_array

# Pasta onde estão as novas imagens
new_images_dir = 'dataset/test_images'
new_images = []

for img_file in os.listdir(new_images_dir):
    img_path = os.path.join(new_images_dir, img_file)
    img_array = load_and_preprocess_image(img_path)
    new_images.append(img_array)

new_images = np.array(new_images)

# Passo 1: Carregar o conjunto de dados MNIST
(x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()

# Passo 2: Normalizar as imagens para o intervalo de 0 a 1
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Passo 3: Ajustar a forma das entradas
x_train = x_train.reshape((x_train.shape[0], 28, 28, 1))
x_test = x_test.reshape((x_test.shape[0], 28, 28, 1))

# Adicionar novas imagens ao conjunto de treinamento
# Assumindo que você já tem as labels para as novas imagens
# Exemplo: se suas novas imagens são dos dígitos 0 a 9, você deve criar as labels correspondentes
new_labels = []

for img_file in os.listdir(new_images_dir):
    # Supondo que o nome dos arquivos seja no formato 'numero_contador' (ex: '1_1', '2_3', etc.)
    label = int(img_file.split('_')[0])  # Extrai o número antes do primeiro underscore
    new_labels.append(label)

new_labels = np.array(new_labels)

# Passo 4: Combinar os dados
x_train = np.concatenate((x_train, new_images), axis=0)
y_train = np.concatenate((y_train, new_labels), axis=0)

datagen = ImageDataGenerator(
    rotation_range=10,       # Rotaciona as imagens em até 10 graus
    zoom_range=0.1,          # Aplica um zoom aleatório de até 10%
    width_shift_range=0.1,   # Desloca a imagem horizontalmente em até 10% do tamanho
    height_shift_range=0.1,  # Desloca a imagem verticalmente em até 10% do tamanho
    shear_range=0.1,         # Aplica uma transformação de cisalhamento
    fill_mode='nearest'      # Preenchimento das lacunas criadas pela transformação
)

# Passo 5: Criar o modelo
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# Passo 6: Compilar o modelo
model.compile(optimizer=Adam(learning_rate=0.0001), 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])

# Passo 7: Treinar o modelo e armazenar o histórico
datagen.fit(x_train)

history = model.fit(datagen.flow(x_train, y_train, batch_size=16), epochs=10, validation_data=(x_test, y_test))

# Passo 8: Avaliar o modelo
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Test accuracy: {test_acc}")

# Passo 9: Fazer previsões
predictions = model.predict(x_test)
predicted_classes = tf.argmax(predictions, axis=1)

model.save('models/garrancho_mnist_v2_model.keras')

# Passo 10: Exibir as imagens de teste e previsões
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

# Passo 11: Gerar gráficos de acurácia e perda
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