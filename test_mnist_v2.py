import os
import numpy as np
from PIL import Image
from tensorflow import keras
from tensorflow.keras.preprocessing import image # type: ignore
import matplotlib.pyplot as plt

def resize_images(images, target_size=(28, 28)):
    """Redimensiona uma lista de imagens para o tamanho especificado."""
    resized_images = []
    for img in images:
        if img.ndim == 2:  # Se a imagem é 2D
            img = np.expand_dims(img, axis=-1)  # Adiciona uma dimensão para o canal
        img = (img * 255).astype(np.uint8)
        img_resized = Image.fromarray(img.squeeze()).resize(target_size)
        resized_images.append(np.array(img_resized))
    return np.array(resized_images)

def load_and_preprocess_image(img_path):
    """Carrega e processa uma imagem."""
    img = image.load_img(img_path, target_size=(64, 64), color_mode='grayscale')  # Carrega com 64x64
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0  # Normaliza a imagem
    return img_array

# Diretório onde estão as imagens de teste
test_images_dir = 'dataset/test_images_v2'

test_images = []

# Carrega e processa as imagens
for img_file in os.listdir(test_images_dir):
    img_path = os.path.join(test_images_dir, img_file)
    img_array = load_and_preprocess_image(img_path)
    test_images.append(img_array)

test_images = np.array(test_images)

# Redimensiona as imagens para 28x28
test_images_resized = resize_images(test_images)

# Carrega o modelo treinado
model = keras.models.load_model('models/garrancho_mnist_v2_model.keras')

# Faz previsões
predictions = model.predict(test_images_resized)

# Obtém as classes previstas
predicted_classes = np.argmax(predictions, axis=1)

# Visualiza as imagens e as previsões
num_images = len(test_images_resized)
cols = 4  # Definindo 4 colunas
rows = (num_images // cols) + (num_images % cols > 0)  # Calcula o número de linhas necessárias

plt.figure(figsize=(10, 10))
for i, img_file in enumerate(os.listdir(test_images_dir)):
    plt.subplot(rows, cols, i + 1)  # Configura as linhas e colunas
    plt.imshow(test_images_resized[i].reshape(28, 28), cmap='gray')
    plt.title(f'Previsão: {predicted_classes[i]}')
    plt.axis('off')

plt.tight_layout()  # Ajusta o layout para evitar sobreposição
plt.show()