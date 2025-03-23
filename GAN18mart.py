from tensorflow.keras.models import load_model
import numpy as np
import os
import pandas as pd
import numpy as np
from tensorflow.keras import layers, models, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Dense, Dropout, Flatten, Concatenate
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import GlobalAveragePooling2D
from PIL import Image
from natsort import natsorted 

import os
import pandas as pd
import numpy as np

# CSV dosyalarını oku
def load_csv_data(folder_path):
    data = []
    # Dosyaları doğal sıralama ile sırala
    csv_files = natsorted(
        [f for f in os.listdir(folder_path) if f.endswith('.csv')],
        key=lambda x: x.lower()  # Büyük/küçük harf duyarlılığını kaldır
    )
    for file_name in csv_files:
        file_path = os.path.join(folder_path, file_name)
        df = pd.read_csv(file_path, header=None)
        df = df.apply(pd.to_numeric, errors='coerce')  # Hatalı verileri NaN yap
        df = df.dropna()  # NaN değerleri temizle
        data.append(df.values)
    return np.array(data)

def load_images_from_folder(folder_path):
    images = []
    # Dosyaları doğal sıralama ile sırala
    image_files = natsorted(
        [f for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg'))],
        key=lambda x: x.lower()  # Büyük/küçük harf duyarlılığını kaldır
    )
    
    for file_name in image_files:
        file_path = os.path.join(folder_path, file_name)
        
        # Görseli aç
        image = Image.open(file_path).convert('RGB')  # RGB formatına çevir
        
        # Orijinal genişlik ve yükseklik
        w, h = image.size
        
        # Sol üst çeyrek bölgeyi al
        quarter_image = image.crop((0, 0, w // 2, h // 2))
        
        # Çeyrek bölgeyi 64x64 boyutuna getir
        resized_image = quarter_image.resize((64, 64))
        
        # NumPy array'e çevir ve normalleştir
        image_array = np.array(resized_image) / 255.0  # [0, 1] aralığına normalleştir
        images.append(image_array)
    
    return np.array(images)

pattern_folder = r"C:\Users\atade\Desktop\10440_veri\input_Resim"
patterns = load_images_from_folder(pattern_folder)
# Desenleri ve S21 grafiklerini yükle
csv_folder = r"C:\Users\atade\Desktop\10440_veri\csv" # CSV dosyalarının bulunduğu klasör
s21_data = load_csv_data(csv_folder)

# Veriyi normalize et (örneğin, [-1, 1] aralığına)
s21_data = (s21_data - np.min(s21_data)) / (np.max(s21_data) - np.min(s21_data)) * 2 - 1



# Eğitim verilerini hazırla
X_train = patterns  # Desenler (64x64x3)
y_train = s21_data  # S21 grafikleri (202 boyutlu vektör)
print(f"X_train boyutu: {X_train.shape}")
print(f"y_train boyutu: {y_train.shape}")
# Verileri eğitim ve test setlerine ayır
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)








def weighted_loss(y_true, y_pred):
    error = K.abs(y_true - y_pred)
    weight = K.exp(-0.07 * K.abs(y_true))
    dy_dx_true = K.abs(y_true[:, 1:] - y_true[:, :-1])
    dy_dx_pred = K.abs(y_pred[:, 1:] - y_pred[:, :-1])
    gradient_penalty = K.mean(K.abs(dy_dx_true - dy_dx_pred))
    loss = K.mean(weight * error) + (0.2 * gradient_penalty)
    return loss

# VGG16 modelini yükleyelim
model = load_model(
    "C:/Users/atade/Desktop/test_sonuçları/VGG16+TEST/model/Yeni7231_64x64+15katman+ınterpolatıon7.keras",
    custom_objects={'weighted_loss': weighted_loss}
)

from tensorflow.keras import layers, models, optimizers
import numpy as np

# Üretici (Generator) Modeli
def build_generator(latent_dim):
    model = models.Sequential([
        layers.Dense(128 * 16 * 16, input_dim=latent_dim),  # 16x16 feature map
        layers.Reshape((16, 16, 128)),
        layers.Conv2DTranspose(128, kernel_size=4, strides=2, padding='same'),  # 32x32
        layers.BatchNormalization(),
        layers.LeakyReLU(alpha=0.2),
        layers.Conv2DTranspose(64, kernel_size=4, strides=2, padding='same'),  # 64x64
        layers.BatchNormalization(),
        layers.LeakyReLU(alpha=0.2),
        layers.Conv2DTranspose(3, kernel_size=4, strides=1, padding='same', activation='tanh')  # 64x64x3
    ])
    return model

# Ayırt Edici (Discriminator) Modeli
def build_discriminator(input_shape):
    model = models.Sequential([
        layers.Conv2D(64, kernel_size=4, strides=2, padding='same', input_shape=input_shape),  # 32x32
        layers.LeakyReLU(alpha=0.2),
        layers.Conv2D(128, kernel_size=4, strides=2, padding='same'),  # 16x16
        layers.LeakyReLU(alpha=0.2),
        layers.Conv2D(256, kernel_size=4, strides=2, padding='same'),  # 8x8
        layers.LeakyReLU(alpha=0.2),
        layers.Flatten(),
        layers.Dense(1, activation='sigmoid')  # Gerçek veya sahte
    ])
    return model


# GAN Modeli
def build_gan(generator, discriminator):
    discriminator.trainable = False  # Ayırt ediciyi dondur
    model = models.Sequential([generator, discriminator])
    return model

# Parametreler
latent_dim = 100  # Latent uzay boyutu
input_shape = (64, 64, 3)  # Desenlerin boyutu

# Modelleri oluştur
generator = build_generator(latent_dim)
discriminator = build_discriminator(input_shape)
gan = build_gan(generator, discriminator)

# Optimizer'lar
discriminator.compile(optimizer=optimizers.Adam(learning_rate=0.0002, beta_1=0.5), loss='binary_crossentropy')
gan.compile(optimizer=optimizers.Adam(learning_rate=0.0002, beta_1=0.5), loss='binary_crossentropy')

# Eğitim parametreleri
epochs = 5000
batch_size = 8
half_batch = batch_size // 2

# Eğitim döngüsü
for epoch in range(epochs):
    # Gerçek desenleri seç
    idx = np.random.randint(0, X_train.shape[0], half_batch)
    real_patterns = X_train[idx]
    real_labels = np.ones((half_batch, 1))  # Gerçek desenler için etiket: 1

    # Sahte desenler üret
    noise = np.random.normal(0, 1, (half_batch, latent_dim))
    fake_patterns = generator.predict(noise)
    fake_labels = np.zeros((half_batch, 1))  # Sahte desenler için etiket: 0

    # Ayırt ediciyi eğit
    d_loss_real = discriminator.train_on_batch(real_patterns, real_labels)
    d_loss_fake = discriminator.train_on_batch(fake_patterns, fake_labels)
    d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

    # Üreticiyi eğit
    noise = np.random.normal(0, 1, (batch_size, latent_dim))
    valid_labels = np.ones((batch_size, 1))  # GAN için etiket: 1
    g_loss = gan.train_on_batch(noise, valid_labels)

    # İlerlemeyi yazdır
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, D Loss: {d_loss}, G Loss: {g_loss}")





        # Dip noktası için özgün desenler üret
def generate_patterns_for_dip(generator, latent_dim, n_samples):
    noise = np.random.normal(0, 1, (n_samples, latent_dim))
    generated_patterns = generator.predict(noise)
    return generated_patterns

# 10 özgün desen üret
n_samples = 10
dip_patterns = generate_patterns_for_dip(generator, latent_dim, n_samples)

# Desenleri görselleştir
import matplotlib.pyplot as plt
for i, pattern in enumerate(dip_patterns):
    plt.figure()
    plt.imshow((pattern + 1) / 2)  # [-1, 1] aralığını [0, 1] aralığına dönüştür
    plt.title(f"Generated Pattern {i+1} for Dip Point")
    plt.show()