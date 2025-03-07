import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import (Dense, Dropout, GlobalAveragePooling2D, Input, Reshape, Conv2D, 
                                     LeakyReLU, UpSampling2D, Embedding, Concatenate, Flatten)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

# GPU Bellek Yönetimi
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# Özel Kayıp Fonksiyonu (weighted_loss)
def weighted_loss(y_true, y_pred):
    error = K.abs(y_true - y_pred)
    weight = K.exp(-0.07 * K.abs(y_true))
    dy_dx_true = K.abs(y_true[:, 1:] - y_true[:, :-1])
    dy_dx_pred = K.abs(y_pred[:, 1:] - y_pred[:, :-1])
    gradient_penalty = K.mean(K.abs(dy_dx_true - dy_dx_pred))
    loss = K.mean(weight * error) + (0.2 * gradient_penalty)
    return loss

# Generator Modeli (CGAN)
def build_generator(latent_dim, num_classes):
    # Koşul girişi
    condition_input = Input(shape=(1,))
    condition = Dense(64, activation='relu')(condition_input)
    condition = Reshape((1, 1, 64))(condition)
    
    # Gürültü girişi
    noise_input = Input(shape=(latent_dim,))
    noise = Dense(8 * 8 * 256)(noise_input)
    noise = Reshape((8, 8, 256))(noise)
    
    # Koşul ve gürültüyü birleştir
    merged = Concatenate()([noise, condition])
    
    # CNN Katmanları
    x = UpSampling2D()(merged)
    x = Conv2D(128, kernel_size=3, padding='same')(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = UpSampling2D()(x)
    x = Conv2D(64, kernel_size=3, padding='same')(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Conv2D(1, kernel_size=3, padding='same', activation='sigmoid')(x)
    
    model = Model([noise_input, condition_input], x)
    return model

# Discriminator Modeli (CGAN)
def build_discriminator(model_path, num_classes):
    # Önceden eğitilmiş modeli yükle
    base_model = tf.keras.models.load_model(model_path, custom_objects={'weighted_loss': weighted_loss})
    base_model.trainable = False
    
    # Görüntü girişi
    image_input = Input(shape=(32, 32, 1))
    x = tf.image.resize(image_input, (64, 64))
    x = tf.tile(x, [1, 1, 1, 3])
    x = base_model(x)
    if len(x.shape) == 2:
        x = Reshape((1, 1, x.shape[1]))(x)
    x = GlobalAveragePooling2D()(x)
    
    # Koşul girişi
    condition_input = Input(shape=(1,))
    condition = Dense(64, activation='relu')(condition_input)
    condition = Reshape((1, 1, 64))(condition)
    
    # Görüntü ve koşulu birleştir
    merged = Concatenate()([x, condition])
    merged = Flatten()(merged)
    
    # Ek katmanlar
    x = Dense(128, activation='relu')(merged)
    x = Dropout(0.4)(x)
    outputs = Dense(1, activation='sigmoid')(x)
    
    model = Model([image_input, condition_input], outputs)
    return model

# GAN Modeli (CGAN)
def build_gan(generator, discriminator):
    discriminator.trainable = False
    noise_input = Input(shape=(latent_dim,))
    condition_input = Input(shape=(1,))
    generated_image = generator([noise_input, condition_input])
    validity = discriminator([generated_image, condition_input])
    model = Model([noise_input, condition_input], validity)
    return model

# CSV Dosyalarını Yükleme
def load_csv_data(csv_path):
    data = pd.read_csv(csv_path)
    conditions = data['frequency_range'].values  # Frekans aralığı sütunu
    return conditions

# Model Yolları ve Parametreler
model_path = 'C:/Users/atade/Desktop/test_sonuçları/VGG16+TEST/model/Yeni5500_64x64+15katman+ınterpolatıon7.keras'
csv_path = r"C:\Users\atade\Desktop\5000veri\csv" # CSV dosyası yolu
latent_dim = 100
num_classes = 10  # Frekans aralığı sayısı
epochs = 10000
batch_size = 16

# Verileri Yükle
conditions = load_csv_data(csv_path)
print(f"Yüklenen koşul sayısı: {len(conditions)}")

# Model İnşası
generator = build_generator(latent_dim, num_classes)
discriminator = build_discriminator(model_path, num_classes)
gan = build_gan(generator, discriminator)

# Optimizasyon ve Derleme
opt = Adam(learning_rate=0.0002, beta_1=0.5)
generator.compile(loss='binary_crossentropy', optimizer=opt)
discriminator.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
gan.compile(loss='binary_crossentropy', optimizer=opt)

# Eğitim Döngüsü
for epoch in range(epochs):
    # Gürültü (noise) ve koşul üret
    noise = np.random.normal(0, 1, (batch_size, latent_dim))
    condition = np.random.choice(conditions, batch_size).reshape(-1, 1)
    
    # Sahte görüntüler üret
    generated_samples = generator.predict([noise, condition])
    
    # Gerçek görüntüleri rastgele seç (bu örnekte koşullu değil)
    real_samples = np.random.rand(batch_size, 32, 32, 1)  # Örnek olarak rastgele görüntüler
    
    # Etiketler
    real_labels = np.ones((batch_size, 1))
    fake_labels = np.zeros((batch_size, 1))
    
    # Discriminator eğitimi
    d_loss_real = discriminator.train_on_batch([real_samples, condition], real_labels)
    d_loss_fake = discriminator.train_on_batch([generated_samples, condition], fake_labels)
    
    # Generator eğitimi
    g_loss = gan.train_on_batch([noise, condition], real_labels)
    
    # İlerlemeyi yazdır
    if epoch % 100 == 0:
        print(f"Epoch {epoch}/{epochs}, Discriminator Gerçek Kaybı: {d_loss_real[0]}, "
              f"Discriminator Sahte Kaybı: {d_loss_fake[0]}, Generator Kaybı: {g_loss}")
        
        # Görselleştirme
        if epoch % 500 == 0:
            generated_image = generated_samples[0, :, :, 0]
            plt.imshow(generated_image, cmap='gray')
            plt.title(f"Generated Image - Epoch {epoch}")
            plt.axis('off')
            plt.show()