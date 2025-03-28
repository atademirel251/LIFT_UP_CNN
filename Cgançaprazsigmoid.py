import os
import numpy as np
import tensorflow as tf
from keras import layers
from keras.utils import load_img, img_to_array
from keras.models import Model
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import pandas as pd
from natsort import natsorted
import tensorflow.keras.backend as K


def diagonal_symmetry_loss(y_pred):
    """
    Üretilen çeyrek desenin diagonal (ana köşegen) boyunca simetrik olmasını sağlamak için loss fonksiyonu.
    """
    # Çeyrek görüntünün boyutlarını al
    height, width, channels = K.int_shape(y_pred)[1:]  # Use K.int_shape for symbolic tensors

    # Transpose edilmiş (diagonal simetri yansıtılmış) versiyon
    y_pred_transposed = K.permute_dimensions(y_pred, (0, 2, 1, 3))  # x <-> y değişimi

    # Simetri farkını hesapla
    symmetry_difference = K.abs(y_pred - y_pred_transposed)

    # Ortalamayı al (küçük değerler daha simetrik demek)
    symmetry_loss = K.mean(symmetry_difference)  # Remove axis argument for simplicity
    
    return symmetry_loss  # Return scalar value


def load_images_and_frequencies(image_folder, csv_folder, img_size=(64, 64)):
    images = []
    frequencies = []
    
    # CSV dosyalarını doğal sıralama ile oku
    csv_files = natsorted([f for f in os.listdir(csv_folder) if f.endswith(".csv")])
    
    for csv_file in csv_files:
        # CSV'den frekansı oku ve integer'a yuvarla
        csv_path = os.path.join(csv_folder, csv_file)
        df = pd.read_csv(csv_path)
        
        if df.shape[1] >= 2:
            min_dB_index = df.iloc[:, 1].idxmin()
            freq_value = df.iloc[min_dB_index, 0]
            frequencies.append(int(round(freq_value)))  # Yuvarlama ve integer dönüşüm
        else:
            print(f"Hatalı CSV formatı: {csv_file}. Atlanıyor.")
    
    # Resimleri aynı sırayla yükle
    image_files = natsorted([f for f in os.listdir(image_folder) if f.endswith((".png", ".jpg"))])
    
    for img_file in image_files:
        img_path = os.path.join(image_folder, img_file)
        img = load_img(img_path)
        img = img_to_array(img)
        height, width, _ = img.shape
        img = img[height//2:, width//2:, :]  # Crop the image
        img = tf.image.resize(img, img_size)  # Resize the image
        img = (img - 127.5) / 127.5  # Normalize to [-1, 1]
        images.append(img)
    
    print(f"Yüklenen resim sayısı: {len(images)}, frekans sayısı: {len(frequencies)}")
    
    # Resim ve CSV sayısı eşit mi kontrol et
    if len(images) != len(frequencies):
        raise ValueError(f"Resim sayısı ({len(images)}) ve CSV sayısı ({len(frequencies)}) eşit değil!")
    
    print(f"Örnek frekans değerleri: {np.unique(frequencies)}")  # Benzersiz frekansları göster
    return np.array(images), np.array(frequencies)

# Load images and frequencies
image_folder = r"C:\Users\atade\Desktop\10440_veri\input_Resim"
csv_folder = r"C:\Users\atade\Desktop\10440_veri\csv"
images, frequencies = load_images_and_frequencies(image_folder, csv_folder)

# Normalize frequencies to [-1, 1]
frequencies = (frequencies - np.mean(frequencies)) / np.std(frequencies)

from tensorflow.keras import layers, Model

def build_generator(latent_dim):
    noise = layers.Input(shape=(latent_dim,))
    freq = layers.Input(shape=(1,))
    merged = layers.Concatenate()([noise, freq])

    # 8x8x256 başlangıç boyutu (daha iyi detay için)
    x = layers.Dense(256 * 8 * 8)(merged)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.BatchNormalization(momentum=0.8)(x)
    x = layers.Reshape((8, 8, 256))(x)

    # 8x8 -> 16x16
    x = layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same')(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.BatchNormalization(momentum=0.8)(x)

    # 16x16 -> 32x32
    x = layers.Conv2DTranspose(64, (4, 4), strides=(2, 2), padding='same')(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.BatchNormalization(momentum=0.8)(x)

    # 32x32 -> 64x64 (son çıktı boyutu)
    x = layers.Conv2DTranspose(32, (4, 4), strides=(2, 2), padding='same')(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.BatchNormalization(momentum=0.8)(x)

    # Tek kanallı binary çıktı (1 channel, sigmoid)
    x = layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)
    
    return Model([noise, freq], x, name="Generator")



# Daha az parametreli versiyon
from tensorflow.keras import layers, Model

def build_discriminator(img_shape=(64, 64, 1)):
    img = layers.Input(shape=img_shape)
    freq = layers.Input(shape=(1,))
    
    # 1. Görüntüyü işleme (64x64x1 = 4096)
    x = layers.Flatten()(img)
    
    # 2. Frekans bilgisini EŞİT BOYUTTA genişlet (4096)
    freq_dense = layers.Dense(4096)(freq)  # Kritik nokta!
    freq_dense = layers.LeakyReLU(alpha=0.2)(freq_dense)
    
    # 3. Birleştirme (4096 + 4096 = 8192)
    merged = layers.Concatenate()([x, freq_dense])
    
    # 4. Dense katmanını GİRİŞ BOYUTUNA UYGUN şekilde ayarla
    x = layers.Dense(8192)(merged)  # 8192 giriş, 8192 çıkış
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.Dropout(0.4)(x)
    
    # 5. Çıktı katmanı
    x = layers.Dense(1, activation='sigmoid')(x)
    
    return Model([img, freq], x, name="Discriminator")



# cGAN Modeli
def build_gan(generator, discriminator, latent_dim, lambda_symmetry=10):
    discriminator.trainable = False  # Discriminator'ü dondur
    
    noise = layers.Input(shape=(latent_dim,))
    freq = layers.Input(shape=(1,))
    img = generator([noise, freq])
    valid = discriminator([img, freq])

    # Simetri kaybı hesaplama
    symmetry_penalty = diagonal_symmetry_loss(img)  
    valid_loss = K.mean(K.binary_crossentropy(K.ones_like(valid), valid))  

    # Toplam kaybı hesapla
    total_loss = valid_loss + lambda_symmetry * symmetry_penalty

    # Modeli oluştur ve derle
    gan = Model([noise, freq], valid)
    gan.add_loss(total_loss)
    gan.compile(optimizer=Adam(0.0001, 0.5))  

    return gan

# Model parameters
img_shape = images.shape[1:]  # Görüntü boyutları
latent_dim = 150  # Latent dim boyutu

# Build models
discriminator = build_discriminator(img_shape)
discriminator.compile(loss='binary_crossentropy', optimizer=Adam(0.0001, 0.5), metrics=['accuracy'])

generator = build_generator(latent_dim)
gan = build_gan(generator, discriminator, latent_dim)
gan.compile(loss='binary_crossentropy', optimizer=Adam(0.0001, 0.5))

# Training function
def train_gan(gan, generator, discriminator, images, frequencies, latent_dim, epochs=3000, batch_size=8, save_interval=500):
    half_batch = batch_size // 2

    for epoch in range(epochs):
        # Gerçek veri seç
        idx = np.random.randint(0, images.shape[0], half_batch)
        real_images = images[idx]
        real_frequencies = frequencies[idx]
        
        # Fake veri üret
        noise = np.random.normal(0, 1, (half_batch, latent_dim))
        fake_frequencies = np.random.choice(frequencies, half_batch)
        fake_images = generator.predict([noise, fake_frequencies])

        # Discriminator eğitimi
        d_loss_real = discriminator.train_on_batch([real_images, real_frequencies], np.ones((half_batch, 1)))
        d_loss_fake = discriminator.train_on_batch([fake_images, fake_frequencies], np.zeros((half_batch, 1)))
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        # Generator eğitimi (Burada valid_y kullanılmaz çünkü gan.add_loss zaten kaybı içeriyor)
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        sampled_frequencies = np.random.choice(frequencies, batch_size)
        g_loss = gan.train_on_batch([noise, sampled_frequencies])

        # Durumu ekrana yazdır
        print(f"{epoch} [D loss: {d_loss[0]:.4f}, acc.: {100 * d_loss[1]:.2f}%] [G loss: {g_loss:.4f}]")

        # Görselleri kaydet
        if epoch % save_interval == 0:
            save_images(generator, epoch, latent_dim, frequencies)

def save_images(generator, epoch, latent_dim, frequencies, examples=10):
    noise = np.random.normal(0, 1, (examples, latent_dim))
    sampled_frequencies = np.random.choice(frequencies, examples)
    generated_images = generator.predict([noise, sampled_frequencies])
    generated_images = (generated_images * 255).astype(np.uint8)  # 0-255 aralığına çevir

    for i in range(examples):
        img = generated_images[i, :, :, 0]  # Tek kanal olduğu için [:,:,0] alınır
        img_flipped_x = np.flip(img, axis=1)
        top_row = np.concatenate((img, img_flipped_x), axis=1)
        bottom_row = np.flip(top_row, axis=0)
        pattern_image = np.concatenate((top_row, bottom_row), axis=0)

        plt.figure()
        plt.imshow(pattern_image, cmap='gray', interpolation='nearest')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(f"CCDgan_patternSON_epoch_{epoch}_sample_{i}.png", cmap='gray')
        plt.close()


# Train GAN model
train_gan(gan, generator, discriminator, images, frequencies, latent_dim)
generator.save("generator_model3.h5")