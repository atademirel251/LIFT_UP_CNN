import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

from PIL import Image  # PNG kaydetmek için
import csv
tf.keras.mixed_precision.set_global_policy('float32')

# Giriş verileri (CSV dosyası)
input_csv_path = r"C:\Users\atade\Desktop\ahmet.csv"  # CSV dosyasını kullanıyoruz

# Çıktı klasörü
output_img_folder = r"C:\Users\atade\Desktop\gan_img_output"
os.makedirs(output_img_folder, exist_ok=True)

# CSV kaydetme için klasör
csv_file_path = r"C:\Users\atade\Desktop\input_images.csv"

# CSV dosyasından verileri yükleme
def load_csv_data(file_path):
    data = np.loadtxt(file_path, delimiter=',')  # CSV dosyasını yükle
    data = data.reshape((-1, 16, 16, 1))  # 16x16 boyutunda olacak şekilde şekil değiştir
    return data.astype(np.uint8)

# Görüntüleri yükle
images = load_csv_data(input_csv_path)

# Giriş resimlerini CSV dosyasına kaydetme
def save_images_to_csv(images, csv_file_path):
    with open(csv_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        for image in images:
            writer.writerow(image.flatten())  # 16x16'lık her resmi satır olarak yaz

# Giriş resimlerini CSV'ye kaydet
save_images_to_csv(images, csv_file_path)

# GAN Modeli
def build_generator(latent_dim):
    model = Sequential([
        layers.Dense(128 * 4 * 4, input_dim=latent_dim),
        layers.LeakyReLU(alpha=0.2),
        layers.Reshape((4, 4, 128)),
        layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same'),
        layers.LeakyReLU(alpha=0.2),
        layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same'),
        layers.LeakyReLU(alpha=0.2),
        layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')  # Grayscale ve 0-1 arasında çıktı
    ])
    return model

def build_discriminator(img_shape):
    model = Sequential([
        layers.Conv2D(64, (3, 3), strides=(2, 2), padding='same', input_shape=img_shape),
        layers.LeakyReLU(alpha=0.2),
        layers.Dropout(0.25),
        layers.Conv2D(128, (3, 3), strides=(2, 2), padding='same'),
        layers.LeakyReLU(alpha=0.2),
        layers.Dropout(0.25),
        layers.Flatten(),
        layers.Dense(1, activation='sigmoid')
    ])
    return model

def build_gan(generator, discriminator):
    discriminator.trainable = False
    model = Sequential([generator, discriminator])
    return model

# Model parametreleri
img_shape = (16, 16, 1)  # 16x16 grayscale görüntüler
latent_dim = 200

# Modelleri oluştur
discriminator = build_discriminator(img_shape)
discriminator.compile(loss='binary_crossentropy', optimizer=Adam(0.0001, 0.5), metrics=['accuracy'])

generator = build_generator(latent_dim)
gan = build_gan(generator, discriminator)
gan.compile(loss='binary_crossentropy', optimizer=Adam(0.0001, 0.5))

# Üretilen görüntüleri kaydetme fonksiyonu (PNG formatında)
def save_generated_images(generator, epoch, latent_dim, output_folder, examples=10):
    noise = np.random.normal(0, 1, (examples, latent_dim))
    generated_images = generator.predict(noise)  
    generated_images = (generated_images * 255).astype(np.uint8)  # 0-255 formatına getir

    for i in range(examples):
        img = generated_images[i].squeeze()  # (16, 16, 1) -> (16, 16)
        img_pil = Image.fromarray(img, mode='L')  # Grayscale (L modu)
        img_path = os.path.join(output_folder, f"gan_epoch_{epoch}_sample_{i}.png")
        img_pil.save(img_path)
    
    print(f"Üretilen görüntüler {output_folder} klasörüne kaydedildi.")

# Eğitim fonksiyonu
def train_gan(gan, generator, discriminator, images, latent_dim, epochs=20000, batch_size=8, save_interval=100):
    half_batch = batch_size // 2
    for epoch in range(epochs):
        idx = np.random.randint(0, images.shape[0], half_batch)
        real_images = images[idx]

        noise = np.random.normal(0, 1, (half_batch, latent_dim))
        fake_images = generator.predict(noise)

        d_loss_real = discriminator.train_on_batch(real_images, np.ones((half_batch, 1)))
        d_loss_fake = discriminator.train_on_batch(fake_images, np.zeros((half_batch, 1)))
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        valid_y = np.array([1] * batch_size)
        g_loss = gan.train_on_batch(noise, valid_y)

        print(f"{epoch} [D loss: {d_loss[0]}, acc.: {100*d_loss[1]}%] [G loss: {g_loss}]")

        # 100 epoch'da bir görüntüleri kaydet
        if epoch % save_interval == 0:
            save_generated_images(generator, epoch, latent_dim, output_img_folder)

# GAN'ı eğit
train_gan(gan, generator, discriminator, images, latent_dim, epochs=20000, batch_size=8, save_interval=100)