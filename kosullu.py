import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

# Görüntüleri ve S21 grafiklerini yükle
import os
import numpy as np
import tensorflow as tf
from natsort import natsorted  # Doğal sıralama için

def load_data(image_folder, s21_folder):
    images = []
    s21_values = []

    # Dosyaları doğal sıralama ile sırala
    image_files = natsorted([f for f in os.listdir(image_folder) if f.endswith(".png")], key=lambda x: x.lower())
    s21_files = natsorted([f for f in os.listdir(s21_folder) if f.endswith(".csv")], key=lambda x: x.lower())

    # Görüntü ve S21 dosyalarını eşleştir
    for img_file, s21_file in zip(image_files, s21_files):
        # Görüntüyü yükle ve ön işle
        img = tf.keras.preprocessing.image.load_img(os.path.join(image_folder, img_file))
        img = tf.keras.preprocessing.image.img_to_array(img)
        img = (img - 127.5) / 127.5  # [-1, 1] aralığına normalleştir
        images.append(img)

        # S21 grafiğini yükle
        s21_data = np.loadtxt(os.path.join(s21_folder, s21_file), delimiter=",", skiprows=1)
        
        # 2. sütundaki en küçük değeri (dip değer) ve karşılık gelen 1. sütun değerini bul
        dip_index = np.argmin(s21_data[:, 1])  # 2. sütundaki en küçük değerin indeksi
        dip_value = s21_data[dip_index, 0]  # 1. sütundaki karşılık gelen değer
        s21_values.append(dip_value)
    
    return np.array(images), np.array(s21_values)
# Koşullu Generator

# Generator modeli
def build_generator(latent_dim, num_classes):
    noise_input = layers.Input(shape=(latent_dim,))
    condition_input = layers.Input(shape=(num_classes,))
    
    x = layers.concatenate([noise_input, condition_input])
    x = layers.Dense(128 * 16 * 16)(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.Reshape((16, 16, 128))(x)
    x = layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same')(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same')(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.Conv2D(3, (3, 3), activation='tanh', padding='same')(x)  # (64, 64, 3) boyutunda çıktı
    
    model = models.Model([noise_input, condition_input], x)
    return model

# Discriminator modeli
def build_discriminator(img_shape, num_classes):
    img_input = layers.Input(shape=img_shape)
    condition_input = layers.Input(shape=(num_classes,))
    
    x = layers.Conv2D(64, (3, 3), strides=(2, 2), padding='same')(img_input)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.Dropout(0.25)(x)
    x = layers.Conv2D(128, (3, 3), strides=(2, 2), padding='same')(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.Dropout(0.25)(x)
    x = layers.Flatten()(x)
    
    condition = layers.Dense(128)(condition_input)
    condition = layers.LeakyReLU(alpha=0.2)(condition)
    x = layers.concatenate([x, condition])
    
    x = layers.Dense(1, activation='sigmoid')(x)
    
    model = models.Model([img_input, condition_input], x)
    return model

# GAN modeli
def build_gan(generator, discriminator):
    discriminator.trainable = False
    noise_input = layers.Input(shape=(latent_dim,))
    condition_input = layers.Input(shape=(num_classes,))
    generated_image = generator([noise_input, condition_input])
    validity = discriminator([generated_image, condition_input])
    model = models.Model([noise_input, condition_input], validity)
    return model

# Parametreler
latent_dim = 100
num_classes = 1
img_shape = (64, 64, 3)

# Modelleri oluştur
generator = build_generator(latent_dim, num_classes)
discriminator = build_discriminator(img_shape, num_classes)
gan = build_gan(generator, discriminator)

# Optimizer'lar
discriminator.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5), metrics=['accuracy'])
gan.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5))

# Eğitim
def train_gan(gan, generator, discriminator, images, s21_values, latent_dim, epochs=20000, batch_size=8, save_interval=500):
    half_batch = batch_size // 2
    for epoch in range(epochs):
        # Gerçek görüntüleri seç
        idx = np.random.randint(0, images.shape[0], half_batch)
        real_images = images[idx]
        real_conditions = s21_values[idx].reshape(-1, 1)
        real_labels = np.ones((half_batch, 1))

        # Sahte görüntüler üret
        noise = np.random.normal(0, 1, (half_batch, latent_dim))
        fake_conditions = np.random.uniform(low=np.min(s21_values), high=np.max(s21_values), size=(half_batch, 1))
        fake_images = generator.predict([noise, fake_conditions])
        fake_labels = np.zeros((half_batch, 1))

        # Discriminator'ı eğit
        d_loss_real = discriminator.train_on_batch([real_images, real_conditions], real_labels)
        d_loss_fake = discriminator.train_on_batch([fake_images, fake_conditions], fake_labels)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        # GAN'ı eğit
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        valid_y = np.ones((batch_size, 1))
        target_conditions = np.random.uniform(low=np.min(s21_values), high=np.max(s21_values), size=(batch_size, 1))
        g_loss = gan.train_on_batch([noise, target_conditions], valid_y)

        # İlerlemeyi yazdır
        print(f"{epoch} [D loss: {d_loss[0]}, acc.: {100*d_loss[1]}%] [G loss: {g_loss}]")

        # Belirli aralıklarla görüntüleri kaydet
        if epoch % save_interval == 0:
            save_images(generator, epoch, latent_dim, output_folder=output_folder)

# Görüntüleri kaydetme fonksiyonu
def save_images(generator, epoch, latent_dim, output_folder, examples=10):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    noise = np.random.normal(0, 1, (examples, latent_dim))
    target_conditions = np.full((examples, 1), 10.0)  # Örnek koşul (S21 dip değeri)
    generated_images = generator.predict([noise, target_conditions])
    generated_images = 0.5 * generated_images + 0.5  # [0, 1] aralığına ölçeklendir

    for i in range(examples):
        plt.figure()
        plt.imshow(generated_images[i], interpolation='nearest')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, f"gan_epoch_{epoch}_sample_{i}.png"))
        plt.close()

# Eğitimi başlat
output_folder = r"C:\Users\atade\Desktop\img_Eren"
train_gan(gan, generator, discriminator, images, s21_values, latent_dim, epochs=20000, batch_size=8, save_interval=500)