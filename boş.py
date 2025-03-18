import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

# Görüntülerin ve CSV dosyalarının bulunduğu klasör
image_folder = r"C:\Users\atade\Desktop\9683_veri\input_Resim"
csv_folder = r"C:\Users\atade\Desktop\9683_veri\csv"

# CSV dosyalarını oku ve koşul frekansını hesapla
def load_condition_from_csv(csv_folder):
    conditions = []
    for csv_file in os.listdir(csv_folder):
        if csv_file.endswith(".csv"):
            # CSV dosyasını oku (başlık satırı yok)
            df = pd.read_csv(os.path.join(csv_folder, csv_file), header=None)
            # 2. sütunun en dip değerine karşılık gelen 1. sütun değerini al
            # Verileri sayısal olarak işle
            df = df.apply(pd.to_numeric, errors='coerce')  # Sayısal olmayan değerleri NaN yap
            df = df.dropna()  # NaN değerleri temizle
            if not df.empty:  # Eğer dosya boş değilse
                condition_freq = df.iloc[df.iloc[:, 1].idxmin(), 0]
                # Koşul frekansını 8-10 arasında yuvarla
                condition_freq = np.round(condition_freq / 2) * 2  # 8, 10, 12 gibi yuvarla
                condition_freq = np.clip(condition_freq, 8, 10)  # 8-10 arasında sınırla
                conditions.append(condition_freq)
    return np.array(conditions)

# Koşul frekanslarını yükle
conditions = load_condition_from_csv(csv_folder)
print(f"Toplam {len(conditions)} CSV dosyası okundu.")

# Görüntüleri yükleme ve ön işleme
def load_images(image_folder, img_size=(64, 64)):
    images = []
    for filename in os.listdir(image_folder):
        if filename.endswith(".png"):
            img = load_img(os.path.join(image_folder, filename))
            img = img_to_array(img)
            # Görüntünün sağ alt çeyreğini al
            height, width, _ = img.shape
            img = img[height//2:, width//2:, :]
            # Görüntüyü 64x64 boyutuna yeniden boyutlandır
            img = tf.image.resize(img, img_size)
            img = (img - 127.5) / 127.5  # Normalize to [-1, 1]
            images.append(img)
    return np.array(images)

# Görüntüleri yükle
images = load_images(image_folder)

# Koşullu GAN Modeli
def build_generator(latent_dim, condition_dim):
    model = Sequential()
    model.add(layers.Dense(128 * 16 * 16, input_dim=latent_dim + condition_dim))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Reshape((16, 16, 128)))
    model.add(layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Conv2D(3, (3, 3), activation='tanh', padding='same'))
    return model

def build_discriminator(img_shape, condition_dim):
    input_img = layers.Input(shape=img_shape)
    input_cond = layers.Input(shape=(condition_dim,))

    # Görüntü işleme
    x = layers.Conv2D(64, (3, 3), strides=(2, 2), padding='same')(input_img)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.Dropout(0.25)(x)
    x = layers.Conv2D(128, (3, 3), strides=(2, 2), padding='same')(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.Dropout(0.25)(x)
    x = layers.Flatten()(x)

    # Koşul bilgisini ekle
    cond = layers.Dense(128)(input_cond)
    cond = layers.LeakyReLU(alpha=0.2)(cond)
    combined = layers.concatenate([x, cond])

    # Çıkış katmanı
    validity = layers.Dense(1, activation='sigmoid')(combined)

    model = Model([input_img, input_cond], validity)
    return model

def build_gan(generator, discriminator):
    discriminator.trainable = False
    noise = layers.Input(shape=(latent_dim,))
    condition = layers.Input(shape=(condition_dim,))
    img = generator([noise, condition])
    validity = discriminator([img, condition])
    model = Model([noise, condition], validity)
    return model

# Model parametreleri
img_shape = images.shape[1:]
latent_dim = 100
condition_dim = 1  # Koşul boyutu (frekans)

# Modelleri oluştur
generator = build_generator(latent_dim, condition_dim)
discriminator = build_discriminator(img_shape, condition_dim)
discriminator.compile(loss='binary_crossentropy', optimizer=Adam(0.0001, 0.5), metrics=['accuracy'])

gan = build_gan(generator, discriminator)
gan.compile(loss='binary_crossentropy', optimizer=Adam(0.0001, 0.5))

# Eğitim fonksiyonu
def train_gan(gan, generator, discriminator, images, conditions, latent_dim, epochs=10000, batch_size=32, save_interval=500):
    for epoch in range(epochs):
        # Gerçek görüntüleri seç
        idx = np.random.randint(0, images.shape[0], batch_size)
        real_images = images[idx]
        real_conditions = conditions[idx].reshape(-1, 1)  # Koşul bilgisi

        # Sahte görüntüler üret
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        fake_conditions = conditions[idx].reshape(-1, 1)  # Koşul bilgisi
        fake_images = generator.predict([noise, fake_conditions])

        # Discriminator'ı eğit
        d_loss_real = discriminator.train_on_batch([real_images, real_conditions], np.ones((batch_size, 1)))
        d_loss_fake = discriminator.train_on_batch([fake_images, fake_conditions], np.zeros((batch_size, 1)))
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        # GAN'ı eğit
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        valid_y = np.array([1] * batch_size)
        g_loss = gan.train_on_batch([noise, real_conditions], valid_y)

        # İlerlemeyi yazdır
        print(f"{epoch} [D loss: {d_loss[0]}, acc.: {100*d_loss[1]}%] [G loss: {g_loss}]")

        # Belirli aralıklarla görüntüleri kaydet
        if epoch % save_interval == 0:
            save_images(generator, epoch, latent_dim, conditions)

def save_images(generator, epoch, latent_dim, conditions, examples=10):
    noise = np.random.normal(0, 1, (examples, latent_dim))
    sample_conditions = conditions[:examples].reshape(-1, 1)  # Koşul bilgisi
    generated_images = generator.predict([noise, sample_conditions])
    generated_images = 0.5 * generated_images + 0.5  # Rescale to [0, 1]

    # Her bir görüntüyü ayrı ayrı kaydet
    for i in range(examples):
        plt.figure()
        plt.imshow(generated_images[i], interpolation='nearest')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(f"gan_conditioned_epoch_{epoch}_sample_{i}.png")
        plt.close()

# GAN'ı eğit
train_gan(gan, generator, discriminator, images, conditions, latent_dim, epochs=20000, batch_size=8, save_interval=500)