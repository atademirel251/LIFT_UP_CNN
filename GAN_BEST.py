import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import cv2

# Görüntülerin bulunduğu klasör
image_folder = r"C:\Users\atade\Desktop\9683_veri\input_Resim"



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

# GAN Modeli
def build_generator(latent_dim):
    model = Sequential()
    model.add(layers.Dense(128 * 16 * 16, input_dim=latent_dim))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Reshape((16, 16, 128)))
    model.add(layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Conv2D(3, (3, 3), activation='tanh', padding='same'))
    return model

def build_discriminator(img_shape):
    model = Sequential()
    model.add(layers.Conv2D(64, (3, 3), strides=(2, 2), padding='same', input_shape=img_shape))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Dropout(0.25))
    model.add(layers.Conv2D(128, (3, 3), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Dropout(0.25))
    model.add(layers.Flatten())
    model.add(layers.Dense(1, activation='sigmoid'))
    return model

def build_gan(generator, discriminator):
    discriminator.trainable = False
    model = Sequential()
    model.add(generator)
    model.add(discriminator)
    return model

# Model parametreleri
img_shape = images.shape[1:]
latent_dim = 100

# Modelleri oluştur
discriminator = build_discriminator(img_shape)
discriminator.compile(loss='binary_crossentropy', optimizer=Adam(0.0001, 0.5), metrics=['accuracy'])

generator = build_generator(latent_dim)
gan = build_gan(generator, discriminator)
gan.compile(loss='binary_crossentropy', optimizer=Adam(0.0001, 0.5))

# Eğitim fonksiyonu
def train_gan(gan, generator, discriminator, images, latent_dim, epochs=10000, batch_size=32, save_interval=500):
    half_batch = batch_size // 2
    for epoch in range(epochs):
        # Gerçek görüntüleri seç
        idx = np.random.randint(0, images.shape[0], half_batch)
        real_images = images[idx]
        # Sahte görüntüler üret
        noise = np.random.normal(0, 1, (half_batch, latent_dim))
        fake_images = generator.predict(noise)
        # Discriminator'ı eğit
        d_loss_real = discriminator.train_on_batch(real_images, np.ones((half_batch, 1)))
        d_loss_fake = discriminator.train_on_batch(fake_images, np.zeros((half_batch, 1)))
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
        # GAN'ı eğit
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        valid_y = np.array([1] * batch_size)
        g_loss = gan.train_on_batch(noise, valid_y)
        # İlerlemeyi yazdır
        print(f"{epoch} [D loss: {d_loss[0]}, acc.: {100*d_loss[1]}%] [G loss: {g_loss}]")
        # Belirli aralıklarla görüntüleri kaydet
        if epoch % save_interval == 0:
            save_images(generator, epoch, latent_dim)

""" def save_images(generator, epoch, latent_dim, examples=10):
    noise = np.random.normal(0, 1, (examples, latent_dim))
    generated_images = generator.predict(noise)
    generated_images = 0.5 * generated_images + 0.5  # Rescale to [0, 1]

    # Her bir görüntüyü ayrı ayrı kaydet
    for i in range(examples):
        # Görüntünün simetrisini al
        img = generated_images[i]
        img_flipped = np.flip(img, axis=0)  # Y ekseninde simetri
        img_flipped = np.flip(img_flipped, axis=1)  # X ekseninde simetri

        plt.figure()
        plt.imshow(img_flipped, interpolation='nearest')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(f"gan_generated_image2_epoch_{epoch}_sample_{i}.png")
        plt.close()
 """

def save_images(generator, epoch, latent_dim, examples=10):
    noise = np.random.normal(0, 1, (examples, latent_dim))
    generated_images = generator.predict(noise)
    generated_images = 0.5 * generated_images + 0.5  # Rescale to [0, 1]

    # Her bir görüntüyü ayrı ayrı kaydet
    for i in range(examples):
        # Görüntünün x ekseninde simetrisini al
        img = generated_images[i]
        img_flipped_x = np.flip(img, axis=1)  # X ekseninde simetri

        # Üst satır: Orijinal görüntü + x ekseninde simetrisi
        top_row = np.concatenate((img, img_flipped_x), axis=1)

        # Alt satır: Üst satırın y ekseninde simetrisi
        bottom_row = np.flip(top_row, axis=0)

        # Tüm görüntüyü birleştir
        pattern_image = np.concatenate((top_row, bottom_row), axis=0)

        # Görüntüyü kaydet
        plt.figure()
        plt.imshow(pattern_image, interpolation='nearest')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(f"gan_patternSON_epoch_{epoch}_sample_{i}.png")
        plt.close()


# GAN'ı eğit
train_gan(gan, generator, discriminator, images, latent_dim, epochs=20000, batch_size=8, save_interval=500)