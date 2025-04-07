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
    Üretilen 64x64 görüntünün çapraz simetrik olmasını sağlamak için loss fonksiyonu.
    Orjinal görüntü ve transpozu arasındaki farkı minimize eder.
    """
    # Transpose işlemi ile çapraz simetri yansıtması
    y_pred_transposed = tf.transpose(y_pred, perm=[0, 2, 1, 3])
    
    # Simetri farkını hesapla (mutlak fark)
    symmetry_diff = tf.abs(y_pred - y_pred_transposed)
    
    # Köşegen maskesi oluştur (köşegen dışındakiler 1, köşegendekiler 0.5)
    height = tf.shape(y_pred)[1]
    mask = 1.0 - 0.5 * tf.eye(height)  # Köşegen elemanlarına 0.5 ağırlık
    
    # Maskeyi genişlet (batch ve channel boyutları ekle)
    mask = tf.expand_dims(mask, axis=0)  # batch boyutu
    mask = tf.expand_dims(mask, axis=-1)  # channel boyutu
    
    # Ağırlıklı simetri kaybı hesapla
    weighted_diff = symmetry_diff * mask
    symmetry_loss = tf.reduce_mean(weighted_diff)
    
    return symmetry_loss

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
        img = load_img(img_path,color_mode='grayscale')
        img = img_to_array(img)
        height, width, _ = img.shape
        img = img[height//2:, width//2:, :]  # Crop the image
        img = tf.image.resize(img, img_size,method='nearest')  # Resize the image
        img = tf.cast(img > 127, tf.float32)  # sadece 0 ve 1 olacak şekilde binarize et
 # Normalize to [-1, 1]
        images.append(img)
    
    print(f"Yüklenen resim sayısı: {len(images)}, frekans sayısı: {len(frequencies)}")
    
    # Resim ve CSV sayısı eşit mi kontrol et
    if len(images) != len(frequencies):
        raise ValueError(f"Resim sayısı ({len(images)}) ve CSV sayısı ({len(frequencies)}) eşit değil!")
    
    print(f"Örnek frekans değerleri: {np.unique(frequencies)}")  # Benzersiz frekansları göster
    return np.array(images), np.array(frequencies)

# Load images and frequencies
image_folder = r"C:\Users\atade\Desktop\12386_veri\resim128_opencv_rgb"
csv_folder = r"C:\Users\atade\Desktop\12386_veri\Tüm_csv"
images, frequencies = load_images_and_frequencies(image_folder, csv_folder)

# Normalize frequencies to [-1, 1]
frequencies = (frequencies - np.mean(frequencies)) / np.std(frequencies)

class BinaryActivation(layers.Layer):
    def __init__(self, threshold=0.9):
        super(BinaryActivation, self).__init__()
        self.threshold = threshold

    def call(self, inputs):
        binary = tf.where(inputs > self.threshold, 1.0, 0.0)
        return inputs + tf.stop_gradient(binary - inputs)


def build_generator(latent_dim):
    noise = layers.Input(shape=(latent_dim,))
    freq = layers.Input(shape=(1,))
    merged = layers.Concatenate()([noise, freq])
    
    x = layers.Dense(128 * 16 * 16)(merged)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.BatchNormalization(momentum=0.8)(x)
    x = layers.Reshape((16, 16, 128))(x)
    
    x = layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same')(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.BatchNormalization(momentum=0.8)(x)
    
    x = layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same')(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.BatchNormalization(momentum=0.8)(x)
    
    # Son katman ve binary aktivasyon
    x = layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x) # Tek kanal grayscale
    x = BinaryActivation()(x)  
    
    return Model([noise, freq], x)

def build_discriminator(img_shape):
    # img_shape = (64, 64, 3) -> 1 kanal olacak
    img = layers.Input(shape=(64, 64, 1))  # ya da img_shape

    freq = layers.Input(shape=(1,))
    
    # Girdiyi binary yap (eğer değilse)
    binary_img = BinaryActivation()(img)
    
    freq_expanded = layers.Dense(img_shape[0] * img_shape[1])(freq)
    freq_expanded = layers.Reshape((img_shape[0], img_shape[1], 1))(freq_expanded)
    
    merged = layers.Concatenate()([binary_img, freq_expanded])
    
    x = layers.Conv2D(64, (3, 3), strides=(2, 2), padding='same')(merged)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.Dropout(0.3)(x)
    
    x = layers.Conv2D(128, (3, 3), strides=(2, 2), padding='same')(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.Dropout(0.3)(x)
    
    x = layers.Flatten()(x)
    x = layers.Dense(1, activation='sigmoid')(x)
    
    return Model([img, freq], x)

def build_gan(generator, discriminator, latent_dim, lambda_symmetry=10):
    discriminator.trainable = False

    noise = layers.Input(shape=(latent_dim,))
    freq = layers.Input(shape=(1,))
    img = generator([noise, freq])
    valid = discriminator([img, freq])

    # Simetri Cezası
    symmetry_penalty = diagonal_symmetry_loss(img)  
    
    # Binary Crossentropy Loss
    valid_loss = K.binary_crossentropy(K.ones_like(valid), valid)
    valid_loss = K.mean(valid_loss)  

    # Toplam Kaybı Hesapla
    total_loss = valid_loss + lambda_symmetry * symmetry_penalty

    # Modeli oluştur ve derle
    gan = Model([noise, freq], valid)
    gan.add_loss(total_loss)
    gan.compile(optimizer=Adam(0.0001, 0.5))

    return gan

# Model parameters
img_shape = (64, 64, 1)  # Görüntü boyutları
latent_dim = 100  # Latent dim boyutu

# Build models
discriminator = build_discriminator(img_shape)
discriminator.compile(loss='binary_crossentropy', optimizer=Adam(0.0001, 0.5), metrics=['accuracy'])

generator = build_generator(latent_dim)
gan = build_gan(generator, discriminator, latent_dim)
gan.compile(loss='binary_crossentropy', optimizer=Adam(0.0001, 0.5))

def train_gan(gan, generator, discriminator, images, frequencies, latent_dim, epochs=20000, batch_size=8, save_interval=500):
    half_batch = batch_size // 2
    for epoch in range(epochs):
        # Select real images and corresponding frequencies
        idx = np.random.randint(0, images.shape[0], half_batch)
        real_images = images[idx]
        real_frequencies = frequencies[idx]
        
        # Generate fake images
        noise = np.random.normal(0, 1, (half_batch, latent_dim))
        fake_frequencies = np.random.choice(frequencies, half_batch)
        fake_images = generator.predict([noise, fake_frequencies])
        
        # Train discriminator
        d_loss_real = discriminator.train_on_batch([real_images, real_frequencies], np.ones((half_batch, 1)))
        d_loss_fake = discriminator.train_on_batch([fake_images, fake_frequencies], np.zeros((half_batch, 1)))
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
        
        # Train generator
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        valid_y = np.array([1] * batch_size)
        sampled_frequencies = np.random.choice(frequencies, batch_size)
        g_loss = gan.train_on_batch([noise, sampled_frequencies], valid_y)
        
        # Print progress
        print(f"{epoch} [D loss: {d_loss[0]}, acc.: {100*d_loss[1]}%] [G loss: {g_loss}]")
        
        # Save images at intervals
        if epoch % save_interval == 0:
            save_images(generator, epoch, latent_dim, frequencies)

def save_images(generator, epoch, latent_dim, frequencies, examples=4):
    noise = np.random.normal(0, 1, (examples, latent_dim))
    sampled_frequencies = np.random.choice(frequencies, examples)
    generated_images = generator.predict([noise, sampled_frequencies])

    # 0-1 --> 0-255 çevir
    generated_images = (generated_images * 255).astype(np.uint8)

    for i in range(examples):
        # Tek kanallı grayscale görüntüyü RGB'ye dönüştür (aynı kanal 3 kere)
        gray = generated_images[i, :, :, 0]
        rgb_image = np.stack([gray] * 3, axis=-1)

        # Simetrik pattern oluşturma (opsiyonel)
        img_flipped_x = np.flip(rgb_image, axis=1)
        top_row = np.concatenate((rgb_image, img_flipped_x), axis=1)
        bottom_row = np.flip(top_row, axis=0)
        pattern_image = np.concatenate((top_row, bottom_row), axis=0)

        plt.figure()
        plt.imshow(pattern_image, cmap='gray', interpolation='nearest', vmin=0, vmax=255)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(f"binary_ACDgan_epoch_{epoch}_sample_{i}.png")
        plt.close()


# Train GAN model
train_gan(gan, generator, discriminator, images, frequencies, latent_dim)
generator.save("binary_generator_model.h5")