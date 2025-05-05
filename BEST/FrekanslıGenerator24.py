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


def block_homogeneity_loss(y_pred, block_size=8):
    """
    8x8 blokların içindeki piksellerin homojen olmasını sağlamak için loss.
    Her bloğun varyansını minimize eder (siyah/beyaz bloklar için).
    """
    # Gri tonlamaya çevir (RGB ise)
    if y_pred.shape[-1] == 3:
        y_pred = tf.reduce_mean(y_pred, axis=-1, keepdims=True)  # RGB -> Gri
    
    # Bloklara böl ve varyans hesapla
    batch_size = tf.shape(y_pred)[0]
    height = tf.shape(y_pred)[1]
    width = tf.shape(y_pred)[2]
    
    # Blok sayısı
    blocks_h = height // block_size
    blocks_w = width // block_size
    
    # Bloklara ayır: [batch, blok_sayısı_h, blok_sayısı_w, block_size, block_size]
    blocks = tf.reshape(y_pred, 
                        [batch_size, blocks_h, block_size, blocks_w, block_size, 1])
    blocks = tf.transpose(blocks, [0, 1, 3, 2, 4, 5])  # Doğru sıralama için
    
    # Varyans hesapla: [batch, blok_sayısı_h, blok_sayısı_w]
    block_var = tf.math.reduce_variance(blocks, axis=[3,4,5])
    
    # Ortalama varyans kaybı
    loss = tf.reduce_mean(block_var)
    return loss



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
        img = load_img(img_path)
        img = img_to_array(img)
        height, width, _ = img.shape
        #img = img[height//2:, width//2:, :]  # Crop the image
        img = img[height//2:, width//2:, :]

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
image_folder = r"C:\Users\atade\Desktop\14348_VERi\resim128_NET"
csv_folder = r"C:\Users\atade\Desktop\14348_VERi\Tüm_csv"
images, frequencies = load_images_and_frequencies(image_folder, csv_folder)

# Normalize frequencies to [-1, 1]
frequencies = (frequencies - np.mean(frequencies)) / np.std(frequencies)

def build_generator(latent_dim):
    noise = layers.Input(shape=(latent_dim,))
    freq = layers.Input(shape=(1,))
    merged = layers.Concatenate()([noise, freq])

    x = layers.Dense(128 * 16 * 16)(merged)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.BatchNormalization(momentum=0.8)(x)
    x = layers.Reshape((16, 16, 128))(x)

    def add_freq_info(x, freq, spatial_size):
        freq_map = layers.Dense(spatial_size * spatial_size)(freq)
        freq_map = layers.Reshape((spatial_size, spatial_size, 1))(freq_map)
        x = layers.Concatenate()([x, freq_map])
        return x

    # Katman 1: 16x16 -> 32x32
    x = add_freq_info(x, freq, 16)
    x = layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same')(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.BatchNormalization(momentum=0.8)(x)

    # Katman 2: 32x32 -> 64x64
    x = add_freq_info(x, freq, 32)
    x = layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same')(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.BatchNormalization(momentum=0.8)(x)

    # Ekstra Katman (64x64 -> 64x64)
    x = add_freq_info(x, freq, 64)
    x = layers.Conv2D(128, (3, 3), padding='same')(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.BatchNormalization(momentum=0.8)(x)

    # Çıkış Katmanı
    x = layers.Conv2D(3, (3, 3), activation='tanh', padding='same')(x)

    return Model([noise, freq], x)



# Discriminator Model
def build_discriminator(img_shape):
    img = layers.Input(shape=img_shape)
    freq = layers.Input(shape=(1,))
    
    # Process the frequency map (expand it first)
    freq_expanded = layers.Dense(img_shape[0] * img_shape[1])(freq)
    freq_expanded = layers.Reshape((img_shape[0], img_shape[1], 1))(freq_expanded)
    
    # Upsample the frequency map to match the image dimensions (256x256)
    freq_expanded = layers.UpSampling2D(size=(img_shape[0] // 64, img_shape[1] // 64))(freq_expanded)
    
    # Concatenate the image and upsampled frequency map
    merged = layers.Concatenate()([img, freq_expanded])
    
    # Add convolutional layers
    x = layers.Conv2D(32, (3, 3), strides=(2, 2), padding='same')(merged)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Dropout(0.5)(x)
    
    x = layers.Conv2D(64, (3, 3), strides=(2, 2), padding='same')(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Dropout(0.5)(x)
    
    x = layers.Flatten()(x)
    x = layers.Dense(1, activation='sigmoid')(x)
    
    return Model([img, freq], x)



# cGAN Modeli
def build_gan(generator, discriminator, latent_dim, lambda_symmetry=10, lambda_homogeneity=5):
    discriminator.trainable = False

    noise = layers.Input(shape=(latent_dim,))
    freq = layers.Input(shape=(1,))
    img = generator([noise, freq])
    valid = discriminator([img, freq])

    # Mevcut kayıplar
    symmetry_penalty = diagonal_symmetry_loss(img)  
    valid_loss = K.binary_crossentropy(K.ones_like(valid), valid)
    valid_loss = K.mean(valid_loss)  

    # Yeni blok homojenliği kaybı
    homogeneity_penalty = block_homogeneity_loss(img)  # 🔹 Yeni eklenen loss

    # Toplam kayıp (homojenlik kaybını da ekle)
    total_loss = valid_loss + lambda_symmetry * symmetry_penalty + lambda_homogeneity * homogeneity_penalty

    gan = Model([noise, freq], valid)
    gan.add_loss(total_loss)
    gan.compile(optimizer=Adam(0.0005, 0.5))
    
    return gan
# Model parameters
# Model parameters
img_shape = images.shape[1:]
latent_dim = 32

# Build models
discriminator = build_discriminator(img_shape)
discriminator.compile(loss='binary_crossentropy', optimizer=Adam(0.0001, 0.5), metrics=['accuracy'])

generator = build_generator(latent_dim)
gan = build_gan(generator, discriminator, latent_dim, lambda_symmetry=10, lambda_homogeneity=5)  # 🔹 lambda_homogeneity eklendi
gan.compile(loss='binary_crossentropy', optimizer=Adam(0.0008, 0.5))

# Training function
def train_gan(gan, generator, discriminator, images, frequencies, latent_dim, epochs=10000, batch_size=16, save_interval=500):
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
    generated_images = 0.5 * generated_images + 0.5  # Rescale to [0, 1]

    for i in range(examples):
        img = generated_images[i]  # Sağ alt çeyrek
        img_flipped_x = np.flip(img, axis=0)  # Y ekseninde flip (aşağı yukarı)

        # Sağ alt ve sağ üstü birleştir
        right_half = np.concatenate((img_flipped_x, img), axis=0)

        right_half_flipped_y = np.flip(right_half, axis=1)  # X ekseninde flip (sağ-sol)

        # Sağ ve sol yarıları birleştir
        pattern_image = np.concatenate((right_half_flipped_y, right_half), axis=1)

        plt.figure()
        plt.imshow(pattern_image, interpolation='nearest')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(f"TALHAgan_patternSON_epoch_{epoch}_sample_{i}.png")
        plt.close()

# Train GAN model
train_gan(gan, generator, discriminator, images, frequencies, latent_dim)
generator.save("Frekanslıgenerator_model28_nisan.h5")